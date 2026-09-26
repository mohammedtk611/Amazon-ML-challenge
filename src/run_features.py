import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import csv

from .config import TRAIN_FILES
from .data_loader import load_train_data
from .preprocessing import apply_preprocessing
from .features import build_pair_features

def run_feature_generation(train_data: dict, candidate_file: Path, output_dir: Path):
    print("Preparing normalized datasets for feature generation...")
    
    # Preprocess
    s1 = apply_preprocessing(train_data["source1"]).set_index("entity_id")
    s2 = apply_preprocessing(train_data["source2"]).set_index("entity_id")
    s3 = apply_preprocessing(train_data["source3"]).set_index("entity_id")
    gt = train_data.get("ground_truth", pd.DataFrame())
    
    gt_pairs = set()
    if not gt.empty:
        for _, row in gt.iterrows():
            s1_id = row['source1_entity_id']
            matches = [m for m in str(row['matched_entity_ids']).split(",") if m]
            for m in matches:
                gt_pairs.add((s1_id, m))
                
    if not candidate_file.exists():
        print(f"Error: Candidate file {candidate_file} not found. Run blocking first.")
        return

    features_list = []
    labels = []
    pair_ids = []
    hard_negatives = []
    
    total_pairs = 0
    start_time = time.time()
    
    print("Generating features...")
    # Read candidates
    with open(candidate_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            s1_id = row['source1_entity_id']
            cands = [c for c in row['candidate_entity_ids'].split(",") if c]
            
            if s1_id not in s1.index:
                continue
                
            s1_row = s1.loc[s1_id]
            
            for c_id in cands:
                if c_id in s2.index:
                    c_row = s2.loc[c_id]
                    c_source = 's2'
                elif c_id in s3.index:
                    c_row = s3.loc[c_id]
                    c_source = 's3'
                else:
                    continue
                    
                is_match = 1 if (s1_id, c_id) in gt_pairs else 0
                
                # We assume no provenance saved in TSV for now
                feats = build_pair_features(s1_row, c_row, cand_source=c_source, provenance=[])
                
                features_list.append(feats)
                labels.append(is_match)
                pair_ids.append((s1_id, c_id))
                total_pairs += 1
                
                # Hard negative criteria: High name similarity but not a match
                if not is_match and feats.get('name_jaro_winkler', 0) > 0.85:
                    hard_negatives.append({
                        "s1_id": s1_id,
                        "cand_id": c_id,
                        "s1_name": s1_row.get("business_name"),
                        "c_name": c_row.get("business_name"),
                        "s1_address": s1_row.get("business_address"),
                        "c_address": c_row.get("business_address"),
                        "jaro": feats.get('name_jaro_winkler')
                    })
                
                # For demo purposes, we break if it gets too large
                if total_pairs > 100000: # chunk boundary
                    pass
                    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Processed {total_pairs} pairs in {duration:.2f} seconds ({total_pairs/max(duration, 1):.2f} pairs/sec).")
    
    df_feats = pd.DataFrame(features_list)
    df_feats['is_match'] = labels
    
    # Generate feature statistics
    report_path = output_dir / "feature_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 4 — Pair Feature Engineering Report\n\n")
        f.write(f"**Total candidate pairs processed:** {total_pairs}\n")
        f.write(f"**Processing Time:** {duration:.2f} seconds\n")
        f.write(f"**Throughput:** {total_pairs/max(duration, 1):.2f} pairs/sec\n")
        f.write(f"**Total Features:** {df_feats.shape[1] - 1}\n\n")
        
        # Check for constant features
        constant_features = [col for col in df_feats.columns if col != 'is_match' and df_feats[col].nunique() <= 1]
        if constant_features:
            f.write("### Constant Features\n")
            f.write("These features had only one unique value across all pairs (may be due to missing provenance):\n")
            for col in constant_features:
                f.write(f"- {col}\n")
            f.write("\n")
            
        f.write("### MATCH vs NON-MATCH Distributions\n")
        f.write("Comparing key similarities:\n\n")
        
        key_features = ['name_jaro_winkler', 'name_token_jaccard', 'address_token_jaccard', 'address_levenshtein', 'postal_exact_match']
        for col in key_features:
            if col in df_feats.columns:
                match_median = df_feats[df_feats['is_match'] == 1][col].median()
                nonmatch_median = df_feats[df_feats['is_match'] == 0][col].median()
                
                f.write(f"**{col}**:\n")
                f.write(f"- MATCH Median: {match_median:.3f}\n")
                f.write(f"- NON-MATCH Median: {nonmatch_median:.3f}\n\n")
                
    print(f"Wrote {report_path}")
    
    # Generate hard negatives
    hn_path = output_dir / "potential_hard_negatives.md"
    with open(hn_path, "w", encoding="utf-8") as f:
        f.write("# Potential Hard Negatives\n\n")
        f.write("These pairs have high name similarity (>0.85 Jaro-Winkler) but are ground-truth non-matches.\n\n")
        
        for idx, hn in enumerate(hard_negatives[:50]): # limit to 50 for report
            f.write(f"### Example {idx+1}: {hn['s1_id']} <-> {hn['cand_id']}\n")
            f.write(f"- **Jaro-Winkler**: {hn['jaro']:.3f}\n")
            f.write(f"- **S1**: {hn['s1_name']} | {hn['s1_address']}\n")
            f.write(f"- **Cand**: {hn['c_name']} | {hn['c_address']}\n\n")
            
    print(f"Wrote {hn_path}")

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Pair Feature Engineering")
    parser.add_argument("--data-root", type=str, default="dataset", help="Root directory for the dataset")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for reports")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cand_file = output_dir / "candidate_pairs.tsv"

    print("Loading training datasets for feature engineering...")
    try:
        train_data = load_train_data(train_dir, TRAIN_FILES)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
        
    run_feature_generation(train_data, cand_file, output_dir)

if __name__ == "__main__":
    main()
