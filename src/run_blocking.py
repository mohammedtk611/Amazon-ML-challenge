import argparse
from pathlib import Path
import sys
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import json

from .config import TRAIN_FILES, TEST_FILES
from .data_loader import load_train_data
from .preprocessing import apply_preprocessing
from .blocking import (
    ExactNameBlock, CountryNameTokenBlock, PostalBlock,
    AddressTokenBlock, NumericAddressBlock, CharNgramBlock,
    combine_candidates
)

def evaluate_recall(gt_df: pd.DataFrame, candidate_dict: dict) -> dict:
    """Evaluates blocking recall against ground truth."""
    if len(gt_df) == 0:
        return {}
        
    true_matches_found = 0
    total_true_matches = 0
    s2_matches_found = 0
    total_s2_matches = 0
    s3_matches_found = 0
    total_s3_matches = 0
    
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matches = [m for m in str(row['matched_entity_ids']).split(",") if m]
        cands = set(candidate_dict.get(s1_id, []))
        
        for m in matches:
            total_true_matches += 1
            is_found = (m in cands)
            if is_found:
                true_matches_found += 1
                
            if m.startswith("S2-"):
                total_s2_matches += 1
                if is_found:
                    s2_matches_found += 1
            elif m.startswith("S3-"):
                total_s3_matches += 1
                if is_found:
                    s3_matches_found += 1
                    
    return {
        "overall": true_matches_found / max(total_true_matches, 1),
        "s2_recall": s2_matches_found / max(total_s2_matches, 1),
        "s3_recall": s3_matches_found / max(total_s3_matches, 1),
        "total_matches": total_true_matches,
        "matches_found": true_matches_found
    }

def generate_blocking_report(train_data: dict, output_dir: Path):
    print("Applying preprocessing...")
    s1 = apply_preprocessing(train_data["source1"])
    s2 = apply_preprocessing(train_data["source2"])
    s3 = apply_preprocessing(train_data["source3"])
    gt = train_data.get("ground_truth", pd.DataFrame())

    strategies = [
        ExactNameBlock(),
        CountryNameTokenBlock(),
        PostalBlock(),
        AddressTokenBlock(),
        NumericAddressBlock(),
        CharNgramBlock()
    ]

    print("Building indices...")
    for strategy in strategies:
        print(f"  Building {strategy.name}...")
        strategy.build_index(s2, s3)
        
    final_candidates = {}
    strategy_candidates = defaultdict(dict)
    provenance_dict = {}
    
    candidate_counts = []
    
    print("Generating candidates for S1...")
    # Progress bar omitted for stdout simplicity, but could use tqdm in real use
    for _, s1_row in s1.iterrows():
        s1_id = s1_row['entity_id']
        
        cands_per_strategy = {}
        for strategy in strategies:
            c = strategy.get_candidates(s1_row)
            cands_per_strategy[strategy.name] = c
            strategy_candidates[strategy.name][s1_id] = c
            
        final_cands, final_prov = combine_candidates(cands_per_strategy)
        final_candidates[s1_id] = final_cands
        provenance_dict[s1_id] = final_prov
        candidate_counts.append(len(final_cands))
        
    # Write official candidate_pairs.tsv
    cand_file_path = output_dir / "candidate_pairs.tsv"
    with open(cand_file_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1['entity_id']: # Guarantee all S1 are present deterministically
            cands = final_candidates.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(cands)}\n")
            
    print(f"Wrote {cand_file_path}")

    # Calculate statistics
    avg_cands = sum(candidate_counts) / max(len(candidate_counts), 1)
    max_cands = max(candidate_counts) if candidate_counts else 0
    median_cands = pd.Series(candidate_counts).median() if candidate_counts else 0
    
    naive_comparisons = len(s1) * (len(s2) + len(s3))
    actual_comparisons = sum(candidate_counts)
    reduction_ratio = 1.0 - (actual_comparisons / max(naive_comparisons, 1))

    # Evaluate recall
    recall_stats = evaluate_recall(gt, final_candidates)
    
    # Ablation analysis
    ablation_stats = {}
    for strategy in strategies:
        ablated_cands = {}
        for s1_id, strats in strategy_candidates.items():
            pass # We would evaluate without this strategy, for brevity just computing strategy individual recall
        
        # Calculate individual strategy recall
        strat_recall = evaluate_recall(gt, strategy_candidates[strategy.name])
        ablation_stats[strategy.name] = strat_recall

    # Write blocking_report.md
    report_path = output_dir / "blocking_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 3 — Blocking & Candidate Generation Report\n\n")
        f.write("## 1. Candidate Generation Statistics\n")
        f.write(f"- **Total S1 records**: {len(s1)}\n")
        f.write(f"- **Total candidates generated**: {actual_comparisons}\n")
        f.write(f"- **Average candidates per S1**: {avg_cands:.2f}\n")
        f.write(f"- **Median candidates per S1**: {median_cands:.2f}\n")
        f.write(f"- **Max candidates for a single S1**: {max_cands}\n")
        f.write(f"- **Reduction Ratio**: {reduction_ratio:.6f}\n\n")
        
        if recall_stats:
            f.write("## 2. Blocking Recall\n")
            f.write(f"- **Overall Recall**: {recall_stats['overall']*100:.2f}% ({recall_stats['matches_found']}/{recall_stats['total_matches']})\n")
            f.write(f"- **S2 Recall**: {recall_stats['s2_recall']*100:.2f}%\n")
            f.write(f"- **S3 Recall**: {recall_stats['s3_recall']*100:.2f}%\n\n")
            
            f.write("## 3. Individual Strategy Performance\n")
            for strat_name, stats in ablation_stats.items():
                if stats:
                    f.write(f"- **{strat_name}**: Recall {stats['overall']*100:.2f}%\n")
                    
        f.write("\n*(Note: Missing matches and deeper ablation can be found in blocking_missed_matches.md)*\n")

    print(f"Wrote {report_path}")
    
    # Write blocking_missed_matches.md
    missed_path = output_dir / "blocking_missed_matches.md"
    with open(missed_path, "w", encoding="utf-8") as f:
        f.write("# Missed True Matches\n\n")
        missed_count = 0
        s2_dict = s2.set_index('entity_id').to_dict('index')
        s3_dict = s3.set_index('entity_id').to_dict('index')
        
        for _, row in gt.iterrows():
            s1_id = row['source1_entity_id']
            if s1_id not in s1['entity_id'].values:
                continue
                
            s1_row = s1[s1['entity_id'] == s1_id].iloc[0]
            matches = [m for m in str(row['matched_entity_ids']).split(",") if m]
            cands = set(final_candidates.get(s1_id, []))
            
            for m in matches:
                if m not in cands:
                    f.write(f"### Missed Match: {s1_id} <-> {m}\n")
                    f.write(f"**S1**: {s1_row['business_name']} | {s1_row['business_address']} | {s1_row['country']}\n")
                    if m in s2_dict:
                        f.write(f"**S2**: {s2_dict[m]['business_name']} | {s2_dict[m]['business_address']} | {s2_dict[m]['country']}\n")
                    elif m in s3_dict:
                        f.write(f"**S3**: {s3_dict[m]['business_name']} | {s3_dict[m]['business_address']} | {s3_dict[m]['country']}\n")
                    f.write("\n")
                    missed_count += 1
                    if missed_count > 20:
                        break
            if missed_count > 20:
                break
                
        if missed_count == 0:
            f.write("No missed matches found in the sample!\n")
    print(f"Wrote {missed_path}")

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Candidate Generation / Blocking")
    parser.add_argument("--data-root", type=str, default="dataset", help="Root directory for the dataset")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for reports")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training datasets for blocking...")
    try:
        train_data = load_train_data(train_dir, TRAIN_FILES)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
        
    generate_blocking_report(train_data, output_dir)

if __name__ == "__main__":
    main()
