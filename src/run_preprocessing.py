import argparse
from pathlib import Path
import sys

from .config import TRAIN_FILES, TEST_FILES
from .data_loader import load_train_data, load_test_data
from .preprocessing import apply_preprocessing
import pandas as pd

def compute_collision_stats(df: pd.DataFrame, raw_col: str, norm_col: str) -> str:
    lines = []
    # Find distinct raw values mapping to the same normalized value
    # exclude empties
    valid = df[(df[raw_col] != "") & (df[norm_col] != "")]
    
    unique_raw_per_norm = valid.groupby(norm_col)[raw_col].nunique()
    collisions = unique_raw_per_norm[unique_raw_per_norm > 1]
    
    lines.append(f"- **Total distinct normalized values with collisions**: {len(collisions)}")
    
    # Show a few examples
    if len(collisions) > 0:
        lines.append("#### Examples of Collisions:")
        # Take top 5 with most collisions
        top_collisions = collisions.sort_values(ascending=False).head(5)
        for norm_val in top_collisions.index:
            raw_vals = valid[valid[norm_col] == norm_val][raw_col].unique()
            lines.append(f"- Normalized: **'{norm_val}'** <- Raw: {list(raw_vals)[:5]}")
            
    return "\n".join(lines)

def generate_normalization_report(train_data: dict, output_path: Path):
    lines = ["# Phase 2 — Normalization Report\n"]
    
    if not train_data:
        lines.append("No data available.")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return
        
    s1_raw = train_data["source1"]
    if len(s1_raw) == 0:
        lines.append("Source 1 is empty.")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return

    print("Applying preprocessing to Source 1 for report...")
    s1 = apply_preprocessing(s1_raw)
    
    lines.append("## 1. Before / After Examples\n")
    sample = s1.dropna(subset=['business_name']).head(10)
    for _, row in sample.iterrows():
        lines.append(f"**RAW NAME**: {row['business_name']}")
        lines.append(f"-> NORMALIZED: {row['business_name_normalized']}")
        lines.append(f"-> CORE: {row['business_name_core']}")
        lines.append(f"**RAW ADDRESS**: {row['business_address']}")
        lines.append(f"-> NORMALIZED: {row['business_address_normalized']}")
        lines.append("")
        
    lines.append("## 2. Extraction Statistics\n")
    postal_count = (s1['business_address_postal'] != "").sum()
    numeric_count = s1['business_address_numeric_tokens'].apply(lambda x: len(x) > 0).sum()
    lines.append(f"- **Postal codes extracted**: {postal_count} ({(postal_count/len(s1)*100):.2f}%)")
    lines.append(f"- **Addresses with numeric tokens**: {numeric_count} ({(numeric_count/len(s1)*100):.2f}%)")
    lines.append("")

    lines.append("## 3. Collision Analysis\n")
    lines.append("### Business Name Collisions (Normalized)")
    lines.append(compute_collision_stats(s1, "business_name", "business_name_normalized"))
    lines.append("")
    
    lines.append("### Business Name Collisions (Core)")
    lines.append(compute_collision_stats(s1, "business_name", "business_name_core"))
    lines.append("")
    
    lines.append("### Address Collisions")
    lines.append(compute_collision_stats(s1, "business_address", "business_address_normalized"))
    lines.append("")
    
    lines.append("## 4. Ground-Truth-Aware Analysis\n")
    gt = train_data.get("ground_truth", pd.DataFrame())
    if len(gt) > 0:
        print("Applying preprocessing to Source 2 for GT analysis...")
        s2 = apply_preprocessing(train_data["source2"])
        
        # Build dictionary for quick lookup
        s1_dict = s1.set_index('entity_id')[['business_name', 'business_address', 'business_name_normalized', 'business_address_normalized']].to_dict('index')
        s2_dict = s2.set_index('entity_id')[['business_name', 'business_address', 'business_name_normalized', 'business_address_normalized']].to_dict('index')
        
        raw_name_match = 0
        norm_name_match = 0
        raw_addr_match = 0
        norm_addr_match = 0
        total_pairs = 0
        
        for _, row in gt.iterrows():
            s1_id = row["source1_entity_id"]
            if s1_id not in s1_dict:
                continue
            matches = [m for m in str(row["matched_entity_ids"]).split(",") if m]
            for m in matches:
                if m in s2_dict:
                    total_pairs += 1
                    s1_r = s1_dict[s1_id]
                    s2_r = s2_dict[m]
                    
                    if s1_r['business_name'] == s2_r['business_name'] and s1_r['business_name'] != "":
                        raw_name_match += 1
                    if s1_r['business_name_normalized'] == s2_r['business_name_normalized'] and s1_r['business_name_normalized'] != "":
                        norm_name_match += 1
                        
                    if s1_r['business_address'] == s2_r['business_address'] and s1_r['business_address'] != "":
                        raw_addr_match += 1
                    if s1_r['business_address_normalized'] == s2_r['business_address_normalized'] and s1_r['business_address_normalized'] != "":
                        norm_addr_match += 1
                        
        if total_pairs > 0:
            lines.append(f"- **Total True Match S1-S2 Pairs evaluated**: {total_pairs}")
            lines.append(f"- **Raw exact name agreement**: {(raw_name_match/total_pairs*100):.2f}%")
            lines.append(f"- **Normalized exact name agreement**: {(norm_name_match/total_pairs*100):.2f}%")
            lines.append(f"- **Raw exact address agreement**: {(raw_addr_match/total_pairs*100):.2f}%")
            lines.append(f"- **Normalized exact address agreement**: {(norm_addr_match/total_pairs*100):.2f}%")
        else:
            lines.append("No valid S1-S2 pairs found in Ground Truth.")
    else:
        lines.append("No ground truth available for analysis.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
def main():
    parser = argparse.ArgumentParser(description="Phase 2: Normalization and Preprocessing")
    parser.add_argument("--data-root", type=str, default="dataset", help="Root directory for the dataset")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for reports")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training datasets for normalization report...")
    try:
        train_data = load_train_data(train_dir, TRAIN_FILES)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Normalization report generation aborted.")
        sys.exit(1)
        
    print("Generating Normalization Report...")
    output_path = output_dir / "normalization_report.md"
    generate_normalization_report(train_data, output_path)
    print(f"Normalization report generated at {output_path}")

if __name__ == "__main__":
    main()
