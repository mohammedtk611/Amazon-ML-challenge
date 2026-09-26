import argparse
from pathlib import Path
import sys
import pandas as pd
import json

from src.config import TRAIN_FILES, TEST_FILES
from src.data_loader import load_train_data, load_test_data
from src.training import build_ground_truth_index
from src.threshold_optimization import search_thresholds, select_best_threshold, apply_threshold
from src.inference import run_test_inference
from src.entity_matching import write_candidate_pairs, write_matching_results, verify_output_subset

def run_phase6(train_data: dict, test_data: dict, output_dir: Path, model_dir: Path):
    print("Loading Validation Predictions...")
    val_pred_path = output_dir / "validation_predictions.tsv"
    if not val_pred_path.exists():
        print(f"Error: {val_pred_path} not found. Run Phase 5 first.")
        sys.exit(1)
        
    val_df = pd.read_csv(val_pred_path, sep='\t')
    
    print("Reconstructing Entity-Level Ground Truth...")
    gt_df = train_data["ground_truth"]
    gt_index = build_ground_truth_index(gt_df)
    
    print("Optimizing Threshold via Entity-Level F0.5...")
    results_df = search_thresholds(val_df, gt_index)
    results_df.to_csv(output_dir / "threshold_results.csv", index=False)
    
    best_threshold = select_best_threshold(results_df)
    best_row = results_df.loc[results_df['threshold'] == best_threshold].iloc[0]
    
    print(f"Selected Validation Threshold: {best_threshold:.2f}")
    print(f"Validation F0.5: {best_row['f0_5']:.4f}")
    
    threshold_meta = {
        "threshold": best_threshold,
        "metric": "F0.5",
        "validation_f0_5": best_row['f0_5'],
        "selection_method": "entity_level_validation_grid_search"
    }
    with open(model_dir / "threshold.json", "w") as f:
        json.dump(threshold_meta, f, indent=4)
        
    print("Running Test Inference...")
    test_cands, test_df, all_source1_ids = run_test_inference(test_data, model_dir)
    
    print("Applying Threshold to Test Predictions...")
    if not test_df.empty:
        # We must align columns dynamically to the validation threshold application
        # test_df has 'source1_entity_id', 'candidate_entity_id', 'match_probability'
        test_df = test_df.rename(columns={'s1_id': 'source1_entity_id', 'cand_id': 'candidate_entity_id'})
        test_matches = apply_threshold(test_df, best_threshold)
    else:
        test_matches = {s1_id: set() for s1_id in all_source1_ids}
        
    # Ensure all S1 entities exist in matches dict
    for s1_id in all_source1_ids:
        if s1_id not in test_matches:
            test_matches[s1_id] = set()
            
    print("Verifying Subset Constraints...")
    if not verify_output_subset(test_matches, test_cands):
        print("CRITICAL ERROR: Matches are not a subset of candidates.")
        sys.exit(1)
        
    print("Writing Final Challenge Outputs...")
    cand_out = output_dir / "candidate_pairs.tsv"
    write_candidate_pairs(test_cands, all_source1_ids, cand_out)
    
    match_out = output_dir / "matching_results.tsv"
    write_matching_results(test_matches, all_source1_ids, match_out)
    
    # Custom Validator Checks
    df_match = pd.read_csv(match_out, sep='\t')
    df_cand = pd.read_csv(cand_out, sep='\t')
    
    assert len(df_match) == len(all_source1_ids), "Missing S1 entities in match output!"
    assert len(df_cand) == len(all_source1_ids), "Missing S1 entities in candidate output!"
    assert df_match['source1_entity_id'].is_unique, "Duplicate S1 IDs in match output!"
    assert df_cand['source1_entity_id'].is_unique, "Duplicate S1 IDs in candidate output!"
    
    # Statistics
    total_matches = df_match['candidate_entity_ids'].fillna("").apply(lambda x: len(x.split(";")) if x else 0).sum()
    total_cands = df_cand['candidate_entity_ids'].fillna("").apply(lambda x: len(x.split(";")) if x else 0).sum()
    s1_with_cands = (df_cand['candidate_entity_ids'].fillna("") != "").sum()
    
    stats_path = output_dir / "final_matching_stats.md"
    with open(stats_path, "w") as f:
        f.write("# Final Matching Statistics\n\n")
        f.write(f"- **Total Test S1 Entities:** {len(all_source1_ids)}\n")
        f.write(f"- **Total Candidate Pairs Generated:** {total_cands}\n")
        f.write(f"- **Total Predicted Matches:** {total_matches}\n")
        f.write(f"- **S1 Entities with at least 1 candidate:** {s1_with_cands}\n")
        f.write(f"- **Selected Threshold:** {best_threshold:.2f}\n")
        
    print(f"Phase 6 complete. Statistics saved to {stats_path}")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    output_dir = Path(args.output_dir)
    model_dir = Path("models")
    
    try:
        train_data = load_train_data(train_dir, TRAIN_FILES)
        test_data = load_test_data(test_dir, TEST_FILES)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
        
    run_phase6(train_data, test_data, output_dir, model_dir)

if __name__ == "__main__":
    main()
