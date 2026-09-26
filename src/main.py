import argparse
from pathlib import Path
import sys
import json
import pandas as pd

from .config import TEST_FILES
from .data_loader import load_test_data
from .inference import run_test_inference
from .threshold_optimization import apply_threshold
from .entity_matching import write_candidate_pairs, write_matching_results, verify_output_subset

def run_submission_pipeline(data_root: Path, output_dir: Path, model_dir: Path):
    test_dir = data_root / "test"
    
    print("1. Loading Input Test Data...")
    try:
        test_data = load_test_data(test_dir, TEST_FILES)
    except Exception as e:
        print(f"Error loading test data: {e}")
        sys.exit(1)
        
    print("2-6. Preprocessing, Candidate Generation, Feature Engineering, and ML Inference...")
    test_cands, test_df, all_source1_ids = run_test_inference(test_data, model_dir)
    
    print("7. Loading Final Threshold...")
    try:
        with open(model_dir / "final_threshold.json", "r") as f:
            thresh_meta = json.load(f)
        threshold = thresh_meta["threshold"]
    except Exception as e:
        print(f"Error loading final_threshold.json, falling back to threshold.json: {e}")
        with open(model_dir / "threshold.json", "r") as f:
            thresh_meta = json.load(f)
        threshold = thresh_meta["threshold"]
        
    print(f"Using Threshold: {threshold}")
    
    print("8. Generating Final Matches...")
    if not test_df.empty:
        test_df = test_df.rename(columns={'s1_id': 'source1_entity_id', 'cand_id': 'candidate_entity_id'})
        test_matches = apply_threshold(test_df, threshold)
    else:
        test_matches = {s1_id: set() for s1_id in all_source1_ids}
        
    for s1_id in all_source1_ids:
        if s1_id not in test_matches:
            test_matches[s1_id] = set()
            
    print("9. Validating Output Constraints...")
    if not verify_output_subset(test_matches, test_cands):
        print("CRITICAL ERROR: Matches are not a strict subset of candidate pairs.")
        sys.exit(1)
        
    print("10. Writing Final Outputs...")
    output_dir.mkdir(parents=True, exist_ok=True)
    cand_out = output_dir / "candidate_pairs.tsv"
    match_out = output_dir / "matching_results.tsv"
    
    write_candidate_pairs(test_cands, all_source1_ids, cand_out)
    write_matching_results(test_matches, all_source1_ids, match_out)
    
    print("Validating with official utils validator if present...")
    utils_val = Path("utils/validate_submission.py")
    if utils_val.exists():
        import subprocess
        subprocess.run([sys.executable, str(utils_val)], check=False)
        
    print("Pipeline Complete! Output ready in output/")

def main():
    parser = argparse.ArgumentParser(description="Final Reproducibility Pipeline")
    parser.add_argument("--data-root", type=str, default="dataset", help="Root directory containing /test")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    model_dir = Path("models")
    
    run_submission_pipeline(data_root, output_dir, model_dir)

if __name__ == "__main__":
    main()
