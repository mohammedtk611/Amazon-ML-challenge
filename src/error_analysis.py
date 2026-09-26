import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json

def generate_baseline_metrics(val_df: pd.DataFrame, ground_truth: Dict[str, Set[str]], best_threshold: float, output_path: Path):
    """Generates Phase 6 baseline metrics json."""
    from .threshold_optimization import evaluate_predictions, apply_threshold
    
    val_s1_ids = set(val_df['source1_entity_id'].unique())
    val_gt = {k: v for k, v in ground_truth.items() if k in val_s1_ids}
    
    preds = apply_threshold(val_df, best_threshold)
    metrics = evaluate_predictions(preds, val_gt)
    
    total_matches = sum(len(p) for p in preds.values())
    cand_count = len(val_df)
    
    # Calculate blocking recall
    gt_pairs = sum(len(v) for v in val_gt.values())
    if gt_pairs == 0:
        blocking_recall = 1.0
    else:
        # Find which true pairs are in val_df
        # Create a set of (s1, s2/s3) pairs from val_df
        val_pairs = set(zip(val_df['source1_entity_id'], val_df['candidate_entity_id']))
        true_found = 0
        for s1, cands in val_gt.items():
            for c in cands:
                if (s1, c) in val_pairs:
                    true_found += 1
        blocking_recall = true_found / gt_pairs

    stats = {
        "candidate_recall": blocking_recall,
        "validation_precision": metrics['precision'],
        "validation_recall": metrics['recall'],
        "validation_f0_5": metrics['f05'],
        "selected_threshold": best_threshold,
        "candidate_count": cand_count,
        "predicted_match_count": total_matches
    }
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    return stats

def run_error_analysis(val_df: pd.DataFrame, ground_truth: Dict[str, Set[str]], best_threshold: float, output_dir: Path):
    """Runs structural error analysis for Phase 7."""
    
    # Analyze Classifier Failures (FNs where pair IS in candidate set but prob < threshold)
    fn_list = val_df[(val_df['label'] == 1) & (val_df['match_probability'] < best_threshold)]
    with open(output_dir / "classifier_error_analysis.md", "w") as f:
        f.write("# Classifier Error Analysis\n\n")
        f.write(f"Total False Negatives in Candidate Set: {len(fn_list)}\n")
        
    # Analyze False Positives
    fp_list = val_df[(val_df['label'] == 0) & (val_df['match_probability'] >= best_threshold)]
    with open(output_dir / "entity_error_analysis.md", "w") as f:
        f.write("# False Positive Analysis\n\n")
        f.write(f"Total False Positives: {len(fp_list)}\n")
        
    # Blocking failures (Pairs in GT missing from val_df)
    val_s1_ids = set(val_df['source1_entity_id'].unique())
    val_pairs = set(zip(val_df['source1_entity_id'], val_df['candidate_entity_id']))
    blocking_misses = 0
    for s1, cands in ground_truth.items():
        if s1 not in val_s1_ids:
            continue
        for c in cands:
            if (s1, c) not in val_pairs:
                blocking_misses += 1
                
    with open(output_dir / "blocking_error_analysis.md", "w") as f:
        f.write("# Blocking Error Analysis\n\n")
        f.write(f"Total True Matches Missed by Blocking: {blocking_misses}\n")
        
    # Singleton, Multi-match, No-match analysis
    with open(output_dir / "singleton_analysis.md", "w") as f:
        f.write("# Singleton Analysis\n\nTo be filled by detailed row-level inspection.\n")
    with open(output_dir / "no_match_analysis.md", "w") as f:
        f.write("# No-Match Analysis\n\nTo be filled by detailed row-level inspection.\n")
