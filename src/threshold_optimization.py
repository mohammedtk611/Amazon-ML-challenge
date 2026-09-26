import pandas as pd
import numpy as np
from typing import Dict, Set, Tuple, List
import json

def calculate_entity_metrics(predicted: Set[str], actual: Set[str]) -> Tuple[float, float, float]:
    """
    Calculates Precision, Recall, and F0.5 for a single Source 1 entity.
    Handles empty ground-truth securely based on standard challenge rules.
    """
    if len(actual) == 0 and len(predicted) == 0:
        return 1.0, 1.0, 1.0
    elif len(actual) == 0 and len(predicted) > 0:
        return 0.0, 0.0, 0.0
    elif len(actual) > 0 and len(predicted) == 0:
        return 0.0, 0.0, 0.0
        
    tp = len(predicted.intersection(actual))
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / ((0.25 * precision) + recall)
        
    return precision, recall, f05

def evaluate_predictions(predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]]) -> Dict[str, float]:
    """Macro-averaged metrics across all S1 entities."""
    total_p, total_r, total_f05 = 0.0, 0.0, 0.0
    n = len(ground_truth)
    
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f05": 0.0}
        
    for s1_id, actual in ground_truth.items():
        pred = predictions.get(s1_id, set())
        p, r, f = calculate_entity_metrics(pred, actual)
        total_p += p
        total_r += r
        total_f05 += f
        
    return {
        "precision": total_p / n,
        "recall": total_r / n,
        "f05": total_f05 / n
    }

def apply_threshold(val_df: pd.DataFrame, threshold: float) -> Dict[str, Set[str]]:
    """Applies a probability threshold to validation pairs and groups by S1."""
    filtered = val_df[val_df['match_probability'] >= threshold]
    
    # Use groupby to collect candidates
    predictions = {}
    
    # Ensure all S1 from the dataframe have an entry, even if empty
    for s1_id in val_df['source1_entity_id'].unique():
        predictions[s1_id] = set()
        
    if not filtered.empty:
        grouped = filtered.groupby('source1_entity_id')['candidate_entity_id'].apply(set).to_dict()
        for k, v in grouped.items():
            predictions[k] = v
            
    return predictions

def search_thresholds(val_df: pd.DataFrame, ground_truth: Dict[str, Set[str]]) -> pd.DataFrame:
    """Performs a grid search for the optimal threshold (0.01 steps)."""
    results = []
    
    thresholds = np.arange(0.0, 1.01, 0.01)
    
    # Pre-extract unique S1 IDs to ensure we evaluate over the exact validation set
    val_s1_ids = set(val_df['source1_entity_id'].unique())
    val_gt = {k: v for k, v in ground_truth.items() if k in val_s1_ids}
    
    for t in thresholds:
        preds = apply_threshold(val_df, t)
        
        # Calculate matching stats
        matched_s1_count = sum(1 for p in preds.values() if len(p) > 0)
        total_pred_matches = sum(len(p) for p in preds.values())
        avg_matches = total_pred_matches / max(len(preds), 1)
        
        metrics = evaluate_predictions(preds, val_gt)
        
        results.append({
            "threshold": round(t, 2),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f0_5": metrics["f05"],
            "predicted_match_count": total_pred_matches,
            "matched_source1_count": matched_s1_count,
            "average_matches_per_source1": avg_matches
        })
        
    return pd.DataFrame(results)

def select_best_threshold(results_df: pd.DataFrame) -> float:
    """Selects the threshold that maximizes F0.5."""
    best_row = results_df.loc[results_df['f0_5'].idxmax()]
    return float(best_row['threshold'])
