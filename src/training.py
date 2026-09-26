import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, precision_score, recall_score, f1_score, confusion_matrix
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any, Set

from .preprocessing import apply_preprocessing
from .blocking import (
    ExactNameBlock, CountryNameTokenBlock, PostalBlock,
    AddressTokenBlock, NumericAddressBlock, CharNgramBlock,
    combine_candidates
)
from .features import build_pair_features

def create_entity_split(s1: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits S1 into train and validation sets based on entity_id."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(s1, groups=s1['entity_id']))
    
    train_s1 = s1.iloc[train_idx].copy()
    val_s1 = s1.iloc[val_idx].copy()
    
    # Assert no leakage
    overlap = set(train_s1['entity_id']).intersection(set(val_s1['entity_id']))
    assert len(overlap) == 0, f"Entity leakage detected: {len(overlap)} entities in both splits."
    
    return train_s1, val_s1

def generate_training_candidates(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> Dict[str, Tuple[List[str], Dict[str, List[str]]]]:
    """Generates candidates using Phase 3 blocking strategies."""
    strategies = [
        ExactNameBlock(), CountryNameTokenBlock(), PostalBlock(),
        AddressTokenBlock(), NumericAddressBlock(), CharNgramBlock()
    ]
    
    for strategy in strategies:
        strategy.build_index(s2, s3)
        
    results = {}
    for _, row in s1.iterrows():
        s1_id = row['entity_id']
        cands_per_strategy = {}
        for strategy in strategies:
            cands_per_strategy[strategy.name] = strategy.get_candidates(row)
            
        final_cands, final_prov = combine_candidates(cands_per_strategy)
        results[s1_id] = (final_cands, final_prov)
        
    return results

def build_ground_truth_index(gt: pd.DataFrame) -> Dict[str, Set[str]]:
    """Builds a dictionary of S1 ID to a set of true matching candidate IDs."""
    gt_dict = {}
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        val = row.get('matched_entity_ids', "")
        if pd.isna(val) or not str(val).strip():
            gt_dict[s1_id] = set()
        else:
            matches = set(m for m in str(val).split(",") if m)
            gt_dict[s1_id] = matches
    return gt_dict

def label_candidate_pairs(
    candidates_dict: Dict[str, Tuple[List[str], Dict[str, List[str]]]], 
    gt_index: Dict[str, Set[str]]
) -> Tuple[List[Dict], int, int]:
    """Extracts positive pairs and missing positives from candidates."""
    positive_pairs = []
    missing_positive_count = 0
    total_ground_truth_positives = 0
    
    for s1_id, (cands, prov) in candidates_dict.items():
        matches = gt_index.get(s1_id, set())
        total_ground_truth_positives += len(matches)
        
        cands_set = set(cands)
        for m in matches:
            if m in cands_set:
                positive_pairs.append({
                    "s1_id": s1_id,
                    "cand_id": m,
                    "provenance": prov.get(m, []),
                    "label": 1
                })
            else:
                missing_positive_count += 1
                
    return positive_pairs, total_ground_truth_positives, missing_positive_count

def sample_negative_pairs(
    candidates_dict: Dict[str, Tuple[List[str], Dict[str, List[str]]]],
    gt_index: Dict[str, Set[str]],
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    num_random: int = 2,
    num_hard_name: int = 1,
    random_state: int = 42
) -> List[Dict]:
    """Samples negative candidate pairs avoiding ground truth leakage."""
    rng = np.random.default_rng(random_state)
    negative_pairs = []
    
    # We need quick name lookups for hard negatives
    s1_names = s1.set_index('entity_id')['business_name_normalized'].to_dict()
    s2_s3_names = pd.concat([
        s2[['entity_id', 'business_name_normalized']], 
        s3[['entity_id', 'business_name_normalized']]
    ]).set_index('entity_id')['business_name_normalized'].to_dict()
    
    # Simple Jaccard for quick hard negative check
    def quick_sim(n1, n2):
        s_1, s_2 = set(str(n1).split()), set(str(n2).split())
        return len(s_1.intersection(s_2)) / max(len(s_1.union(s_2)), 1)
    
    for s1_id, (cands, prov) in candidates_dict.items():
        true_matches = gt_index.get(s1_id, set())
        negatives = [c for c in cands if c not in true_matches]
        
        if not negatives:
            continue
            
        # Random negatives
        if len(negatives) <= num_random:
            chosen_random = negatives
        else:
            chosen_random = rng.choice(negatives, size=num_random, replace=False).tolist()
            
        for c in chosen_random:
            negative_pairs.append({
                "s1_id": s1_id,
                "cand_id": c,
                "provenance": prov.get(c, []),
                "label": 0
            })
            
        # Hard name negatives
        n1 = s1_names.get(s1_id, "")
        if n1:
            scored_negs = []
            for n in negatives:
                if n in chosen_random: continue
                n2 = s2_s3_names.get(n, "")
                scored_negs.append((n, quick_sim(n1, n2)))
            
            scored_negs.sort(key=lambda x: x[1], reverse=True)
            for c, _ in scored_negs[:num_hard_name]:
                negative_pairs.append({
                    "s1_id": s1_id,
                    "cand_id": c,
                    "provenance": prov.get(c, []),
                    "label": 0
                })
                
    return negative_pairs

def build_training_matrix(
    pair_dicts: List[Dict],
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame
) -> pd.DataFrame:
    """Computes features for pairs."""
    features_list = []
    
    s1_idx = s1.set_index('entity_id')
    s2_idx = s2.set_index('entity_id')
    s3_idx = s3.set_index('entity_id')
    
    for pair in pair_dicts:
        s1_id = pair['s1_id']
        c_id = pair['cand_id']
        
        s1_row = s1_idx.loc[s1_id]
        if c_id in s2_idx.index:
            c_row = s2_idx.loc[c_id]
            c_source = 's2'
        elif c_id in s3_idx.index:
            c_row = s3_idx.loc[c_id]
            c_source = 's3'
        else:
            continue
            
        feats = build_pair_features(s1_row, c_row, cand_source=c_source, provenance=pair.get('provenance', []))
        feats['label'] = pair['label']
        feats['s1_id'] = s1_id
        feats['cand_id'] = c_id
        feats['candidate_source'] = c_source
        
        features_list.append(feats)
        
    return pd.DataFrame(features_list)

def train_match_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, random_state: int = 42) -> lgb.Booster:
    """Trains the LightGBM classifier."""
    pos_ratio = sum(y_train) / len(y_train)
    scale_pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
    
    print(f"Training Data - Positive Ratio: {pos_ratio:.4f}, Suggested scale_pos_weight: {scale_pos_weight:.2f}")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': random_state,
        'scale_pos_weight': scale_pos_weight
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    return model

def calculate_f05(y_true, y_pred_prob, threshold):
    y_pred = (y_pred_prob >= threshold).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    if p + r == 0:
        return 0.0
    return (1.25 * p * r) / ((0.25 * p) + r)

def evaluate_classifier(y_true, y_pred_prob):
    """Calculates comprehensive metrics for binary classification."""
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_pred_prob),
        "pr_auc": average_precision_score(y_true, y_pred_prob),
        "log_loss": log_loss(y_true, y_pred_prob)
    }
    
    # Calculate F0.5 at diagnostic thresholds
    f05_results = {}
    for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
        f05_results[f"t={t}"] = calculate_f05(y_true, y_pred_prob, t)
        
    metrics["f05_diagnostic"] = f05_results
    
    # Reference threshold metrics (t=0.5)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    metrics["ref_precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["ref_recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["ref_f1"] = f1_score(y_true, y_pred, zero_division=0)
    
    return metrics
