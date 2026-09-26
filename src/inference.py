import pandas as pd
import lightgbm as lgb
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .preprocessing import apply_preprocessing
from .training import generate_training_candidates
from .features import build_pair_features

def generate_test_features(
    s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, 
    candidates: Dict[str, Tuple[List[str], Dict]], 
    feature_schema: List[str]
) -> pd.DataFrame:
    """Builds the feature matrix for test candidate pairs."""
    features_list = []
    
    s1_idx = s1.set_index('entity_id')
    s2_idx = s2.set_index('entity_id')
    s3_idx = s3.set_index('entity_id')
    
    for s1_id, (cands, prov) in candidates.items():
        s1_row = s1_idx.loc[s1_id]
        
        for c_id in cands:
            if c_id in s2_idx.index:
                c_row = s2_idx.loc[c_id]
                c_source = 's2'
            elif c_id in s3_idx.index:
                c_row = s3_idx.loc[c_id]
                c_source = 's3'
            else:
                continue
                
            feats = build_pair_features(s1_row, c_row, cand_source=c_source, provenance=prov.get(c_id, []))
            feats['s1_id'] = s1_id
            feats['cand_id'] = c_id
            
            features_list.append(feats)
            
    df = pd.DataFrame(features_list)
    if df.empty:
        return df
        
    # Reorder columns to strictly match training schema
    meta_cols = ['s1_id', 'cand_id']
    df = df[feature_schema + meta_cols]
    
    return df

def run_test_inference(test_data: dict, model_dir: Path) -> Tuple[Dict, pd.DataFrame, List[str]]:
    """Runs the full pipeline on test data."""
    print("Preprocessing Test Datasets...")
    s1 = apply_preprocessing(test_data["source1"])
    s2 = apply_preprocessing(test_data["source2"])
    s3 = apply_preprocessing(test_data["source3"])
    
    all_source1_ids = s1['entity_id'].tolist()
    
    print("Generating Test Candidates...")
    test_cands = generate_training_candidates(s1, s2, s3)
    
    with open(model_dir / "feature_columns.json", "r") as f:
        feature_schema = json.load(f)
        
    print("Extracting Test Features...")
    test_df = generate_test_features(s1, s2, s3, test_cands, feature_schema)
    
    if test_df.empty:
        print("Warning: Zero test candidates generated.")
        return test_cands, test_df, all_source1_ids
        
    print("Loading LightGBM Model...")
    model = lgb.Booster(model_file=str(model_dir / "lightgbm_matcher.txt"))
    
    X_test = test_df.drop(columns=['s1_id', 'cand_id'])
    
    print("Predicting Probabilities...")
    preds = model.predict(X_test)
    test_df['match_probability'] = preds
    
    return test_cands, test_df, all_source1_ids
