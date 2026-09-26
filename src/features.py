import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein, JaroWinkler
from typing import Dict, List, Any

def _jaccard(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def _overlap_coeff(set1: set, set2: set) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    min_len = min(len(set1), len(set2))
    return intersection / min_len if min_len > 0 else 0.0

def _normalized_similarity(s1: str, s2: str, func) -> float:
    if not s1 or not s2:
        return 0.0
    val = func.normalized_similarity(s1, s2)
    # RapidFuzz can return 0-100 or 0-1 depending on func and version.
    # Safely normalize to [0, 1]
    if val > 1.0:
        return val / 100.0
    return val

def name_features(r1: pd.Series, r2: pd.Series) -> Dict[str, Any]:
    n1 = str(r1.get('business_name_normalized', ''))
    n2 = str(r2.get('business_name_normalized', ''))
    
    c1 = str(r1.get('business_name_core', ''))
    c2 = str(r2.get('business_name_core', ''))
    
    t1 = r1.get('business_name_tokens', [])
    t2 = r2.get('business_name_tokens', [])
    
    if not isinstance(t1, list): t1 = []
    if not isinstance(t2, list): t2 = []
    
    s1, s2 = set(t1), set(t2)
    
    feats = {
        'name_missing_s1': 1 if not n1 else 0,
        'name_missing_candidate': 1 if not n2 else 0,
        
        'name_exact': 1 if n1 and n2 and n1 == n2 else 0,
        'name_core_exact': 1 if c1 and c2 and c1 == c2 else 0,
        
        'name_levenshtein': _normalized_similarity(n1, n2, Levenshtein),
        'name_jaro_winkler': _normalized_similarity(n1, n2, JaroWinkler),
        
        'name_token_intersection_count': len(s1.intersection(s2)),
        'name_token_union_count': len(s1.union(s2)),
        'name_token_jaccard': _jaccard(s1, s2),
        'name_token_overlap': _overlap_coeff(s1, s2),
        
        'name_length_difference': abs(len(n1) - len(n2)),
        'name_length_ratio': min(len(n1), len(n2)) / max(len(n1), len(n2)) if max(len(n1), len(n2)) > 0 else 0.0,
        
        'name_token_count_difference': abs(len(t1) - len(t2))
    }
    return feats

def address_features(r1: pd.Series, r2: pd.Series) -> Dict[str, Any]:
    a1 = str(r1.get('business_address_normalized', ''))
    a2 = str(r2.get('business_address_normalized', ''))
    
    t1 = r1.get('business_address_tokens', [])
    t2 = r2.get('business_address_tokens', [])
    
    if not isinstance(t1, list): t1 = []
    if not isinstance(t2, list): t2 = []
    
    s1, s2 = set(t1), set(t2)
    
    feats = {
        'address_missing_s1': 1 if not a1 else 0,
        'address_missing_candidate': 1 if not a2 else 0,
        
        'address_exact': 1 if a1 and a2 and a1 == a2 else 0,
        
        'address_levenshtein': _normalized_similarity(a1, a2, Levenshtein),
        'address_jaro_winkler': _normalized_similarity(a1, a2, JaroWinkler),
        
        'address_token_intersection_count': len(s1.intersection(s2)),
        'address_token_union_count': len(s1.union(s2)),
        'address_token_jaccard': _jaccard(s1, s2),
        'address_token_overlap': _overlap_coeff(s1, s2),
        
        'address_length_difference': abs(len(a1) - len(a2)),
        'address_length_ratio': min(len(a1), len(a2)) / max(len(a1), len(a2)) if max(len(a1), len(a2)) > 0 else 0.0,
        
        'address_token_count_difference': abs(len(t1) - len(t2))
    }
    return feats

def postal_features(r1: pd.Series, r2: pd.Series) -> Dict[str, Any]:
    p1 = str(r1.get('business_address_postal', ''))
    p2 = str(r2.get('business_address_postal', ''))
    
    both_present = 1 if p1 and p2 else 0
    exact = 1 if both_present and p1 == p2 else 0
    mismatch = 1 if both_present and p1 != p2 else 0
    
    return {
        'postal_missing_s1': 1 if not p1 else 0,
        'postal_missing_candidate': 1 if not p2 else 0,
        'postal_both_present': both_present,
        'postal_exact_match': exact,
        'postal_mismatch': mismatch
    }

def numeric_features(r1: pd.Series, r2: pd.Series) -> Dict[str, Any]:
    t1 = r1.get('business_address_numeric_tokens', [])
    t2 = r2.get('business_address_numeric_tokens', [])
    
    if not isinstance(t1, list): t1 = []
    if not isinstance(t2, list): t2 = []
    
    s1, s2 = set(t1), set(t2)
    intersection = len(s1.intersection(s2))
    
    return {
        'numeric_token_intersection_count': intersection,
        'numeric_token_jaccard': _jaccard(s1, s2),
        'numeric_token_overlap': _overlap_coeff(s1, s2),
        'has_common_numeric_token': 1 if intersection > 0 else 0
    }

def country_features(r1: pd.Series, r2: pd.Series) -> Dict[str, Any]:
    c1 = str(r1.get('country_normalized', ''))
    c2 = str(r2.get('country_normalized', ''))
    
    both_present = 1 if c1 and c2 else 0
    exact = 1 if both_present and c1 == c2 else 0
    mismatch = 1 if both_present and c1 != c2 else 0
    
    return {
        'country_missing_s1': 1 if not c1 else 0,
        'country_missing_candidate': 1 if not c2 else 0,
        'country_both_present': both_present,
        'country_exact_match': exact,
        'country_mismatch': mismatch
    }

def structural_features(r1: pd.Series, r2: pd.Series, s2_or_s3: str) -> Dict[str, Any]:
    return {
        'candidate_source_is_s2': 1 if s2_or_s3 == 's2' else 0,
        'candidate_source_is_s3': 1 if s2_or_s3 == 's3' else 0
    }

def build_pair_features(r1: pd.Series, r2: pd.Series, cand_source: str, provenance: List[str]) -> Dict[str, Any]:
    """Builds a complete numerical feature vector for a candidate pair."""
    feats = {}
    feats.update(name_features(r1, r2))
    feats.update(address_features(r1, r2))
    feats.update(postal_features(r1, r2))
    feats.update(numeric_features(r1, r2))
    feats.update(country_features(r1, r2))
    feats.update(structural_features(r1, r2, cand_source))
    
    # provenance features
    prov_set = set(provenance)
    for block_type in ["exact_name", "country_name_token", "postal", "address_token", "numeric_address", "char_ngram"]:
        feats[f"blocked_by_{block_type}"] = 1 if block_type in prov_set else 0
        
    return feats
