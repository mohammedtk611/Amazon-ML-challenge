import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from tqdm import tqdm

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .config import (
    MAX_BLOCK_FREQUENCY, MAX_CANDIDATES_PER_SOURCE, 
    MIN_TOKEN_LENGTH, TOP_K_NGRAM, NGRAM_RANGE
)

class BlockingStrategy:
    def __init__(self, name: str):
        self.name = name
        self.index = defaultdict(list)
        self.token_freq = defaultdict(int)

    def build_index(self, s2: pd.DataFrame, s3: pd.DataFrame):
        raise NotImplementedError

    def get_candidates(self, s1_row: pd.Series) -> List[str]:
        raise NotImplementedError

class ExactNameBlock(BlockingStrategy):
    def __init__(self):
        super().__init__("exact_name")

    def build_index(self, s2: pd.DataFrame, s3: pd.DataFrame):
        for df in [s2, s3]:
            for _, row in df.iterrows():
                if row['business_name_normalized']:
                    self.index[row['business_name_normalized']].append(row['entity_id'])
                if row['business_name_core'] and row['business_name_core'] != row['business_name_normalized']:
                    self.index[row['business_name_core']].append(row['entity_id'])

    def get_candidates(self, s1_row: pd.Series) -> List[str]:
        cands = set()
        if s1_row['business_name_normalized']:
            cands.update(self.index.get(s1_row['business_name_normalized'], []))
        if s1_row['business_name_core']:
            cands.update(self.index.get(s1_row['business_name_core'], []))
        return list(cands)

class CountryNameTokenBlock(BlockingStrategy):
    def __init__(self):
        super().__init__("country_name_token")

    def build_index(self, s2: pd.DataFrame, s3: pd.DataFrame):
        # First pass to compute frequency
        for df in [s2, s3]:
            for _, row in df.iterrows():
                country = row['country_normalized']
                tokens = row['business_name_tokens']
                if not country or not isinstance(tokens, list):
                    continue
                for t in tokens:
                    if len(t) >= MIN_TOKEN_LENGTH:
                        self.token_freq[(country, t)] += 1
                        
        # Second pass to build index
        for df in [s2, s3]:
            for _, row in df.iterrows():
                country = row['country_normalized']
                tokens = row['business_name_tokens']
                if not country or not isinstance(tokens, list):
                    continue
                for t in tokens:
                    if len(t) >= MIN_TOKEN_LENGTH and self.token_freq[(country, t)] <= MAX_BLOCK_FREQUENCY:
                        self.index[(country, t)].append(row['entity_id'])

    def get_candidates(self, s1_row: pd.Series) -> List[str]:
        country = s1_row['country_normalized']
        tokens = s1_row['business_name_tokens']
        if not country or not isinstance(tokens, list):
            return []
            
        cands = set()
        for t in tokens:
            if len(t) >= MIN_TOKEN_LENGTH:
                cands.update(self.index.get((country, t), []))
        return list(cands)

class PostalBlock(BlockingStrategy):
    def __init__(self):
        super().__init__("postal")

    def build_index(self, s2: pd.DataFrame, s3: pd.DataFrame):
        for df in [s2, s3]:
            for _, row in df.iterrows():
                postal = row['business_address_postal']
                if postal:
                    self.index[postal].append(row['entity_id'])

    def get_candidates(self, s1_row: pd.Series) -> List[str]:
        postal = s1_row['business_address_postal']
        if not postal:
            return []
        return self.index.get(postal, [])

class AddressTokenBlock(BlockingStrategy):
    def __init__(self):
        super().__init__("address_token")

    def build_index(self, s2: pd.DataFrame, s3: pd.DataFrame):
        for df in [s2, s3]:
            for _, row in df.iterrows():
                tokens = row['business_address_tokens']
                if not isinstance(tokens, list):
                    continue
                for t in set(tokens):
                    if len(t) >= MIN_TOKEN_LENGTH:
                        self.token_freq[t] += 1
                        
        for df in [s2, s3]:
            for _, row in df.iterrows():
                tokens = row['business_address_tokens']
                if not isinstance(tokens, list):
                    continue
                for t in set(tokens):
                    if len(t) >= MIN_TOKEN_LENGTH and self.token_freq[t] <= MAX_BLOCK_FREQUENCY:
                        self.index[t].append(row['entity_id'])

    def get_candidates(self, s1_row: pd.Series) -> List[str]:
        tokens = s1_row['business_address_tokens']
        if not isinstance(tokens, list):
            return []
            
        cands = set()
        for t in tokens:
            if len(t) >= MIN_TOKEN_LENGTH:
                cands.update(self.index.get(t, []))
        return list(cands)

class NumericAddressBlock(BlockingStrategy):
    def __init__(self):
        super().__init__("numeric_address")

    def build_index(self, s2: pd.DataFrame, s3: pd.DataFrame):
        for df in [s2, s3]:
            for _, row in df.iterrows():
                num_tokens = row['business_address_numeric_tokens']
                if not isinstance(num_tokens, list):
                    continue
                for t in set(num_tokens):
                    self.index[t].append(row['entity_id'])

    def get_candidates(self, s1_row: pd.Series) -> List[str]:
        num_tokens = s1_row['business_address_numeric_tokens']
        if not isinstance(num_tokens, list):
            return []
            
        cands = set()
        for t in num_tokens:
            cands.update(self.index.get(t, []))
        return list(cands)

class CharNgramBlock(BlockingStrategy):
    def __init__(self):
        super().__init__("char_ngram")
        self.vectorizer = None
        self.tfidf_matrix = None
        self.candidate_ids = []

    def build_index(self, s2: pd.DataFrame, s3: pd.DataFrame):
        if not SKLEARN_AVAILABLE:
            print("Sklearn not available. Skipping CharNgramBlock.")
            return
            
        df = pd.concat([s2, s3], ignore_index=True)
        # Use business_name_normalized
        names = df['business_name_normalized'].fillna("").tolist()
        self.candidate_ids = df['entity_id'].tolist()
        
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=NGRAM_RANGE, min_df=2)
        self.tfidf_matrix = self.vectorizer.fit_transform(names)

    def get_candidates(self, s1_row: pd.Series) -> List[str]:
        if not SKLEARN_AVAILABLE or self.vectorizer is None:
            return []
            
        name = s1_row['business_name_normalized']
        if not name:
            return []
            
        vec = self.vectorizer.transform([name])
        # Compute cosine similarity
        similarities = cosine_similarity(vec, self.tfidf_matrix).flatten()
        
        # Get top k indices
        top_indices = similarities.argsort()[-TOP_K_NGRAM:][::-1]
        
        cands = []
        for idx in top_indices:
            if similarities[idx] > 0.0:  # Only if there's some similarity
                cands.append(self.candidate_ids[idx])
        return cands

def combine_candidates(candidate_lists: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, List[str]]]:
    """Combines candidates from multiple strategies and enforces caps while preserving provenance."""
    combined_scores = defaultdict(int)
    provenance = defaultdict(list)
    
    # Priority weighting for capping tie-breakers
    strategy_weights = {
        "exact_name": 100,
        "postal": 50,
        "country_name_token": 30,
        "address_token": 20,
        "char_ngram": 10,
        "numeric_address": 5
    }
    
    for strategy, cands in candidate_lists.items():
        weight = strategy_weights.get(strategy, 1)
        for cand in cands:
            combined_scores[cand] += weight
            provenance[cand].append(strategy)
            
    # Apply cap deterministically
    sorted_cands = sorted(combined_scores.keys(), key=lambda c: (-combined_scores[c], c))
    final_cands = sorted_cands[:MAX_CANDIDATES_PER_SOURCE]
    
    # Filter provenance to only final candidates
    final_provenance = {c: provenance[c] for c in final_cands}
    
    return final_cands, final_provenance
