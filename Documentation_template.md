# Business Entity Resolution

## Executive Summary
This project solves the Amazon ML Challenge 2026 Business Entity Resolution problem. We engineered a scalable data pipeline spanning structural normalization, targeted blocking, and LightGBM-based probabilistic matching, specifically prioritizing entity-level F0.5 metrics in our evaluation.

## Methodology
The pipeline consists of:
1. **Normalization**: Extracting postal codes and numeric tokens to standard forms while leaving original data untouched.
2. **Candidate Generation (Blocking)**: Limiting the O(N x M) candidate space by enforcing structural agreement via multi-key block joining.
3. **Feature Engineering**: Calculating string distance metrics (Jaro-Winkler, Levenshtein, Token Set Ratios) via `rapidfuzz`.
4. **Machine Learning Model**: Training a LightGBM classifier with carefully sampled hard negatives from the generated candidates.
5. **Entity-Level Matching**: Optimizing the final classification threshold explicitly against the challenge F0.5 metric, macro-averaged over Source 1 entities.

## Problem Analysis
The principal challenge is the immense cardinality of potential pairs across three sources. Real-world business data incorporates severe naming truncations, missing tokens, transliteration variance, and country-specific formatting.

## Solution Strategy
We bypassed computationally exhaustive cross-joins by developing a specialized multi-pass blocking strategy.

## Candidate Generation / Blocking
We employed strict string matching on structural primitives (e.g. Normalized Name + Normalized Country, Postal Code + Numeric Address Tokens) to construct highly overlapping blocks.

## Matching Model
We selected LightGBM due to its capability for handling structured tabular features swiftly, its efficiency with imbalanced data distributions, and its robust handling of missing numerical representations. 

## Results & Error Analysis
*(Run `src.run_phase7` for dynamic results on your validation hold-out set)*

## Conclusion
The system provides a scalable, deterministic entity resolution architecture free of external API dependencies, explicitly optimized for the challenge's F0.5 criteria.

## Appendix / Code Artefacts
- `src/main.py`: Pipeline entry point.
- `src/preprocessing.py`: String normalization routines.
- `src/blocking.py`: Candidate generator.
- `src/features.py`: Pairwise similarity scoring.
- `src/training.py`: Supervised learning architecture.
- `src/threshold_optimization.py`: F0.5 calculation.
