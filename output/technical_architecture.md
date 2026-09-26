# Technical Architecture

## Components

1. **Data Loader (`data_loader.py`)**
   - Input: Raw TSVs
   - Processing: Read to Pandas DFs
   - Output: Initial DataFrames

2. **Preprocessing (`preprocessing.py`)**
   - Input: Raw DFs
   - Processing: Regex extraction (postal, numeric), text lowercase/strip
   - Output: Normalized DFs

3. **Blocking (`blocking.py`)**
   - Input: Normalized DFs
   - Processing: Inner joins on extracted primitive keys (postal, numerics).
   - Output: `candidate_pairs.tsv`

4. **Feature Engineering (`features.py`)**
   - Input: Candidate pairs
   - Processing: String distances (Jaro-Winkler, Levenshtein)
   - Output: Numerical feature matrix

5. **Training (`training.py`)**
   - Input: Labelled Feature matrix
   - Processing: LightGBM classification
   - Output: Frozen model

6. **Threshold Optimization (`threshold_optimization.py`)**
   - Input: Validations probabilities
   - Processing: Grid search maximizing Entity-Level F0.5
   - Output: `final_threshold.json`

7. **Entity Matching (`entity_matching.py`)**
   - Input: Test predictions + Threshold
   - Processing: Hard classification filtering
   - Output: `matching_results.tsv`