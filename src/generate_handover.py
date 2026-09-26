import os
from pathlib import Path

def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content.strip())

def main():
    out_dir = Path("output")
    
    write_file(out_dir / "project_explanation.md", """
# Business Entity Resolution — Complete Explanation

## 1. Problem
Business Entity Resolution is the task of identifying whether two or more distinct textual records refer to the same real-world business entity. In this challenge, Source 1 (S1) contains unique reference records. Source 2 (S2) and Source 3 (S3) contain potentially duplicate, noisy records. The records are difficult to match due to severe OCR noise, transliteration variations across languages, missing fields, and structural inconsistencies in address formatting.

## 2. Overall Idea
Raw Records -> Normalization -> Blocking -> Candidate Pairs -> Feature Engineering -> LightGBM -> Probability -> Threshold -> Entity-Level Matching -> Final Output

## 3. Why We Cannot Compare Everything
An $O(N^2)$ cross-join across millions of records is computationally infeasible. Blocking restricts the search space by only generating Candidate Pairs for records that share a high-confidence exact match on cheap-to-compute structural keys (like normalized country, postal codes, or numeric structural elements).

## 4. Normalization
- **Business names**: Lowercased, stripped of punctuation, and basic legal suffixes standardized.
- **Addresses**: Lowercased, stripped of common punctuation.
- **Countries**: ISO mappings applied where possible.
- **Postal codes / Numerics**: Extracted explicitly via regex into independent columns.
Raw values are preserved so that features can calculate string distances without losing native signal.

## 5. Blocking
- Token blocking (matching significant unigrams)
- Postal blocking
- Numeric-address blocking
These are combined using a logical UNION. If any strategy hits, the pair enters the candidate set.

## 6. Features
- Name similarity (RapidFuzz Jaro-Winkler)
- Address similarity (RapidFuzz Levenshtein)
- Numeric token containment
- Exact column matches

## 7. LightGBM
Input: A 1D vector of numerical similarity and categorical boolean features.
Output: A probability $P(Match=1)$.
It is a binary classification problem because any given S1-candidate pair is either fundamentally the same entity (1) or different (0).

## 8. Threshold
The probability must be binarized into a strict decision. A threshold of 0.5 is mathematically arbitrary; threshold optimization calibrates this cutoff to specifically maximize the F0.5 metric, balancing the challenge's high penalty on false positives (precision).

## 9. Entity-Level Matching
One S1 record can map to 0, 1, or N S2/S3 records. 

## 10. Evaluation
- Precision: How many predicted matches were true?
- Recall: How many true matches were predicted?
- F0.5: Harmonic mean of precision and recall, with precision weighted twice as much. F0.5 is the official challenge metric.

## 11. Final Outputs
- `candidate_pairs.tsv`: The unioned output of the blocking stage.
- `matching_results.tsv`: The final threshold-filtered subset of the candidate pairs.
    """)

    write_file(out_dir / "technical_architecture.md", """
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
    """)

    write_file(out_dir / "code_walkthrough.md", """
# Code Walkthrough

## src/preprocessing.py
- **Purpose**: Cleans strings and extracts discrete blocking primitives.
- **Functions**: `normalize_string`, `extract_postal`, `extract_numeric`

## src/blocking.py
- **Purpose**: Generates scalable subsets of candidate matching pairs.
- **Functions**: `generate_candidates`, `merge_blocks`

## src/features.py
- **Purpose**: Calculates mathematical similarities.
- **Functions**: `compute_features`, `jarowinkler_similarity`

## src/training.py
- **Purpose**: Executes gradient boosted trees.
- **Functions**: `train_lightgbm`

## src/main.py
- **Purpose**: The reproducible CLI entry point tying all modules sequentially.
- **Functions**: `main`, `run_submission_pipeline`
    """)

    write_file(out_dir / "data_flow.md", """
# Data Flow

1. S1/S2/S3 TSV -> `pandas.read_csv`
2. DataFrames -> `preprocessing.py`
3. Normalized DataFrames -> `blocking.py`
4. Blocking keys -> Inner Joins
5. Candidate pairs -> `features.py`
6. Feature matrix -> `LightGBM.predict`
7. Model probabilities -> `threshold_optimization.py` (Filtering)
8. Thresholded pairs -> Output Generation
9. TSV outputs
    """)

    write_file(out_dir / "model_card.md", """
# Model Card

## Model
LightGBM binary classifier

## Task
Business entity pair classification

## Input
Pair-level numerical features (RapidFuzz distances)

## Output
Match probability

## Training
Trained on synthetic negative pairs sampled from the blocking candidate universe.

## Validation
Strict Source-1 entity-level split.

## Threshold
Grid search maximized for Entity-Level F0.5.

## Known Limitations
- Bounded entirely by Blocking Recall ceiling.
- Extremely noise-sensitive to transliteration that bypasses string distance metrics.
    """)

    write_file(out_dir / "presentation_outline.md", """
# Presentation Outline

- Slide 1: Problem (Entity Resolution context)
- Slide 2: Challenge (O(N^2) scale, F0.5 precision-heavy metric)
- Slide 3: Solution Architecture (Pipeline Flowchart)
- Slide 4: Data Preprocessing (Normalization rules)
- Slide 5: Candidate Generation (UNION blocking strategies)
- Slide 6: Feature Engineering (String distance algorithms)
- Slide 7: LightGBM Model (Binary classification formulation)
- Slide 8: Threshold + Entity Matching (Probability -> Binary logic)
- Slide 9: Validation Results (Entity-level splits)
- Slide 10: Error Analysis (False negatives due to blocking)
- Slide 11: Final Submission (Deterministic CLI generation)
- Slide 12: Future Improvements (Hard negative mining, LLM usage)
    """)
    
    write_file(out_dir / "interview_questions.md", """
# Interview Questions

## Basic
1. Why blocking?
2. Why LightGBM?
...

## Intermediate
1. Why use F0.5 instead of Accuracy?
2. How did you extract postal codes?

## Advanced
1. How did you prevent entity leakage?
2. What are hard negatives?

## Project-Specific
1. Why UNION rather than intersection for blocking?
2. What happens when there are zero matches?
    """)
    
    write_file(out_dir / "interview_answers.md", """
# Interview Answers

- **Why blocking?**: Because N^2 cross joins crash on millions of records.
- **Why UNION?**: Intersections limit recall too heavily on noisy data.
- **Entity Leakage?**: We split strictly by `source1_entity_id`, never row-level.
- **Zero matches?**: We explicitly allow S1 entities to have empty sets to maximize precision if probabilities fall below the threshold.
- **LLM Usage?**: That was not implemented in the final system.
    """)

    write_file(out_dir / "two_minute_explanation.md", """
# 2-Minute Explanation
We built a highly scalable ML pipeline to match noisy business entities across 3 distinct datasets. To prevent the computational explosion of comparing every single record, we implemented a deterministic blocking system using postal codes and extracted numbers. This generates a subset of candidate pairs. For these pairs, we extract numerical string distance features (like Jaro-Winkler) and pass them to a LightGBM classifier. Because the competition heavily penalizes false positives, we execute a custom grid search to pick the optimal probability threshold maximizing the F0.5 metric on a holdout set. The final predictions strictly adhere to challenge outputs and are 100% deterministically reproducible.
    """)

    write_file(out_dir / "thirty_second_explanation.md", """
# 30-Second Explanation
I built an end-to-end Entity Resolution pipeline. It leverages multi-key structural blocking to generate scalable candidate pairs, computes pairwise text similarity features, and scores them using a LightGBM classifier. The threshold is heavily biased toward precision to optimize the F0.5 challenge metric, outputting perfectly formatted TSV mappings.
    """)

    write_file(out_dir / "technical_deep_dive.md", """
# Technical Deep-Dive

1. **Why blocking was necessary**: Scale.
2. **Why multiple blocks**: No single noisy key is reliable enough.
3. **Blocking recall**: The model cannot match what the blocker drops.
4. **Leakage prevention**: S1 GroupKFold splitting.
5. **Why LightGBM**: Fast, natively handles missing values.
    """)

    write_file(out_dir / "limitations.md", """
# Limitations

## KNOWN LIMITATIONS
- Hard blocking drops extremely noisy true-positive pairs.
- Translitarations (e.g. Arabic to Latin) bypass string distance algorithms.

## POSSIBLE FUTURE IMPROVEMENTS
- Implementing contrastive learning embeddings.
- LLM zero-shot verification on edge cases.
    """)

    write_file(out_dir / "reproduction_guide.md", """
# Reproduction Guide
1. `pip install -r requirements.txt`
2. Place test data in `dataset/test/`
3. `python -m src.main --data-root dataset --output-dir output`
4. Outputs will generate in `output/`
    """)

    write_file(out_dir / "troubleshooting.md", """
# Troubleshooting

- **Problem**: Missing `dataset/test` files.
- **Solution**: Ensure the TSVs match the challenge specifications.

- **Problem**: Metric is 0 for precision.
- **Solution**: The threshold is too high, blocking all candidates.
    """)

    write_file(out_dir / "project_index.md", """
# Project Index

- `src/main.py`: Entry point.
- `models/final_threshold.json`: Frozen threshold.
- `output/candidate_pairs.tsv`: Subset pairs.
- `submissions/v1/`: Final zip artifact.
    """)

    write_file(out_dir / "phase14_status.md", """
# Phase 14 Status

## Project Status
COMPLETE

## Final Architecture
Normalization -> Blocking -> Features -> LightGBM -> Thresholding

## Final Model
LightGBM binary classifier

## Submission Version
v1

## Documentation
Fully generated and available in `output/`

## Known Limitations
Bounded by blocking recall.

## Reproducibility
Deterministic CLI entry point provided.
    """)

if __name__ == "__main__":
    main()
