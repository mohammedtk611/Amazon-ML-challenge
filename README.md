# Business Entity Resolution

## 1. Project Overview
This project tackles the Amazon ML Challenge 2026 Business Entity Resolution task. We developed an end-to-end Machine Learning pipeline that standardizes noisy entity records, applies multi-key blocking strategies to identify candidate sets, calculates pairwise similarity features, and scores matches with a LightGBM classifier optimized for the challenge's distinct entity-level F0.5 metric.

## 2. Solution Architecture

```
Source Records
↓
Normalization
↓
Multi-Blocking
↓
Candidate Pairs
↓
Pair Features
↓
LightGBM
↓
Threshold Optimization
↓
Entity-Level Matching
↓
Submission Outputs
```

## 3. Project Structure
- `src/` - Complete pipeline source code (Normalization, Blocking, Features, ML Inference).
- `models/` - Final configuration, frozen models, and threshold mappings.
- `output/` - System logs, metrics, error analysis, and the generated submission files.
- `tests/` - Unit tests ensuring strict pipeline reliability and output metric validation.

## 4. Requirements
The project solely relies on standard Data Science libraries:
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `rapidfuzz>=3.0.0`
- `lightgbm>=4.0.0`

## 5. Installation
```bash
pip install -r requirements.txt
```

## 6. Running the Pipeline
To produce the final reproducible submission output from scratch on your datasets:
```bash
python -m src.main --data-root dataset --output-dir output
```

## 7. Input Data
The pipeline expects TSV files in a structured format:
- `dataset/test/test_source1.tsv`
- `dataset/test/test_source2.tsv`
- `dataset/test/test_source3.tsv`

## 8. Output Files
- `output/candidate_pairs.tsv`: The full pre-ML subset of potential matches per Source 1 entity.
- `output/matching_results.tsv`: The exact final predicted subsets strictly bound by the optimum probability threshold.

## 9. Methodology
- **Normalization:** Non-destructive structural parsing to expose names, numerical values, and postal tokens natively.
- **Blocking:** Restrictive but multi-layered inner joins (on country, basic structural tokens) to prevent full O(N^2) explosion while maintaining acceptable recall.
- **Features:** Efficient calculation of text distance metrics alongside discrete exact-token containment interactions.
- **Model:** A binary classification LightGBM model trained on synthetic negatives drawn directly from the block candidates.
- **Threshold:** Entity-level Macro F0.5 Grid Search. 

## 10. Validation
We validated using strict Source-1 entity splits to prevent target label leakage. F0.5 grid searches and metric generations handled 0-match ground-truth entities accurately per challenge specification.

## 11. Final Validation Results
*(To be generated upon running the pipeline against your populated test datasets.)*

## 12. Limitations
- Test ground truth is strictly unobserved, meaning final test performance is inherently estimated.
- The threshold selection is tightly bound to validation data.
- Extremely noisy business records with heavy transliterations and zero intersecting geographic overlap will be permanently omitted by the initial strict Blocking constraints.

## 13. Reproducibility
The absolute, deterministic mechanism for regenerating all final outputs is via:
```bash
python -m src.main --data-root dataset --output-dir output
```

## 14. License / Model Constraint
This pipeline fulfills all challenge requirements (uses LightGBM which operates substantially below the 8B parameter limit constraint) and leverages universally acceptable open-source libraries (MIT/Apache 2.0).
