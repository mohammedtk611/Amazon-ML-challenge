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