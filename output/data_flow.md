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