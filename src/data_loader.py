import pandas as pd
from pathlib import Path
from typing import Dict
from .config import EXPECTED_SOURCE_COLUMNS, EXPECTED_GT_COLUMNS

def load_source_file(file_path: Path) -> pd.DataFrame:
    """Loads a single source TSV file and validates its columns."""
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required file: {file_path}")
    
    # Use keep_default_na=False to read empty strings as '' rather than NaN
    df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)
    
    actual_columns = list(df.columns)
    if actual_columns != EXPECTED_SOURCE_COLUMNS:
        raise ValueError(
            f"Incorrect columns in {file_path.name}.\n"
            f"Expected: {EXPECTED_SOURCE_COLUMNS}\n"
            f"Actual: {actual_columns}"
        )
        
    return df

def load_ground_truth(file_path: Path) -> pd.DataFrame:
    """Loads the ground truth TSV file and validates its columns."""
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required file: {file_path}")
    
    df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)
    
    actual_columns = list(df.columns)
    if actual_columns != EXPECTED_GT_COLUMNS:
        raise ValueError(
            f"Incorrect columns in {file_path.name}.\n"
            f"Expected: {EXPECTED_GT_COLUMNS}\n"
            f"Actual: {actual_columns}"
        )
        
    return df

def load_train_data(train_dir: Path, files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Loads all training files into a dictionary of dataframes."""
    data = {}
    for key, filename in files.items():
        file_path = train_dir / filename
        if key == "ground_truth":
            data[key] = load_ground_truth(file_path)
        else:
            data[key] = load_source_file(file_path)
    return data

def load_test_data(test_dir: Path, files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Loads all test files into a dictionary of dataframes."""
    data = {}
    for key, filename in files.items():
        file_path = test_dir / filename
        data[key] = load_source_file(file_path)
    return data
