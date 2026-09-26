from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = ROOT_DIR / "output"

# File names
TRAIN_FILES = {
    "source1": "train_source1.tsv",
    "source2": "train_source2.tsv",
    "source3": "train_source3.tsv",
    "ground_truth": "train_ground_truth.tsv",
}

TEST_FILES = {
    "source1": "test_source1.tsv",
    "source2": "test_source2.tsv",
    "source3": "test_source3.tsv",
}

EXPECTED_SOURCE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country"
]

EXPECTED_GT_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids"
]

# Blocking Configuration
MAX_BLOCK_FREQUENCY = 1000       # If a token appears in more than this many records, skip it
MAX_CANDIDATES_PER_SOURCE = 50   # Max candidates to keep per S1 record
MIN_TOKEN_LENGTH = 3             # Minimum length of token to be considered for blocking
TOP_K_NGRAM = 20                 # Number of candidates to retrieve from n-gram block
NGRAM_RANGE = (2, 4)             # N-gram range for character TF-IDF

