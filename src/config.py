from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATASET_DIR = PROJECT_ROOT / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Train files
TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"

TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

# Test files
TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# output dir
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Creating directories if needed
for directory in [OUTPUT_DIR, MODELS_DIR, EXPERIMENTS_DIR, REPORTS_DIR]:
    directory.mkdir(parents = True, exist_ok = True)