import pandas as pd
from tqdm import tqdm

from .config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
)


def load_tsv(file_path):

    print(f"\nLoading: {file_path.name}")

    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(f"  ✓ Loaded {len(df):,} rows × {len(df.columns):,} columns")

    return df


def load_train_source1():
    return load_tsv(TRAIN_SOURCE1)


def load_train_source2():
    return load_tsv(TRAIN_SOURCE2)


def load_train_source3():
    return load_tsv(TRAIN_SOURCE3)


def load_ground_truth():
    return load_tsv(TRAIN_GROUND_TRUTH)


def load_test_source1():
    return load_tsv(TEST_SOURCE1)


def load_test_source2():
    return load_tsv(TEST_SOURCE2)


def load_test_source3():
    return load_tsv(TEST_SOURCE3)


def load_all_train_data():

    print("\n========== LOADING TRAIN DATA ==========")

    files = {
        "source1": TRAIN_SOURCE1,
        "source2": TRAIN_SOURCE2,
        "source3": TRAIN_SOURCE3,
        "ground_truth": TRAIN_GROUND_TRUTH,
    }

    data = {}

    for name, path in tqdm(
        files.items(),
        desc="Train files",
        unit="file",
    ):
        data[name] = load_tsv(path)

    return data


def load_all_test_data():

    print("\n========== LOADING TEST DATA ==========")

    files = {
        "source1": TEST_SOURCE1,
        "source2": TEST_SOURCE2,
        "source3": TEST_SOURCE3,
    }

    data = {}

    for name, path in tqdm(
        files.items(),
        desc="Test files",
        unit="file",
    ):
        data[name] = load_tsv(path)

    return data


def print_dataset_summary():

    train_data = load_all_train_data()
    test_data = load_all_test_data()

    print("\n========== DATASET SUMMARY ==========")

    print("\nTRAIN:")

    for name, df in train_data.items():
        print(
            f"  {name:15s} → "
            f"{len(df):,} rows × {len(df.columns):,} columns"
        )

    print("\nTEST:")

    for name, df in test_data.items():
        print(
            f"  {name:15s} → "
            f"{len(df):,} rows × {len(df.columns):,} columns"
        )

    print("\n✓ Dataset loading completed successfully.")


if __name__ == "__main__":
    print_dataset_summary()