import os
from typing import List, Tuple

from joblib import load  # type: ignore

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load_training_data() -> Tuple[List[str], List[List[str]]]:
    """Returns (X_train, y_train, mlb)."""
    return load(os.path.join(DATA_DIR, "train.joblib"))


def load_validation_data() -> Tuple[List[str], List[List[str]]]:
    """Returns (X_val, y_val, mlb)."""
    return load(os.path.join(DATA_DIR, "val.joblib"))


def load_test_data() -> List[str]:
    return load(os.path.join(DATA_DIR, "test.joblib"))
