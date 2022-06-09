"""
Helper functions.
"""

import os
from typing import Any, AnyStr

import pandas as pd
from joblib import dump
from joblib import load as _load


def store_dataframe(df: pd.DataFrame, *path: AnyStr) -> None:
    """
    Stores a dataframe as a csv file.

    Creates the parent directories if needed.
    """
    p = os.path.join(*path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_csv(p, mode="w", index=False)


def store(obj: Any, *path: AnyStr) -> None:
    """
    Stores an object.

    Creates the parent directories if needed.
    """
    p = os.path.join(*path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    dump(obj, p)


def load(*path: AnyStr) -> Any:
    """Loads an object."""
    return _load(os.path.join(*path))