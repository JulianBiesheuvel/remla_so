import os
from typing import Any, AnyStr

from joblib import dump
from joblib import load as _load


def store(obj: Any, *path: AnyStr):
    """Stores an object."""
    p = os.path.join(*path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    dump(obj, p)


def load(*path: AnyStr):
    """Loads an object."""
    return _load(os.path.join(*path))
