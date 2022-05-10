from joblib import load
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

def load_training_data():
    """Returns (X_train, y_train, mlb)."""
    return load(os.path.join(DATA_DIR, 'train.joblib'))

def load_validation_data():
    """Returns (X_val, y_val, mlb)."""
    return load('data/val.joblib')

def load_test_data():
    return load('data/test.joblib')