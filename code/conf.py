"""Configuration."""
import os

DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

OUTPUT_DIR = "output"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
