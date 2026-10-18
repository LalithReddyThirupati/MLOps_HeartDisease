"""Cleaning utilities for the heart disease dataset.

Missing values are deliberately left as NaN: imputation happens inside the
training pipeline so the same logic is reused at serving time.

Usage:
    python -m src.data_processing
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/heart_raw.csv")
PROCESSED_PATH = Path("data/processed/heart_clean.csv")

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
RAW_TARGET = "num"   # 0 = no disease, 1-4 = increasing severity
TARGET = "target"    # binary: 0 = absent, 1 = present


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy: numeric features, NaN for missing, binary target."""
    expected = set(FEATURES + [RAW_TARGET])
    absent = expected - set(df.columns)
    if absent:
        raise ValueError(f"Input is missing expected columns: {sorted(absent)}")

    out = df.copy().replace("?", np.nan)
    out[FEATURES] = out[FEATURES].apply(pd.to_numeric)  # raises on bad values
    out[TARGET] = (pd.to_numeric(out[RAW_TARGET]) > 0).astype(int)

    out = out[FEATURES + [TARGET]].drop_duplicates().reset_index(drop=True)
    return out


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    cleaned = clean(load_raw())
    save_processed(cleaned)
    print(f"Cleaned shape: {cleaned.shape}")
    print("Missing values:\n", cleaned.isna().sum()[lambda s: s > 0])
    print("Class balance:\n", cleaned[TARGET].value_counts())
