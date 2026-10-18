"""Download the UCI Heart Disease (Cleveland) dataset and save an untouched raw copy.

Usage:
    python -m src.download_data
"""
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

UCI_DATASET_ID = 45  # Heart Disease, Cleveland subset
RAW_PATH = Path("data/raw/heart_raw.csv")


def download(dest: Path = RAW_PATH) -> pd.DataFrame:
    """Fetch the dataset from UCI and write it to `dest` without modification."""
    heart = fetch_ucirepo(id=UCI_DATASET_ID)
    df = pd.concat([heart.data.features, heart.data.targets], axis=1)

    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return df


if __name__ == "__main__":
    df = download()
    print(f"Saved {len(df)} rows x {df.shape[1]} columns to {RAW_PATH}")
    missing = df.isna().sum()
    print("Missing values per column:")
    print(missing[missing > 0] if missing.any() else "  none")
