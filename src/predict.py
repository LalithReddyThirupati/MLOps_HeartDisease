"""Inference helpers used by the API and from the command line.

Usage:
    python -m src.predict samples/sample_request.json
"""
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.data_processing import FEATURES

DEFAULT_MODEL_PATH = "models/model.joblib"
THRESHOLD = 0.5


def model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))


def load_model(path: Path | None = None):
    return joblib.load(path or model_path())


def load_metadata(path: Path | None = None) -> dict:
    meta = (path or model_path()).with_name("metadata.json")
    return json.loads(meta.read_text()) if meta.exists() else {}


def predict_one(model, record: dict) -> dict:
    """Predict for one patient. Missing optional fields (ca, thal) may be None."""
    df = pd.DataFrame([record], columns=FEATURES, dtype=float)
    p_disease = float(model.predict_proba(df)[0, 1])
    prediction = int(p_disease >= THRESHOLD)
    return {
        "prediction": prediction,
        "label": "heart disease" if prediction else "no heart disease",
        "probability_disease": round(p_disease, 4),
        "confidence": round(max(p_disease, 1 - p_disease), 4),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m src.predict <request.json>")
    payload = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(predict_one(load_model(), payload), indent=2))
