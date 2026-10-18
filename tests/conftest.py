"""Shared fixtures. Tests use synthetic data shaped like the UCI file, so the
suite runs offline and does not depend on the downloaded dataset."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data_processing import FEATURES
from src.features import build_pipeline


def make_raw(n: int = 150, seed: int = 0) -> pd.DataFrame:
    """Synthetic records in raw UCI format (target column `num`, 0-4)."""
    rng = np.random.default_rng(seed)
    risk = rng.random(n)
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n),
        "sex": rng.integers(0, 2, n),
        "cp": np.where(risk > 0.5, 4, rng.integers(1, 4, n)),
        "trestbps": rng.integers(95, 190, n),
        "chol": rng.integers(130, 400, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": (190 - 60 * risk + rng.normal(0, 10, n)).round(),
        "exang": (risk > 0.6).astype(int),
        "oldpeak": (3 * risk + rng.normal(0, 0.5, n)).clip(0).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3, 6, 7], n).astype(float),
        "num": np.where(risk > 0.5, rng.integers(1, 5, n), 0),
    })
    df.loc[[3, 17], "ca"] = np.nan
    df.loc[[8], "thal"] = np.nan
    return df


@pytest.fixture
def raw_df():
    return make_raw()


@pytest.fixture
def clean_df(raw_df):
    from src.data_processing import clean
    return clean(raw_df)


@pytest.fixture
def fitted_model(clean_df):
    model = build_pipeline(LogisticRegression(max_iter=1000))
    return model.fit(clean_df[FEATURES], clean_df["target"])


@pytest.fixture
def sample_record():
    return {"age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260, "fbs": 0,
            "restecg": 2, "thalach": 120, "exang": 1, "oldpeak": 2.4, "slope": 2,
            "ca": 2, "thal": 7}


@pytest.fixture
def model_file(tmp_path, fitted_model, monkeypatch):
    """Saved model + metadata in a temp dir, with MODEL_PATH pointing at it."""
    from src.train import save_model
    path = save_model(fitted_model, {"model_name": "test_lr", "model_version": "test"},
                      tmp_path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    return path
