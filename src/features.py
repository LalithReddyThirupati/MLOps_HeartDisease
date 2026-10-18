"""Feature engineering shared by training and serving.

Everything here ends up inside one scikit-learn Pipeline that is saved with the
model, so the API applies exactly the same transformations as training did.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Feature groups (UCI coding: cp 1-4, restecg 0-2, slope 1-3, thal 3/6/7, ca 0-3)
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak", "pct_max_hr"]
COUNT = ["ca"]                     # small ordinal count -> mode impute, then scale
BINARY = ["sex", "fbs", "exang"]   # already 0/1 -> impute only
NOMINAL = ["cp", "restecg", "slope", "thal"]  # unordered codes -> one-hot


class ClinicalFeatures(BaseEstimator, TransformerMixin):
    """Adds derived clinical features.

    pct_max_hr: peak heart rate achieved during the exercise test as a fraction
    of the age-predicted maximum (220 - age). A low value suggests the patient
    could not reach normal exertion, which is a known risk indicator.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X["pct_max_hr"] = X["thalach"] / (220 - X["age"])
        return X


def build_preprocessor() -> ColumnTransformer:
    continuous = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    count = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("scale", StandardScaler()),
    ])
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("continuous", continuous, CONTINUOUS),
        ("count", count, COUNT),
        ("binary", SimpleImputer(strategy="most_frequent"), BINARY),
        ("nominal", nominal, NOMINAL),
    ])


def build_pipeline(estimator) -> Pipeline:
    """Full pipeline: derived features -> impute/scale/encode -> classifier."""
    return Pipeline([
        ("clinical", ClinicalFeatures()),
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])
