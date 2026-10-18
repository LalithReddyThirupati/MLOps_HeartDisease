import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src.data_processing import FEATURES
from src.features import ClinicalFeatures, build_pipeline, build_preprocessor


def test_pct_max_hr_value(clean_df):
    out = ClinicalFeatures().transform(clean_df[FEATURES])
    row = clean_df.iloc[0]
    assert out["pct_max_hr"].iloc[0] == pytest.approx(row["thalach"] / (220 - row["age"]))


def test_preprocessor_outputs_no_nans(clean_df):
    X = ClinicalFeatures().transform(clean_df[FEATURES])
    Xt = build_preprocessor().fit_transform(X)
    assert not np.isnan(np.asarray(Xt, dtype=float)).any()


def test_one_hot_expands_nominal_columns(clean_df):
    X = ClinicalFeatures().transform(clean_df[FEATURES])
    pre = build_preprocessor().fit(X)
    names = pre.get_feature_names_out()
    assert any(n.startswith("nominal__cp_") for n in names)
    assert not any(n == "nominal__cp" for n in names)


def test_pipeline_handles_missing_and_unseen_values(clean_df):
    model = build_pipeline(DummyClassifier(strategy="prior"))
    model.fit(clean_df[FEATURES], clean_df["target"])
    X_new = clean_df[FEATURES].head(2).copy()
    X_new.loc[X_new.index[0], ["ca", "thal"]] = np.nan
    X_new.loc[X_new.index[1], "restecg"] = 9   # category never seen in training
    proba = model.predict_proba(X_new)
    assert proba.shape == (2, 2)
