import numpy as np
import pandas as pd
import pytest

from src.data_processing import FEATURES, TARGET, clean


def test_clean_binarises_target(raw_df):
    out = clean(raw_df)
    assert set(out[TARGET].unique()) <= {0, 1}
    assert (out[TARGET] == (raw_df["num"] > 0).astype(int)).all()


def test_clean_keeps_expected_columns(raw_df):
    assert list(clean(raw_df).columns) == FEATURES + [TARGET]


def test_question_marks_become_nan(raw_df):
    raw = raw_df.astype(object)
    raw.loc[0, "ca"] = "?"
    out = clean(raw)
    assert np.isnan(out.loc[0, "ca"])


def test_missing_values_are_not_imputed_here(raw_df):
    assert clean(raw_df)[["ca", "thal"]].isna().sum().sum() == 3


def test_missing_column_raises(raw_df):
    with pytest.raises(ValueError, match="thal"):
        clean(raw_df.drop(columns="thal"))


def test_non_numeric_garbage_raises(raw_df):
    raw = raw_df.astype(object)
    raw.loc[0, "chol"] = "high"
    with pytest.raises((ValueError, TypeError)):
        clean(raw)


def test_duplicates_removed(raw_df):
    doubled = pd.concat([raw_df, raw_df.head(5)], ignore_index=True)
    assert len(clean(doubled)) == len(clean(raw_df))
