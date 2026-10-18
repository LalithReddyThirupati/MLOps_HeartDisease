import json

import joblib
from sklearn.linear_model import LogisticRegression

from src.data_processing import FEATURES
from src.train import cv_summary, evaluate, feature_importance, run_search, save_model


def test_run_search_returns_fitted_best_model(clean_df):
    search = run_search(LogisticRegression(max_iter=1000), {"model__C": [0.1, 1.0]},
                        clean_df[FEATURES], clean_df["target"], n_splits=3)
    summary = cv_summary(search)
    assert 0.0 <= summary["cv_roc_auc_mean"] <= 1.0
    assert {"cv_accuracy_mean", "cv_recall_std"} <= summary.keys()


def test_evaluate_reports_all_metrics(fitted_model, clean_df):
    metrics = evaluate(fitted_model, clean_df[FEATURES], clean_df["target"])
    assert set(metrics) == {"test_accuracy", "test_precision", "test_recall",
                            "test_f1", "test_roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_model_learns_signal(fitted_model, clean_df):
    """Synthetic data has a planted signal; a working pipeline must beat chance."""
    assert evaluate(fitted_model, clean_df[FEATURES], clean_df["target"])["test_roc_auc"] > 0.7


def test_feature_importance_names_match(fitted_model):
    imp = feature_importance(fitted_model)
    assert len(imp) == len(fitted_model.named_steps["preprocess"].get_feature_names_out())


def test_save_model_roundtrip(tmp_path, fitted_model, clean_df):
    path = save_model(fitted_model, {"model_name": "lr"}, tmp_path)
    reloaded = joblib.load(path)
    X = clean_df[FEATURES].head(5)
    assert (reloaded.predict(X) == fitted_model.predict(X)).all()
    assert json.loads((tmp_path / "metadata.json").read_text())["model_name"] == "lr"
