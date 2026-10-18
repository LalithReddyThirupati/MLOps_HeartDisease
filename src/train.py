"""Train, tune and compare classifiers; track every run in MLflow.

For each model family a grid search with stratified 5-fold CV is run. Every
grid candidate is logged as a child MLflow run; the best candidate per family
is logged in the parent run with hold-out metrics and plots. The overall best
model (by CV ROC-AUC) is saved to models/ for serving.

Usage:
    python -m src.train                # full grids
    python -m src.train --quick        # tiny grids, for CI smoke runs
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.data_processing import FEATURES, PROCESSED_PATH, TARGET
from src.features import build_pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def model_space(quick: bool = False) -> dict:
    """Candidate estimators and their hyperparameter grids."""
    if quick:
        return {
            "logistic_regression": (
                LogisticRegression(max_iter=2000, solver="liblinear"),
                {"model__C": [0.1, 1.0]},
            ),
            "random_forest": (
                RandomForestClassifier(random_state=SEED),
                {"model__n_estimators": [50], "model__max_depth": [4]},
            ),
        }
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, solver="liblinear"),
            {"model__C": [0.01, 0.1, 1.0, 10.0], "model__penalty": ["l1", "l2"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=SEED),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 4, 8],
                "model__min_samples_leaf": [1, 3],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=SEED),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
    }


def load_dataset(path: Path = PROCESSED_PATH):
    df = pd.read_csv(path)
    return df[FEATURES], df[TARGET]


def run_search(estimator, grid: dict, X, y, n_splits: int = 5) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        build_pipeline(estimator), grid, scoring=SCORING, refit="roc_auc",
        cv=cv, n_jobs=-1, return_train_score=False,
    )
    search.fit(X, y)
    return search


def cv_summary(search: GridSearchCV) -> dict:
    """Mean and std of every CV metric for the best candidate."""
    i = search.best_index_
    res = search.cv_results_
    out = {}
    for m in SCORING:
        out[f"cv_{m}_mean"] = float(res[f"mean_test_{m}"][i])
        out[f"cv_{m}_std"] = float(res[f"std_test_{m}"][i])
    return out


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "test_accuracy": accuracy_score(y, pred),
        "test_precision": precision_score(y, pred, zero_division=0),
        "test_recall": recall_score(y, pred, zero_division=0),
        "test_f1": f1_score(y, pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y, proba),
    }


def make_plots(model, X, y, name: str, out_dir: Path) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X, y, ax=ax, cmap="Blues",
                                          display_labels=["No disease", "Disease"])
    ax.set_title(f"Confusion matrix - {name}")
    p = out_dir / f"{name}_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_title(f"ROC curve - {name}")
    p = out_dir / f"{name}_roc_curve.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    importance = feature_importance(model)
    if importance is not None:
        top = importance.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(top.index, top.values, color="#2a6f97")
        ax.set_title(f"Top features - {name}")
        p = out_dir / f"{name}_feature_importance.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def feature_importance(model):
    """Importances (trees) or absolute coefficients (linear), by output feature."""
    clf = model.named_steps["model"]
    names = model.named_steps["preprocess"].get_feature_names_out()
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        values = np.abs(clf.coef_[0])
    else:
        return None
    return pd.Series(values, index=names).sort_values(ascending=False)


def save_model(model, metadata: dict, model_dir: Path = MODEL_DIR) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "model.joblib"
    joblib.dump(model, path)
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return path


def log_to_mlflow(name, search, cv_metrics, test_metrics, plots, X_train):
    import mlflow
    import mlflow.sklearn

    with mlflow.start_run(run_name=name) as parent:
        mlflow.set_tag("model_family", name)
        res = search.cv_results_
        for i, params in enumerate(res["params"]):
            with mlflow.start_run(run_name=f"{name}-candidate-{i}", nested=True):
                mlflow.log_params({k.replace("model__", ""): v for k, v in params.items()})
                mlflow.log_metrics({f"cv_{m}_mean": float(res[f"mean_test_{m}"][i])
                                    for m in SCORING})
        mlflow.log_params({k.replace("model__", ""): v
                           for k, v in search.best_params_.items()})
        mlflow.log_metrics({**cv_metrics, **test_metrics})
        for p in plots:
            mlflow.log_artifact(str(p), artifact_path="plots")
        mlflow.sklearn.log_model(search.best_estimator_, artifact_path="model",
                                 input_example=X_train.head(3))
        return parent.info.run_id


def main(quick: bool = False, data_path: Path = PROCESSED_PATH,
         experiment: str = "heart-disease", use_mlflow: bool = True):
    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    if use_mlflow:
        import mlflow
        mlflow.set_experiment(experiment)

    rows, fitted = [], {}
    for name, (estimator, grid) in model_space(quick).items():
        print(f"\n=== {name}: {np.prod([len(v) for v in grid.values()])} candidates ===")
        search = run_search(estimator, grid, X_train, y_train)
        cv_metrics = cv_summary(search)
        test_metrics = evaluate(search.best_estimator_, X_test, y_test)
        plots = make_plots(search.best_estimator_, X_test, y_test, name,
                           REPORT_DIR / "figures")
        run_id = (log_to_mlflow(name, search, cv_metrics, test_metrics, plots, X_train)
                  if use_mlflow else None)

        print(f"best params: {search.best_params_}")
        print(f"CV ROC-AUC {cv_metrics['cv_roc_auc_mean']:.3f} "
              f"± {cv_metrics['cv_roc_auc_std']:.3f} | "
              f"test ROC-AUC {test_metrics['test_roc_auc']:.3f}")
        rows.append({"model": name, "mlflow_run_id": run_id,
                     **cv_metrics, **test_metrics})
        fitted[name] = search

    comparison = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
    REPORT_DIR.mkdir(exist_ok=True)
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    best_name = comparison.iloc[0]["model"]
    best = fitted[best_name]
    metadata = {
        "model_name": best_name,
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_params": {k.replace("model__", ""): v for k, v in best.best_params_.items()},
        "features": FEATURES,
        "metrics": comparison.iloc[0].drop(["model", "mlflow_run_id"]).to_dict(),
        "mlflow_run_id": comparison.iloc[0]["mlflow_run_id"],
    }
    path = save_model(best.best_estimator_, metadata)
    print(f"\nSelected {best_name} (highest CV ROC-AUC). Saved to {path}")
    print(comparison[["model", "cv_roc_auc_mean", "test_roc_auc",
                      "test_accuracy", "test_recall"]].to_string(index=False))
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="tiny grids for CI")
    parser.add_argument("--data", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--experiment", default="heart-disease")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    main(args.quick, args.data, args.experiment, not args.no_mlflow)
