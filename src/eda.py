"""Exploratory data analysis: generates every EDA figure used in the report.

Usage:
    python -m src.eda        # writes PNGs to reports/figures/eda/
The notebook notebooks/01_eda.ipynb calls the same functions interactively.
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

from src.data_processing import PROCESSED_PATH, TARGET

if "ipykernel" not in sys.modules:   # headless when run as a script / in CI
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

OUT_DIR = Path("reports/figures/eda")
LABELS = {0: "No disease", 1: "Disease"}
PALETTE = {"No disease": "#4c956c", "Disease": "#c44536"}

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
CODEBOOK = {
    "sex": {0: "female", 1: "male"},
    "fbs": {0: "<=120", 1: ">120"},
    "exang": {0: "no", 1: "yes"},
    "cp": {1: "typical", 2: "atypical", 3: "non-anginal", 4: "asymptomatic"},
    "restecg": {0: "normal", 1: "ST-T abn.", 2: "LV hypertrophy"},
    "slope": {1: "up", 2: "flat", 3: "down"},
    "thal": {3: "normal", 6: "fixed", 7: "reversible"},
}


def _save(fig, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return path


def class_balance(df, out_dir=OUT_DIR):
    counts = df[TARGET].map(LABELS).value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[i] for i in counts.index])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v} ({v / len(df):.0%})",
                ha="center")
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    return fig, _save(fig, "class_balance", out_dir)


def missing_values(df, out_dir=OUT_DIR):
    miss = df.isna().sum()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(miss.index, miss.values, color="#6c757d")
    ax.set_title(f"Missing values ({int(miss.sum())} cells of {df.size})")
    ax.set_ylabel("Count")
    return fig, _save(fig, "missing_values", out_dir)


def numeric_distributions(df, out_dir=OUT_DIR):
    data = df.assign(diagnosis=df[TARGET].map(LABELS))
    order = list(LABELS.values())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, NUMERIC):
        sns.histplot(data=data, x=col, hue="diagnosis", hue_order=order, palette=PALETTE,
                     kde=True, element="step", stat="density", common_norm=False, ax=ax)
        ax.set_title(col)
    sns.boxplot(data=data, x="diagnosis", y="oldpeak", hue="diagnosis", order=order,
                palette=PALETTE, legend=False, ax=axes.flat[-1])
    axes.flat[-1].set_title("ST depression (oldpeak) by class")
    fig.suptitle("Numeric feature distributions by diagnosis", fontsize=13)
    return fig, _save(fig, "numeric_distributions", out_dir)


def categorical_rates(df, out_dir=OUT_DIR):
    """Disease rate within each category - more informative than raw counts."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, col in zip(axes.flat, CATEGORICAL):
        sub = df[[col, TARGET]].dropna()
        rate = sub.groupby(col)[TARGET].agg(["mean", "size"])
        labels = [CODEBOOK.get(col, {}).get(int(k), str(int(k))) for k in rate.index]
        ax.bar(labels, rate["mean"], color="#c44536", alpha=0.85)
        for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
            ax.text(i, m + 0.02, f"n={n}", ha="center", fontsize=8)
        ax.axhline(df[TARGET].mean(), ls="--", color="grey", lw=1)
        ax.set_ylim(0, 1.1)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=30)
    axes.flat[0].set_ylabel("Disease rate")
    fig.suptitle("Disease rate per category (dashed = overall rate)", fontsize=13)
    return fig, _save(fig, "categorical_disease_rates", out_dir)


def correlation_heatmap(df, out_dir=OUT_DIR):
    corr = df.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Spearman correlation (robust to the ordinal codes)")
    return fig, _save(fig, "correlation_heatmap", out_dir)


def target_correlations(df):
    return (df.corr(method="spearman")[TARGET].drop(TARGET)
            .sort_values(key=abs, ascending=False))


def run_all(df: pd.DataFrame, out_dir: Path = OUT_DIR) -> list:
    paths = []
    for fn in (class_balance, missing_values, numeric_distributions,
               categorical_rates, correlation_heatmap):
        fig, path = fn(df, out_dir)
        plt.close(fig)
        paths.append(path)
    return paths


if __name__ == "__main__":
    data = pd.read_csv(PROCESSED_PATH)
    for p in run_all(data):
        print("wrote", p)
    print("\nSpearman correlation with target:\n", target_correlations(data).round(3))
