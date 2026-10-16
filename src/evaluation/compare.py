"""Compare all evaluated runs: mean ± std over seeds for every model and configuration.

    python -m src.evaluation.compare

Run src.evaluation.evaluate first. Writes:
    results/tables/comparison/all_runs.csv        one row per run
    results/tables/comparison/summary.csv         mean / std per (dataset, labels, model, train_fraction)
    results/figures/comparison/<dataset>_<labels>_data_size.png
"""
import json

import pandas as pd

from src.config.config import FIGURES_DIR, RUNS_DIR, TABLES_DIR
from src.models.registry import MODELS
from src.visualization import plots

GROUP = ["dataset", "labels", "model", "train_fraction"]
METRICS = ["test_accuracy", "test_macro_f1", "train_seconds", "ms_per_image"]


def load_runs():
    rows = [json.loads(p.read_text()) for p in sorted(RUNS_DIR.glob("*/test_metrics.json"))]
    runs = pd.DataFrame(rows).drop(columns="per_class_recall")
    # Keep the registry order (scratch_cnn, mobilenetv2, resnet50) in every table and figure.
    runs["model"] = pd.Categorical(runs["model"], categories=list(MODELS), ordered=True)
    return runs.sort_values(GROUP + ["seed"])


def summarise(runs):
    summary = runs.groupby(GROUP, observed=True).agg(
        **{f"{m}_{stat}": (m, stat) for m in METRICS for stat in ["mean", "std"]},
        params=("params", "first"),
        seeds=("seed", "nunique"),
    )
    return summary.reset_index().round(4)


def main():
    runs = load_runs()
    summary = summarise(runs)

    table_dir = TABLES_DIR / "comparison"
    table_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(table_dir / "all_runs.csv", index=False)
    summary.to_csv(table_dir / "summary.csv", index=False)
    print(f"Saved {table_dir / 'summary.csv'}")

    for (dataset, labels), rows in summary.groupby(["dataset", "labels"]):
        name = f"{dataset}_{labels}"
        cols = ["model", "train_fraction", "test_accuracy_mean", "test_accuracy_std",
                "test_macro_f1_mean", "test_macro_f1_std", "params", "ms_per_image_mean", "seeds"]
        print(f"\n{name}\n{rows[cols].to_string(index=False)}")
        plots.plot_data_size(rows, FIGURES_DIR / "comparison" / f"{name}_data_size.png",
                             title=f"Macro-F1 vs training-set size ({dataset}, {labels})")


if __name__ == "__main__":
    main()
