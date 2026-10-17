"""Compare all evaluated runs: mean ± std over seeds for every model and configuration.

    python -m src.evaluation.compare

Run src.evaluation.evaluate first. Writes:
    results/tables/comparison/all_runs.csv          one row per run
    results/tables/comparison/summary.csv           mean / std per (dataset, labels, model, train_fraction)
    results/tables/comparison/fine_tuning.csv       transfer runs: validation accuracy before / after fine-tuning
    results/figures/comparison/model_comparison_macro_f1.png and _accuracy.png
    results/figures/comparison/fine_tuning_gain.png
    results/figures/comparison/<dataset>_<labels>_data_size.png   (only when several training fractions exist)
"""
import json

import pandas as pd

from src.config.config import FIGURES_DIR, RUNS_DIR, TABLES_DIR
from src.models.registry import MODELS
from src.visualization import plots

GROUP = ["dataset", "labels", "model", "train_fraction"]
METRICS = ["test_accuracy", "test_macro_f1", "train_seconds", "ms_per_image"]
DATASET_NAMES = {"fer2013": "FER-2013", "rafdb": "RAF-DB"}


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


def setting_name(dataset, labels):
    return f"{DATASET_NAMES.get(dataset, dataset)} {labels}"


def fine_tuning_table(runs):
    """Validation accuracy at the end of the frozen-backbone phase vs the best epoch, per transfer run."""
    rows = []
    for run in runs.itertuples():
        summary = json.loads((RUNS_DIR / run.run / "train_summary.json").read_text())
        fine_tune_start = summary.get("fine_tune_start_epoch")
        if not fine_tune_start:
            continue
        history = pd.read_csv(RUNS_DIR / run.run / "history.csv")
        frozen = history.iloc[: fine_tune_start - 1]
        rows.append({
            "run": run.run,
            "setting": setting_name(run.dataset, run.labels),
            "model": run.model,
            "frozen_val_accuracy": round(float(frozen["val_accuracy"].max()), 4),
            "best_val_accuracy": summary["best_val_accuracy"],
        })
    return pd.DataFrame(rows)


def main():
    runs = load_runs()
    summary = summarise(runs)

    table_dir = TABLES_DIR / "comparison"
    figure_dir = FIGURES_DIR / "comparison"
    table_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(table_dir / "all_runs.csv", index=False)
    summary.to_csv(table_dir / "summary.csv", index=False)
    print(f"Saved {table_dir / 'summary.csv'}")

    full_data = summary[summary["train_fraction"] == 1.0].copy()
    full_data["setting"] = [setting_name(d, l) for d, l in zip(full_data["dataset"], full_data["labels"])]
    plots.plot_model_comparison(full_data, figure_dir / "model_comparison_macro_f1.png")
    plots.plot_model_comparison(full_data, figure_dir / "model_comparison_accuracy.png",
                                metric="test_accuracy", title="Test accuracy per model")

    fine_tuning = fine_tuning_table(runs[runs["train_fraction"] == 1.0])
    if not fine_tuning.empty:
        fine_tuning.to_csv(table_dir / "fine_tuning.csv", index=False)
        plots.plot_fine_tuning_gain(fine_tuning, figure_dir / "fine_tuning_gain.png")

    for (dataset, labels), rows in summary.groupby(["dataset", "labels"]):
        name = f"{dataset}_{labels}"
        cols = ["model", "train_fraction", "test_accuracy_mean", "test_accuracy_std",
                "test_macro_f1_mean", "test_macro_f1_std", "params", "ms_per_image_mean", "seeds"]
        print(f"\n{name}\n{rows[cols].to_string(index=False)}")
        if rows["train_fraction"].nunique() > 1:
            plots.plot_data_size(rows, figure_dir / f"{name}_data_size.png",
                                 title=f"Macro-F1 vs training-set size ({dataset}, {labels})")


if __name__ == "__main__":
    main()
