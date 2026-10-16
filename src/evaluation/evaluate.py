"""Evaluate trained runs on the test set.

    python -m src.evaluation.evaluate                                    # every run not evaluated yet
    python -m src.evaluation.evaluate --run fer2013_resnet50_emotion7_f1_s42
    python -m src.evaluation.evaluate --overwrite                        # re-evaluate every run

Writes to each run folder:
    test_metrics.json       accuracy, macro-F1, per-class recall, parameters, timings
    confusion_matrix.png
    predictions.png         examples of correct and incorrect predictions
Inference time depends on the hardware, so only compare runs evaluated on the same device.
"""
import argparse
import json
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from tensorflow import keras

import src.models.transfer  # noqa: F401  registers ImageNetPreprocess so transfer models can be loaded
from src.config.config import RUNS_DIR
from src.data.dataset import load_split, make_dataset
from src.visualization import plots


def predict_timed(model, ds):
    """Predicted class ids and the average inference time per image in milliseconds."""
    model.predict(ds.take(1), verbose=0)  # warm-up: the first call builds the graph
    start = time.perf_counter()
    probs = model.predict(ds, verbose=0)
    ms_per_image = 1000 * (time.perf_counter() - start) / len(probs)
    return probs.argmax(axis=1), ms_per_image


def evaluate_run(run_dir):
    config = json.loads((run_dir / "config.json").read_text())
    train_summary = json.loads((run_dir / "train_summary.json").read_text())
    data = load_split(config["dataset"], config["labels"])
    class_names = data["class_names"]
    y_true = data["y_test"]

    model = keras.models.load_model(run_dir / "model.keras")
    test_ds = make_dataset(data["x_test"], y_true, config["batch_size"],
                           img_size=config["img_size"], channels=config["channels"])
    y_pred, ms_per_image = predict_timed(model, test_ds)

    recalls = recall_score(y_true, y_pred, labels=range(len(class_names)), average=None, zero_division=0)
    metrics = {
        "run": run_dir.name,
        "dataset": config["dataset"],
        "model": config["model"],
        "labels": config["labels"],
        "train_fraction": config["train_fraction"],
        "seed": config["seed"],
        "test_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "test_macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "per_class_recall": {name: round(float(r), 4) for name, r in zip(class_names, recalls)},
        "params": train_summary["params"],
        "train_seconds": train_summary["train_seconds"],
        "ms_per_image": round(ms_per_image, 3),
        "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plots.plot_confusion_matrix(cm, class_names, run_dir / "confusion_matrix.png",
                                title=f"{config['model']} · {config['labels']} (test)")
    plots.plot_predictions(data["x_test"], y_true, y_pred, class_names, run_dir / "predictions.png")

    print(f"{run_dir.name}: accuracy {metrics['test_accuracy']:.4f}, "
          f"macro-F1 {metrics['test_macro_f1']:.4f}, {metrics['ms_per_image']:.2f} ms/image")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", help="Run folder name (repeatable). Default: all runs")
    p.add_argument("--overwrite", action="store_true", help="Re-evaluate runs that already have results")
    args = p.parse_args()

    if args.run:
        run_dirs = [RUNS_DIR / name for name in args.run]
    else:
        run_dirs = sorted(d for d in RUNS_DIR.glob("*") if (d / "model.keras").exists())
        if not args.overwrite:
            run_dirs = [d for d in run_dirs if not (d / "test_metrics.json").exists()]
    if not run_dirs:
        print("Nothing to evaluate (use --overwrite to re-evaluate existing runs).")
    for run_dir in run_dirs:
        evaluate_run(run_dir)


if __name__ == "__main__":
    main()
