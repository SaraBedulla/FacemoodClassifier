"""Train one model on one configuration.

    python -m src.training.train --model scratch_cnn --labels emotion7
    python -m src.training.train --model scratch_cnn --labels affect3
    python -m src.training.train --model scratch_cnn --train-fraction 0.25 --seed 1
    python -m src.training.train --dataset rafdb

Each run writes to results/runs/<dataset>_<model>_<labels>_f<fraction>_s<seed>/:
    model.keras         best weights (lowest validation loss)
    config.json         every setting needed to rebuild the run (img_size, channels, labels, ...)
    history.csv         per-epoch loss/accuracy for train and validation
    train_summary.json  best validation scores, parameter count, training time
    training_curves.png
Test-set evaluation is done separately (src/evaluation/).
"""
import argparse
import json
import time

import pandas as pd
from tensorflow import keras

from src.config.config import DATASETS, DEFAULT_DATASET, RUNS_DIR
from src.data.dataset import class_weights, load_split, make_dataset
from src.models import MODELS
from src.visualization import plots


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(DATASETS))
    p.add_argument("--model", default="scratch_cnn", choices=list(MODELS))
    p.add_argument("--labels", default="emotion7", choices=["emotion7", "affect3"])
    p.add_argument("--train-fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10, help="Early-stopping patience (epochs)")
    p.add_argument("--no-class-weights", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)

    run_name = f"{args.dataset}_{args.model}_{args.labels}_f{args.train_fraction:g}_s{args.seed}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    data = load_split(args.dataset, args.labels, args.train_fraction, args.seed)
    num_classes = len(data["class_names"])
    builder, img_size, channels = MODELS[args.model]
    img_size = img_size or DATASETS[args.dataset]["img_size"]
    channels = channels or DATASETS[args.dataset]["channels"]

    train_ds = make_dataset(data["x_train"], data["y_train"], args.batch_size, training=True,
                            img_size=img_size, channels=channels, seed=args.seed)
    val_ds = make_dataset(data["x_val"], data["y_val"], args.batch_size,
                          img_size=img_size, channels=channels)

    model = builder(num_classes, input_shape=(img_size, img_size, channels))
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=max(1, args.patience // 2), min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(run_dir / "model.keras", monitor="val_loss",
                                        save_best_only=True),
        keras.callbacks.CSVLogger(run_dir / "history.csv"),
    ]
    weights = None if args.no_class_weights else class_weights(data["y_train"], num_classes)

    config = vars(args) | {"img_size": img_size, "channels": channels,
                           "train_size": len(data["y_train"]), "class_weights": weights}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    start = time.perf_counter()
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              class_weight=weights, callbacks=callbacks)
    train_seconds = time.perf_counter() - start

    history = pd.read_csv(run_dir / "history.csv")
    plots.plot_history(history, run_dir / "training_curves.png")

    best = history.loc[history["val_loss"].idxmin()]
    summary = {
        "run": run_name,
        "best_epoch": int(best["epoch"]) + 1,
        "epochs_trained": len(history),
        "best_val_loss": round(float(best["val_loss"]), 4),
        "best_val_accuracy": round(float(best["val_accuracy"]), 4),
        "params": int(model.count_params()),
        "train_seconds": round(train_seconds, 1),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nBest epoch {summary['best_epoch']}: val accuracy {summary['best_val_accuracy']:.4f}")
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
