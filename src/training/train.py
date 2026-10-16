"""Train one model on one configuration.

    python -m src.training.train --model scratch_cnn --labels emotion7
    python -m src.training.train --model scratch_cnn --labels affect3
    python -m src.training.train --model scratch_cnn --train-fraction 0.25 --seed 1
    python -m src.training.train --model mobilenetv2 --labels emotion7
    python -m src.training.train --model resnet50 --labels affect3 --batch-size 32
    python -m src.training.train --dataset rafdb

Transfer-learning models (mobilenetv2, resnet50) train in two phases:
    1. head:      backbone frozen, only the new head trains (--lr, up to --head-epochs epochs)
    2. fine-tune: the top --fine-tune-layers of the backbone are unfrozen and trained
                  together with the head (--fine-tune-lr, until --epochs in total)

Each run writes to results/runs/<dataset>_<model>_<labels>_f<fraction>_s<seed>/:
    model.keras         best weights (lowest validation loss, over both phases)
    config.json         every setting needed to rebuild the run (img_size, channels, labels, ...)
    history.csv         per-epoch loss/accuracy for train and validation
    train_summary.json  best validation scores, parameter counts, training time
    training_curves.png
Test-set evaluation is done separately (src/evaluation/).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from tensorflow import keras

from src.config.config import DATASETS, DEFAULT_DATASET, RUNS_DIR
from src.data.dataset import class_weights, load_split, make_dataset
from src.models.registry import MODELS
from src.models.transfer import unfreeze_top
from src.visualization import plots


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(DATASETS))
    p.add_argument("--model", default="scratch_cnn", choices=list(MODELS))
    p.add_argument("--labels", default="emotion7", choices=["emotion7", "affect3"])
    p.add_argument("--train-fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60, help="Maximum epochs in total")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10, help="Early-stopping patience (epochs)")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--img-size", type=int, help="Override the model's input size (e.g. 128 for faster transfer runs)")
    p.add_argument("--head-epochs", type=int, default=10, help="Transfer models: epochs with frozen backbone")
    p.add_argument("--fine-tune-lr", type=float, default=1e-5)
    p.add_argument("--fine-tune-layers", type=int,
                   help="Transfer models: backbone layers to unfreeze (default per model, 0 = never)")
    return p.parse_args()


def make_callbacks(run_dir, patience, checkpoint, append_history):
    """Fresh early stopping / LR schedule for each phase; the checkpoint is shared so it keeps
    the best model over all phases."""
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=max(1, patience // 2), min_lr=1e-6),
        checkpoint,
        keras.callbacks.CSVLogger(run_dir / "history.csv", append=append_history),
    ]


def compile_and_fit(model, lr, train_ds, val_ds, class_weight, callbacks, epochs, initial_epoch=0):
    """Train until `epochs` (in total) or early stopping. Returns the number of epochs done so far."""
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, initial_epoch=initial_epoch,
                        class_weight=class_weight, callbacks=callbacks)
    return initial_epoch + len(history.history["loss"])


def count_trainable_params(model):
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)

    run_name = f"{args.dataset}_{args.model}_{args.labels}_f{args.train_fraction:g}_s{args.seed}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    data = load_split(args.dataset, args.labels, args.train_fraction, args.seed)
    num_classes = len(data["class_names"])
    spec = MODELS[args.model]
    img_size = args.img_size or spec.img_size or DATASETS[args.dataset]["img_size"]
    channels = spec.channels or DATASETS[args.dataset]["channels"]
    fine_tune_layers = spec.fine_tune_layers if args.fine_tune_layers is None else args.fine_tune_layers

    train_ds = make_dataset(data["x_train"], data["y_train"], args.batch_size, training=True,
                            img_size=img_size, channels=channels, seed=args.seed)
    val_ds = make_dataset(data["x_val"], data["y_val"], args.batch_size,
                          img_size=img_size, channels=channels)

    model = spec.builder(num_classes, input_shape=(img_size, img_size, channels))
    model.summary()

    weights = None if args.no_class_weights else class_weights(data["y_train"], num_classes)
    config = vars(args) | {"img_size": img_size, "channels": channels,
                           "fine_tune_layers": fine_tune_layers,
                           "train_size": len(data["y_train"]), "class_weights": weights}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    checkpoint = keras.callbacks.ModelCheckpoint(run_dir / "model.keras", monitor="val_loss",
                                                 save_best_only=True)
    fit = dict(train_ds=train_ds, val_ds=val_ds, class_weight=weights)

    start = time.perf_counter()
    fine_tune_start = None
    if fine_tune_layers:
        print("\nPhase 1: training the new head (backbone frozen)")
        fine_tune_start = compile_and_fit(
            model, args.lr, **fit, epochs=args.head_epochs,
            callbacks=make_callbacks(run_dir, args.patience, checkpoint, append_history=False))

        print(f"\nPhase 2: fine-tuning the top {fine_tune_layers} backbone layers")
        unfreeze_top(model, fine_tune_layers)
        compile_and_fit(
            model, args.fine_tune_lr, **fit, epochs=args.epochs, initial_epoch=fine_tune_start,
            callbacks=make_callbacks(run_dir, args.patience, checkpoint, append_history=True))
    else:
        compile_and_fit(
            model, args.lr, **fit, epochs=args.epochs,
            callbacks=make_callbacks(run_dir, args.patience, checkpoint, append_history=False))
    train_seconds = time.perf_counter() - start

    history = pd.read_csv(run_dir / "history.csv")
    plots.plot_history(history, run_dir / "training_curves.png", fine_tune_start=fine_tune_start)

    best = history.loc[history["val_loss"].idxmin()]
    summary = {
        "run": run_name,
        "best_epoch": int(best["epoch"]) + 1,
        "epochs_trained": len(history),
        "fine_tune_start_epoch": None if fine_tune_start is None else fine_tune_start + 1,
        "best_val_loss": round(float(best["val_loss"]), 4),
        "best_val_accuracy": round(float(best["val_accuracy"]), 4),
        "params": int(model.count_params()),
        "trainable_params": count_trainable_params(model),
        "train_seconds": round(train_seconds, 1),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nBest epoch {summary['best_epoch']}: val accuracy {summary['best_val_accuracy']:.4f}")
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
