"""Exploratory data analysis: class balance, sample images, mean faces.

    python -m src.data.eda                      # FER-2013 (default)
    python -m src.data.eda --dataset rafdb
Writes figures to results/figures/eda/<dataset>/ and class-count tables to results/tables/eda/<dataset>/.
"""
import argparse

import numpy as np
import pandas as pd

from src.config.config import DATASETS, DEFAULT_DATASET, FIGURES_DIR, TABLES_DIR
from src.data.dataset import load_split
from src.data.labels import AFFECT_CLASSES, EMOTIONS, to_scheme
from src.visualization import plots


def main(dataset):
    out_dir = FIGURES_DIR / "eda" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir = TABLES_DIR / "eda" / dataset
    table_dir.mkdir(parents=True, exist_ok=True)
    data = load_split(dataset, "emotion7")
    splits = {"train": data["y_train"], "val": data["y_val"], "test": data["y_test"]}

    for scheme, names in [("emotion7", EMOTIONS), ("affect3", AFFECT_CLASSES)]:
        counts = {s: np.bincount(to_scheme(y, scheme), minlength=len(names)) for s, y in splits.items()}
        table = pd.DataFrame(counts, index=names)
        table["total"] = table.sum(axis=1)
        table["share_%"] = (100 * table["total"] / table["total"].sum()).round(1)
        print(f"\n{scheme}\n{table}")
        table.to_csv(table_dir / f"class_counts_{scheme}.csv")
        plots.plot_class_distribution(counts, names, out_dir / f"class_distribution_{scheme}.png")

    imbalance = np.bincount(data["y_train"]).max() / np.bincount(data["y_train"]).min()
    print(f"\nLargest/smallest class ratio in train: {imbalance:.1f}x")
    print(f"Pixel mean {data['x_train'].mean():.1f}, std {data['x_train'].std():.1f}")

    plots.plot_samples(data["x_train"], data["y_train"], EMOTIONS, out_dir / "samples.png")
    plots.plot_mean_faces(data["x_train"], data["y_train"], EMOTIONS, out_dir / "mean_faces.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(DATASETS))
    main(parser.parse_args().dataset)
