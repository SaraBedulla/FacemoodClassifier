"""Read a raw dataset and save fixed train/val/test splits (run once per dataset).

    python -m src.data.prepare --dataset rafdb   --source data/raw/rafdb
    python -m src.data.prepare --dataset fer2013 --source data/raw/fer2013

RAF-DB (basic emotions), either layout works:
    Kaggle version:   <source>/train/<1-7>/*.jpg and <source>/test/<1-7>/*.jpg
    Official version: <source>/EmoLabel/list_patition_label.txt and <source>/Image/aligned/*.jpg
FER-2013:
    Folder version:   <source>/train/<emotion>/*.png and <source>/test/<emotion>/*.png
    CSV version:      fer2013.csv with columns emotion, pixels, Usage

Datasets without a validation set get a stratified 10% of train held out.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src.config.config import DATASETS, DEFAULT_DATASET, SEED, processed_path
from src.data.labels import EMOTIONS

# RAF-DB label ids: 1 surprise, 2 fear, 3 disgust, 4 happiness, 5 sadness, 6 anger, 7 neutral.
RAFDB_LABELS = {1: "surprise", 2: "fear", 3: "disgust", 4: "happy", 5: "sad", 6: "angry", 7: "neutral"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _load_image(path, img_size, channels):
    img = Image.open(path).convert("RGB" if channels == 3 else "L").resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.uint8)
    return arr if channels == 3 else arr[..., None]


def _class_to_label(folder_name):
    """Map a class folder name (emotion word or RAF-DB number) to the canonical label id."""
    name = folder_name.lower()
    if name.isdigit():
        name = RAFDB_LABELS[int(name)]
    return EMOTIONS.index(name)


def _read_folder_split(split_dir, img_size, channels):
    images, labels = [], []
    for class_dir in sorted(p for p in Path(split_dir).iterdir() if p.is_dir()):
        label = _class_to_label(class_dir.name)
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_SUFFIXES:
                images.append(_load_image(img_path, img_size, channels))
                labels.append(label)
    return np.stack(images), np.array(labels, dtype=np.int64)


def _read_rafdb_official(root, img_size, channels):
    images = {"train": [], "test": []}
    labels = {"train": [], "test": []}
    for line in (root / "EmoLabel" / "list_patition_label.txt").read_text().split("\n"):
        if not line.strip():
            continue
        filename, raf_label = line.split()
        split = "train" if filename.startswith("train") else "test"
        aligned = root / "Image" / "aligned" / filename.replace(".jpg", "_aligned.jpg")
        images[split].append(_load_image(aligned, img_size, channels))
        labels[split].append(EMOTIONS.index(RAFDB_LABELS[int(raf_label)]))
    return (np.stack(images["train"]), np.array(labels["train"], dtype=np.int64),
            np.stack(images["test"]), np.array(labels["test"], dtype=np.int64))


def _read_fer2013_csv(csv_path):
    df = pd.read_csv(csv_path)
    pixels = np.stack(
        [np.array(p.split(), dtype=np.uint8).reshape(48, 48) for p in df["pixels"]]
    )[..., None]
    labels = df["emotion"].to_numpy(dtype=np.int64)
    # Standard FER-2013 protocol: PublicTest = validation, PrivateTest = test.
    usage = df["Usage"].to_numpy()
    out = []
    for name in ["Training", "PublicTest", "PrivateTest"]:
        mask = usage == name
        out += [pixels[mask], labels[mask]]
    return tuple(out)


def _find_root(source):
    """Kaggle zips sometimes nest everything in one extra folder (e.g. DATASET/)."""
    if (source / "train").is_dir() or (source / "EmoLabel").is_dir():
        return source
    subdirs = [p for p in source.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return _find_root(subdirs[0])
    raise FileNotFoundError(f"Could not find train/ and test/ (or EmoLabel/) inside {source}")


def read_dataset(dataset, source, val_fraction):
    fmt = DATASETS[dataset]
    source = Path(source)
    if source.suffix == ".csv":
        return _read_fer2013_csv(source)

    root = _find_root(source)
    if (root / "EmoLabel").is_dir():
        x_train, y_train, x_test, y_test = _read_rafdb_official(root, **fmt)
    else:
        x_train, y_train = _read_folder_split(root / "train", **fmt)
        x_test, y_test = _read_folder_split(root / "test", **fmt)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_fraction, stratify=y_train, random_state=SEED
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


def build_splits(dataset, source, val_fraction=0.1):
    """Read `dataset` from `source` and save fixed splits to data/processed/."""
    x_train, y_train, x_val, y_val, x_test, y_test = read_dataset(dataset, source, val_fraction)
    out_path = processed_path(dataset)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,
    )
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(f"{name:>5}: {len(y):6d} images")
    print(f"Image shape {x_train.shape[1:]}. Saved splits to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build fixed train/val/test splits.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(DATASETS))
    parser.add_argument("--source", required=True, help="Path to the downloaded dataset")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Share of train held out for validation")
    args = parser.parse_args()
    build_splits(args.dataset, args.source, args.val_fraction)
