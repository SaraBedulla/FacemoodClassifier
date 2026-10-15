"""Load the fixed splits and turn them into tf.data pipelines.

Images are kept as raw 0-255 values in each dataset's native format
(RAF-DB: 100x100 RGB, FER-2013: 48x48 grayscale). Each model does its own
normalisation, so the transfer-learning models can apply their own preprocess_input.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.config.config import DEFAULT_DATASET, SEED, processed_path
from src.data.labels import LABEL_SCHEMES, to_scheme


def subsample(x, y, fraction, seed):
    """Stratified subset of the training data (for the data-size experiment)."""
    if fraction >= 1.0:
        return x, y
    x_sub, _, y_sub, _ = train_test_split(
        x, y, train_size=fraction, stratify=y, random_state=seed
    )
    return x_sub, y_sub


def load_split(dataset=DEFAULT_DATASET, scheme="emotion7", train_fraction=1.0, seed=SEED):
    """Load the fixed splits. Returns a dict with x_/y_ train/val/test and class_names.

    Only the training set is subsampled by `train_fraction`; val and test stay
    the same for every run so all results are comparable.
    """
    path = processed_path(dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.data.prepare --dataset {dataset} --source <path>"
        )
    d = np.load(path)
    data = {k: d[k] for k in d.files}
    for split in ["train", "val", "test"]:
        data[f"y_{split}"] = to_scheme(data[f"y_{split}"], scheme)
    data["x_train"], data["y_train"] = subsample(
        data["x_train"], data["y_train"], train_fraction, seed
    )
    data["class_names"] = LABEL_SCHEMES[scheme]
    return data


def class_weights(y, num_classes):
    """'Balanced' class weights to counter class imbalance (e.g. few 'disgust' or 'fear' images)."""
    present = np.unique(y)
    weights = compute_class_weight("balanced", classes=present, y=y)
    out = {i: 1.0 for i in range(num_classes)}
    out.update({int(c): float(w) for c, w in zip(present, weights)})
    return out


def make_dataset(x, y, batch_size=64, training=False, img_size=None, channels=None, seed=SEED):
    """Batched tf.data pipeline yielding float32 images in the 0-255 range.

    img_size / channels convert the images if a model needs a different input,
    e.g. make_dataset(..., img_size=224, channels=3) for ResNet50.
    None keeps the dataset's native size / channels.
    Augmentation is applied only when training=True.
    """
    import tensorflow as tf
    from tensorflow import keras

    augment = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=seed),
            keras.layers.RandomRotation(0.05, seed=seed),
            keras.layers.RandomZoom(0.1, seed=seed),
            keras.layers.RandomTranslation(0.1, 0.1, seed=seed),
        ],
        name="augmentation",
    )

    native_size, native_channels = x.shape[1], x.shape[-1]

    def prepare(img, label):
        img = tf.cast(img, tf.float32)
        if img_size and img_size != native_size:
            img = tf.image.resize(img, (img_size, img_size))
        if channels == 3 and native_channels == 1:
            img = tf.image.grayscale_to_rgb(img)
        elif channels == 1 and native_channels == 3:
            img = tf.image.rgb_to_grayscale(img)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    if training:
        ds = ds.map(lambda img, label: (augment(img, training=True), label),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
