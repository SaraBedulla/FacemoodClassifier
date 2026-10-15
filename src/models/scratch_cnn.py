"""Baseline CNN trained from scratch (RAF-DB 100x100 RGB or FER-2013 48x48 grayscale)."""
from tensorflow import keras
from tensorflow.keras import layers


def conv_block(x, filters, dropout):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    return layers.Dropout(dropout)(x)


def build_scratch_cnn(num_classes, input_shape=(100, 100, 3), filters=(64, 128, 256, 512)):
    """Four VGG-style blocks (2x conv + BN + ReLU, max-pool, dropout), then a dense head.

    Expects raw 0-255 pixels; rescaling happens inside the model.
    Each block halves the spatial size (RAF-DB: 100 -> 50 -> 25 -> 12 -> 6).
    Global average pooling keeps the head small for any input size.
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    for f in filters:
        x = conv_block(x, f, dropout=0.25)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="scratch_cnn")
