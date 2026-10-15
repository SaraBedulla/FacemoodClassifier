"""Model registry. Add the transfer-learning builders here so training can use them.

Each builder takes (num_classes, input_shape) and returns an uncompiled keras.Model.
img_size / channels are the input the model needs; None means the dataset's native
format (RAF-DB: 100x100x3, FER-2013: 48x48x1).
"""
from src.models.scratch_cnn import build_scratch_cnn

MODELS = {
    # name: (builder, img_size, channels)
    "scratch_cnn": (build_scratch_cnn, None, None),
    # "mobilenetv2": (build_mobilenetv2, 224, 3),
    # "resnet50": (build_resnet50, 224, 3),
}
