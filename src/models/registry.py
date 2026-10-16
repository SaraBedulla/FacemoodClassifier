"""Model registry: every model that src.training.train can use.

Each builder takes (num_classes, input_shape) and returns an uncompiled keras.Model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.models.scratch_cnn import build_scratch_cnn
from src.models.transfer import build_mobilenetv2, build_resnet50


@dataclass(frozen=True)
class ModelSpec:
    builder: Callable
    img_size: int | None = None  # None = dataset's native size (RAF-DB 100, FER-2013 48)
    channels: int | None = None  # None = dataset's native channels (RAF-DB 3, FER-2013 1)
    fine_tune_layers: int = 0    # > 0: two-phase transfer learning, unfreezing this many backbone layers


MODELS = {
    "scratch_cnn": ModelSpec(build_scratch_cnn),
    # Last 3 inverted-residual stages (block_13 onwards) of 154 layers.
    "mobilenetv2": ModelSpec(build_mobilenetv2, img_size=224, channels=3, fine_tune_layers=38),
    # Last residual stage (conv5) of 175 layers.
    "resnet50": ModelSpec(build_resnet50, img_size=224, channels=3, fine_tune_layers=32),
}
