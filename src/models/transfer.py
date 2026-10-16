"""Transfer-learning models: an ImageNet-pretrained backbone with a new classification head.

Training happens in two phases (see src/training/train.py):
    1. the backbone is frozen and only the new head is trained;
    2. unfreeze_top() makes the top layers of the backbone trainable and they are
       fine-tuned together with the head at a small learning rate.
"""
from tensorflow import keras
from tensorflow.keras import layers

BACKBONES = {
    "mobilenetv2": (keras.applications.MobileNetV2, keras.applications.mobilenet_v2.preprocess_input),
    "resnet50": (keras.applications.ResNet50, keras.applications.resnet50.preprocess_input),
}


@keras.utils.register_keras_serializable(package="facemood")
class ImageNetPreprocess(layers.Layer):
    """The backbone's own preprocess_input as a layer, so the model takes raw 0-255 pixels.

    MobileNetV2 scales pixels to [-1, 1]; ResNet50 converts RGB to BGR and subtracts the
    ImageNet channel means. A named layer (rather than a Lambda) keeps model.keras loadable.
    """

    def __init__(self, backbone_name, **kwargs):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name

    def call(self, inputs):
        _, preprocess_input = BACKBONES[self.backbone_name]
        return preprocess_input(inputs)

    def get_config(self):
        return super().get_config() | {"backbone_name": self.backbone_name}


def build_transfer_model(backbone_name, num_classes, input_shape=(224, 224, 3), dropout=0.3):
    """Pretrained backbone (frozen) -> global average pooling -> dropout -> softmax."""
    backbone_class, _ = BACKBONES[backbone_name]
    backbone = backbone_class(include_top=False, weights="imagenet",
                              input_shape=input_shape, name="backbone")
    backbone.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = ImageNetPreprocess(backbone_name)(inputs)
    # training=False keeps BatchNorm in inference mode, also later during fine-tuning,
    # so the small face dataset does not overwrite the ImageNet statistics.
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name=backbone_name)


def build_mobilenetv2(num_classes, input_shape=(224, 224, 3)):
    return build_transfer_model("mobilenetv2", num_classes, input_shape)


def build_resnet50(num_classes, input_shape=(224, 224, 3)):
    return build_transfer_model("resnet50", num_classes, input_shape)


def unfreeze_top(model, n_layers):
    """Make the last n_layers of the backbone trainable for fine-tuning.

    BatchNorm layers stay frozen. Recompile the model afterwards for the change to take effect.
    """
    backbone = model.get_layer("backbone")
    backbone.trainable = True
    first_trainable = len(backbone.layers) - n_layers
    for i, layer in enumerate(backbone.layers):
        is_batch_norm = isinstance(layer, layers.BatchNormalization)
        layer.trainable = i >= first_trainable and not is_batch_norm
