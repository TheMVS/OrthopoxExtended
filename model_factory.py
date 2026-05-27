"""
Model factory.

This file defines all neural models and feature extractors used in the project.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import tensorflow as tf
import Config


@dataclass
class BuiltModel:
    model: tf.keras.Model
    preprocess_fn: Callable
    last_conv_layer_name: str


def get_backbone_constructor_and_preprocess(backbone_name: str):
    """Return constructor, preprocessing function, and a default Grad-CAM layer."""
    if backbone_name == "EfficientNetV2B0":
        from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input
        return EfficientNetV2B0, preprocess_input, "top_activation"

    if backbone_name == "EfficientNetV2S":
        from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input
        return EfficientNetV2S, preprocess_input, "top_activation"

    if backbone_name == "ConvNeXtTiny":
        from tensorflow.keras.applications.convnext import ConvNeXtTiny, preprocess_input
        return ConvNeXtTiny, preprocess_input, "convnext_tiny_stage_3_block_2_depthwise_conv"

    if backbone_name == "ConvNeXtSmall":
        from tensorflow.keras.applications.convnext import ConvNeXtSmall, preprocess_input
        return ConvNeXtSmall, preprocess_input, "convnext_small_stage_3_block_2_depthwise_conv"

    if backbone_name in ("DenseNet201", "MonkeyNetDenseNet201"):
        from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
        return DenseNet201, preprocess_input, "conv5_block32_concat"

    if backbone_name == "ResNet50V2":
        from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
        return ResNet50V2, preprocess_input, "conv5_block3_out"

    if backbone_name == "Xception":
        from tensorflow.keras.applications.xception import Xception, preprocess_input
        return Xception, preprocess_input, "block14_sepconv2_act"

    if backbone_name == "MobileNetV3Large":
        from tensorflow.keras.applications.mobilenet_v3 import MobileNetV3Large, preprocess_input
        return MobileNetV3Large, preprocess_input, "top_activation"

    raise ValueError(f"Unsupported backbone: {backbone_name}")


def set_trainable_tail(base_model: tf.keras.Model, unfreeze_blocks: int) -> None:
    """
    Freeze the full backbone and then unfreeze a small tail.

    This is safer for small datasets than training all layers end-to-end.
    """
    for layer in base_model.layers:
        layer.trainable = False

    if unfreeze_blocks <= 0:
        return

    n_layers = len(base_model.layers)
    start = max(0, n_layers - int(unfreeze_blocks) * 20)

    for layer in base_model.layers[start:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True


def sparse_categorical_crossentropy_with_optional_smoothing(label_smoothing: float = 0.0):
    """
    Sparse categorical crossentropy compatible with TensorFlow versions where
    SparseCategoricalCrossentropy does not support label_smoothing.
    """
    label_smoothing = float(label_smoothing)

    if label_smoothing <= 0.0:
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    def _loss(y_true, y_pred):
        y_true_int = tf.cast(y_true, tf.int32)
        num_classes = tf.shape(y_pred)[-1]

        y_onehot = tf.one_hot(y_true_int, num_classes)
        y_smooth = (
            y_onehot * (1.0 - label_smoothing)
            + label_smoothing / tf.cast(num_classes, tf.float32)
        )

        y_pred_safe = tf.clip_by_value(
            y_pred,
            tf.keras.backend.epsilon(),
            1.0,
        )

        return -tf.reduce_sum(y_smooth * tf.math.log(y_pred_safe), axis=-1)

    return _loss


def focal_loss(gamma: float, label_smoothing: float = 0.0):
    """
    Sparse categorical focal loss for multiclass classification.

    This implementation keeps sparse integer labels but applies optional
    label smoothing manually, so it works with older TensorFlow/Keras versions.
    """
    gamma = float(gamma)
    label_smoothing = float(label_smoothing)

    def _loss(y_true, y_pred):
        y_true_int = tf.cast(y_true, tf.int32)
        num_classes = tf.shape(y_pred)[-1]

        y_onehot = tf.one_hot(y_true_int, num_classes)

        if label_smoothing > 0.0:
            y_target = (
                y_onehot * (1.0 - label_smoothing)
                + label_smoothing / tf.cast(num_classes, tf.float32)
            )
        else:
            y_target = y_onehot

        y_pred_safe = tf.clip_by_value(
            y_pred,
            tf.keras.backend.epsilon(),
            1.0,
        )

        ce_loss = -tf.reduce_sum(y_target * tf.math.log(y_pred_safe), axis=-1)

        pt = tf.reduce_sum(y_onehot * y_pred_safe, axis=-1)
        modulating = tf.pow(1.0 - pt, gamma)

        return modulating * ce_loss

    return _loss


def build_model(config: dict, num_classes: int) -> BuiltModel:
    """Build one fine-tuned DL candidate configuration."""
    backbone_name = config["backbone"]
    ctor, preprocess_fn, gradcam_layer = get_backbone_constructor_and_preprocess(backbone_name)

    inputs = tf.keras.Input(
        shape=Config.IMAGE_SIZE + (Config.CHANNELS,),
        name="image",
    )

    base = ctor(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )

    set_trainable_tail(base, int(config.get("unfreeze_blocks", 1)))

    x = base.output

    if backbone_name == "MonkeyNetDenseNet201":
        x = tf.keras.layers.Conv2D(
            256,
            3,
            padding="same",
            activation=None,
            name="monkeynet_extra_conv",
        )(x)
        x = tf.keras.layers.BatchNormalization(name="monkeynet_extra_bn")(x)
        x = tf.keras.layers.Activation("relu", name="monkeynet_extra_relu")(x)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=2,
            padding="same",
            name="monkeynet_extra_pool",
        )(x)
        gradcam_layer = "monkeynet_extra_conv"

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(float(config["dropout"]), name="dropout")(x)
    x = tf.keras.layers.Dense(
        int(config["dense_units"]),
        activation="relu",
        name="dense",
    )(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="softmax",
    )(x)

    model = tf.keras.Model(inputs, outputs, name=backbone_name)

    return BuiltModel(
        model=model,
        preprocess_fn=preprocess_fn,
        last_conv_layer_name=gradcam_layer,
    )


def compile_model(model: tf.keras.Model, config: dict, stage: str) -> None:
    """Compile model for either head training or fine-tuning."""
    lr = (
        config["learning_rate_head"]
        if stage == "head"
        else config["learning_rate_finetune"]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=float(lr))
    label_smoothing = float(config.get("label_smoothing", 0.0))

    if config.get("use_focal_loss", False):
        loss = focal_loss(
            gamma=float(config["focal_gamma"]),
            label_smoothing=label_smoothing,
        )
    else:
        loss = sparse_categorical_crossentropy_with_optional_smoothing(
            label_smoothing=label_smoothing,
        )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )


def build_feature_extractor(backbone_name: str) -> BuiltModel:
    """Build a frozen CNN feature extractor for ML baselines."""
    ctor, preprocess_fn, gradcam_layer = get_backbone_constructor_and_preprocess(backbone_name)

    inputs = tf.keras.Input(
        shape=Config.IMAGE_SIZE + (Config.CHANNELS,),
        name="image",
    )

    base = ctor(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )

    for layer in base.layers:
        layer.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(base.output)

    model = tf.keras.Model(
        inputs,
        x,
        name=f"{backbone_name}_frozen_embeddings",
    )

    return BuiltModel(
        model=model,
        preprocess_fn=preprocess_fn,
        last_conv_layer_name=gradcam_layer,
    )