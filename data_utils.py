"""
Data loading and data transformation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

import Config


@dataclass
class DatasetBundle:
    images: np.ndarray
    labels: np.ndarray
    class_names: list[str]
    filepaths: list[str]


def load_image_dataset(base_path: str, variant_name: str) -> DatasetBundle:
    """Load images from base_path/variant_name/class_name/image_file."""
    root = os.path.join(base_path, variant_name)

    if not os.path.isdir(root):
        if os.path.isdir(base_path) and any(
            os.path.isdir(os.path.join(base_path, d)) for d in os.listdir(base_path)
        ):
            root = base_path
        else:
            raise FileNotFoundError(f"Dataset folder not found: {root}")

    class_names = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )

    if not class_names:
        raise ValueError(f"No class folders found under: {root}")

    images, labels, filepaths = [], [], []

    for class_id, class_name in enumerate(class_names):
        cdir = os.path.join(root, class_name)

        for fn in sorted(os.listdir(cdir)):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                continue

            path = os.path.join(cdir, fn)
            img = Image.open(path).convert("RGB").resize(Config.IMAGE_SIZE)

            images.append(np.asarray(img, dtype=np.float32))
            labels.append(class_id)
            filepaths.append(path)

    if not images:
        raise ValueError(f"No valid images found under: {root}")

    return DatasetBundle(
        images=np.stack(images).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        class_names=class_names,
        filepaths=filepaths,
    )


def compute_balanced_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Compute inverse-frequency class weights for imbalanced datasets."""
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )

    return {int(c): float(w) for c, w in zip(classes, weights)}


def build_augmentation_layer(policy_name: str) -> tf.keras.Sequential | None:
    """
    Build a Keras augmentation layer for training only.

    If the selected policy has no augmentation operations, return None.
    This avoids creating an empty Sequential model, which raises an error
    in recent Keras versions.
    """
    p = Config.AUGMENTATION[policy_name]
    layers = []

    if p["flip_left_right"]:
        layers.append(tf.keras.layers.RandomFlip("horizontal"))

    if p["rotation"] > 0:
        layers.append(tf.keras.layers.RandomRotation(factor=p["rotation"]))

    if p["zoom"] > 0:
        layers.append(
            tf.keras.layers.RandomZoom(
                height_factor=p["zoom"],
                width_factor=p["zoom"],
            )
        )

    if p["contrast"] > 0:
        layers.append(tf.keras.layers.RandomContrast(factor=p["contrast"]))

    if p["brightness"] > 0:
        layers.append(
            tf.keras.layers.Lambda(
                lambda x: tf.image.random_brightness(
                    x,
                    max_delta=p["brightness"],
                )
            )
        )

    if not layers:
        return None

    return tf.keras.Sequential(layers, name=f"augmentation_{policy_name}")


def make_tf_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    preprocess_fn: Callable,
    batch_size: int,
    training: bool,
    augmentation_layer: tf.keras.Model | None = None,
) -> tf.data.Dataset:
    """Create a tf.data dataset with optional training-only augmentation."""
    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(images),
            reshuffle_each_iteration=True,
        )

    def _map(x, y):
        x = tf.cast(x, tf.float32)

        if training and augmentation_layer is not None:
            x = augmentation_layer(x, training=True)

        x = preprocess_fn(x)

        return x, tf.cast(y, tf.int32)

    return (
        ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def make_tta_batch(image_batch: np.ndarray, transform_name: str) -> np.ndarray:
    """Apply deterministic test-time augmentation to a batch of raw images."""
    x = np.asarray(image_batch, dtype=np.float32)

    if transform_name == "identity":
        return x

    if transform_name == "flip_left_right":
        return x[:, :, ::-1, :]

    if transform_name == "flip_up_down":
        return x[:, ::-1, :, :]

    if transform_name == "rot90":
        return np.rot90(x, k=1, axes=(1, 2)).copy().astype(np.float32)

    if transform_name == "rot180":
        return np.rot90(x, k=2, axes=(1, 2)).copy().astype(np.float32)

    if transform_name == "rot270":
        return np.rot90(x, k=3, axes=(1, 2)).copy().astype(np.float32)

    raise ValueError(f"Unknown TTA transform: {transform_name}")


def deterministic_offline_augment(image: np.ndarray, transform_id: int) -> np.ndarray:
    """Deterministic augmentation used only for the paper-like protocol comparison."""
    x = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)

    if transform_id == 0:
        y = tf.image.flip_left_right(x)
    elif transform_id == 1:
        y = tf.image.flip_up_down(x)
    elif transform_id == 2:
        y = tf.image.rot90(x, k=1)
    elif transform_id == 3:
        y = tf.image.rot90(x, k=2)
    elif transform_id == 4:
        y = tf.image.rot90(x, k=3)
    elif transform_id == 5:
        y = tf.image.adjust_brightness(x, delta=0.12)
    elif transform_id == 6:
        y = tf.image.adjust_brightness(x, delta=-0.12)
    elif transform_id == 7:
        y = tf.image.adjust_contrast(x, contrast_factor=1.2)
    elif transform_id == 8:
        y = tf.image.adjust_contrast(x, contrast_factor=0.8)
    elif transform_id == 9:
        y = tf.image.central_crop(x, central_fraction=0.88)
        y = tf.image.resize(y, Config.IMAGE_SIZE)
    elif transform_id == 10:
        y = tf.image.resize_with_crop_or_pad(
            x,
            Config.IMAGE_SIZE[0] + 16,
            Config.IMAGE_SIZE[1] + 16,
        )
        y = tf.image.resize(y, Config.IMAGE_SIZE)
    else:
        y = x

    return tf.clip_by_value(y, 0.0, 255.0).numpy()[0].astype(np.float32)


def make_paper_like_augmented_dataset(images, labels, filepaths, repeats: int):
    """
    Create an augmented dataset before splitting.

    This is deliberately used only to reproduce a paper-like optimistic protocol.
    It should not be used for the rigorous evaluation.
    """
    out_images = [images.astype(np.float32)]
    out_labels = [labels.astype(np.int64)]
    out_paths = list(filepaths)

    for rep in range(repeats):
        aug = np.stack(
            [
                deterministic_offline_augment(img, rep % 11)
                for img in images
            ]
        ).astype(np.float32)

        out_images.append(aug)
        out_labels.append(labels.astype(np.int64))
        out_paths.extend(
            [f"{p}::offline_aug_{rep:02d}" for p in filepaths]
        )

    return (
        np.concatenate(out_images),
        np.concatenate(out_labels),
        out_paths,
    )
