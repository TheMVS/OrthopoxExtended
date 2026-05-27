"""
Visual explainability utilities.

Implemented methods:
1. Grad-CAM: gradient-based CNN localization.
2. Grad-CAM++: sharper gradient-based localization for small/multiple regions.
3. Occlusion sensitivity: perturbation-based sanity check.

SHAP is intentionally not included by default because, for CNN image models, it
is slow and often less clinically intuitive than heatmap-based image XAI.
"""
from __future__ import annotations
import os
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Normalize heatmap to [0, 1]."""
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap

def make_gradcam_heatmap(model: tf.keras.Model, img_tensor: np.ndarray, last_conv_layer_name: str, pred_index: int | None = None) -> np.ndarray:
    """Compute standard Grad-CAM for one preprocessed image tensor."""
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    return _normalize_heatmap(heatmap.numpy())

def make_gradcam_plus_plus_heatmap(model: tf.keras.Model, img_tensor: np.ndarray, last_conv_layer_name: str, pred_index: int | None = None) -> np.ndarray:
    """Compute Grad-CAM++ heatmap using higher-order gradient weights."""
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                conv_outputs, predictions = grad_model(img_tensor)
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                score = predictions[:, pred_index]
            first_grads = tape3.gradient(score, conv_outputs)
        second_grads = tape2.gradient(first_grads, conv_outputs)
    third_grads = tape1.gradient(second_grads, conv_outputs)
    conv_outputs = conv_outputs[0]
    first_grads = first_grads[0]
    second_grads = second_grads[0]
    third_grads = third_grads[0]
    eps = 1e-8
    denominator = 2.0 * second_grads + conv_outputs * third_grads
    denominator = tf.where(tf.abs(denominator) > eps, denominator, tf.ones_like(denominator) * eps)
    alphas = second_grads / denominator
    weights = tf.reduce_sum(alphas * tf.nn.relu(first_grads), axis=(0, 1))
    heatmap = tf.reduce_sum(conv_outputs * weights, axis=-1)
    return _normalize_heatmap(heatmap.numpy())

def make_occlusion_sensitivity_heatmap(model: tf.keras.Model, raw_image: np.ndarray, preprocess_fn, pred_index: int, patch_size: int = 32, stride: int = 32, baseline_value: float = 0.0) -> np.ndarray:
    """Hide patches and measure the predicted-class probability drop."""
    image = np.asarray(raw_image, dtype=np.float32)
    h, w = image.shape[:2]
    original_prob = float(model.predict(preprocess_fn(image[None, ...].copy()), verbose=0)[0, pred_index])
    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2, x2 = min(y + patch_size, h), min(x + patch_size, w)
            occluded = image.copy()
            occluded[y:y2, x:x2, :] = baseline_value
            prob = float(model.predict(preprocess_fn(occluded[None, ...]), verbose=0)[0, pred_index])
            drop = max(0.0, original_prob - prob)
            heatmap[y:y2, x:x2] += drop
            counts[y:y2, x:x2] += 1.0
    counts[counts == 0] = 1.0
    return _normalize_heatmap(heatmap / counts)

def overlay_heatmap_on_image(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """Overlay a red heatmap on the original RGB image."""
    base = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGBA")
    heat = Image.fromarray(np.uint8(255 * heatmap)).resize(base.size, resample=Image.BILINEAR).convert("L")
    color = Image.new("RGBA", base.size, (255, 0, 0, 0))
    color.putalpha(heat.point(lambda p: int(p * alpha)))
    return Image.alpha_composite(base, color).convert("RGB")

def save_xai_png(image: np.ndarray, overlay: Image.Image, output_path: str, true_label: str, pred_label: str, confidence: float, method_name: str) -> None:
    """Save original image and XAI overlay side by side."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    original = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    w, h = original.size
    canvas = Image.new("RGB", (w * 2, h + 60), "white")
    canvas.paste(original, (0, 60))
    canvas.paste(overlay, (w, 60))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 20), f"{method_name} | True: {true_label} | Pred: {pred_label} | Conf: {confidence:.3f}", fill=(0, 0, 0))
    canvas.save(output_path, format="PNG")

def save_gradcam_png(image, overlay, output_path, true_label, pred_label, confidence):
    """Backward-compatible wrapper."""
    save_xai_png(image, overlay, output_path, true_label, pred_label, confidence, method_name="Grad-CAM")
