"""
Metrics used throughout the project.

The metric set is intentionally broader than accuracy.

Why?
Medical image datasets are often imbalanced. Accuracy can look high even when
minority classes are poorly recognized. For that reason, macro and calibration
metrics are essential.

Metric interpretation:

- Accuracy:
  Global/micro metric. Useful for readers but not the main criterion.

- Macro F1:
  Main metric. Computes F1 per class and averages classes equally.

- Balanced accuracy:
  Mean recall across classes. It answers whether the model detects each class
  fairly.

- Cohen's kappa:
  Agreement corrected for chance. Useful when class prevalence can inflate raw
  accuracy.

- ECE:
  Expected Calibration Error. Measures whether predicted confidence matches
  empirical correctness.

- Brier score:
  Squared error of predicted probabilities. Lower values mean better probability
  quality.

- Per-class metrics:
  Required for clinical interpretation and error analysis.
"""


from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """Compute multiclass ECE using max probability as confidence."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    conf = np.max(proba, axis=1)
    pred = np.argmax(proba, axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if np.any(mask):
            ece += (np.sum(mask) / len(y_true)) * abs(np.mean(correct[mask]) - np.mean(conf[mask]))
    return float(ece)


def brier_multiclass(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    """Compute multiclass Brier score."""
    y_onehot = np.eye(n_classes, dtype=np.float32)[np.asarray(y_true)]
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))


def compute_core_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict:
    """Compute the project-wide metrics."""
    n_classes = proba.shape[1]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "ece": float(expected_calibration_error(y_true, proba)),
        "brier": float(brier_multiclass(y_true, proba, n_classes=n_classes)),
    }


def classwise_rows(y_true, y_pred, class_names, metadata: dict) -> list[dict]:
    """Return one precision/recall/F1 row per class."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    rows = []
    for i, name in enumerate(class_names):
        rows.append({
            **metadata,
            "class_id": i,
            "class_name": name,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        })
    return rows


def confusion_matrix_long_rows(y_true, y_pred, class_names, metadata: dict) -> list[dict]:
    """Return confusion matrix in long/tabular format."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    rows = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            rows.append({
                **metadata,
                "true_class_id": i,
                "true_class_name": true_name,
                "pred_class_id": j,
                "pred_class_name": pred_name,
                "count": int(cm[i, j]),
            })
    return rows


def confused_pair_rows(y_true, y_pred, class_names, metadata: dict) -> list[dict]:
    """Return only off-diagonal confusion pairs for clinical error analysis."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    rows = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                rows.append({
                    **metadata,
                    "true_class": true_name,
                    "predicted_class": pred_name,
                    "count": int(cm[i, j]),
                })
    return rows



def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for NumPy arrays."""
    logits = np.asarray(logits, dtype=np.float32)
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def apply_temperature_to_proba(proba: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply fixed temperature scaling to probability outputs.

    The model currently outputs softmax probabilities, not logits. Therefore we
    use log-probabilities as pseudo-logits and apply softmax(log(p) / T).
    This is a deterministic post-processing step and does not retrain the model.
    """
    proba = np.asarray(proba, dtype=np.float32)
    eps = 1e-7
    pseudo_logits = np.log(np.clip(proba, eps, 1.0))
    return softmax_np(pseudo_logits / float(temperature)).astype(np.float32)


def predictive_entropy(proba: np.ndarray) -> np.ndarray:
    """Return Shannon entropy for each probability vector."""
    p = np.asarray(proba, dtype=np.float32)
    eps = 1e-7
    return -np.sum(p * np.log(np.clip(p, eps, 1.0)), axis=1)


def entropy_rejection_summary(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    """Metrics for accepted samples under entropy-based abstention."""
    entropy = predictive_entropy(proba)
    accepted = entropy <= float(threshold)
    pred = np.argmax(proba, axis=1)
    coverage = float(np.mean(accepted)) if len(accepted) else 0.0
    if np.any(accepted):
        accepted_accuracy = float(accuracy_score(np.asarray(y_true)[accepted], pred[accepted]))
    else:
        accepted_accuracy = np.nan
    return {
        f"entropy_threshold_{threshold}_coverage": coverage,
        f"entropy_threshold_{threshold}_accepted_accuracy": accepted_accuracy,
    }
