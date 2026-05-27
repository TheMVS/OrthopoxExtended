
"""
Automatic paper-oriented plots generated from evaluation CSV outputs.

These functions do not train models. They only consume CSV files produced by
`evaluate.py` and save lightweight PNG summaries for inspection and paper drafts.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_col(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns and not df.empty


def plot_family_macro_f1(summary_df: pd.DataFrame, out_dir: str) -> None:
    """Bar plot of baseline macro-F1 by model family/backbone."""
    if summary_df.empty or "macro_f1_mean" not in summary_df.columns:
        return

    df = summary_df.copy()
    if "inference_method" in df.columns:
        df = df[df["inference_method"] == "baseline"]
    if df.empty:
        return

    label_col = "model_family" if "model_family" in df.columns else "backbone" if "backbone" in df.columns else "config_name"
    group_cols = [label_col]
    if "condition" in df.columns:
        group_cols.append("condition")

    plot_df = (
        df.groupby(group_cols, as_index=False)["macro_f1_mean"]
        .max()
        .sort_values("macro_f1_mean", ascending=False)
    )
    if plot_df.empty:
        return

    labels = plot_df[group_cols].astype(str).agg(" | ".join, axis=1)
    values = plot_df["macro_f1_mean"].values

    plt.figure(figsize=(max(8, len(labels) * 0.55), 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=45, ha="right")
    plt.ylabel("Macro-F1 mean")
    plt.title("Best baseline Macro-F1 by model family/condition")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "family_macro_f1_bar.png"), dpi=200)
    plt.close()


def plot_inference_method_comparison(summary_df: pd.DataFrame, out_dir: str) -> None:
    """Compare post-training inference methods using mean macro-F1 and ECE."""
    if summary_df.empty or "inference_method" not in summary_df.columns:
        return

    metric = "macro_f1_mean"
    if metric not in summary_df.columns:
        return

    df = (
        summary_df.groupby("inference_method", as_index=False)[metric]
        .max()
        .sort_values(metric, ascending=False)
    )
    if df.empty:
        return

    plt.figure(figsize=(9, 5))
    plt.bar(range(len(df)), df[metric].values)
    plt.xticks(range(len(df)), df["inference_method"].astype(str), rotation=45, ha="right")
    plt.ylabel("Best Macro-F1 mean")
    plt.title("Post-training inference strategy comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "inference_method_macro_f1_bar.png"), dpi=200)
    plt.close()

    if "ece_mean" in summary_df.columns:
        ece_df = (
            summary_df.groupby("inference_method", as_index=False)["ece_mean"]
            .min()
            .sort_values("ece_mean", ascending=True)
        )
        plt.figure(figsize=(9, 5))
        plt.bar(range(len(ece_df)), ece_df["ece_mean"].values)
        plt.xticks(range(len(ece_df)), ece_df["inference_method"].astype(str), rotation=45, ha="right")
        plt.ylabel("Best / lowest ECE mean")
        plt.title("Calibration comparison by inference strategy")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "inference_method_ece_bar.png"), dpi=200)
        plt.close()


def plot_calibration_curve(pred_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: str, n_bins: int = 10) -> None:
    """Reliability diagram for the best available configuration/method."""
    required = {"config_name", "condition", "inference_method", "confidence", "correct"}
    if pred_df.empty or not required.issubset(set(pred_df.columns)):
        return

    if not summary_df.empty and "macro_f1_mean" in summary_df.columns:
        best = summary_df.sort_values("macro_f1_mean", ascending=False).iloc[0]
        mask = pred_df["config_name"].astype(str).eq(str(best.get("config_name")))
        if "condition" in pred_df.columns and "condition" in best:
            mask &= pred_df["condition"].astype(str).eq(str(best.get("condition")))
        if "inference_method" in pred_df.columns and "inference_method" in best:
            mask &= pred_df["inference_method"].astype(str).eq(str(best.get("inference_method")))
        df = pred_df[mask].copy()
    else:
        df = pred_df.copy()

    if df.empty:
        df = pred_df.copy()

    conf = df["confidence"].astype(float).values
    corr = df["correct"].astype(float).values
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, accs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if np.any(m):
            bin_centers.append((lo + hi) / 2.0)
            accs.append(float(np.mean(corr[m])))
            counts.append(int(np.sum(m)))

    if not bin_centers:
        return

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(bin_centers, accs, marker="o", label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Calibration curve / reliability diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "calibration_curve_best_model.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(range(len(counts)), counts)
    plt.xticks(range(len(counts)), [f"{c:.2f}" for c in bin_centers], rotation=45, ha="right")
    plt.xlabel("Confidence bin center")
    plt.ylabel("Samples")
    plt.title("Calibration bin support")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "calibration_bin_counts.png"), dpi=200)
    plt.close()


def plot_confusion_matrix_best(cm_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: str) -> None:
    """Plot confusion matrix for best summary row when long-format CM is available."""
    needed = {"true_class_name", "pred_class_name", "count", "config_name"}
    if cm_df.empty or not needed.issubset(set(cm_df.columns)):
        return

    df = cm_df.copy()
    if not summary_df.empty and "macro_f1_mean" in summary_df.columns:
        best = summary_df.sort_values("macro_f1_mean", ascending=False).iloc[0]
        mask = df["config_name"].astype(str).eq(str(best.get("config_name")))
        for col in ["condition", "inference_method"]:
            if col in df.columns and col in best:
                mask &= df[col].astype(str).eq(str(best.get(col)))
        df = df[mask].copy()
    if df.empty:
        return

    classes = sorted(set(df["true_class_name"].astype(str)).union(set(df["pred_class_name"].astype(str))))
    mat = np.zeros((len(classes), len(classes)), dtype=float)
    idx = {c: i for i, c in enumerate(classes)}
    for _, row in df.iterrows():
        mat[idx[str(row["true_class_name"])] , idx[str(row["pred_class_name"])] ] += float(row["count"])

    plt.figure(figsize=(7, 6))
    plt.imshow(mat)
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix for best configuration")
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, int(mat[i, j]), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_best_model.png"), dpi=200)
    plt.close()


def plot_protocol_comparison(protocol_summary_df: pd.DataFrame, out_dir: str) -> None:
    """Bar plot for protocol-comparison macro-F1."""
    if protocol_summary_df.empty or "protocol" not in protocol_summary_df.columns or "macro_f1_mean" not in protocol_summary_df.columns:
        return

    df = protocol_summary_df.sort_values("macro_f1_mean", ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(df)), df["macro_f1_mean"].values)
    plt.xticks(range(len(df)), df["protocol"].astype(str), rotation=45, ha="right")
    plt.ylabel("Macro-F1 mean")
    plt.title("Protocol comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "protocol_comparison_macro_f1.png"), dpi=200)
    plt.close()


def generate_all_report_figures(
    summary_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    protocol_summary_df: pd.DataFrame | None,
    out_dir: str,
    calibration_bins: int = 10,
) -> None:
    """Generate all automatic figures available from final evaluation outputs."""
    _ensure_dir(out_dir)
    plot_family_macro_f1(summary_df, out_dir)
    plot_inference_method_comparison(summary_df, out_dir)
    plot_calibration_curve(prediction_df, summary_df, out_dir, n_bins=calibration_bins)
    plot_confusion_matrix_best(confusion_df, summary_df, out_dir)
    if protocol_summary_df is not None:
        plot_protocol_comparison(protocol_summary_df, out_dir)
