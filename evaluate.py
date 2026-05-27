"""
Final evaluation stage.

This version trains each selected configuration once per condition/seed/fold and
then evaluates multiple post-training inference strategies without retraining:
- baseline
- simple TTA
- advanced TTA
- weighted TTA
- MC Dropout
- fixed temperature scaling
- entropy-based rejection metrics
"""

from __future__ import annotations
import json
import os
import shutil
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import tensorflow as tf

import Config
from data_utils import (
    load_image_dataset,
    make_tf_dataset,
    build_augmentation_layer,
    compute_balanced_class_weights,
    make_tta_batch,
    make_paper_like_augmented_dataset,
)
from gradcam_utils import (
    make_gradcam_heatmap,
    make_gradcam_plus_plus_heatmap,
    make_occlusion_sensitivity_heatmap,
    overlay_heatmap_on_image,
    save_xai_png,
)
from metrics import (
    compute_core_metrics,
    classwise_rows,
    confusion_matrix_long_rows,
    confused_pair_rows,
    apply_temperature_to_proba,
    entropy_rejection_summary,
    predictive_entropy,
)
from model_factory import build_model, compile_model
from stats_utils import friedman_from_seed_table, pairwise_comparisons, mean_ci95, paired_test
from ml_baselines import run_ml_baselines
from report_utils import generate_all_report_figures


def make_callbacks():
    """Callbacks for final evaluation. No checkpoint is required here."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=Config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=Config.REDUCE_LR_PATIENCE,
            verbose=0,
        ),
    ]


def _predict_for_transforms(built, images: np.ndarray, batch_size: int, transforms: list[str], weights: dict[str, float] | None = None) -> np.ndarray:
    """Predict with deterministic transforms while accumulating probabilities incrementally."""
    proba_sum = None
    total_weight = 0.0

    for tname in transforms:
        transformed = make_tta_batch(images, tname)
        ds = make_tf_dataset(
            transformed,
            np.zeros(len(transformed), dtype=np.int64),
            preprocess_fn=built.preprocess_fn,
            batch_size=batch_size,
            training=False,
            augmentation_layer=None,
        )
        proba = built.model.predict(ds, verbose=0).astype(np.float32)
        weight = 1.0 if weights is None else float(weights.get(tname, 1.0))
        proba_sum = proba * weight if proba_sum is None else proba_sum + proba * weight
        total_weight += weight

    return (proba_sum / max(total_weight, 1e-8)).astype(np.float32)


def _predict_mc_dropout(built, images: np.ndarray, batch_size: int, passes: int) -> np.ndarray:
    """Monte Carlo Dropout inference with dropout active and no gradient computation."""
    proba_sum = None
    n = len(images)
    for _ in range(int(passes)):
        pass_preds = []
        for start in range(0, n, batch_size):
            batch = images[start:start + batch_size].astype(np.float32)
            batch = built.preprocess_fn(batch.copy())
            pred = built.model(batch, training=True).numpy().astype(np.float32)
            pass_preds.append(pred)
        proba = np.concatenate(pass_preds, axis=0)
        proba_sum = proba if proba_sum is None else proba_sum + proba
    return (proba_sum / float(passes)).astype(np.float32)


def predict_probabilities_by_method(built, images: np.ndarray, batch_size: int, method: str, baseline_proba: np.ndarray | None = None) -> np.ndarray:
    """Predict probabilities for one post-training inference strategy."""
    if method == "baseline":
        return _predict_for_transforms(built, images, batch_size, ["identity"])

    if method == "tta_simple":
        return _predict_for_transforms(built, images, batch_size, Config.TTA_SIMPLE_TRANSFORMS)

    if method == "tta_advanced":
        return _predict_for_transforms(built, images, batch_size, Config.TTA_ADVANCED_TRANSFORMS)

    if method == "weighted_tta":
        return _predict_for_transforms(
            built,
            images,
            batch_size,
            Config.WEIGHTED_TTA_TRANSFORMS,
            weights=Config.WEIGHTED_TTA_WEIGHTS,
        )

    if method == "mc_dropout":
        return _predict_mc_dropout(built, images, batch_size, Config.MC_DROPOUT_PASSES)

    if method.startswith("temperature_"):
        if baseline_proba is None:
            baseline_proba = _predict_for_transforms(built, images, batch_size, ["identity"])
        temperature = float(method.replace("temperature_", ""))
        return apply_temperature_to_proba(baseline_proba, temperature)

    raise ValueError(f"Unknown inference method: {method}")


def save_test_gradcam_pngs(built, raw_test_images, y_true, proba, y_pred, class_names, config_name, condition_name, inference_method, seed, fold_id):
    """Save visual explanations for selected test samples."""
    if not Config.SAVE_GRADCAM and not Config.SAVE_GRADCAM_PLUS_PLUS and not Config.SAVE_OCCLUSION_SENSITIVITY:
        return

    indices = list(range(len(raw_test_images)))
    limit = getattr(Config, "XAI_MAX_IMAGES_PER_FOLD", None)
    if limit is None:
        limit = Config.GRADCAM_MAX_IMAGES_PER_FOLD
    if limit is not None:
        indices = indices[:limit]

    for i in indices:
        x = raw_test_images[i:i+1].astype(np.float32)
        x_proc = built.preprocess_fn(x.copy())
        pred_idx = int(y_pred[i])
        conf = float(np.max(proba[i]))
        true_name = class_names[int(y_true[i])]
        pred_name = class_names[pred_idx]
        base_filename = f"img_{i:04d}_true_{true_name}_pred_{pred_name}_conf_{conf:.3f}.png"

        if Config.SAVE_GRADCAM:
            try:
                heatmap = make_gradcam_heatmap(built.model, x_proc, built.last_conv_layer_name, pred_idx)
                overlay = overlay_heatmap_on_image(raw_test_images[i], heatmap)
                out_dir = os.path.join(Config.GRADCAM_DIR, "gradcam", config_name, condition_name, inference_method, f"seed_{seed}", f"fold_{fold_id:02d}")
                save_xai_png(raw_test_images[i], overlay, os.path.join(out_dir, base_filename), true_name, pred_name, conf, "Grad-CAM")
            except Exception as exc:
                print(f"[WARN] Grad-CAM failed for image {i}: {exc}", flush=True)

        if Config.SAVE_GRADCAM_PLUS_PLUS:
            try:
                heatmap = make_gradcam_plus_plus_heatmap(built.model, x_proc, built.last_conv_layer_name, pred_idx)
                overlay = overlay_heatmap_on_image(raw_test_images[i], heatmap)
                out_dir = os.path.join(Config.GRADCAM_DIR, "gradcam_plus_plus", config_name, condition_name, inference_method, f"seed_{seed}", f"fold_{fold_id:02d}")
                save_xai_png(raw_test_images[i], overlay, os.path.join(out_dir, base_filename), true_name, pred_name, conf, "Grad-CAM++")
            except Exception as exc:
                print(f"[WARN] Grad-CAM++ failed for image {i}: {exc}", flush=True)

        if Config.SAVE_OCCLUSION_SENSITIVITY:
            try:
                heatmap = make_occlusion_sensitivity_heatmap(
                    model=built.model,
                    raw_image=raw_test_images[i],
                    preprocess_fn=built.preprocess_fn,
                    pred_index=pred_idx,
                    patch_size=Config.OCCLUSION_PATCH_SIZE,
                    stride=Config.OCCLUSION_STRIDE,
                    baseline_value=Config.OCCLUSION_BASELINE_VALUE,
                )
                overlay = overlay_heatmap_on_image(raw_test_images[i], heatmap)
                out_dir = os.path.join(Config.GRADCAM_DIR, "occlusion_sensitivity", config_name, condition_name, inference_method, f"seed_{seed}", f"fold_{fold_id:02d}")
                save_xai_png(raw_test_images[i], overlay, os.path.join(out_dir, base_filename), true_name, pred_name, conf, "Occlusion")
            except Exception as exc:
                print(f"[WARN] Occlusion sensitivity failed for image {i}: {exc}", flush=True)


def train_and_predict_fold(train_images, train_labels, test_images, test_labels, config, condition_name: str, seed: int, fold_id: int, class_names: list[str]):
    """Train once on one fold and compute all configured inference strategies."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    cfg = deepcopy(config)
    built = build_model(cfg, num_classes=len(class_names))

    if condition_name == "original":
        aug = None
    elif condition_name == "augmented":
        aug = build_augmentation_layer(cfg["augmentation_policy"])
    elif condition_name.startswith("ablation_"):
        policy = condition_name.replace("ablation_", "")
        aug = None if policy == "none" else build_augmentation_layer(policy)
    else:
        aug = None

    class_weights = compute_balanced_class_weights(train_labels) if Config.USE_CLASS_WEIGHTS else None

    sub_train_idx, val_idx = train_test_split(
        np.arange(len(train_labels)),
        test_size=0.15,
        random_state=seed,
        stratify=train_labels,
    )

    train_ds = make_tf_dataset(
        train_images[sub_train_idx], train_labels[sub_train_idx],
        built.preprocess_fn, cfg["batch_size"], training=True, augmentation_layer=aug
    )
    val_ds = make_tf_dataset(
        train_images[val_idx], train_labels[val_idx],
        built.preprocess_fn, cfg["batch_size"], training=False, augmentation_layer=None
    )

    compile_model(built.model, cfg, stage="head")
    built.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(cfg["epochs_head"]),
        verbose=0,
        callbacks=make_callbacks(),
        class_weight=class_weights,
    )

    compile_model(built.model, cfg, stage="finetune")
    built.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(cfg["epochs_finetune"]),
        verbose=0,
        callbacks=make_callbacks(),
        class_weight=class_weights,
    )

    outputs = {}
    baseline_proba = None
    for method in Config.FINAL_INFERENCE_METHODS:
        proba = predict_probabilities_by_method(
            built=built,
            images=test_images,
            batch_size=int(cfg["batch_size"]),
            method=method,
            baseline_proba=baseline_proba,
        )
        if method == "baseline":
            baseline_proba = proba
        pred = np.argmax(proba, axis=1)
        outputs[method] = (pred, proba)

    # Save XAI only for the baseline prediction to keep runtime/storage controlled.
    baseline_pred, baseline_proba = outputs["baseline"]
    save_test_gradcam_pngs(
        built=built,
        raw_test_images=test_images,
        y_true=test_labels,
        proba=baseline_proba,
        y_pred=baseline_pred,
        class_names=class_names,
        config_name=cfg["config_name"],
        condition_name=condition_name,
        inference_method="baseline",
        seed=seed,
        fold_id=fold_id,
    )

    return outputs


def _entropy_extra_metrics(y_true, proba) -> dict:
    """Add entropy and abstention summaries to a metric row."""
    out = {"entropy_mean": float(np.mean(predictive_entropy(proba)))}
    for threshold in Config.ENTROPY_REJECTION_THRESHOLDS:
        out.update(entropy_rejection_summary(y_true, proba, threshold))
    return out


def run_config_across_seeds(bundle, config: dict, condition_name: str):
    """Evaluate one DL configuration across EVAL_SEEDS and OUTER_CV folds."""
    fold_rows, seed_rows, class_rows, cm_rows, error_rows, pred_rows = [], [], [], [], [], []
    images, labels, class_names = bundle.images, bundle.labels, bundle.class_names

    for seed in Config.EVAL_SEEDS:
        skf = StratifiedKFold(n_splits=Config.OUTER_CV, shuffle=True, random_state=seed)
        seed_metric_rows = []

        for fold_id, (tr, te) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
            print(f"[EVAL] {config['config_name']} | {condition_name} | seed={seed} | fold={fold_id}", flush=True)
            outputs = train_and_predict_fold(
                train_images=images[tr],
                train_labels=labels[tr],
                test_images=images[te],
                test_labels=labels[te],
                config=config,
                condition_name=condition_name,
                seed=seed,
                fold_id=fold_id,
                class_names=class_names,
            )

            for inference_method, (pred, proba) in outputs.items():
                m = compute_core_metrics(labels[te], pred, proba)
                m.update(_entropy_extra_metrics(labels[te], proba))
                meta = {
                    "family": "fine_tuned_dl",
                    "model_family": Config.MODEL_FAMILY.get(config["backbone"], "other"),
                    "config_name": config["config_name"],
                    "condition": condition_name,
                    "inference_method": inference_method,
                    "seed": seed,
                    "fold": fold_id,
                    "backbone": config["backbone"],
                }
                row = {**meta, **m}
                fold_rows.append(row)
                seed_metric_rows.append(row)
                class_rows.extend(classwise_rows(labels[te], pred, class_names, meta))
                cm_rows.extend(confusion_matrix_long_rows(labels[te], pred, class_names, meta))
                error_rows.extend(confused_pair_rows(labels[te], pred, class_names, meta))

                te_indices = np.asarray(te)
                confidences = np.max(proba, axis=1)
                entropies = predictive_entropy(proba)
                for local_i, global_i in enumerate(te_indices):
                    pred_rows.append({
                        **meta,
                        "sample_index": int(global_i),
                        "filepath": bundle.filepaths[int(global_i)] if hasattr(bundle, "filepaths") else "",
                        "true_class_id": int(labels[global_i]),
                        "true_class_name": class_names[int(labels[global_i])],
                        "pred_class_id": int(pred[local_i]),
                        "pred_class_name": class_names[int(pred[local_i])],
                        "confidence": float(confidences[local_i]),
                        "entropy": float(entropies[local_i]),
                        "correct": int(pred[local_i] == labels[global_i]),
                    })

            tf.keras.backend.clear_session()

        seed_df = pd.DataFrame(seed_metric_rows)
        for inference_method, sub in seed_df.groupby("inference_method"):
            seed_rows.append({
                "family": "fine_tuned_dl",
                "model_family": Config.MODEL_FAMILY.get(config["backbone"], "other"),
                "config_name": config["config_name"],
                "condition": condition_name,
                "inference_method": inference_method,
                "seed": seed,
                "backbone": config["backbone"],
                "accuracy": float(sub["accuracy"].mean()),
                "macro_f1": float(sub["macro_f1"].mean()),
                "balanced_acc": float(sub["balanced_acc"].mean()),
                "kappa": float(sub["kappa"].mean()),
                "ece": float(sub["ece"].mean()),
                "brier": float(sub["brier"].mean()),
                "entropy_mean": float(sub["entropy_mean"].mean()),
            })

    return fold_rows, seed_rows, class_rows, cm_rows, error_rows, pred_rows


def summarize_seed_results(seed_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated seed results with mean/std/95% CI."""
    metrics = ["accuracy", "macro_f1", "balanced_acc", "kappa", "ece", "brier", "entropy_mean"]
    group_cols = ["family", "config_name"]
    for optional in ["model_family", "backbone", "condition", "inference_method"]:
        if optional in seed_df.columns:
            group_cols.append(optional)

    rows = []
    for keys, sub in seed_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in metrics:
            if metric not in sub.columns:
                continue
            mean, lo, hi = mean_ci95(sub[metric].values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = float(sub[metric].std(ddof=1)) if len(sub) > 1 else 0.0
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["macro_f1_mean", "balanced_acc_mean"], ascending=False)


def run_statistical_analysis(seed_df: pd.DataFrame, out_omnibus: str, out_pairwise: str):
    """Run Friedman and pairwise tests on macro F1."""
    df = seed_df.copy()
    label_cols = ["config_name"]
    for c in ["condition", "inference_method"]:
        if c in df.columns:
            label_cols.append(c)
    df["config_label"] = df[label_cols].astype(str).agg("__".join, axis=1)

    pivot = df.pivot_table(index="seed", columns="config_label", values="macro_f1", aggfunc="mean").sort_index(axis=1)
    stat, p = friedman_from_seed_table(pivot)
    omnibus = pd.DataFrame([{"metric": "macro_f1", "test": "friedman", "statistic": stat, "p_value": p}])
    pairwise_input = df.rename(columns={"config_label": "config_name"})
    pairwise = pairwise_comparisons(pairwise_input[["config_name", "seed", "macro_f1"]], metric_col="macro_f1")
    omnibus.to_csv(out_omnibus, index=False)
    pairwise.to_csv(out_pairwise, index=False)
    return omnibus, pairwise


def _select_configs_for_final(top_configs: list[dict]) -> list[dict]:
    """Optionally select the best configuration per family for final evaluation."""
    if not getattr(Config, "EVALUATE_TOP_PER_FAMILY_ONLY", False):
        return top_configs

    df = pd.DataFrame(top_configs)
    if "model_family" not in df.columns:
        df["model_family"] = df["backbone"].map(Config.MODEL_FAMILY).fillna("other")

    sort_cols = ["search_macro_f1_mean", "search_balanced_acc_mean", "search_kappa_mean"]
    selected = []
    for _, sub in df.groupby("model_family"):
        selected.append(sub.sort_values(sort_cols, ascending=[False, False, False]).head(Config.EVALUATE_TOP_K_PER_FAMILY))
    out = pd.concat(selected, axis=0) if selected else df.head(0)
    if Config.EVALUATE_GLOBAL_TOP_EXTRA > 0:
        out = pd.concat([out, df.sort_values(sort_cols, ascending=[False, False, False]).head(Config.EVALUATE_GLOBAL_TOP_EXTRA)], axis=0)
    out = out.drop_duplicates(subset=["config_name"])
    return out.to_dict(orient="records")


def train_single_split_and_score(train_images, train_labels, test_images, test_labels, class_names, config, condition_name, seed, protocol):
    """Train and evaluate one 80/20 split for protocol comparison using baseline inference."""
    outputs = train_and_predict_fold(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        config=config,
        condition_name=condition_name,
        seed=seed,
        fold_id=1,
        class_names=class_names,
    )
    pred, proba = outputs["baseline"]
    m = compute_core_metrics(test_labels, pred, proba)
    return {"seed": seed, "protocol": protocol, "condition": condition_name, **m}


def run_best_model_protocol_comparison(bundle, best_config: dict):
    """Compare original-only, rigorous train-only augmentation, and paper-like augmentation."""
    rows = []

    for seed in Config.PROTOCOL_COMPARISON_SEEDS:
        tr, te = train_test_split(
            np.arange(len(bundle.labels)),
            test_size=Config.PROTOCOL_COMPARISON_TEST_SIZE,
            random_state=seed,
            stratify=bundle.labels,
        )
        rows.append(train_single_split_and_score(
            bundle.images[tr], bundle.labels[tr],
            bundle.images[te], bundle.labels[te],
            bundle.class_names, best_config, "original", seed, "original_only_clean_split"
        ))
        rows.append(train_single_split_and_score(
            bundle.images[tr], bundle.labels[tr],
            bundle.images[te], bundle.labels[te],
            bundle.class_names, best_config, "augmented", seed, "rigorous_clean_split_train_only_aug"
        ))

        leak_images, leak_labels, _ = make_paper_like_augmented_dataset(
            bundle.images, bundle.labels, bundle.filepaths, repeats=Config.PAPER_LIKE_AUG_REPEATS
        )
        tr_l, te_l = train_test_split(
            np.arange(len(leak_labels)),
            test_size=Config.PROTOCOL_COMPARISON_TEST_SIZE,
            random_state=seed,
            stratify=leak_labels,
        )
        rows.append(train_single_split_and_score(
            leak_images[tr_l], leak_labels[tr_l],
            leak_images[te_l], leak_labels[te_l],
            bundle.class_names, best_config, "original", seed, "paper_like_leaky_augmented_split"
        ))

    runs = pd.DataFrame(rows)
    summary = summarize_seed_results(runs.rename(columns={"protocol": "condition"})).rename(columns={"condition": "protocol"})

    stat_rows = []
    protocols = sorted(runs["protocol"].unique())
    for i, a in enumerate(protocols):
        for b in protocols[i+1:]:
            xa = runs[runs["protocol"] == a].sort_values("seed")["macro_f1"].values
            xb = runs[runs["protocol"] == b].sort_values("seed")["macro_f1"].values
            stat_rows.append({"protocol_a": a, "protocol_b": b, "metric": "macro_f1", **paired_test(xa, xb)})
    stats = pd.DataFrame(stat_rows)

    reference = pd.DataFrame([
        {"reference": Config.REFERENCE_PAPER_NAME, "reported_setting": "original_dataset", "accuracy": Config.REFERENCE_PAPER_REPORTED_ACCURACY_ORIGINAL},
        {"reference": Config.REFERENCE_PAPER_NAME, "reported_setting": "augmented_dataset", "accuracy": Config.REFERENCE_PAPER_REPORTED_ACCURACY_AUGMENTED},
    ])
    return runs, summary, stats, reference


def run_augmentation_ablation(bundle, best_config: dict):
    """Run augmentation-policy ablation for the best model."""
    all_fold, all_seed = [], []
    for policy in Config.AUGMENTATION_ABLATION_POLICIES:
        cfg = deepcopy(best_config)
        cfg["augmentation_policy"] = policy if policy != "none" else "light"
        condition = f"ablation_{policy}"
        fold_rows, seed_rows, _, _, _, _ = run_config_across_seeds(bundle, cfg, condition)
        all_fold.extend(fold_rows)
        all_seed.extend(seed_rows)

    fold_df = pd.DataFrame(all_fold)
    seed_df = pd.DataFrame(all_seed)
    summary_df = summarize_seed_results(seed_df)
    return fold_df, seed_df, summary_df


def main():
    if os.path.isdir(Config.GRADCAM_DIR):
        shutil.rmtree(Config.GRADCAM_DIR)
    os.makedirs(Config.GRADCAM_DIR, exist_ok=True)

    bundle = load_image_dataset(Config.BASE_PATH, Config.VARIANT_NAME)

    with open(Config.TOP_CONFIGS_JSON, "r", encoding="utf-8") as f:
        top_configs = json.load(f)
    top_configs = _select_configs_for_final(top_configs)

    all_fold, all_seed, all_class, all_cm, all_errors, all_predictions = [], [], [], [], [], []

    for config in top_configs:
        cfg = deepcopy(config)
        for condition in ["original", "augmented"]:
            fold_rows, seed_rows, class_rows, cm_rows, error_rows, pred_rows = run_config_across_seeds(bundle, cfg, condition)
            all_fold.extend(fold_rows)
            all_seed.extend(seed_rows)
            all_class.extend(class_rows)
            all_cm.extend(cm_rows)
            all_errors.extend(error_rows)
            all_predictions.extend(pred_rows)

    fold_df = pd.DataFrame(all_fold)
    seed_df = pd.DataFrame(all_seed)
    summary_df = summarize_seed_results(seed_df)

    fold_df.to_csv(Config.EVAL_FOLD_CSV, index=False)
    seed_df.to_csv(Config.EVAL_SEED_CSV, index=False)
    summary_df.to_csv(Config.EVAL_SUMMARY_CSV, index=False)
    pd.DataFrame(all_class).to_csv(Config.EVAL_CLASSWISE_CSV, index=False)
    pd.DataFrame(all_cm).to_csv(Config.EVAL_CONFUSION_LONG_CSV, index=False)
    prediction_df = pd.DataFrame(all_predictions)
    prediction_df.to_csv(Config.EVAL_PREDICTIONS_CSV, index=False)

    err_df = pd.DataFrame(all_errors)
    if not err_df.empty:
        err_df = err_df.groupby(["true_class", "predicted_class"], as_index=False)["count"].sum().sort_values("count", ascending=False)
    err_df.to_csv(Config.EVAL_ERROR_ANALYSIS_CSV, index=False)

    run_statistical_analysis(seed_df, Config.OMNIBUS_STATS_CSV, Config.PAIRWISE_STATS_CSV)

    # Best baseline DL model for ablation and protocol comparison.
    baseline_summary = summary_df[summary_df["inference_method"] == "baseline"].copy()
    best_row = baseline_summary.iloc[0] if not baseline_summary.empty else summary_df.iloc[0]
    best_name = best_row["config_name"]
    best_config = next(deepcopy(c) for c in top_configs if c["config_name"] == best_name)

    # Augmentation ablation.
    aug_fold, aug_seed, aug_summary = run_augmentation_ablation(bundle, best_config)
    aug_fold.to_csv(Config.AUG_ABLATION_FOLD_CSV, index=False)
    aug_seed.to_csv(Config.AUG_ABLATION_SEED_CSV, index=False)
    aug_summary.to_csv(Config.AUG_ABLATION_SUMMARY_CSV, index=False)

    # ML baselines.
    if Config.RUN_ML_BASELINES:
        ml_fold, ml_seed, ml_summary = run_ml_baselines(bundle)
        ml_fold.to_csv(Config.ML_FOLD_CSV, index=False)
        ml_seed.to_csv(Config.ML_SEED_CSV, index=False)
        ml_summary.to_csv(Config.ML_SUMMARY_CSV, index=False)

        combined = pd.concat([summary_df, ml_summary], ignore_index=True, sort=False)
        combined.to_csv(Config.COMBINED_SUMMARY_CSV, index=False)

        combined_seed = pd.concat([seed_df, ml_seed], ignore_index=True, sort=False)
        run_statistical_analysis(combined_seed, Config.COMBINED_OMNIBUS_CSV, Config.COMBINED_PAIRWISE_CSV)

    protocol_summary_for_figures = None

    # Protocol comparison.
    if Config.RUN_PROTOCOL_COMPARISON_FOR_BEST:
        runs, summary, stats, reference = run_best_model_protocol_comparison(bundle, best_config)
        runs.to_csv(Config.PROTOCOL_COMPARISON_RUNS_CSV, index=False)
        summary.to_csv(Config.PROTOCOL_COMPARISON_SUMMARY_CSV, index=False)
        stats.to_csv(Config.PROTOCOL_COMPARISON_STATS_CSV, index=False)
        reference.to_csv(Config.PROTOCOL_COMPARISON_REFERENCE_CSV, index=False)
        protocol_summary_for_figures = summary

    if getattr(Config, "GENERATE_REPORT_FIGURES", False):
        generate_all_report_figures(
            summary_df=summary_df,
            prediction_df=prediction_df,
            confusion_df=pd.DataFrame(all_cm),
            protocol_summary_df=protocol_summary_for_figures,
            out_dir=Config.REPORT_FIGURES_DIR,
            calibration_bins=Config.REPORT_CALIBRATION_BINS,
        )

    print("[DONE] Final evaluation completed.")
    print(f"[DONE] Main summary: {Config.EVAL_SUMMARY_CSV}")
    print(f"[DONE] ML summary: {Config.ML_SUMMARY_CSV}")
    print(f"[DONE] Protocol comparison: {Config.PROTOCOL_COMPARISON_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
