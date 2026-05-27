"""
Configuration search stage.

This version supports:
- discrete choices: ["a", "b", "c"]
- continuous uniform ranges: ("uniform", low, high)
- logarithmic ranges: ("log_uniform", low, high)
- random integer ranges: ("randint", low, high)

This allows learning rates, dropout and unfreeze_blocks to be sampled randomly
inside ranges instead of being manually enumerated in Config.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
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
)
from metrics import compute_core_metrics
from model_factory import build_model, compile_model


def config_hash(cfg: dict) -> str:
    """Create a short deterministic identifier for one configuration."""
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def sample_value(values, rng: random.Random):
    """
    Sample one value from a search-space entry.

    Supported formats:
    - list: random choice
    - ("uniform", low, high): continuous uniform sampling
    - ("log_uniform", low, high): continuous log-scale sampling
    - ("randint", low, high): integer sampling including both limits
    """

    if isinstance(values, list):
        return rng.choice(values)

    if isinstance(values, tuple):
        mode = values[0]

        if mode == "uniform":
            _, low, high = values
            return float(rng.uniform(float(low), float(high)))

        if mode == "log_uniform":
            _, low, high = values
            low = float(low)
            high = float(high)

            if low <= 0 or high <= 0:
                raise ValueError("log_uniform requires positive low and high values.")

            return float(10 ** rng.uniform(np.log10(low), np.log10(high)))

        if mode == "randint":
            _, low, high = values
            return int(rng.randint(int(low), int(high)))

    raise ValueError(f"Unsupported search-space entry: {values}")


def sample_configurations(search_space: dict, max_configs: int, seed: int) -> list[dict]:
    """
    Generate random configurations from Config.SEARCH_SPACE.

    This version forces at least one candidate per model family before filling
    the remaining budget with fully random configurations. This prevents the
    search from accidentally ignoring one architecture family.
    """

    rng = random.Random(seed)
    configs = []
    seen_hashes = set()

    def _make_cfg(forced_backbone: str | None = None) -> dict:
        cfg = {}
        for key, values in search_space.items():
            cfg[key] = sample_value(values, rng)
        if forced_backbone is not None:
            cfg["backbone"] = forced_backbone
        cfg["model_family"] = Config.MODEL_FAMILY.get(cfg["backbone"], "other")
        cfg["config_name"] = config_hash(cfg)
        return cfg

    # 1) Force at least one candidate per model family.
    family_to_backbones = {}
    for backbone in search_space.get("backbone", []):
        family_to_backbones.setdefault(Config.MODEL_FAMILY.get(backbone, "other"), []).append(backbone)

    for family, backbones in sorted(family_to_backbones.items()):
        if len(configs) >= max_configs:
            break
        cfg = _make_cfg(forced_backbone=rng.choice(backbones))
        if cfg["config_name"] not in seen_hashes:
            seen_hashes.add(cfg["config_name"])
            configs.append(cfg)

    # 2) Fill the remaining budget randomly.
    max_attempts = max_configs * 50
    attempts = 0
    while len(configs) < max_configs and attempts < max_attempts:
        attempts += 1
        cfg = _make_cfg()
        if cfg["config_name"] in seen_hashes:
            continue
        seen_hashes.add(cfg["config_name"])
        configs.append(cfg)

    if len(configs) < max_configs:
        print(
            f"[WARN] Only generated {len(configs)} unique configurations "
            f"out of requested {max_configs}.",
            flush=True,
        )

    return configs


def select_top_configs_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Select top-K configurations per family plus optional global extras."""
    df = df.copy()
    if "model_family" not in df.columns:
        df["model_family"] = df["backbone"].map(Config.MODEL_FAMILY).fillna("other")

    sort_cols = ["search_macro_f1_mean", "search_balanced_acc_mean", "search_kappa_mean"]
    selected_parts = []

    for family, sub in df.groupby("model_family"):
        selected_parts.append(
            sub.sort_values(sort_cols, ascending=[False, False, False]).head(Config.TOP_K_PER_FAMILY)
        )

    selected = pd.concat(selected_parts, axis=0) if selected_parts else df.head(0)

    global_extra = df.sort_values(sort_cols, ascending=[False, False, False]).head(Config.TOP_GLOBAL_EXTRA)
    selected = pd.concat([selected, global_extra], axis=0)
    selected = selected.drop_duplicates(subset=["config_name"])
    selected = selected.sort_values(sort_cols, ascending=[False, False, False])

    return selected


def make_callbacks(config_name: str):
    """Training callbacks for one fold/stage."""

    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"{config_name}.weights.h5")

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

        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
    ]


def train_one_fold(train_images, train_labels, val_images, val_labels, class_weights, config):
    """
    Train one candidate on one search fold.

    The outer validation fold is not used for early stopping.
    An internal validation split is created from the training fold.
    """

    tf.keras.backend.clear_session()

    built = build_model(config, num_classes=len(np.unique(train_labels)))

    aug = build_augmentation_layer(config["augmentation_policy"])

    sub_train_idx, inner_val_idx = train_test_split(
        np.arange(len(train_labels)),
        test_size=0.15,
        random_state=Config.SEARCH_SEED,
        stratify=train_labels,
    )

    train_ds = make_tf_dataset(
        train_images[sub_train_idx],
        train_labels[sub_train_idx],
        built.preprocess_fn,
        config["batch_size"],
        training=True,
        augmentation_layer=aug,
    )

    inner_val_ds = make_tf_dataset(
        train_images[inner_val_idx],
        train_labels[inner_val_idx],
        built.preprocess_fn,
        config["batch_size"],
        training=False,
        augmentation_layer=None,
    )

    val_ds = make_tf_dataset(
        val_images,
        val_labels,
        built.preprocess_fn,
        config["batch_size"],
        training=False,
        augmentation_layer=None,
    )

    compile_model(built.model, config, stage="head")

    built.model.fit(
        train_ds,
        validation_data=inner_val_ds,
        epochs=int(config["epochs_head"]),
        verbose=0,
        callbacks=make_callbacks(config["config_name"] + "_head"),
        class_weight=class_weights if Config.USE_CLASS_WEIGHTS else None,
    )

    compile_model(built.model, config, stage="finetune")

    built.model.fit(
        train_ds,
        validation_data=inner_val_ds,
        epochs=int(config["epochs_finetune"]),
        verbose=0,
        callbacks=make_callbacks(config["config_name"] + "_finetune"),
        class_weight=class_weights if Config.USE_CLASS_WEIGHTS else None,
    )

    proba = built.model.predict(val_ds, verbose=0)
    pred = np.argmax(proba, axis=1)

    return pred, proba


def evaluate_config(bundle, config: dict, cv_splits: int, seed: int) -> dict:
    """Cross-validated score for one candidate configuration."""

    labels = bundle.labels
    images = bundle.images

    skf = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=seed,
    )

    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels),
        start=1,
    ):
        print(f"    [FOLD] {fold_idx}/{cv_splits}", flush=True)

        train_images = images[train_idx]
        train_labels = labels[train_idx]
        val_images = images[val_idx]
        val_labels = labels[val_idx]

        class_weights = compute_balanced_class_weights(train_labels)

        pred, proba = train_one_fold(
            train_images=train_images,
            train_labels=train_labels,
            val_images=val_images,
            val_labels=val_labels,
            class_weights=class_weights,
            config=config,
        )

        fold_scores.append(compute_core_metrics(val_labels, pred, proba))

        tf.keras.backend.clear_session()

    out = deepcopy(config)

    out["search_macro_f1_mean"] = float(np.mean([r["macro_f1"] for r in fold_scores]))
    out["search_macro_f1_std"] = float(np.std([r["macro_f1"] for r in fold_scores], ddof=1))
    out["search_balanced_acc_mean"] = float(np.mean([r["balanced_acc"] for r in fold_scores]))
    out["search_kappa_mean"] = float(np.mean([r["kappa"] for r in fold_scores]))
    out["search_ece_mean"] = float(np.mean([r["ece"] for r in fold_scores]))
    out["search_brier_mean"] = float(np.mean([r["brier"] for r in fold_scores]))

    return out


def save_partial_search_results(rows: list[dict]) -> None:
    """Save search_results.csv incrementally after every completed configuration."""

    df = pd.DataFrame(rows).sort_values(
        by=[
            "search_macro_f1_mean",
            "search_balanced_acc_mean",
            "search_kappa_mean",
        ],
        ascending=[False, False, False],
    )

    df.to_csv(Config.SEARCH_RESULTS_CSV, index=False)


def format_seconds(seconds: float) -> str:
    """Format seconds as a readable duration."""

    seconds = int(seconds)

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h:
        return f"{h}h {m}m {s}s"

    if m:
        return f"{m}m {s}s"

    return f"{s}s"


def print_config_summary(config: dict) -> None:
    """Print the most relevant hyperparameters for tracking the search."""

    print(
        f"         backbone={config['backbone']} | "
        f"batch={config['batch_size']} | "
        f"aug={config['augmentation_policy']} | "
        f"focal={config['use_focal_loss']} | "
        f"tta={config['tta']}",
        flush=True,
    )

    print(
        f"         lr_head={config['learning_rate_head']:.2e} | "
        f"lr_finetune={config['learning_rate_finetune']:.2e} | "
        f"dropout={config['dropout']:.3f} | "
        f"unfreeze_blocks={config['unfreeze_blocks']}",
        flush=True,
    )


def main():
    """Run the random hyperparameter search."""

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    bundle = load_image_dataset(Config.BASE_PATH, Config.VARIANT_NAME)

    configs = sample_configurations(
        search_space=Config.SEARCH_SPACE,
        max_configs=Config.MAX_SEARCH_CONFIGS,
        seed=Config.SEARCH_SEED,
    )

    rows = []
    start = time.time()

    for idx, config in enumerate(configs, start=1):
        cfg_start = time.time()

        print(
            f"\n[SEARCH] {idx}/{len(configs)} -> {config['config_name']}",
            flush=True,
        )

        print_config_summary(config)

        row = evaluate_config(
            bundle=bundle,
            config=config,
            cv_splits=Config.SEARCH_CV,
            seed=Config.SEARCH_SEED,
        )

        rows.append(row)

        save_partial_search_results(rows)

        elapsed = time.time() - start
        eta = (elapsed / len(rows)) * (len(configs) - len(rows))

        print(f"[SAVE] Partial CSV saved: {Config.SEARCH_RESULTS_CSV}", flush=True)
        print(
            f"[TIME] current={format_seconds(time.time() - cfg_start)} "
            f"total={format_seconds(elapsed)} "
            f"ETA={format_seconds(eta)}",
            flush=True,
        )

    df = pd.DataFrame(rows).sort_values(
        by=[
            "search_macro_f1_mean",
            "search_balanced_acc_mean",
            "search_kappa_mean",
        ],
        ascending=[False, False, False],
    )

    df.to_csv(Config.SEARCH_RESULTS_CSV, index=False)

    top_df = select_top_configs_by_family(df).copy()

    with open(Config.TOP_CONFIGS_JSON, "w", encoding="utf-8") as f:
        json.dump(top_df.to_dict(orient="records"), f, indent=2)

    summary = {
        "num_candidates_evaluated": int(len(df)),
        "top_n_saved": int(len(top_df)),
        "top_k_per_family": int(Config.TOP_K_PER_FAMILY),
        "top_global_extra": int(Config.TOP_GLOBAL_EXTRA),
        "search_results_csv": Config.SEARCH_RESULTS_CSV,
        "top_configs_json": Config.TOP_CONFIGS_JSON,
        "total_search_time": format_seconds(time.time() - start),
    }

    with open(Config.SEARCH_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] Search results: {Config.SEARCH_RESULTS_CSV}")
    print(f"[DONE] Top configs: {Config.TOP_CONFIGS_JSON}")
    print(f"[DONE] Search summary: {Config.SEARCH_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
