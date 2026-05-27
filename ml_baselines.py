"""
Frozen CNN embeddings + classical ML baselines.

This file implements the hybrid branch requested by reviewers:
    CNN feature extractor + classical ML classifier.

Why frozen embeddings?
They test whether a simple classifier on top of pretrained visual features is
sufficient, or whether fine-tuning the CNN is necessary.

Why SMOTE/SMOTEENN only here?
SMOTE creates synthetic samples by interpolating numeric vectors. This is
meaningful in feature/embedding space, but applying SMOTE directly to raw image
pixels would create artificial images with questionable visual validity.

Leakage-control rule:
For each fold, the sampler is fit only on the training embeddings. Test
embeddings are never resampled.
"""


from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import tensorflow as tf

import Config
from data_utils import make_tf_dataset
from model_factory import build_feature_extractor
from metrics import compute_core_metrics
from stats_utils import mean_ci95


def get_ml_classifier(name: str, seed: int):
    """Return a classical ML classifier with probability outputs."""
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    if name == "SVM":
        return SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
    raise ValueError(f"Unknown ML classifier: {name}")


def build_ml_pipeline(classifier_name: str, sampling: str, seed: int):
    """Build scaler + optional sampler + classifier pipeline."""
    steps = [("scaler", StandardScaler())]
    if sampling == "smote":
        steps.append(("sampler", SMOTE(random_state=seed, k_neighbors=3)))
    elif sampling == "smoteenn":
        steps.append(("sampler", SMOTEENN(random_state=seed)))
    elif sampling != "none":
        raise ValueError(f"Unknown sampling strategy: {sampling}")
    steps.append(("clf", get_ml_classifier(classifier_name, seed)))
    return ImbPipeline(steps) if sampling != "none" else SkPipeline(steps)


def extract_embeddings(bundle, backbone_name: str, batch_size: int = 32) -> np.ndarray:
    """Extract frozen CNN embeddings for all images."""
    cache_path = os.path.join(Config.EMBEDDINGS_CACHE_DIR, f"{backbone_name}_embeddings.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    built = build_feature_extractor(backbone_name)
    ds = make_tf_dataset(
        bundle.images,
        bundle.labels,
        preprocess_fn=built.preprocess_fn,
        batch_size=batch_size,
        training=False,
        augmentation_layer=None,
    )
    embeddings = built.model.predict(ds, verbose=0).astype(np.float32)
    np.save(cache_path, embeddings)
    tf.keras.backend.clear_session()
    return embeddings


def run_ml_baselines(bundle):
    """Run all ML baselines across seeds/folds."""
    fold_rows, seed_rows = [], []
    labels = bundle.labels

    for backbone in Config.ML_EMBEDDING_BACKBONES:
        print(f"[ML] Extracting embeddings: {backbone}", flush=True)
        X = extract_embeddings(bundle, backbone)

        for clf_name in Config.ML_CLASSIFIERS:
            for sampling in Config.ML_SAMPLING_STRATEGIES:
                for seed in Config.EVAL_SEEDS:
                    skf = StratifiedKFold(n_splits=Config.OUTER_CV, shuffle=True, random_state=seed)
                    seed_metrics = []

                    for fold, (tr, te) in enumerate(skf.split(X, labels), start=1):
                        pipe = build_ml_pipeline(clf_name, sampling, seed)
                        pipe.fit(X[tr], labels[tr])
                        pred = pipe.predict(X[te])
                        proba = pipe.predict_proba(X[te])
                        m = compute_core_metrics(labels[te], pred, proba)

                        row = {
                            "family": "frozen_embeddings_ml",
                            "config_name": f"ML__{backbone}__{clf_name}__{sampling}",
                            "backbone": backbone,
                            "classifier": clf_name,
                            "sampling": sampling,
                            "seed": seed,
                            "fold": fold,
                            **m,
                        }
                        fold_rows.append(row)
                        seed_metrics.append(row)

                    seed_df = pd.DataFrame(seed_metrics)
                    seed_rows.append({
                        "family": "frozen_embeddings_ml",
                        "config_name": f"ML__{backbone}__{clf_name}__{sampling}",
                        "backbone": backbone,
                        "classifier": clf_name,
                        "sampling": sampling,
                        "seed": seed,
                        "accuracy": float(seed_df["accuracy"].mean()),
                        "macro_f1": float(seed_df["macro_f1"].mean()),
                        "balanced_acc": float(seed_df["balanced_acc"].mean()),
                        "kappa": float(seed_df["kappa"].mean()),
                        "ece": float(seed_df["ece"].mean()),
                        "brier": float(seed_df["brier"].mean()),
                    })

    fold_df = pd.DataFrame(fold_rows)
    seed_df = pd.DataFrame(seed_rows)

    summary_rows = []
    for cfg, sub in seed_df.groupby("config_name"):
        row = {"family": "frozen_embeddings_ml", "config_name": cfg}
        for metric in ["accuracy", "macro_f1", "balanced_acc", "kappa", "ece", "brier"]:
            mean, lo, hi = mean_ci95(sub[metric].values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = float(sub[metric].std(ddof=1))
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        summary_rows.append(row)

    return fold_df, seed_df, pd.DataFrame(summary_rows).sort_values("macro_f1_mean", ascending=False)
