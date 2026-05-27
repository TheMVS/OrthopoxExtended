"""
Global configuration file for Orthopox image classification experiments.

This version includes:
- family-balanced random search
- additional Keras backbones
- MonkeyNet-style DenseNet201 baseline
- final evaluation with baseline/TTA/weighted-TTA/MC Dropout/temperature scaling
- entropy-based rejection metrics
"""

from __future__ import annotations
import os

# ============================================================
# General paths
# ============================================================

BASE_PATH = "./"
VARIANT_NAME = "data"

OUTPUT_DIR = "./results"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
GRADCAM_DIR = os.path.join(OUTPUT_DIR, "xai")
EMBEDDINGS_CACHE_DIR = os.path.join(OUTPUT_DIR, "embeddings_cache")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_CACHE_DIR, exist_ok=True)

# ============================================================
# Input image configuration
# ============================================================

IMAGE_SIZE = (224, 224)
CHANNELS = 3

# ============================================================
# Execution mode
# ============================================================

RUN_RANDOM_SEARCH = True

# ============================================================
# Model families
# ============================================================

MODEL_FAMILY = {
    "MonkeyNetDenseNet201": "monkeynet_style",
    "DenseNet201": "classic_cnn",
    "ResNet50V2": "classic_cnn",
    "Xception": "classic_cnn",
    "EfficientNetV2B0": "modern_cnn",
    "EfficientNetV2S": "modern_cnn",
    "ConvNeXtTiny": "modern_cnn",
    "ConvNeXtSmall": "modern_cnn",
    "MobileNetV3Large": "lightweight_cnn",
}

DL_BACKBONES = [
    "MonkeyNetDenseNet201",
    "ResNet50V2",
    "Xception",
    "EfficientNetV2B0",
    "EfficientNetV2S",
    "ConvNeXtTiny",
    "ConvNeXtSmall",
    "MobileNetV3Large",
]

# ============================================================
# Search configuration
# ============================================================

SEARCH_SEED = 42
SEARCH_CV = 5

MAX_SEARCH_CONFIGS = 50

# The final selected configurations are family-balanced, not only global-top.
TOP_K_PER_FAMILY = 1
TOP_GLOBAL_EXTRA = 3
TOP_N_TO_EVALUATE = 7  # kept for backward compatibility/logging

SEARCH_RESULTS_CSV = os.path.join(OUTPUT_DIR, "search_results.csv")
TOP_CONFIGS_JSON = os.path.join(OUTPUT_DIR, "top_configs.json")
SEARCH_SUMMARY_JSON = os.path.join(OUTPUT_DIR, "search_summary.json")

SEARCH_SPACE = {
    "backbone": DL_BACKBONES,
    "dropout": ("uniform", 0.15, 0.60),
    "dense_units": [128, 256],
    "learning_rate_head": ("log_uniform", 1e-5, 5e-3),
    "learning_rate_finetune": ("log_uniform", 1e-7, 5e-5),
    "unfreeze_blocks": ("randint", 0, 3),
    "batch_size": [8],
    "epochs_head": [10, 15],
    "epochs_finetune": [15, 20],
    "label_smoothing": [0.0, 0.03, 0.05],
    "use_focal_loss": [False, True],
    "focal_gamma": [1.0, 1.5, 2.0],
    "augmentation_policy": ["none", "light"],

    # Search keeps this field for compatibility, but the final evaluation
    # explicitly compares all inference strategies without retraining.
    "tta": [False],
}

# ============================================================
# Training callbacks
# ============================================================

EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
USE_CLASS_WEIGHTS = True

# ============================================================
# Augmentation policies
# ============================================================

AUGMENTATION = {
    "none": {
        "flip_left_right": False,
        "rotation": 0.0,
        "zoom": 0.0,
        "contrast": 0.0,
        "brightness": 0.0,
    },
    "light": {
        "flip_left_right": True,
        "rotation": 0.05,
        "zoom": 0.05,
        "contrast": 0.05,
        "brightness": 0.05,
    },
    "moderate": {
        "flip_left_right": True,
        "rotation": 0.10,
        "zoom": 0.10,
        "contrast": 0.10,
        "brightness": 0.08,
    },
}

# ============================================================
# Test-time / post-training inference strategies
# ============================================================

TTA_SIMPLE_TRANSFORMS = ["identity", "flip_left_right"]
TTA_ADVANCED_TRANSFORMS = ["identity", "flip_left_right", "flip_up_down", "rot90", "rot270"]
TTA_TRANSFORMS = TTA_SIMPLE_TRANSFORMS  # backward-compatible alias

WEIGHTED_TTA_TRANSFORMS = ["identity", "flip_left_right", "flip_up_down", "rot90", "rot270"]
WEIGHTED_TTA_WEIGHTS = {
    "identity": 0.50,
    "flip_left_right": 0.20,
    "flip_up_down": 0.10,
    "rot90": 0.10,
    "rot270": 0.10,
}

MC_DROPOUT_PASSES = 5
TEMPERATURE_SCALING_VALUES = [1.5, 2.0]
ENTROPY_REJECTION_THRESHOLDS = [0.50, 0.75, 1.00]

FINAL_INFERENCE_METHODS = [
    "baseline",
    "tta_simple",
    "tta_advanced",
    "weighted_tta",
    "mc_dropout",
    "temperature_1.5",
    "temperature_2.0",
]

# ============================================================
# Final evaluation configuration
# ============================================================

EVAL_SEEDS = [11, 22, 33]
OUTER_CV = 5

EVALUATE_TOP_PER_FAMILY_ONLY = True
EVALUATE_TOP_K_PER_FAMILY = 1
EVALUATE_GLOBAL_TOP_EXTRA = 0

EVAL_FOLD_CSV = os.path.join(OUTPUT_DIR, "eval_fold_results.csv")
EVAL_SEED_CSV = os.path.join(OUTPUT_DIR, "eval_seed_results.csv")
EVAL_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "eval_summary.csv")
EVAL_CLASSWISE_CSV = os.path.join(OUTPUT_DIR, "eval_classwise.csv")
EVAL_CONFUSION_LONG_CSV = os.path.join(OUTPUT_DIR, "eval_confusion_long.csv")
EVAL_ERROR_ANALYSIS_CSV = os.path.join(OUTPUT_DIR, "eval_error_analysis.csv")

OMNIBUS_STATS_CSV = os.path.join(OUTPUT_DIR, "omnibus_stats.csv")
PAIRWISE_STATS_CSV = os.path.join(OUTPUT_DIR, "pairwise_stats.csv")

# ============================================================
# Augmentation ablation
# ============================================================

AUGMENTATION_ABLATION_POLICIES = ["none", "light", "moderate"]

AUG_ABLATION_FOLD_CSV = os.path.join(OUTPUT_DIR, "augmentation_ablation_fold.csv")
AUG_ABLATION_SEED_CSV = os.path.join(OUTPUT_DIR, "augmentation_ablation_seed.csv")
AUG_ABLATION_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "augmentation_ablation_summary.csv")

# ============================================================
# ML baselines
# ============================================================

RUN_ML_BASELINES = True

ML_EMBEDDING_BACKBONES = [
    "EfficientNetV2B0",
    "DenseNet201",
    "ResNet50V2",
]

ML_CLASSIFIERS = ["LogisticRegression", "SVM", "RandomForest"]
ML_SAMPLING_STRATEGIES = ["none", "smote", "smoteenn"]

ML_FOLD_CSV = os.path.join(OUTPUT_DIR, "ml_fold_results.csv")
ML_SEED_CSV = os.path.join(OUTPUT_DIR, "ml_seed_results.csv")
ML_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "ml_summary.csv")

COMBINED_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "combined_summary.csv")
COMBINED_OMNIBUS_CSV = os.path.join(OUTPUT_DIR, "combined_omnibus_stats.csv")
COMBINED_PAIRWISE_CSV = os.path.join(OUTPUT_DIR, "combined_pairwise_stats.csv")

# ============================================================
# Protocol comparison
# ============================================================

RUN_PROTOCOL_COMPARISON_FOR_BEST = True

PROTOCOL_COMPARISON_SEEDS = [101, 202, 303]
PROTOCOL_COMPARISON_TEST_SIZE = 0.20
PAPER_LIKE_AUG_REPEATS = 10

PROTOCOL_COMPARISON_RUNS_CSV = os.path.join(OUTPUT_DIR, "protocol_comparison_runs.csv")
PROTOCOL_COMPARISON_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "protocol_comparison_summary.csv")
PROTOCOL_COMPARISON_STATS_CSV = os.path.join(OUTPUT_DIR, "protocol_comparison_stats.csv")
PROTOCOL_COMPARISON_REFERENCE_CSV = os.path.join(OUTPUT_DIR, "protocol_comparison_reference.csv")

REFERENCE_PAPER_NAME = "Bala et al. 2023 MonkeyNet"
REFERENCE_PAPER_REPORTED_ACCURACY_ORIGINAL = 0.9319
REFERENCE_PAPER_REPORTED_ACCURACY_AUGMENTED = 0.9891

# ============================================================
# XAI configuration
# ============================================================

SAVE_GRADCAM = True
SAVE_GRADCAM_PLUS_PLUS = True
SAVE_OCCLUSION_SENSITIVITY = True

GRADCAM_MAX_IMAGES_PER_FOLD = 3
XAI_MAX_IMAGES_PER_FOLD = 3

OCCLUSION_PATCH_SIZE = 32
OCCLUSION_STRIDE = 32
OCCLUSION_BASELINE_VALUE = 0.0

# ============================================================
# Prediction-level outputs and automatic report figures
# ============================================================

EVAL_PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "eval_prediction_level_results.csv")
REPORT_FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(REPORT_FIGURES_DIR, exist_ok=True)

GENERATE_REPORT_FIGURES = True
REPORT_CALIBRATION_BINS = 10
