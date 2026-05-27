"""
Statistical analysis utilities.

This file contains the inferential tests used to support model comparisons.

Important:
The statistical unit used here is the seed-level mean, not individual images.
This avoids treating thousands of image predictions as independent observations
when they are produced by the same trained model/fold procedure.

Tests:
- Friedman:
  Omnibus test for comparing more than two configurations.

- Shapiro-Wilk:
  Checks whether paired differences are approximately normal.

- Paired t-test:
  Used if paired differences are compatible with normality.

- Wilcoxon signed-rank:
  Non-parametric paired alternative.

- Holm correction:
  Controls family-wise error rate across multiple pairwise comparisons.

- Cohen's dz:
  Effect size for paired designs.

- 95% CI:
  Reports uncertainty around repeated-seed means.
"""


from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, shapiro, ttest_rel, wilcoxon, t


def mean_ci95(values) -> tuple[float, float, float]:
    """Return mean, lower 95% CI, upper 95% CI."""
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x))
    if len(x) < 2:
        return mean, mean, mean
    se = np.std(x, ddof=1) / np.sqrt(len(x))
    crit = t.ppf(0.975, df=len(x) - 1)
    return mean, float(mean - crit * se), float(mean + crit * se)


def friedman_from_seed_table(seed_metric_table: pd.DataFrame) -> tuple[float, float]:
    """Run Friedman test on seed x configuration table."""
    cols = [seed_metric_table[c].dropna().values for c in seed_metric_table.columns]
    if len(cols) < 3:
        return np.nan, np.nan
    stat, p = friedmanchisquare(*cols)
    return float(stat), float(p)


def cohen_dz(diff: np.ndarray) -> float:
    """Cohen's dz = mean paired difference / SD paired difference."""
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return 0.0
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def paired_test(x, y) -> dict:
    """Choose paired t-test or Wilcoxon after Shapiro on paired differences."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y

    if len(diff) >= 3:
        shapiro_stat, shapiro_p = shapiro(diff)
    else:
        shapiro_stat, shapiro_p = np.nan, 0.0

    if np.isfinite(shapiro_p) and shapiro_p > 0.05:
        stat, p = ttest_rel(x, y)
        test_name = "paired_t_test"
    else:
        try:
            stat, p = wilcoxon(x, y, zero_method="wilcox", correction=False)
            test_name = "wilcoxon"
        except ValueError:
            stat, p = 0.0, 1.0
            test_name = "wilcoxon_degenerate"

    return {
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p),
        "shapiro_stat": float(shapiro_stat) if np.isfinite(shapiro_stat) else np.nan,
        "shapiro_p_value": float(shapiro_p) if np.isfinite(shapiro_p) else np.nan,
        "effect_size_dz": cohen_dz(diff),
    }


def holm_correction(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda z: z[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (idx, p) in enumerate(indexed, start=1):
        adj = (m - rank + 1) * p
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def pairwise_comparisons(seed_df: pd.DataFrame, metric_col: str = "macro_f1") -> pd.DataFrame:
    """Pairwise comparisons using matched seed-level means."""
    rows, raw_p = [], []
    configs = sorted(seed_df["config_name"].unique().tolist())
    for a, b in itertools.combinations(configs, 2):
        sub_a = seed_df[seed_df["config_name"] == a].sort_values("seed")
        sub_b = seed_df[seed_df["config_name"] == b].sort_values("seed")
        common = sorted(set(sub_a["seed"]).intersection(set(sub_b["seed"])))
        xa = sub_a[sub_a["seed"].isin(common)][metric_col].values
        xb = sub_b[sub_b["seed"].isin(common)][metric_col].values
        result = paired_test(xa, xb)
        rows.append({"config_a": a, "config_b": b, "metric": metric_col, **result})
        raw_p.append(result["p_value"])

    adjusted = holm_correction(raw_p) if raw_p else []
    for row, p_adj in zip(rows, adjusted):
        row["p_value_holm"] = float(p_adj)
        row["significant_0_05"] = bool(p_adj < 0.05)
    return pd.DataFrame(rows)
