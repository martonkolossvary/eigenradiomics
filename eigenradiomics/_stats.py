"""Shared statistical primitives used across eigenradiomics analysis modules.

These helpers are deliberately dependency-light (numpy / scipy / pandas only) so
that both the reproducibility and batch-effect modules can share a single,
tested implementation of each statistic.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.typing import NDArray

# ----------------------------------------------------------------------
# Multiple-testing correction
# ----------------------------------------------------------------------


def _fdr_correct(p_values: NDArray) -> NDArray:
    """Apply Benjamini-Hochberg False Discovery Rate (FDR) multiple testing correction."""
    n = len(p_values)
    if n == 0:
        return p_values

    valid_mask = ~np.isnan(p_values)
    valid_p = p_values[valid_mask]
    if len(valid_p) == 0:
        return p_values

    sorted_indices = np.argsort(valid_p)
    sorted_p = valid_p[sorted_indices]

    q_values = np.zeros_like(valid_p)
    prev_q = 1.0
    for rank in range(len(valid_p) - 1, -1, -1):
        p = sorted_p[rank]
        q = p * len(valid_p) / (rank + 1)
        q = min(q, prev_q)
        q_values[rank] = q
        prev_q = q

    corrected_p = np.zeros_like(p_values, dtype=float)
    corrected_p[~valid_mask] = np.nan
    corrected_p[valid_mask] = q_values[np.argsort(sorted_indices)]
    return corrected_p


# ----------------------------------------------------------------------
# Correlation confidence intervals & deterministic seeding
# ----------------------------------------------------------------------


def _get_deterministic_seed(feature_name: str, base_seed: int = 42) -> int:
    """Generate a unique 32-bit integer seed for a feature name deterministically."""
    hasher = hashlib.blake2b(key=str(base_seed).encode())
    hasher.update(feature_name.encode())
    return int(hasher.hexdigest(), 16) % (2**32)


def _fisher_ci(r: float, n: int, is_spearman: bool = False) -> tuple[float, float]:
    """Compute Fisher-transformed 95% Confidence Interval for a correlation coefficient."""
    if n <= 3 or np.isnan(r):
        return np.nan, np.nan

    # Clip r to prevent infinite values in arctanh
    r_clipped = np.clip(r, -0.9999, 0.9999)
    z = np.arctanh(r_clipped)

    # Standard error adjustments
    se = (1.03 if is_spearman else 1.0) / np.sqrt(n - 3)

    z_low = z - 1.96 * se
    z_high = z + 1.96 * se

    return float(np.tanh(z_low)), float(np.tanh(z_high))


# ----------------------------------------------------------------------
# Intraclass correlation (ICC(2,1)) + bootstrap CI
# ----------------------------------------------------------------------


def _icc_2_1_estimate(Y: NDArray) -> dict[str, float]:
    """Calculate two-way random-effects, absolute-agreement, single-measure ICC(2,1).

    Parameters
    ----------
    Y : ndarray of shape (n_samples, n_observers)
        Matrix of measurements.

    Returns
    -------
    stats : dict[str, float]
        Dictionary of computed ANOVA MS values and the ICC estimate.
    """
    n, k = Y.shape
    if n < 2 or k < 2:
        return {
            "icc": np.nan,
            "ms_between_subjects": np.nan,
            "ms_between_observers": np.nan,
            "ms_error": np.nan,
            "f_stat": np.nan,
            "p_value": np.nan,
        }

    # Grand mean
    grand_mean = np.mean(Y)

    # Sum of squares
    ss_total = np.sum((Y - grand_mean) ** 2)

    # Between subjects
    subject_means = np.mean(Y, axis=1)
    ss_between_subjects = k * np.sum((subject_means - grand_mean) ** 2)
    df_between_subjects = n - 1
    ms_between_subjects = ss_between_subjects / df_between_subjects

    # Between observers
    observer_means = np.mean(Y, axis=0)
    ss_between_observers = n * np.sum((observer_means - grand_mean) ** 2)
    df_between_observers = k - 1
    ms_between_observers = ss_between_observers / df_between_observers

    # Residual / Error
    ss_error = ss_total - ss_between_subjects - ss_between_observers
    df_error = df_between_subjects * df_between_observers
    ms_error = max(ss_error / df_error, 1e-15)  # Avoid division by zero

    # F-statistic and p-value for subjects
    f_stat = ms_between_subjects / ms_error
    p_value = stats.f.sf(f_stat, df_between_subjects, df_error)

    # ICC(2,1) formula
    denominator = (
        ms_between_subjects
        + (k - 1) * ms_error
        + (k / n) * (ms_between_observers - ms_error)
    )
    if denominator <= 0 or np.isnan(denominator):
        icc = np.nan
    else:
        icc = (ms_between_subjects - ms_error) / denominator

    return {
        "icc": float(icc),
        "ms_between_subjects": float(ms_between_subjects),
        "ms_between_observers": float(ms_between_observers),
        "ms_error": float(ms_error),
        "f_stat": float(f_stat),
        "p_value": float(p_value),
    }


def _bootstrap_icc_ci(
    Y: NDArray,
    feature_name: str,
    iterations: int = 1000,
    base_seed: int = 42,
) -> tuple[float, float]:
    """Deterministically estimate the 95% Confidence Interval for ICC(2,1) via bootstrapping."""
    n, k = Y.shape
    if n < 3 or k < 2 or iterations <= 0:
        return np.nan, np.nan

    seed = _get_deterministic_seed(feature_name, base_seed=base_seed)
    rng = np.random.default_rng(seed)

    boot_estimates = []
    for _ in range(iterations):
        boot_indices = rng.choice(n, size=n, replace=True)
        Y_boot = Y[boot_indices]
        est = _icc_2_1_estimate(Y_boot)
        if not np.isnan(est["icc"]):
            boot_estimates.append(est["icc"])

    if len(boot_estimates) < min(10, iterations // 2):
        return np.nan, np.nan

    ci_low = float(np.percentile(boot_estimates, 2.5))
    ci_high = float(np.percentile(boot_estimates, 97.5))
    return ci_low, ci_high


# ----------------------------------------------------------------------
# Group-difference tests (batch-effect feature-level statistics)
# ----------------------------------------------------------------------


def anova_effect(groups: list[NDArray]) -> tuple[float, float, float]:
    """Calculate one-way ANOVA F-statistic, p-value, and eta-squared effect size."""
    groups = [g[np.isfinite(g)] for g in groups if np.isfinite(g).sum() > 0]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan
    values = np.concatenate(groups)
    if len(values) <= len(groups) or np.nanvar(values) == 0:
        return np.nan, np.nan, np.nan

    f_stat, p_value = stats.f_oneway(*groups)
    grand = np.nanmean(values)
    ss_between = sum(len(g) * (np.nanmean(g) - grand) ** 2 for g in groups)
    ss_total = np.nansum((values - grand) ** 2)
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    return float(f_stat), float(p_value), float(eta2)


def kruskal_effect(groups: list[NDArray]) -> tuple[float, float, float]:
    """Calculate Kruskal-Wallis H-statistic, p-value, and epsilon-squared effect size."""
    groups = [g[np.isfinite(g)] for g in groups if np.isfinite(g).sum() > 0]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan
    values = np.concatenate(groups)
    if len(values) <= len(groups):
        return np.nan, np.nan, np.nan

    h_stat, p_value = stats.kruskal(*groups)
    n = len(values)
    k = len(groups)
    epsilon2 = max((h_stat - k + 1) / (n - k), 0) if n > k else np.nan
    return float(h_stat), float(p_value), float(epsilon2)


def levene_test(groups: list[NDArray]) -> tuple[float, float]:
    """Calculate Brown-Forsythe/Levene variance homogeneity test statistic and p-value."""
    groups = [g[np.isfinite(g)] for g in groups if np.isfinite(g).sum() > 1]
    if len(groups) < 2:
        return np.nan, np.nan
    stat, p_value = stats.levene(*groups, center="median")
    return float(stat), float(p_value)


def permanova_euclidean(
    values: pd.DataFrame,
    batch: pd.Series,
    permutations: int = 999,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Perform PERMANOVA pseudo-F permutation test on a PCA scores matrix."""
    x = values.to_numpy(float)
    labels = batch.astype(str).to_numpy()
    unique = np.unique(labels)
    n = x.shape[0]
    k = len(unique)

    if k < 2 or n <= k:
        return np.nan, np.nan, np.nan

    grand = x.mean(axis=0)
    sst = float(((x - grand) ** 2).sum())
    if sst == 0:
        return np.nan, np.nan, np.nan

    def pseudo_f(current_labels: np.ndarray) -> tuple[float, float]:
        ss_between = 0.0
        for label in unique:
            group = x[current_labels == label]
            if len(group) == 0:  # pragma: no cover - permutations preserve label counts
                continue
            ss_between += len(group) * float(((group.mean(axis=0) - grand) ** 2).sum())
        ss_within = sst - ss_between
        f_val = (ss_between / (k - 1)) / (ss_within / (n - k)) if ss_within > 0 else np.inf
        return f_val, ss_between / sst

    observed_f, r2 = pseudo_f(labels)
    if permutations == 0 or not np.isfinite(observed_f):
        return float(observed_f), float(r2), np.nan

    rng = np.random.default_rng(random_state)
    exceed = 0
    for _ in range(permutations):
        permuted = rng.permutation(labels)
        perm_f, _ = pseudo_f(permuted)
        exceed += int(perm_f >= observed_f)

    p_value = (exceed + 1) / (permutations + 1)
    return float(observed_f), float(r2), float(p_value)
