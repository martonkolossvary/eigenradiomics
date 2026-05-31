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
    """Fisher-transformed 95% confidence interval for a correlation coefficient.

    For Spearman's rho the standard error is inflated by ``1.03`` (a common
    Bonett-Wright-style approximation). Returns ``(nan, nan)`` when ``n <= 3`` or
    ``|r| >= 1``: the Fisher z-transform diverges at ``|r| = 1``, so the interval
    is not estimable (reporting a narrow band there would be misleading).
    """
    if n <= 3 or np.isnan(r) or abs(r) >= 1.0:
        return np.nan, np.nan

    z = np.arctanh(r)
    se = (1.03 if is_spearman else 1.0) / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def _fisher_mean(coefficients: NDArray | list[float]) -> float:
    """Average correlation coefficients in Fisher z-space.

    The arithmetic mean of correlation coefficients is downward-biased; averaging
    via ``arctanh`` and back-transforming removes that bias. Used to pool the
    pairwise coefficients across three or more observers.
    """
    arr = np.asarray(coefficients, dtype=float)
    z = np.arctanh(np.clip(arr, -0.999999999999, 0.999999999999))
    return float(np.tanh(np.mean(z)))


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

    # Residual / Error. Clamp tiny negative floating-point noise to zero; do NOT
    # inflate to a fake positive (the old 1e-15 floor fabricated a finite F and a
    # near-zero p for perfect-agreement features).
    ss_error = ss_total - ss_between_subjects - ss_between_observers
    df_error = df_between_subjects * df_between_observers
    ms_error = max(ss_error / df_error, 0.0)

    # F-statistic and p-value for a subject effect.
    if ms_error > 0:
        f_stat = ms_between_subjects / ms_error
        p_value = float(stats.f.sf(f_stat, df_between_subjects, df_error))
    elif ms_between_subjects > 0:
        # Perfect agreement with real between-subject variance: subjects are
        # perfectly separable, so the effect is infinitely significant.
        f_stat = np.inf
        p_value = 0.0
    else:
        # No variance anywhere (fully constant input): the test is undefined.
        f_stat = np.nan
        p_value = np.nan

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


def _icc_2_1_batch(Yb: NDArray) -> NDArray:
    """Vectorized ICC(2,1) over a stack of measurement matrices.

    Mirrors :func:`_icc_2_1_estimate` (same SS decomposition, same ``MS_error``
    clamp and ``denominator <= 0 -> NaN`` rule) but over ``Yb`` of shape
    ``(batch, n_samples, n_observers)``, returning one ICC per batch element. Used
    to evaluate all bootstrap resamples at once.
    """
    _, n, k = Yb.shape
    grand = Yb.mean(axis=(1, 2))
    ss_total = ((Yb - grand[:, None, None]) ** 2).sum(axis=(1, 2))
    ss_between_subjects = k * ((Yb.mean(axis=2) - grand[:, None]) ** 2).sum(axis=1)
    ss_between_observers = n * ((Yb.mean(axis=1) - grand[:, None]) ** 2).sum(axis=1)
    ms_between_subjects = ss_between_subjects / (n - 1)
    ms_between_observers = ss_between_observers / (k - 1)
    ms_error = np.maximum(
        (ss_total - ss_between_subjects - ss_between_observers) / ((n - 1) * (k - 1)), 0.0
    )
    denominator = (
        ms_between_subjects + (k - 1) * ms_error + (k / n) * (ms_between_observers - ms_error)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        icc: NDArray = np.where(
            denominator > 0, (ms_between_subjects - ms_error) / denominator, np.nan
        )
    return icc


def _bootstrap_icc_ci(
    Y: NDArray,
    feature_name: str,
    iterations: int = 1000,
    base_seed: int = 42,
) -> tuple[float, float]:
    """Deterministically estimate the 95% Confidence Interval for ICC(2,1) via bootstrapping.

    Note that with only a handful of subjects (``n`` small) the percentile CI is
    unreliable regardless of the number of iterations, since the resampling space
    is tiny; treat such intervals with caution.
    """
    n, k = Y.shape
    if n < 3 or k < 2 or iterations <= 0:
        return np.nan, np.nan

    seed = _get_deterministic_seed(feature_name, base_seed=base_seed)
    rng = np.random.default_rng(seed)

    # Same per-iteration resampling stream as a scalar loop, but the ICC is then
    # evaluated for every resample at once (vectorized) — much faster on wide data.
    indices = np.array([rng.choice(n, size=n, replace=True) for _ in range(iterations)])
    estimates = _icc_2_1_batch(Y[indices])
    boot_estimates = estimates[~np.isnan(estimates)]

    # Require at least half the resamples to yield a valid ICC; fewer means the
    # feature is degenerate and the percentile CI would be meaningless. (The old
    # min(10, iterations // 2) gate accepted as few as 10 successes out of 1000.)
    if boot_estimates.size < max(1, iterations // 2):
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
