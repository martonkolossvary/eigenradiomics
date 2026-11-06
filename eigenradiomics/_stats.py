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

    For Pearson's r the Fisher-z standard error is ``1 / sqrt(n - 3)``. For
    Spearman's rho it uses the Bonett-Wright (2000) variance,
    ``sqrt((1 + r**2 / 2) / (n - 3))``, which widens the interval as ``|r|``
    grows (replacing the previous fixed ``1.03`` inflation factor). Returns
    ``(nan, nan)`` when ``n <= 3`` or ``|r| >= 1``: the Fisher z-transform
    diverges at ``|r| = 1``, so the interval is not estimable (reporting a narrow
    band there would be misleading).
    """
    if n <= 3 or np.isnan(r) or abs(r) >= 1.0:
        return np.nan, np.nan

    z = np.arctanh(r)
    se_scale = np.sqrt(1.0 + r * r / 2.0) if is_spearman else 1.0
    se = se_scale / np.sqrt(n - 3)
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


def _icc_2_1_weighted(Y: NDArray, W: NDArray) -> dict[str, NDArray]:
    """Calculate ICC(2,1) for each row of a matrix of row multiplicities.

    This is the one ICC(2,1) kernel (two-way random effects, absolute agreement,
    single measure; McGraw and Wong ICC(A,1)) for the point estimate and for the
    bootstrap. Row ``i`` of ``Y`` counts ``W[b, i]`` times in draw ``b``, so each
    result is equal to the ICC of the expanded matrix with ``sum(W[b])`` targets.

    The kernel subtracts one value of ``Y`` (a common origin) and divides by a
    power of two. Both steps are exact, so equal measurements give exact zeros.
    The residual sum of squares is a direct sum of squared residuals, not a
    difference of large sums.

    Parameters
    ----------
    Y : ndarray of shape (n_targets, n_observers)
        Finite measurements.
    W : ndarray of shape (n_draws, n_targets)
        Nonnegative row multiplicities.

    Returns
    -------
    dict of ndarray of shape (n_draws,)
        ``icc``, ``ms_between_subjects``, ``ms_between_observers`` and
        ``ms_error``, on the scale of ``Y``.
    """
    W = np.atleast_2d(np.asarray(W, dtype=float))
    n_draws = W.shape[0]
    n, k = Y.shape
    nan = np.full(n_draws, np.nan)
    out = {
        "icc": nan,
        "ms_between_subjects": nan,
        "ms_between_observers": nan,
        "ms_error": nan,
    }
    if n < 1 or k < 2 or not np.isfinite(Y).all():
        return out
    centered = Y - Y.flat[0]
    spread = float(np.abs(centered).max())
    if spread == 0.0:
        return out  # exactly constant input: the ICC is undefined
    # Divide by a power of two (an exact operation) so the squares cannot overflow.
    unit = float(np.ldexp(1.0, int(np.frexp(spread)[1])))
    centered = centered / unit

    total = W.sum(axis=1)  # number of targets in each expanded draw
    row_mean = centered.mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        grand = (W @ row_mean) / total
        col_effect = (W @ centered) / total[:, None] - grand[:, None]
        ss_rows = k * (W * (row_mean[None, :] - grand[:, None]) ** 2).sum(axis=1)
        ss_cols = total * (col_effect**2).sum(axis=1)
        resid = (centered - row_mean[:, None])[None, :, :] - col_effect[:, None, :]
        ss_err = (W[:, :, None] * resid**2).sum(axis=(1, 2))
        df_rows = total - 1
        msr = ss_rows / df_rows
        msc = ss_cols / (k - 1)
        mse = ss_err / (df_rows * (k - 1))
        denom = msr + (k - 1) * mse + (k / total) * (msc - mse)
        icc = np.where((df_rows > 0) & (denom > 0), (msr - mse) / denom, np.nan)
    sq = unit * unit
    out.update(
        icc=icc,
        ms_between_subjects=np.where(df_rows > 0, msr * sq, np.nan),
        ms_between_observers=np.where(df_rows > 0, msc * sq, np.nan),
        ms_error=np.where(df_rows > 0, mse * sq, np.nan),
    )
    return out


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
    res = _icc_2_1_weighted(Y, np.ones((1, n)))
    icc = float(res["icc"][0])
    msr = float(res["ms_between_subjects"][0])
    msc = float(res["ms_between_observers"][0])
    mse = float(res["ms_error"][0])
    if n < 2 or k < 2 or np.isnan(msr):
        f_stat, p_value = np.nan, np.nan
    elif mse > 0:
        f_stat = msr / mse
        p_value = float(stats.f.sf(f_stat, n - 1, (n - 1) * (k - 1)))
    elif msr > 0:
        # Perfect agreement with real between-subject variance: subjects are
        # perfectly separable, so the effect is infinitely significant.
        f_stat, p_value = np.inf, 0.0
    else:
        f_stat, p_value = np.nan, np.nan
    return {
        "icc": icc,
        "ms_between_subjects": msr,
        "ms_between_observers": msc,
        "ms_error": mse,
        "f_stat": float(f_stat),
        "p_value": float(p_value),
    }


def _row_bootstrap_weights(n: int, iterations: int, seed: int) -> NDArray:
    """Return row multiplicities for an ordinary (independent-row) bootstrap."""
    rng = np.random.default_rng(seed)
    indices = np.array([rng.choice(n, size=n, replace=True) for _ in range(iterations)])
    return _indices_to_weights(indices, n)


def _indices_to_weights(indices: NDArray, n: int) -> NDArray:
    """Convert a (draws, size) matrix of row indices into row multiplicities."""
    indices = np.asarray(indices, dtype=int)
    weights = np.zeros((indices.shape[0], n))
    np.add.at(weights, (np.arange(indices.shape[0])[:, None], indices), 1.0)
    return weights


def _percentile_ci(estimates: NDArray, iterations: int) -> tuple[float, float, int]:
    """Return the 95% percentile interval and the number of valid draws.

    At least half of the draws must give a finite ICC. With fewer valid draws
    the feature is degenerate and the function returns a NaN interval.
    """
    valid = estimates[np.isfinite(estimates)]
    if valid.size == 0 or valid.size < max(1, iterations // 2):
        return np.nan, np.nan, int(valid.size)
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)), int(valid.size)


def _bootstrap_icc_ci(
    Y: NDArray,
    feature_name: str,
    iterations: int = 1000,
    base_seed: int = 42,
    bootstrap_indices: NDArray | None = None,
) -> tuple[float, float]:
    """Deterministically estimate the 95% Confidence Interval for ICC(2,1) via bootstrapping.

    Note that with only a handful of subjects (``n`` small) the percentile CI is
    unreliable regardless of the number of iterations, since the resampling space
    is tiny; treat such intervals with caution.
    """
    n, k = Y.shape
    if n < 3 or k < 2 or iterations <= 0:
        return np.nan, np.nan
    if bootstrap_indices is not None:
        weights = _indices_to_weights(bootstrap_indices, n)
    else:
        seed = _get_deterministic_seed(feature_name, base_seed=base_seed)
        weights = _row_bootstrap_weights(n, iterations, seed)
    low, high, _ = _percentile_ci(_icc_2_1_weighted(Y, weights)["icc"], iterations)
    return low, high


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
