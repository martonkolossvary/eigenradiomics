"""Internal helper utilities for WGCNA operations."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray


def _wgcna_compute_eigengene(X_scaled: NDArray) -> tuple[NDArray, NDArray]:
    """Compute the first principal component (eigengene) of scaled data.

    Returns (eigengene, loadings).
    """
    U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    loadings = Vt[0]
    eigengene = X_scaled @ loadings
    # Sign alignment: positive correlation with mean expression.
    avg_expr = X_scaled.mean(axis=1)

    # Bypass np.corrcoef if variance is effectively zero to avoid RuntimeWarning
    if np.std(eigengene) < 1e-12 or np.std(avg_expr) < 1e-12:
        return eigengene, loadings

    if np.corrcoef(eigengene, avg_expr)[0, 1] < -1e-10:
        loadings = -loadings
        eigengene = -eigengene
    return eigengene, loadings


def _wgcna_project_module(
    X_arr: NDArray, feat_idx: NDArray, center: NDArray, scale: NDArray, loading: NDArray
) -> NDArray:
    """Project data for a single module (runs in worker processes)."""
    X_mod = X_arr[:, feat_idx]
    X_scaled = (X_mod - center) / scale
    return np.asarray(X_scaled @ loading)


def _wgcna_fit_single(
    mod: str, X_arr: NDArray, feat_idx: NDArray
) -> tuple[str, NDArray, NDArray, NDArray]:
    """Compute loadings for a single module (runs in worker processes)."""
    X_mod = X_arr[:, feat_idx]
    centers = X_mod.mean(axis=0)
    scales = X_mod.std(axis=0, ddof=1)
    n_const = int((scales == 0).sum())
    if n_const > 0:
        warnings.warn(
            f"Module '{mod}' contains {n_const} zero-variance "
            f"feature(s); their scale is set to 1.0.",
            stacklevel=2,
        )
        scales[scales == 0] = 1.0
    X_scaled = (X_mod - centers) / scales
    _, loadings = _wgcna_compute_eigengene(X_scaled)
    return mod, centers, scales, loadings
