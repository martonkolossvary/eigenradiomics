"""Internal helper utilities for WGCNA operations."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray


def _wgcna_compute_eigengene(
    X_scaled: NDArray,
    n_components: int = 1,
    orient_eigengenes: str | bool = "mean_correlation",
) -> tuple[NDArray, NDArray]:
    """Compute the top n_components principal components (eigengenes) of scaled data.

    Parameters
    ----------
    X_scaled : ndarray of shape (n_samples, n_features)
        Centered and scaled module feature matrix.
    n_components : int, default=1
        Number of principal components (eigengenes) to extract.
    orient_eigengenes : {"mean_correlation", "hub_feature", "sum_loadings", False} or bool
        Direction alignment mode:
        - "mean_correlation" (or True): positive correlation with module mean expression.
        - "hub_feature": positive loading for the dominant feature in that component.
        - "sum_loadings": positive sum of component loadings.
        - False: keep raw SVD sign.

    Returns
    -------
    eigengene : ndarray
    loadings : ndarray
    """
    orient_mode: str | bool = (
        "mean_correlation" if orient_eigengenes is True else orient_eigengenes
    )

    U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)

    if n_components == 1:
        loadings = Vt[0]
        eigengene = X_scaled @ loadings

        if orient_mode is not False and orient_mode is not None:
            if orient_mode == "mean_correlation":
                avg_expr = X_scaled.mean(axis=1)
                if (
                    np.std(eigengene) >= 1e-12
                    and np.std(avg_expr) >= 1e-12
                    and np.corrcoef(eigengene, avg_expr)[0, 1] < -1e-10
                ):
                    loadings = -loadings
                    eigengene = -eigengene
            elif orient_mode == "hub_feature":
                j_max = int(np.argmax(np.abs(loadings)))
                if loadings[j_max] < 0:
                    loadings = -loadings
                    eigengene = -eigengene
            elif orient_mode == "sum_loadings":
                s = float(np.sum(loadings))
                if s < -1e-10:
                    loadings = -loadings
                    eigengene = -eigengene

        return eigengene, loadings

    n_components_eff = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    loadings = Vt[:n_components_eff].T  # shape (n_features, n_components_eff)
    eigengene = X_scaled @ loadings  # shape (n_samples, n_components_eff)

    if orient_mode is not False and orient_mode is not None:
        avg_expr = X_scaled.mean(axis=1)
        for col in range(n_components_eff):
            eg_col = eigengene[:, col]
            v_col = loadings[:, col]
            if orient_mode == "mean_correlation":
                if col == 0:
                    if (
                        np.std(eg_col) >= 1e-12
                        and np.std(avg_expr) >= 1e-12
                        and np.corrcoef(eg_col, avg_expr)[0, 1] < -1e-10
                    ):
                        loadings[:, col] = -v_col
                        eigengene[:, col] = -eg_col
                else:
                    if (
                        np.std(eg_col) >= 1e-12
                        and np.std(avg_expr) >= 1e-12
                        and abs(np.corrcoef(eg_col, avg_expr)[0, 1]) >= 1e-10
                    ):
                        if np.corrcoef(eg_col, avg_expr)[0, 1] < -1e-10:
                            loadings[:, col] = -v_col
                            eigengene[:, col] = -eg_col
                    else:
                        j_max = int(np.argmax(np.abs(v_col)))
                        if v_col[j_max] < 0:
                            loadings[:, col] = -v_col
                            eigengene[:, col] = -eg_col
            elif orient_mode == "hub_feature":
                j_max = int(np.argmax(np.abs(v_col)))
                if v_col[j_max] < 0:
                    loadings[:, col] = -v_col
                    eigengene[:, col] = -eg_col
            elif orient_mode == "sum_loadings":
                s = float(np.sum(v_col))
                if s < -1e-10:
                    loadings[:, col] = -v_col
                    eigengene[:, col] = -eg_col

    return eigengene, loadings


def _wgcna_project_module(
    X_arr: NDArray, feat_idx: NDArray, center: NDArray, scale: NDArray, loading: NDArray
) -> NDArray:
    """Project data for a single module (runs in worker processes)."""
    X_mod = X_arr[:, feat_idx]
    X_scaled = np.nan_to_num((X_mod - center) / scale, nan=0.0)
    return np.asarray(X_scaled @ loading)


def _wgcna_fit_single(
    mod: str,
    X_arr: NDArray,
    feat_idx: NDArray,
    n_components: int = 1,
    orient_eigengenes: str | bool = "mean_correlation",
) -> tuple[str, NDArray, NDArray, NDArray]:
    """Compute loadings for a single module (runs in worker processes)."""
    X_mod = X_arr[:, feat_idx]
    centers = np.nanmean(X_mod, axis=0)
    centers = np.nan_to_num(centers, nan=0.0)
    scales = np.nanstd(X_mod, axis=0, ddof=1)
    scales = np.nan_to_num(scales, nan=1.0)
    n_const = int((scales == 0).sum())
    if n_const > 0:
        warnings.warn(
            f"Module '{mod}' contains {n_const} zero-variance "
            f"feature(s); their scale is set to 1.0.",
            stacklevel=2,
        )
        scales[scales == 0] = 1.0
    X_scaled = np.nan_to_num((X_mod - centers) / scales, nan=0.0)
    _, loadings = _wgcna_compute_eigengene(X_scaled, n_components, orient_eigengenes)
    return mod, centers, scales, loadings
