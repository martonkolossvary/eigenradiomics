"""WGCNAReducer — sklearn-compatible WGCNA dimensionality reduction.

Wraps PyWGCNA's static network-construction functions and adds custom SVD-based
eigengene computation so that ``transform`` can project unseen data.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import warnings
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.cluster.hierarchy import cut_tree, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.utils.validation import check_is_fitted

from eigenradiomics._utils import check_constant_features
from eigenradiomics.reducers._base import BaseReducer
from eigenradiomics.reducers._wgcna_utils import (
    _wgcna_compute_eigengene,
    _wgcna_fit_single,
    _wgcna_project_module,
)

if TYPE_CHECKING:
    import matplotlib.figure

logger = logging.getLogger(__name__)

# Lock shared across all WGCNAReducer instances to serialise fd-level
# stdout redirection, making _capture_output safe in threaded callers.
_capture_lock = threading.Lock()


def _import_pywgcna() -> Any:
    """Lazy import of PyWGCNA with a clear error message."""
    try:
        from PyWGCNA import WGCNA as _WGCNA  # noqa: N811
    except ImportError as exc:
        raise ImportError(
            "PyWGCNA is required for WGCNAReducer. "
            "Install it with: pip install 'eigenradiomics[wgcna]'"
        ) from exc
    return _WGCNA


class WGCNAReducer(BaseReducer):
    """Dimensionality reduction via Weighted Gene Co-expression Network Analysis.

    Builds a co-expression network from the feature matrix, identifies modules
    of correlated features using PyWGCNA's static methods, and represents each
    module by its first principal component (eigengene).  The loadings are
    stored so that ``transform`` can project unseen data without re-running
    the network construction.

    Parameters
    ----------
    network_type : str
        ``"signed hybrid"`` | ``"signed"`` | ``"unsigned"``.
    tom_type : str
        ``"signed"`` | ``"unsigned"``.
    correlation_method : str
        ``"pearson"`` (default) or ``"spearman"``.  Spearman rank-transforms the
        features for network construction (adjacency, TOM, clustering, merging),
        making the network robust to monotone non-linearities; the stored
        eigengene loadings are computed on the original scale so ``transform``
        of unseen data stays leakage-safe.
    soft_power : int or ``"auto"``
        Soft-thresholding power.  ``"auto"`` selects via scale-free topology.
    r_squared_cut : float
        R² threshold for automatic soft-power selection.
    mean_cut : float
        Mean connectivity threshold for soft-power selection.
    power_min, power_max : int
        Inclusive range of candidate powers evaluated when ``soft_power="auto"``.
    min_module_size : int
        Minimum number of features per module.
    me_diss_threshold : float
        Module eigengene dissimilarity threshold for merging (0–1).
    deep_split : int
        ``cutreeHybrid`` sensitivity (0–4).
    pam_respect_dendro : bool
        Passed to ``cutreeHybrid``.
    store_tom : bool
        If True, keep the TOM matrix as ``tom_`` (memory-intensive).
    include_grey : bool
        If True, include unassigned (grey) features as a pseudo-module.
    verbose : int
        0 = suppress PyWGCNA stdout, ≥1 = pass through.
    log_file : str or None
        If set, redirect PyWGCNA stdout to this file.
    n_module_components : int
        Number of principal components (eigengenes) to extract per module (default: 1).
    n_jobs : int or None
        Number of parallel jobs for per-module SVD computation during
        ``fit`` and ``transform``.  ``None`` means 1 (sequential).  -1
        means use all available cores.  Requires ``joblib``.
    missing_strategy : {"raise", "zero", "floor", "mean", "median", "pairwise"}, default="raise"
        How to handle missing values (NaNs) in the feature matrix:
        - "raise": Raise ValueError on NaN input (strict default).
        - "zero": Impute NaNs with 0.0 (ideal for delta / change matrices).
        - "floor": Impute NaNs with the training feature-wise minimum.
        - "mean": Impute NaNs with the training feature-wise mean.
        - "median": Impute NaNs with the training feature-wise median.
        - "pairwise": Compute adjacency via pairwise-complete correlation,
          falling back to training-mean imputation for SVD projection.
    orient_eigengenes : {"mean_correlation", "hub_feature", "sum_loadings", False} or bool,
        default="mean_correlation"
        Sign alignment mode for module eigengenes:
        - "mean_correlation" (or True): Ensure positive correlation with module mean expression.
        - "hub_feature": Ensure positive loading for dominant feature in each component.
        - "sum_loadings": Ensure positive sum of loadings.
        - False: Retain raw SVD sign.
    """

    _reducer_prefix: str = "wgcna"

    def __init__(
        self,
        network_type: str = "signed hybrid",
        tom_type: str = "signed",
        correlation_method: str = "pearson",
        soft_power: int | str = "auto",
        r_squared_cut: float = 0.9,
        mean_cut: float = 100,
        power_min: int = 1,
        power_max: int = 30,
        min_module_size: int = 50,
        me_diss_threshold: float = 0.2,
        deep_split: int = 2,
        pam_respect_dendro: bool = False,
        store_tom: bool = False,
        include_grey: bool = False,
        verbose: int = 0,
        log_file: str | None = None,
        n_jobs: int | None = None,
        n_module_components: int = 1,
        missing_strategy: str = "raise",
        orient_eigengenes: str | bool = "mean_correlation",
    ) -> None:
        self.network_type = network_type
        self.tom_type = tom_type
        self.correlation_method = correlation_method
        self.soft_power = soft_power
        self.r_squared_cut = r_squared_cut
        self.mean_cut = mean_cut
        self.power_min = power_min
        self.power_max = power_max
        self.min_module_size = min_module_size
        self.me_diss_threshold = me_diss_threshold
        self.deep_split = deep_split
        self.pam_respect_dendro = pam_respect_dendro
        self.store_tom = store_tom
        self.include_grey = include_grey
        self.verbose = verbose
        self.log_file = log_file
        self.n_jobs = n_jobs
        self.n_module_components = n_module_components
        self.missing_strategy = missing_strategy
        self.orient_eigengenes = orient_eigengenes

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def _get_tags(self) -> dict[str, bool]:
        """Return scikit-learn tags for estimator checks (sklearn < 1.6)."""
        tags = super()._get_tags()
        tags["allow_nan"] = self.missing_strategy != "raise"
        return tags

    def __sklearn_tags__(self) -> Any:
        """Return scikit-learn tags for estimator checks (sklearn >= 1.6)."""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = self.missing_strategy != "raise"
        return tags

    def _validate_params(self) -> None:
        """Validate constructor parameters at fit time."""
        if self.soft_power != "auto" and (
            not isinstance(self.soft_power, int) or self.soft_power < 1
        ):
            raise ValueError(
                f"soft_power must be a positive integer or 'auto', got {self.soft_power!r}."
            )
        if not 0 < self.r_squared_cut <= 1:
            raise ValueError(f"r_squared_cut must be in (0, 1], got {self.r_squared_cut}.")
        if self.mean_cut <= 0:
            raise ValueError(f"mean_cut must be positive, got {self.mean_cut}.")
        if not isinstance(self.min_module_size, int) or self.min_module_size < 1:
            raise ValueError(
                f"min_module_size must be a positive integer, got {self.min_module_size}."
            )
        if not 0 <= self.me_diss_threshold <= 1:
            raise ValueError(f"me_diss_threshold must be in [0, 1], got {self.me_diss_threshold}.")
        if not isinstance(self.n_module_components, int) or self.n_module_components < 1:
            raise ValueError(
                f"n_module_components must be a positive integer, got {self.n_module_components}."
            )
        if self.n_jobs is not None and (not isinstance(self.n_jobs, int) or self.n_jobs == 0):
            raise ValueError(f"n_jobs must be a non-zero integer or None, got {self.n_jobs!r}.")
        if self.deep_split not in {0, 1, 2, 3, 4}:
            raise ValueError(f"deep_split must be 0, 1, 2, 3, or 4, got {self.deep_split}.")
        if not isinstance(self.verbose, int) or self.verbose < 0:
            raise ValueError(f"verbose must be a non-negative integer, got {self.verbose}.")
        valid_network_types = {"signed hybrid", "signed", "unsigned"}
        if self.network_type not in valid_network_types:
            raise ValueError(
                f"network_type must be one of {valid_network_types}, got {self.network_type!r}."
            )
        valid_tom_types = {"signed", "unsigned"}
        if self.tom_type not in valid_tom_types:
            raise ValueError(f"tom_type must be one of {valid_tom_types}, got {self.tom_type!r}.")
        valid_correlation_methods = {"pearson", "spearman"}
        if self.correlation_method not in valid_correlation_methods:
            raise ValueError(
                f"correlation_method must be one of {valid_correlation_methods}, "
                f"got {self.correlation_method!r}."
            )
        if not isinstance(self.power_min, int) or self.power_min < 1:
            raise ValueError(f"power_min must be a positive integer, got {self.power_min}.")
        if not isinstance(self.power_max, int) or self.power_max < self.power_min:
            raise ValueError(
                f"power_max must be an integer >= power_min ({self.power_min}), "
                f"got {self.power_max}."
            )
        valid_missing_strategies = {"raise", "zero", "floor", "mean", "median", "pairwise"}
        if self.missing_strategy not in valid_missing_strategies:
            raise ValueError(
                f"missing_strategy must be one of {valid_missing_strategies}, "
                f"got {self.missing_strategy!r}."
            )
        valid_orient_modes = {"mean_correlation", "hub_feature", "sum_loadings", True, False}
        if self.orient_eigengenes not in valid_orient_modes:
            raise ValueError(
                f"orient_eigengenes must be one of {valid_orient_modes}, "
                f"got {self.orient_eigengenes!r}."
            )

    def _clear_fitted_attributes(self) -> None:
        """Remove all fitted attributes so a refit starts clean."""
        fitted_attrs = [
            "soft_power_",
            "soft_power_table_",
            "dendrogram_",
            "tom_",
            "module_colors_",
            "module_names_",
            "module_assignments_",
            "module_centers_",
            "module_scales_",
            "module_loadings_",
            "n_components_",
            "feature_names_in_",
            "n_features_in_",
            "impute_values_",
        ]
        for attr in fitted_attrs:
            if hasattr(self, attr):
                delattr(self, attr)

    def _guard_all_nan_features(self, feature_names: Sequence[str]) -> None:
        """Check for all-NaN features, replace impute value with 0.0, and warn."""
        if self.impute_values_ is None:
            return
        nan_mask = np.isnan(self.impute_values_)
        if np.any(nan_mask):
            n_nan = int(np.sum(nan_mask))
            names = ", ".join(str(feature_names[i]) for i in np.flatnonzero(nan_mask)[:5])
            suffix = " ..." if n_nan > 5 else ""
            warnings.warn(
                f"{n_nan} feature(s) are entirely NaN across all training samples; "
                f"their impute values are set to 0.0: {names}{suffix}.",
                UserWarning,
                stacklevel=3,
            )
            self.impute_values_[nan_mask] = 0.0

    def fit(self, X: NDArray | pd.DataFrame, y: None = None) -> WGCNAReducer:
        """Build the WGCNA network and compute module eigengene loadings.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored
        """
        self._validate_params()
        self._clear_fitted_attributes()

        WGCNA = _import_pywgcna()

        X_arr = self._validate_input(X, reset=True, allow_nan=(self.missing_strategy != "raise"))
        feature_names = self.feature_names_in_
        n_samples, n_features = X_arr.shape
        if n_samples < 3:
            raise ValueError(
                "WGCNAReducer requires n_samples >= 3 to estimate correlations; "
                f"got n_samples = {n_samples}."
            )
        if n_features < 3:
            raise ValueError(
                "WGCNAReducer requires n_features >= 3 to build a correlation network; "
                f"got n_features = {n_features}."
            )

        # 1. Compute impute values
        if self.missing_strategy == "raise":
            self.impute_values_ = None
            X_imputed = X_arr
        elif self.missing_strategy == "zero":
            self.impute_values_ = np.zeros(n_features, dtype=float)
            X_imputed = np.where(np.isnan(X_arr), 0.0, X_arr)
        elif self.missing_strategy == "floor":
            self.impute_values_ = np.nanmin(X_arr, axis=0)
            self._guard_all_nan_features(feature_names)
            X_imputed = np.where(np.isnan(X_arr), self.impute_values_, X_arr)
        elif self.missing_strategy in ("mean", "pairwise"):
            self.impute_values_ = np.nanmean(X_arr, axis=0)
            self._guard_all_nan_features(feature_names)
            if self.missing_strategy == "pairwise":
                X_imputed = X_arr
            else:
                X_imputed = np.where(np.isnan(X_arr), self.impute_values_, X_arr)
        elif self.missing_strategy == "median":
            self.impute_values_ = np.nanmedian(X_arr, axis=0)
            self._guard_all_nan_features(feature_names)
            X_imputed = np.where(np.isnan(X_arr), self.impute_values_, X_arr)

        # Constant features check
        if self.missing_strategy != "pairwise":
            check_mat = X_imputed
        else:
            assert self.impute_values_ is not None
            check_mat = np.where(np.isnan(X_arr), self.impute_values_, X_arr)
        constant_mask = check_constant_features(check_mat, threshold=0.0)
        constant = np.flatnonzero(constant_mask)
        if constant.size > 0:
            names = ", ".join(str(feature_names[i]) for i in constant[:5])
            suffix = " ..." if constant.size > 5 else ""
            warnings.warn(
                f"{constant.size} feature(s) have zero variance and will not enter the "
                f"correlation network (left unassigned): {names}{suffix}.",
                stacklevel=2,
            )

        if self.correlation_method == "spearman":
            X_net = pd.DataFrame(X_imputed).rank(axis=0).to_numpy()
        else:
            X_net = X_imputed

        with self._capture_output():
            if self.missing_strategy == "pairwise":
                df_net = pd.DataFrame(X_net, columns=feature_names)
                corr_df = df_net.corr(method="pearson")
                corr_mat = np.nan_to_num(corr_df.to_numpy(), nan=0.0)
                corr_mat = 0.5 * (corr_mat + corr_mat.T)
                np.fill_diagonal(corr_mat, 1.0)

                if self.network_type == "unsigned":
                    sim_mat = np.abs(corr_mat)
                elif self.network_type == "signed":
                    sim_mat = 0.5 * (1.0 + corr_mat)
                else:  # "signed hybrid"
                    sim_mat = np.maximum(0.0, corr_mat)
                sim_mat = 0.5 * (sim_mat + sim_mat.T)
                np.fill_diagonal(sim_mat, 1.0)
                sim_df = pd.DataFrame(sim_mat, columns=feature_names, index=feature_names)

                # --- 1. Soft power selection ---
                if self.soft_power == "auto":
                    power_est, power_table = WGCNA.pickSoftThreshold(
                        sim_df,
                        dataIsExpr=False,
                        RsquaredCut=self.r_squared_cut,
                        MeanCut=self.mean_cut,
                        powerVector=list(range(self.power_min, self.power_max + 1)),
                        networkType=self.network_type,
                    )
                    if power_est is None:
                        raise ValueError(
                            "Automatic soft-power selection failed — no power "
                            "reached the R² threshold. Try lowering r_squared_cut "
                            "or setting soft_power to an explicit integer."
                        )
                    self.soft_power_ = int(power_est)
                    self.soft_power_table_ = power_table
                    logger.info("Auto-selected soft power: %d", self.soft_power_)
                else:
                    self.soft_power_ = int(self.soft_power)
                    self.soft_power_table_ = None

                # --- 2. Adjacency matrix ---
                adj_mat = (sim_mat**self.soft_power_).astype(float)
                np.fill_diagonal(adj_mat, 1.0)

                # --- 3. TOM ---
                tom = WGCNA.TOMsimilarity(adj_mat, TOMType=self.tom_type)
                tom_arr = tom.values if isinstance(tom, pd.DataFrame) else np.asarray(tom)

            else:
                df = pd.DataFrame(X_net, columns=feature_names)

                # --- 1. Soft power selection ---
                if self.soft_power == "auto":
                    power_est, power_table = WGCNA.pickSoftThreshold(
                        df,
                        dataIsExpr=True,
                        RsquaredCut=self.r_squared_cut,
                        MeanCut=self.mean_cut,
                        powerVector=list(range(self.power_min, self.power_max + 1)),
                        networkType=self.network_type,
                    )
                    if power_est is None:
                        raise ValueError(
                            "Automatic soft-power selection failed — no power "
                            "reached the R² threshold. Try lowering r_squared_cut "
                            "or setting soft_power to an explicit integer."
                        )
                    self.soft_power_ = int(power_est)
                    self.soft_power_table_ = power_table
                    logger.info("Auto-selected soft power: %d", self.soft_power_)
                else:
                    self.soft_power_ = int(self.soft_power)
                    self.soft_power_table_ = None

                # --- 2. Adjacency matrix ---
                adj = WGCNA.adjacency(
                    df,
                    adjacencyType=self.network_type,
                    power=self.soft_power_,
                )

                # --- 3. TOM ---
                tom = WGCNA.TOMsimilarity(adj, TOMType=self.tom_type)
                tom_arr = tom.values if isinstance(tom, pd.DataFrame) else np.asarray(tom)
                del adj

            if self.store_tom:
                self.tom_ = tom_arr

            # --- 4. Hierarchical clustering on 1 - TOM ---
            dist = 1.0 - tom_arr
            np.fill_diagonal(dist, 0.0)
            dist_condensed = dist[np.triu_indices(n_features, k=1)]
            dendro = linkage(dist_condensed, method="average")
            del dist_condensed

            self.dendrogram_ = dendro

            # --- 5. Dynamic tree cut ---
            dist_df = pd.DataFrame(dist, columns=feature_names, index=feature_names)
            cut_result = WGCNA.cutreeHybrid(
                dendro,
                dist_df,
                minClusterSize=self.min_module_size,
                deepSplit=self.deep_split,
                pamRespectsDendro=self.pam_respect_dendro,
            )
            del dist, dist_df
            if isinstance(cut_result, pd.DataFrame):
                numeric_labels = cut_result.iloc[:, 0].values
            else:
                numeric_labels = np.asarray(cut_result)

            # --- 6. Map numeric labels → colors ---
            label_df = pd.DataFrame({"Value": numeric_labels})
            colors = WGCNA.labels2colors(label_df)
            color_list = colors.tolist() if isinstance(colors, np.ndarray) else list(colors)

        # Identify the unassigned ("grey") module
        numeric_labels_arr = np.asarray(numeric_labels)
        grey_color: str | None = None
        if (numeric_labels_arr == 0).any():
            grey_idx = int(np.flatnonzero(numeric_labels_arr == 0)[0])
            grey_color = color_list[grey_idx]

        # --- 7. Merge close modules ---
        merged_colors = self._merge_close_modules(X_net, color_list, protect=grey_color)

        # --- 8. Build module assignments ---
        unique_modules = sorted(set(merged_colors))
        if not self.include_grey and grey_color is not None and grey_color in unique_modules:
            unique_modules.remove(grey_color)

        self.module_colors_ = merged_colors
        self.module_names_ = unique_modules
        self.module_assignments_: dict[str, NDArray] = {}
        for mod in unique_modules:
            self.module_assignments_[mod] = np.array(
                [i for i, c in enumerate(merged_colors) if c == mod]
            )

        # --- 9. Compute eigengene loadings via SVD ---
        self._compute_module_loadings(X_arr, self.n_module_components)

        self.n_components_ = sum(
            self.module_loadings_[mod].shape[1] if self.module_loadings_[mod].ndim == 2 else 1
            for mod in unique_modules
        )
        if self.n_components_ == 0:
            warnings.warn(
                "WGCNA found no modules (every feature was assigned to the grey "
                "unassigned group); transform will return a (n_samples, 0) matrix. "
                "Lower min_module_size or set include_grey=True.",
                stacklevel=2,
            )
        return self

    def transform(self, X: NDArray | pd.DataFrame) -> NDArray:
        """Project data into the module eigengene space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        Y : ndarray of shape (n_samples, n_components_)
        """
        check_is_fitted(self, "module_loadings_")
        X_arr = self._validate_input(X, reset=False, allow_nan=(self.missing_strategy != "raise"))
        if self.missing_strategy != "raise" and self.impute_values_ is not None:
            X_arr = np.where(np.isnan(X_arr), self.impute_values_, X_arr)

        n_samples = X_arr.shape[0]

        effective_jobs = self.n_jobs
        if effective_jobs is not None and effective_jobs != 1 and len(self.module_names_) > 1:
            from joblib import Parallel, delayed

            columns = Parallel(n_jobs=effective_jobs)(
                delayed(_wgcna_project_module)(
                    X_arr,
                    self.module_assignments_[mod],
                    self.module_centers_[mod],
                    self.module_scales_[mod],
                    self.module_loadings_[mod],
                )
                for mod in self.module_names_
            )
        else:
            columns = [
                _wgcna_project_module(
                    X_arr,
                    self.module_assignments_[mod],
                    self.module_centers_[mod],
                    self.module_scales_[mod],
                    self.module_loadings_[mod],
                )
                for mod in self.module_names_
            ]

        result: NDArray = np.column_stack(columns) if columns else np.empty((n_samples, 0))
        return result

    def inverse_transform(self, Y: NDArray) -> NDArray:
        """Map reduced data back to the original space via the SVD loadings.

        Parameters
        ----------
        Y : array-like of shape (n_samples, n_components_)

        Returns
        -------
        X_recon : ndarray of shape (n_samples, n_features)
        """
        check_is_fitted(self, "module_loadings_")
        Y_arr = np.asarray(Y, dtype=float)
        if Y_arr.ndim != 2 or Y_arr.shape[1] != self.n_components_:
            raise ValueError(
                f"Expected 2-D array with {self.n_components_} columns, got {Y_arr.shape}."
            )

        n_samples = Y_arr.shape[0]
        n_features = len(self.feature_names_in_)
        X_recon = np.zeros((n_samples, n_features))

        col_offset = 0
        for mod in self.module_names_:
            feat_idx = self.module_assignments_[mod]
            loadings = self.module_loadings_[mod]
            if loadings.ndim == 2:
                k_i = loadings.shape[1]
                Y_mod = Y_arr[:, col_offset : col_offset + k_i]
                X_mod_scaled = Y_mod @ loadings.T
                col_offset += k_i
            else:
                Y_mod = Y_arr[:, col_offset : col_offset + 1]
                loadings_row = loadings.reshape(1, -1)
                X_mod_scaled = Y_mod @ loadings_row
                col_offset += 1
            # Unscale and recenter
            X_recon[:, feat_idx] = (
                X_mod_scaled * self.module_scales_[mod] + self.module_centers_[mod]
            )

        return X_recon

    def get_catalog_enrichment(
        self,
        catalog: Any,
        columns: str | Sequence[str] | None = None,
        *,
        include_grey: bool = False,
        method: str = "fisher",
        alternative: str = "greater",
    ) -> pd.DataFrame:
        """Compute over-representation analysis of catalog metadata within WGCNA modules.

        Parameters
        ----------
        catalog : FeatureCatalog, DataFrame, or CSV path
            Catalog describing features.
        columns : str or sequence of str, optional
            Catalog columns to test (e.g. ``["family", "family_group"]``). If None,
            tests all eligible string/categorical columns in the catalog.
        include_grey : bool, default=False
            Whether to include the unassigned 'grey' module in enrichment calculations.
        method : {"fisher", "chi2", "hypergeometric"}, default="fisher"
            Statistical test for over-representation.
        alternative : {"greater", "less", "two-sided"}, default="greater"
            Alternative hypothesis for the test.

        Returns
        -------
        enrichment : pandas.DataFrame
            Detailed over-representation results table.
        """
        check_is_fitted(self, "module_assignments_")
        assignments = {}
        for mod, idxs in self.module_assignments_.items():
            for i in idxs:
                assignments[self.feature_names_in_[i]] = mod

        labels_series = pd.Series(assignments)
        if not include_grey:
            labels_series = labels_series[labels_series != "grey"]

        from eigenradiomics.analysis import compute_group_enrichment

        return compute_group_enrichment(
            labels_series,
            catalog,
            columns=columns,
            method=method,
            alternative=alternative,
        )

    # ------------------------------------------------------------------
    # WGCNA-specific public methods (all prefixed wgcna_)
    # ------------------------------------------------------------------

    def wgcna_get_module_assignments(self) -> dict[str, list[str]]:
        """Return mapping of module colour → list of input feature names.

        Returns
        -------
        assignments : dict[str, list[str]]
        """
        check_is_fitted(self, "module_assignments_")
        return {
            mod: self.feature_names_in_[idx].tolist()
            for mod, idx in self.module_assignments_.items()
        }

    def wgcna_get_module_sizes(self) -> dict[str, int]:
        """Return number of features in each module.

        Returns
        -------
        sizes : dict[str, int]
        """
        check_is_fitted(self, "module_assignments_")
        return {mod: len(idx) for mod, idx in self.module_assignments_.items()}

    def wgcna_get_soft_power_table(self) -> pd.DataFrame | None:
        """Return the soft-power analysis table (only when ``soft_power="auto"``).

        Returns
        -------
        table : DataFrame or None
        """
        check_is_fitted(self, "soft_power_")
        return self.soft_power_table_

    def wgcna_get_feature_importances(self, *, normalize: bool = True) -> dict[str, pd.DataFrame]:
        """Return per-module feature importances based on SVD loadings.

        Parameters
        ----------
        normalize : bool
            If True (default), loadings are rescaled so that each module's
            importances sum to 1.

        Returns
        -------
        importances : dict[str, DataFrame]
            Mapping of module colour → DataFrame with columns
            ``["feature", "loading", "importance"]``, sorted by
            ``importance`` descending.
        """
        check_is_fitted(self, "module_loadings_")
        result: dict[str, pd.DataFrame] = {}
        for mod in self.module_names_:
            feat_idx = self.module_assignments_[mod]
            names = self.feature_names_in_[feat_idx]
            loadings = self.module_loadings_[mod]
            if loadings.ndim == 2:
                abs_load = np.sqrt((loadings**2).sum(axis=1))
                primary_loading = loadings[:, 0]
            else:
                abs_load = np.abs(loadings)
                primary_loading = loadings
            if normalize and abs_load.sum() > 0:
                importance = abs_load / abs_load.sum()
            else:
                importance = abs_load
            df = pd.DataFrame(
                {
                    "feature": names,
                    "loading": primary_loading,
                    "importance": importance,
                }
            )
            result[mod] = df.sort_values("importance", ascending=False).reset_index(drop=True)
        return result

    def wgcna_plot_soft_power(
        self, figsize: tuple[int, int] = (10, 5)
    ) -> matplotlib.figure.Figure:
        """Diagnostic plot for scale-free topology fit."""
        import matplotlib.pyplot as plt

        check_is_fitted(self, "soft_power_")
        table = self.soft_power_table_
        if table is None:
            raise RuntimeError(
                "Soft power table not available — fit with soft_power='auto' first."
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        power = table["Power"]
        ax1.scatter(power, table["SFT.R.sq"], c="steelblue", s=30)
        ax1.axhline(self.r_squared_cut, color="red", ls="--", lw=0.8)
        ax1.axvline(self.soft_power_, color="green", ls=":", lw=0.8)
        ax1.set_xlabel("Soft Threshold (power)")
        ax1.set_ylabel("Scale-Free Topology Model Fit (R²)")
        ax1.set_title("Scale independence")

        ax2.scatter(power, table["mean(k)"], c="steelblue", s=30)
        ax2.axvline(self.soft_power_, color="green", ls=":", lw=0.8)
        ax2.set_xlabel("Soft Threshold (power)")
        ax2.set_ylabel("Mean Connectivity")
        ax2.set_title("Mean connectivity")

        fig.tight_layout()
        return fig

    def wgcna_plot_dendrogram(
        self,
        figsize: tuple[int, int] = (12, 4),
    ) -> matplotlib.figure.Figure:
        """Plot the feature dendrogram with module colour bar."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba
        from scipy.cluster.hierarchy import dendrogram as scipy_dendro

        check_is_fitted(self, "dendrogram_")

        fig, (ax_dendro, ax_colors) = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": [4, 0.4]}
        )

        d = scipy_dendro(self.dendrogram_, ax=ax_dendro, no_labels=True)
        ax_dendro.set_title("Feature Dendrogram")
        ax_dendro.set_ylabel("Height")

        leaves = d["leaves"]
        colors_rgba = []
        for idx in leaves:
            c = self.module_colors_[idx]
            try:
                colors_rgba.append(to_rgba(c))
            except ValueError:
                colors_rgba.append(to_rgba("lightgrey"))

        ax_colors.imshow(
            [colors_rgba],
            aspect="auto",
            interpolation="nearest",
        )
        ax_colors.set_yticks([])
        ax_colors.set_xticks([])
        ax_colors.set_ylabel("Module")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # reduction-artifact hooks (override BaseReducer defaults)
    # ------------------------------------------------------------------

    def _artifact_similarity(self) -> pd.DataFrame | None:
        """Topological Overlap Matrix (only available when ``store_tom=True``)."""
        if not hasattr(self, "tom_"):
            return None
        names = self.feature_names_in_
        return pd.DataFrame(self.tom_, index=names, columns=names)

    def _artifact_linkage(self) -> NDArray | None:
        """Average-linkage hierarchical clustering on ``1 - TOM``."""
        linkage_matrix: NDArray = self.dendrogram_
        return linkage_matrix

    def _artifact_cluster_labels(self) -> pd.Series | None:
        """Per-feature merged module colour (includes any unassigned features)."""
        return pd.Series(self.module_colors_, index=self.feature_names_in_, name="module")

    def _artifact_feature_order(self) -> NDArray | None:
        """Feature names in dendrogram-leaf order."""
        order = leaves_list(self.dendrogram_)
        ordered: NDArray = np.asarray(self.feature_names_in_)[order]
        return ordered

    def _artifact_feature_importances(self) -> pd.DataFrame | None:
        """Per-module SVD-loading importances flattened into one table."""
        frames = []
        for module, frame in self.wgcna_get_feature_importances().items():
            annotated = frame.copy()
            annotated["module"] = module
            frames.append(annotated)
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _merge_close_modules(
        self, X_arr: NDArray, color_list: list[str], protect: str | None = None
    ) -> list[str]:
        """Iteratively merge modules via hierarchical clustering of eigengene dissimilarity."""
        colors = list(color_list)

        while True:
            unique_mods = sorted(set(colors))
            if protect is not None and protect in unique_mods:
                unique_mods.remove(protect)
            n_mods = len(unique_mods)
            if n_mods <= 1:
                break

            # Compute eigengenes per module using NaN-safe math
            eigengenes = np.empty((X_arr.shape[0], n_mods))
            for col, mod in enumerate(unique_mods):
                idx = np.array([i for i, c in enumerate(colors) if c == mod])
                X_mod = X_arr[:, idx]
                centers = np.nanmean(X_mod, axis=0)
                centers = np.nan_to_num(centers, nan=0.0)
                scales = np.nanstd(X_mod, axis=0, ddof=1)
                scales = np.nan_to_num(scales, nan=1.0)
                n_const = int((scales == 0).sum())
                if n_const > 0:
                    logger.debug(
                        "Module '%s' has %d zero-variance feature(s) during merge.",
                        mod,
                        n_const,
                    )
                    scales[scales == 0] = 1.0
                X_scaled = np.nan_to_num((X_mod - centers) / scales, nan=0.0)
                eg, _ = _wgcna_compute_eigengene(X_scaled, 1, self.orient_eigengenes)
                eigengenes[:, col] = eg

            # Dissimilarity matrix: 1 - |cor| (abs for radiomics)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                corr_mat = np.corrcoef(eigengenes, rowvar=False)

            n_nan = np.isnan(corr_mat).sum()
            if n_nan > 0:
                warnings.warn(
                    f"Eigengene correlation matrix contains {n_nan} NaN "
                    f"entries (likely from constant-variance modules). "
                    f"These are set to 0 for merging.",
                    stacklevel=2,
                )
                np.nan_to_num(corr_mat, copy=False, nan=0.0)
            diss_mat = 1.0 - np.abs(corr_mat)
            np.fill_diagonal(diss_mat, 0.0)

            # Hierarchical clustering (average linkage) + cut at threshold
            diss_condensed = squareform(diss_mat, checks=False)
            me_dendro = linkage(diss_condensed, method="average")
            branch_labels = cut_tree(me_dendro, height=self.me_diss_threshold).ravel()

            # Determine merges: modules sharing a branch are merged
            n_new_mods = len(set(branch_labels))
            if n_new_mods >= n_mods:
                break

            branch_to_color: dict[int, str] = {}
            for branch_id in sorted(set(branch_labels)):
                members = [unique_mods[k] for k, b in enumerate(branch_labels) if b == branch_id]
                best = max(members, key=lambda m: sum(1 for c in colors if c == m))
                branch_to_color[branch_id] = best

            mod_to_merged: dict[str, str] = {}
            for k, mod in enumerate(unique_mods):
                mod_to_merged[mod] = branch_to_color[branch_labels[k]]
            colors = [mod_to_merged.get(c, c) for c in colors]

        return colors

    def _compute_module_loadings(self, X_arr: NDArray, n_module_components: int = 1) -> None:
        """Compute and store SVD loadings for each module."""
        self.module_centers_: dict[str, NDArray] = {}
        self.module_scales_: dict[str, NDArray] = {}
        self.module_loadings_: dict[str, NDArray] = {}

        effective_jobs = self.n_jobs
        if effective_jobs is not None and effective_jobs != 1 and len(self.module_names_) > 1:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=effective_jobs)(
                delayed(_wgcna_fit_single)(
                    mod,
                    X_arr,
                    self.module_assignments_[mod],
                    n_module_components,
                    self.orient_eigengenes,
                )
                for mod in self.module_names_
            )
        else:
            results = [
                _wgcna_fit_single(
                    mod,
                    X_arr,
                    self.module_assignments_[mod],
                    n_module_components,
                    self.orient_eigengenes,
                )
                for mod in self.module_names_
            ]

        for mod, centers, scales, loadings in results:
            self.module_centers_[mod] = centers
            self.module_scales_[mod] = scales
            self.module_loadings_[mod] = loadings

    @contextlib.contextmanager
    def _capture_output(self) -> Iterator[None]:
        """Context manager to capture or suppress PyWGCNA stdout."""
        import multiprocessing

        if self.verbose >= 1 and self.log_file is None:
            yield
            return

        if multiprocessing.current_process().name != "MainProcess":
            yield
            return

        target = self.log_file if self.log_file is not None else os.devnull

        old_stdout_fd = None
        with _capture_lock:
            try:
                old_stdout_fd = os.dup(1)
                with open(target, "a", encoding="utf-8") as stream:
                    os.dup2(stream.fileno(), 1)
                    with contextlib.redirect_stdout(stream):
                        yield
                    stream.flush()
            finally:
                if old_stdout_fd is not None:
                    os.dup2(old_stdout_fd, 1)
                    os.close(old_stdout_fd)
