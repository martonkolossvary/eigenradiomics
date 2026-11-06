"""Abstract base class for all reducers in eigenradiomics."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from eigenradiomics._utils import validate_estimator_input
from eigenradiomics.artifacts import PairedReductionResult, ReductionArtifacts


class BaseReducer(ABC, TransformerMixin, BaseEstimator):
    """Abstract base for dimensionality-reduction transformers.

    Subclasses must implement ``fit`` and ``transform``.  They should also set
    the class-level ``_reducer_prefix`` attribute (e.g. ``"wgcna"``, ``"pca"``)
    which controls output feature naming.

    Attributes (set after ``fit``)
    ------------------------------
    n_components_ : int
        Number of output dimensions.
    feature_names_in_ : ndarray of str
        Names of the input features seen during ``fit``.
    """

    _reducer_prefix: str = "reducer"  # override in subclass

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def _get_tags(self) -> dict[str, bool]:
        """Return scikit-learn tags for estimator checks (sklearn < 1.6)."""
        tags = super()._get_tags() if hasattr(super(), "_get_tags") else {}  # type: ignore[misc]
        tags.update(
            {
                "requires_y": False,
            }
        )
        return tags

    def __sklearn_tags__(self) -> Any:
        """Return scikit-learn tags for estimator checks (sklearn >= 1.6)."""
        if hasattr(super(), "__sklearn_tags__"):
            tags = super().__sklearn_tags__()  # type: ignore[misc]
            tags.target_tags.required = False
            return tags
        return (
            self._get_tags()
        )  # pragma: no cover  # Fallback for type checkers / older sklearn that somehow call it

    @abstractmethod
    def fit(self, X: NDArray | pd.DataFrame, y: None = None) -> BaseReducer:
        """Fit the reducer to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored
        """

    @abstractmethod
    def transform(self, X: NDArray | pd.DataFrame) -> NDArray:
        """Project new data into the reduced space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        Y : ndarray of shape (n_samples, n_components_)
        """

    def transform_paired(
        self,
        X_pre: NDArray | pd.DataFrame,
        X_post: NDArray | pd.DataFrame,
        *,
        relative: bool = False,
        compute_distance: bool = True,
        align_indices: bool = True,
        epsilon: float = 1e-8,
    ) -> PairedReductionResult:
        """Project paired longitudinal datasets into reduced space and compute trajectory metrics.

        Parameters
        ----------
        X_pre : array-like of shape (n_samples, n_features)
            Baseline feature matrix.
        X_post : array-like of shape (n_samples, n_features)
            Follow-up feature matrix.
        relative : bool, default=False
            If True, compute relative change: delta / |baseline|. The value is NaN
            where |baseline| < epsilon. Use it only for scores with a meaningful
            zero; centred scores (PCA, WGCNA) have none, so the function warns.
        compute_distance : bool, default=True
            If True, compute Euclidean trajectory distance ||ME_post - ME_pre||_2 per sample.
        align_indices : bool, default=True
            If True and inputs are DataFrames, pair rows by index label (the
            labels must be unique; unpaired rows are dropped with a warning and
            counted in the result). If False, pair rows by position; the outputs
            carry the ``X_pre`` labels.
        epsilon : float, default=1e-8
            Smallest |baseline| for a relative change. A smaller |baseline| gives NaN.

        Returns
        -------
        result : PairedReductionResult
            Structured container with baseline, followup, delta, relative_change,
            trajectory_distance, and effect_size.
        """
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "n_components_")

        common_idx: Any = None
        is_df = isinstance(X_pre, pd.DataFrame) and isinstance(X_post, pd.DataFrame)

        if isinstance(X_pre, pd.DataFrame) and isinstance(X_post, pd.DataFrame):
            if align_indices:
                for name, frame in (("X_pre", X_pre), ("X_post", X_post)):
                    if not frame.index.is_unique:
                        dup = frame.index[frame.index.duplicated()].unique()[:5].tolist()
                        raise ValueError(
                            f"{name} has duplicate index labels {dup}; keyed pairing needs "
                            "unique labels (or pass align_indices=False for positional pairing)."
                        )
                common_idx = X_pre.index.intersection(X_post.index)
                if len(common_idx) == 0:
                    raise ValueError("No common index entries between X_pre and X_post.")
                n_dropped_pre = len(X_pre) - len(common_idx)
                n_dropped_post = len(X_post) - len(common_idx)
                if n_dropped_pre > 0 or n_dropped_post > 0:
                    warnings.warn(
                        f"Index alignment dropped {n_dropped_pre} sample(s) from X_pre and "
                        f"{n_dropped_post} sample(s) from X_post that were not paired.",
                        UserWarning,
                        stacklevel=2,
                    )
                X_pre_aligned: Any = X_pre.loc[common_idx]
                X_post_aligned: Any = X_post.loc[common_idx]
            else:
                if len(X_pre) != len(X_post):
                    raise ValueError(
                        f"Sample count mismatch: X_pre has {len(X_pre)} samples, "
                        f"but X_post has {len(X_post)} samples."
                    )
                common_idx = X_pre.index
                X_pre_aligned = X_pre
                X_post_aligned = X_post
        else:
            if isinstance(X_pre, pd.DataFrame) or isinstance(X_post, pd.DataFrame):
                raise TypeError("X_pre and X_post must both be DataFrames or both be ndarrays.")
            X_pre_arr_in = np.asarray(X_pre)
            X_post_arr_in = np.asarray(X_post)
            if X_pre_arr_in.shape[0] != X_post_arr_in.shape[0]:
                raise ValueError(
                    f"Sample count mismatch: X_pre has {X_pre_arr_in.shape[0]} samples, "
                    f"but X_post has {X_post_arr_in.shape[0]} samples."
                )
            X_pre_aligned = X_pre_arr_in
            X_post_aligned = X_post_arr_in

        try:
            ME_pre_arr = np.asarray(self.transform(X_pre_aligned))
            ME_post_arr = np.asarray(self.transform(X_post_aligned))
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"{type(self).__name__} does not support transform_paired because it cannot "
                "project out-of-sample data. For paired analysis with transductive methods, "
                "use fit_transform on concatenated pre/post matrices or use an inductive "
                "reducer like PCAReducer, WGCNAReducer, or UMAPReducer."
            ) from exc

        delta_arr = ME_post_arr - ME_pre_arr

        relative_arr = None
        if relative:
            warnings.warn(
                "relative change divides by the baseline score. Centred scores (PCA, "
                "WGCNA eigengenes) have no meaningful zero, so the ratio is not "
                "interpretable for them.",
                UserWarning,
                stacklevel=2,
            )
            denom = np.abs(ME_pre_arr)
            relative_arr = np.where(denom < epsilon, np.nan, delta_arr / denom)

        dist_arr = None
        if compute_distance:
            dist_arr = np.linalg.norm(delta_arr, axis=1)

        std_delta = (
            np.std(delta_arr, axis=0, ddof=1)
            if delta_arr.shape[0] > 1
            else np.zeros(delta_arr.shape[1])
        )
        mean_delta = np.mean(delta_arr, axis=0)
        effect_size_arr = np.where(std_delta < 1e-12, np.nan, mean_delta / std_delta)

        feature_names_out = self.get_feature_names_out()

        if is_df:
            baseline_out: pd.DataFrame | NDArray = pd.DataFrame(
                ME_pre_arr, index=common_idx, columns=feature_names_out
            )
            followup_out: pd.DataFrame | NDArray = pd.DataFrame(
                ME_post_arr, index=common_idx, columns=feature_names_out
            )
            delta_out: pd.DataFrame | NDArray = pd.DataFrame(
                delta_arr, index=common_idx, columns=feature_names_out
            )
            relative_out: pd.DataFrame | NDArray | None = (
                pd.DataFrame(relative_arr, index=common_idx, columns=feature_names_out)
                if relative_arr is not None
                else None
            )
            dist_out: pd.Series | NDArray | None = (
                pd.Series(dist_arr, index=common_idx, name="trajectory_distance")
                if dist_arr is not None
                else None
            )
            effect_size_out: pd.Series | NDArray | None = pd.Series(
                effect_size_arr, index=feature_names_out, name="effect_size"
            )
        else:
            baseline_out = ME_pre_arr
            followup_out = ME_post_arr
            delta_out = delta_arr
            relative_out = relative_arr
            dist_out = dist_arr
            effect_size_out = effect_size_arr

        return PairedReductionResult(
            baseline=baseline_out,
            followup=followup_out,
            delta=delta_out,
            relative_change=relative_out,
            trajectory_distance=dist_out,
            effect_size=effect_size_out,
            feature_names_out=feature_names_out,
            n_paired=int(delta_arr.shape[0]),
            n_unpaired_baseline=len(X_pre) - int(delta_arr.shape[0]),
            n_unpaired_followup=len(X_post) - int(delta_arr.shape[0]),
        )

    def inverse_transform(self, Y: NDArray) -> NDArray:
        """Map reduced data back to the original space (if supported).

        Default raises ``NotImplementedError`` because most dimensionality
        reduction methods are lossy and cannot exactly reconstruct the
        original features.  Subclasses may override this when an approximate
        reconstruction is meaningful (e.g. PCA).
        """
        raise NotImplementedError(f"{type(self).__name__} does not support inverse_transform.")

    def get_feature_names_out(self, input_features: NDArray | None = None) -> NDArray:
        """Return output feature names: ``{prefix}_{i}``."""
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "n_components_")
        return np.array(
            [f"{self._reducer_prefix}_{i}" for i in range(self.n_components_)],
            dtype=str,
        )

    # ------------------------------------------------------------------
    # structured reduction artifacts (for downstream plotting / analysis)
    # ------------------------------------------------------------------

    def get_reduction_artifacts(self) -> ReductionArtifacts:
        """Return the reducer's standardized outputs as a :class:`ReductionArtifacts`.

        Subclasses populate whichever elements they can produce by overriding the
        ``_artifact_*`` hooks below; unsupported elements remain ``None``. This
        gives downstream plotting and analysis a single, uniform interface
        regardless of which reducer (or which features) produced the result.
        """
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "feature_names_in_")
        return ReductionArtifacts(
            feature_names=np.asarray(self.feature_names_in_),
            similarity=self._artifact_similarity(),
            linkage=self._artifact_linkage(),
            cluster_labels=self._artifact_cluster_labels(),
            feature_order=self._artifact_feature_order(),
            feature_importances=self._artifact_feature_importances(),
        )

    def _artifact_similarity(self) -> pd.DataFrame | None:
        """Override to expose a symmetric feature-by-feature similarity matrix."""
        return None

    def _artifact_linkage(self) -> NDArray | None:
        """Override to expose a SciPy hierarchical-clustering linkage matrix."""
        return None

    def _artifact_cluster_labels(self) -> pd.Series | None:
        """Override to expose per-feature cluster / module labels."""
        return None

    def _artifact_feature_order(self) -> NDArray | None:
        """Override to expose a display ordering of features."""
        return None

    def _artifact_feature_importances(self) -> pd.DataFrame | None:
        """Override to expose per-feature importances."""
        return None

    # ------------------------------------------------------------------
    # shared helpers
    # ------------------------------------------------------------------

    def _validate_input(
        self,
        X: NDArray | pd.DataFrame,
        *,
        reset: bool = False,
        allow_nan: bool = False,
    ) -> NDArray:
        """Validate *X* and optionally store ``feature_names_in_``.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        reset : bool
            If True, store feature names (call during ``fit``).
            If False, check consistency with stored names (call during
            ``transform``).
        allow_nan : bool
            Allow NaN in input.

        Returns
        -------
        X_arr : ndarray of float64, shape (n, m)
        """
        return validate_estimator_input(self, X, reset=reset, allow_nan=allow_nan)
