"""Abstract base class for all reducers in eigenradiomics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from eigenradiomics._utils import validate_estimator_input


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
