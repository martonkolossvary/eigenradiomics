"""Custom scikit-learn compatible preprocessing transformer for radiomics."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.utils.validation import check_is_fitted

from eigenradiomics._utils import validate_estimator_input


class RadiomicsPrepTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """scikit-learn compatible preprocessor for radiomics features.

    Applies outlier winsorization, Yeo-Johnson power transformation, and z-score
    standardization column-by-column while natively preserving and carrying NaN
    values (avoiding failures caused by standard sklearn transformers on incomplete matrices).

    Like the reducers, the transformer stores the feature names seen during
    ``fit`` and rejects inputs at ``transform`` time whose feature names or
    order differ, preventing silently misapplied per-column parameters.

    Parameters
    ----------
    winsor_lower : float, default=0.01
        Quantile used as the lower clipping boundary for winsorization.
    winsor_upper : float, default=0.99
        Quantile used as the upper clipping boundary for winsorization.
    skip_yeo_johnson : bool, default=False
        If True, skips the Yeo-Johnson power transformation step.
    standardize : bool, default=True
        If True, standardizes features to zero mean and unit variance.
    """

    def __init__(
        self,
        *,
        winsor_lower: float = 0.01,
        winsor_upper: float = 0.99,
        skip_yeo_johnson: bool = False,
        standardize: bool = True,
    ):
        self.winsor_lower = winsor_lower
        self.winsor_upper = winsor_upper
        self.skip_yeo_johnson = skip_yeo_johnson
        self.standardize = standardize

    def _get_tags(self) -> dict[str, Any]:
        """Return scikit-learn tags for estimator checks (sklearn < 1.6)."""
        tags = super()._get_tags() if hasattr(super(), "_get_tags") else {}  # type: ignore[misc]
        tags.update({"allow_nan": True})
        return tags

    def __sklearn_tags__(self) -> Any:
        """Return scikit-learn tags for estimator checks (sklearn >= 1.6)."""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def fit(self, X: Any, y: Any = None) -> RadiomicsPrepTransformer:
        """Fit preprocessing parameters for each feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix containing raw radiomics values, potentially with NaNs.
        y : None
            Ignored.

        Returns
        -------
        self : RadiomicsPrepTransformer
            Fitted transformer.
        """
        # Validate inputs and store feature_names_in_ / n_features_in_.
        X_arr = validate_estimator_input(self, X, reset=True, allow_nan=True)
        n_samples, n_features = X_arr.shape

        self.winsor_bounds_: list[tuple[float, float]] = []
        self.power_transformers_: list[PowerTransformer | None] = []
        self.scales_: list[tuple[float, float]] = []

        for col_idx in range(n_features):
            col_vals = X_arr[:, col_idx]
            valid_mask = ~np.isnan(col_vals)
            valid_vals = col_vals[valid_mask]

            if len(valid_vals) == 0:
                # Handle completely missing columns
                self.winsor_bounds_.append((np.nan, np.nan))
                self.power_transformers_.append(None)
                self.scales_.append((0.0, 1.0))
                continue

            # 1. Winsorization bounds
            lower = float(np.percentile(valid_vals, self.winsor_lower * 100))
            upper = float(np.percentile(valid_vals, self.winsor_upper * 100))
            self.winsor_bounds_.append((lower, upper))

            # Apply winsorization
            clipped = np.clip(valid_vals, lower, upper)

            # 2. Yeo-Johnson
            pt = None
            if not self.skip_yeo_johnson:
                # Skip transformation if column is constant to avoid PowerTransformer error
                if np.nanvar(clipped) > 1e-12:
                    pt = PowerTransformer(method="yeo-johnson", standardize=False)
                    pt.fit(clipped.reshape(-1, 1))
                    transformed = pt.transform(clipped.reshape(-1, 1)).ravel()
                else:
                    transformed = clipped
            else:
                transformed = clipped

            self.power_transformers_.append(pt)

            # 3. Standardization scale
            if self.standardize:
                mean = float(np.nanmean(transformed))
                std = float(np.nanstd(transformed, ddof=0))
                self.scales_.append((mean, std))
            else:
                self.scales_.append((0.0, 1.0))

        return self

    def transform(self, X: Any) -> pd.DataFrame | NDArray:
        """Apply fitted winsorization, Yeo-Johnson, and scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix containing raw radiomics values.

        Returns
        -------
        X_trans : pd.DataFrame or ndarray
            Preprocessed feature matrix. Returns a pandas DataFrame if *X* was a DataFrame,
            otherwise returns a numpy array.
        """
        check_is_fitted(self, "n_features_in_")
        X_arr = validate_estimator_input(self, X, reset=False, allow_nan=True)

        n_samples, n_features = X_arr.shape
        X_trans = np.empty_like(X_arr, dtype=float)

        for col_idx in range(n_features):
            col_vals = X_arr[:, col_idx]
            valid_mask = ~np.isnan(col_vals)
            valid_vals = col_vals[valid_mask]

            # Initialize column to NaN
            X_trans[:, col_idx] = np.nan

            if len(valid_vals) == 0:
                continue

            # Fetch fitted parameters
            lower, upper = self.winsor_bounds_[col_idx]
            pt = self.power_transformers_[col_idx]
            mean, std = self.scales_[col_idx]

            # 1. Apply winsorization
            clipped = np.clip(valid_vals, lower, upper)

            # 2. Apply Yeo-Johnson
            if pt is not None:
                transformed = pt.transform(clipped.reshape(-1, 1)).ravel()
            else:
                transformed = clipped

            # 3. Apply standard scaling
            if self.standardize and std > 1e-12:
                scaled = (transformed - mean) / std
            else:
                scaled = transformed

            X_trans[valid_mask, col_idx] = scaled

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_trans, index=X.index, columns=X.columns)

        return cast(NDArray, X_trans)
