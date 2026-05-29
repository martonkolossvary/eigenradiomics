"""Shared validation utilities for eigenradiomics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
import scipy.sparse
from numpy.typing import NDArray
from sklearn.utils.validation import check_is_fitted, validate_data  # noqa: F401


class _SupportsFeatureNames(Protocol):
    feature_names_in_: NDArray


def validate_feature_matrix(
    X: NDArray | pd.DataFrame,
    *,
    ensure_2d: bool = True,
    allow_nan: bool = False,
    min_samples: int = 1,
    min_features: int = 1,
) -> NDArray:
    """Validate and coerce a feature matrix to a 2-D float numpy array.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    ensure_2d : bool
        Raise if X is not two-dimensional.
    allow_nan : bool
        If False, raise on NaN values.
    min_samples, min_features : int
        Minimum required dimensions.

    Returns
    -------
    X_arr : ndarray of shape (n_samples, n_features)
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.to_numpy(dtype=float, na_value=np.nan)
    else:
        X_arr = np.asarray(X, dtype=float)

    if ensure_2d and X_arr.ndim != 2:
        raise ValueError(f"Expected 2-D input, got {X_arr.ndim}-D array.")

    n, m = X_arr.shape
    if n < min_samples:
        raise ValueError(f"Expected at least {min_samples} samples, got {n}.")
    if m < min_features:
        raise ValueError(f"Expected at least {min_features} features, got {m}.")

    if not allow_nan and np.isnan(X_arr).any():
        raise ValueError("Input contains NaN values. Set allow_nan=True to permit them.")

    return cast(NDArray, X_arr)


def extract_feature_names(
    X: NDArray | pd.DataFrame,
    n_features: int | None = None,
) -> NDArray:
    """Return feature names from a DataFrame or generate them for arrays.

    Parameters
    ----------
    X : array-like
        Input data.
    n_features : int, optional
        Override number of features (used when X is not available).

    Returns
    -------
    names : ndarray of str
    """
    if isinstance(X, pd.DataFrame):
        return np.asarray(X.columns, dtype=str)
    m = n_features if n_features is not None else np.asarray(X).shape[1]
    return np.array([f"feature_{i}" for i in range(m)], dtype=str)


def _assert_dense_matrix(X: Any) -> None:
    """Raise TypeError if X is a sparse matrix."""
    if scipy.sparse.issparse(X):
        raise TypeError(
            "Sparse matrices are not supported. "
            "Convert to a dense array or DataFrame before passing to the estimator."
        )


def validate_estimator_input(
    estimator: Any,
    X: NDArray | pd.DataFrame,
    *,
    reset: bool,
    allow_nan: bool = False,
    min_samples: int = 1,
    min_features: int = 1,
) -> NDArray:
    """Validate estimator input and enforce feature identity across fit/transform.

    This wraps scikit-learn's ``validate_data`` so estimators set
    ``n_features_in_`` correctly while still supporting generated feature names
    for numpy inputs.
    """
    _assert_dense_matrix(X)
    feature_name_estimator = cast(_SupportsFeatureNames, estimator)
    has_dataframe_names = isinstance(X, pd.DataFrame)
    had_feature_names = hasattr(estimator, "feature_names_in_")
    stored_feature_names = getattr(estimator, "feature_names_in_", None)

    if not reset and had_feature_names and not has_dataframe_names:
        delattr(estimator, "feature_names_in_")

    try:
        X_arr = validate_data(
            estimator,
            X=X,
            y="no_validation",
            reset=reset,
            dtype=float,
            ensure_2d=True,
            ensure_all_finite="allow-nan" if allow_nan else True,
            ensure_min_samples=min_samples,
            ensure_min_features=min_features,
        )
    finally:
        if not reset and had_feature_names and not has_dataframe_names:
            feature_name_estimator.feature_names_in_ = cast(NDArray, stored_feature_names)

    names = extract_feature_names(X, n_features=X_arr.shape[1])

    if reset:
        feature_name_estimator.feature_names_in_ = names
        return cast(NDArray, X_arr)

    check_is_fitted(estimator, "feature_names_in_")
    _check_feature_names(
        feature_name_estimator.feature_names_in_,
        names,
        type(estimator).__name__,
    )
    return cast(NDArray, X_arr)


def _check_feature_names(
    expected: NDArray,
    actual: NDArray,
    estimator_name: str,
) -> None:
    """Raise when transform-time feature names differ from fit-time names."""
    if np.array_equal(expected, actual):
        return

    expected_counter = Counter(expected.tolist())
    actual_counter = Counter(actual.tolist())
    missing = list((expected_counter - actual_counter).elements())
    unexpected = list((actual_counter - expected_counter).elements())

    if missing or unexpected:
        missing_str = ", ".join(map(str, missing[:5])) or "none"
        unexpected_str = ", ".join(map(str, unexpected[:5])) or "none"
        raise ValueError(
            f"{estimator_name} expected the same feature names seen during fit. "
            f"Missing: {missing_str}. Unexpected: {unexpected_str}."
        )

    raise ValueError(
        f"{estimator_name} requires input features to be in the same order as during fit."
    )
