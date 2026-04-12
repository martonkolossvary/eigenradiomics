"""Direct unit tests for eigenradiomics._utils public functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from eigenradiomics._utils import (
    _check_feature_names,
    extract_feature_names,
    validate_estimator_input,
    validate_feature_matrix,
)


class TestValidateFeatureMatrix:
    """Tests for the standalone validate_feature_matrix function."""

    def test_accepts_2d_ndarray(self):
        X = np.random.default_rng(0).standard_normal((10, 5))
        result = validate_feature_matrix(X)
        assert result.dtype == float
        assert result.shape == (10, 5)

    def test_accepts_dataframe(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = validate_feature_matrix(df)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            validate_feature_matrix(np.array([1, 2, 3]))

    def test_rejects_3d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            validate_feature_matrix(np.zeros((2, 3, 4)))

    def test_nan_rejected_by_default(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.raises(ValueError, match="NaN"):
            validate_feature_matrix(X, allow_nan=False)

    def test_nan_allowed(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        result = validate_feature_matrix(X, allow_nan=True)
        assert np.isnan(result[0, 1])

    def test_min_samples_enforced(self):
        X = np.random.default_rng(0).standard_normal((2, 5))
        with pytest.raises(ValueError, match="at least 3 samples"):
            validate_feature_matrix(X, min_samples=3)

    def test_min_features_enforced(self):
        X = np.random.default_rng(0).standard_normal((5, 1))
        with pytest.raises(ValueError, match="at least 2 features"):
            validate_feature_matrix(X, min_features=2)

    def test_coerces_int_to_float(self):
        X = np.array([[1, 2], [3, 4]])
        result = validate_feature_matrix(X)
        assert result.dtype == float


class TestExtractFeatureNames:
    """Tests for extract_feature_names."""

    def test_from_dataframe(self):
        df = pd.DataFrame({"alpha": [1], "beta": [2], "gamma": [3]})
        names = extract_feature_names(df)
        assert list(names) == ["alpha", "beta", "gamma"]

    def test_from_ndarray(self):
        X = np.zeros((2, 4))
        names = extract_feature_names(X)
        assert list(names) == ["feature_0", "feature_1", "feature_2", "feature_3"]

    def test_n_features_override(self):
        X = np.zeros((2, 4))
        names = extract_feature_names(X, n_features=3)
        assert list(names) == ["feature_0", "feature_1", "feature_2"]


class TestCheckFeatureNames:
    """Tests for _check_feature_names."""

    def test_matching_names_pass(self):
        names = np.array(["a", "b", "c"])
        _check_feature_names(names, names, "TestEstimator")  # no error

    def test_missing_feature_raises(self):
        expected = np.array(["a", "b", "c"])
        actual = np.array(["a", "b", "d"])
        with pytest.raises(ValueError, match="Missing"):
            _check_feature_names(expected, actual, "TestEstimator")

    def test_reordered_features_raises(self):
        expected = np.array(["a", "b", "c"])
        actual = np.array(["c", "b", "a"])
        with pytest.raises(ValueError, match="same order"):
            _check_feature_names(expected, actual, "TestEstimator")

    def test_extra_feature_raises(self):
        expected = np.array(["a", "b"])
        actual = np.array(["a", "b", "c"])
        with pytest.raises(ValueError, match="Unexpected"):
            _check_feature_names(expected, actual, "TestEstimator")


class TestValidateEstimatorInputSparse:
    """Sparse matrix rejection in validate_estimator_input."""

    def test_sparse_csr_raises(self):
        from sklearn.base import BaseEstimator

        est = BaseEstimator()
        X_sparse = scipy.sparse.csr_matrix(np.eye(5))
        with pytest.raises(TypeError, match="Sparse matrices are not supported"):
            validate_estimator_input(est, X_sparse, reset=True)

    def test_sparse_csc_raises(self):
        from sklearn.base import BaseEstimator

        est = BaseEstimator()
        X_sparse = scipy.sparse.csc_matrix(np.eye(5))
        with pytest.raises(TypeError, match="Sparse matrices are not supported"):
            validate_estimator_input(est, X_sparse, reset=True)
