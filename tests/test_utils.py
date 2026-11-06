"""Direct unit tests for eigenradiomics._utils public functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from sklearn.base import BaseEstimator

from eigenradiomics._utils import (
    _check_feature_names,
    _save_figure,
    check_constant_features,
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


class _FitOnly(BaseEstimator):
    """Minimal estimator: check_is_fitted needs a fit method."""

    def fit(self, X, y=None):
        return self


class TestValidateEstimatorInputNames:
    """Feature-name checks across numpy fit and DataFrame transform."""

    def test_numpy_fit_dataframe_transform_warns(self):
        est = _FitOnly()
        X = np.arange(12.0).reshape(4, 3)
        validate_estimator_input(est, X, reset=True)
        assert list(est.feature_names_in_) == ["feature_0", "feature_1", "feature_2"]
        df = pd.DataFrame(X, columns=["a", "b", "c"])
        with pytest.warns(UserWarning, match="fitted without feature names"):
            out = validate_estimator_input(est, df, reset=False)
        assert out.shape == (4, 3)
        assert list(est.feature_names_in_) == ["feature_0", "feature_1", "feature_2"]

    def test_dataframe_fit_wrong_names_still_raises(self):
        est = _FitOnly()
        df = pd.DataFrame(np.arange(12.0).reshape(4, 3), columns=["a", "b", "c"])
        validate_estimator_input(est, df, reset=True)
        validate_estimator_input(est, df.to_numpy(), reset=True)
        validate_estimator_input(est, df, reset=True)
        with pytest.raises(ValueError, match="feature names"):
            validate_estimator_input(est, df[["b", "a", "c"]], reset=False)


class TestCheckConstantFeatures:
    """Tests for check_constant_features helper."""

    def test_identifies_constant_columns(self):
        X = np.array(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.5, 3.0],
                [1.0, 2.1, 3.0],
            ]
        )
        mask = check_constant_features(X)
        np.testing.assert_array_equal(mask, [True, False, True])

    def test_handles_nan_correctly(self):
        X = np.array(
            [
                [1.0, 2.0, np.nan],
                [1.0, 2.5, np.nan],
                [np.nan, 2.1, np.nan],
            ]
        )
        mask = check_constant_features(X)
        np.testing.assert_array_equal(mask, [True, False, False])


class TestSaveFigure:
    """Tests for _save_figure environment variable branches."""

    def test_save_figure_env_vars(self, tmp_path, monkeypatch):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        out_png = tmp_path / "plot.png"
        monkeypatch.setenv("SAVE_PDF", "1")
        monkeypatch.setenv("SAVE_TIFF", "true")
        monkeypatch.setenv("TIFF_DPI", "150")

        _save_figure(fig, out_png)
        plt.close(fig)

        assert out_png.exists()
        assert (tmp_path / "plot.pdf").exists()
        assert (tmp_path / "plot.tiff").exists()

    def test_save_figure_invalid_tiff_dpi(self, tmp_path, monkeypatch):
        fig, ax = plt.subplots()
        out_png = tmp_path / "plot2.png"
        monkeypatch.setenv("SAVE_TIFF", "true")
        monkeypatch.setenv("TIFF_DPI", "invalid_number")

        _save_figure(fig, out_png)
        plt.close(fig)
        assert (tmp_path / "plot2.tiff").exists()
