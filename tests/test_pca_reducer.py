"""Tests for eigenradiomics.reducers.PCAReducer and SparsePCAReducer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eigenradiomics.reducers import PCAReducer, SparsePCAReducer


@pytest.fixture()
def small_feature_matrix():
    """Simple 2D matrix for testing."""
    rng = np.random.default_rng(42)
    latent1 = rng.standard_normal(40)
    latent2 = rng.standard_normal(40)
    X = np.zeros((40, 10))
    for i in range(5):
        X[:, i] = latent1 + rng.standard_normal(40) * 0.1
    for i in range(5, 10):
        X[:, i] = latent2 + rng.standard_normal(40) * 0.1
    return X


@pytest.fixture()
def small_feature_df(small_feature_matrix):
    """DataFrame wrapper around small_feature_matrix."""
    columns = [f"feat_{i}" for i in range(small_feature_matrix.shape[1])]
    return pd.DataFrame(small_feature_matrix, columns=columns)


class TestPCAReducer:
    """Core PCAReducer behavior."""

    def test_fit_returns_self(self, small_feature_matrix):
        pca = PCAReducer(n_components=2)
        assert pca.fit(small_feature_matrix) is pca

    def test_output_shape(self, small_feature_df):
        pca = PCAReducer(n_components=3)
        Y = pca.fit_transform(small_feature_df)
        assert Y.shape == (small_feature_df.shape[0], 3)
        assert pca.n_components_ == 3

    def test_transform_new_data(self, small_feature_matrix):
        pca = PCAReducer(n_components=2)
        pca.fit(small_feature_matrix)
        rng = np.random.default_rng(100)
        X_new = small_feature_matrix + rng.standard_normal(small_feature_matrix.shape) * 0.1
        Y_new = pca.transform(X_new)
        assert Y_new.shape == (small_feature_matrix.shape[0], 2)

    def test_inverse_transform(self, small_feature_matrix):
        pca = PCAReducer(n_components=5)
        Y = pca.fit_transform(small_feature_matrix)
        X_reconstructed = pca.inverse_transform(Y)
        assert X_reconstructed.shape == small_feature_matrix.shape

    def test_dataframe_feature_names(self, small_feature_df):
        pca = PCAReducer(n_components=2)
        pca.fit(small_feature_df)
        assert hasattr(pca, "feature_names_in_")
        assert list(pca.feature_names_in_[:3]) == ["feat_0", "feat_1", "feat_2"]
        names_out = pca.get_feature_names_out()
        assert list(names_out) == ["pca_0", "pca_1"]

    def test_dataframe_column_order_mismatch(self, small_feature_df):
        pca = PCAReducer(n_components=2)
        pca.fit(small_feature_df)
        # Reordering column names must trigger ValueError (contract validation)
        with pytest.raises(ValueError, match="same order"):
            pca.transform(small_feature_df.iloc[:, ::-1])

    def test_reduction_artifacts(self, small_feature_df):
        pca = PCAReducer(n_components=2)
        pca.fit(small_feature_df)
        artifacts = pca.get_reduction_artifacts()
        assert artifacts is not None
        assert list(artifacts.feature_names) == list(small_feature_df.columns)
        importances = artifacts.feature_importances
        assert importances is not None
        assert "importance" in importances.columns
        # components_ (2) + importance (1)
        assert importances.shape == (small_feature_df.shape[1], 3)


class TestSparsePCAReducer:
    """Core SparsePCAReducer behavior."""

    def test_fit_returns_self(self, small_feature_matrix):
        spca = SparsePCAReducer(n_components=2, random_state=42)
        assert spca.fit(small_feature_matrix) is spca

    def test_output_shape(self, small_feature_df):
        spca = SparsePCAReducer(n_components=3, random_state=42)
        Y = spca.fit_transform(small_feature_df)
        assert Y.shape == (small_feature_df.shape[0], 3)
        assert spca.n_components_ == 3

    def test_transform_new_data(self, small_feature_matrix):
        spca = SparsePCAReducer(n_components=2, random_state=42)
        spca.fit(small_feature_matrix)
        rng = np.random.default_rng(100)
        X_new = small_feature_matrix + rng.standard_normal(small_feature_matrix.shape) * 0.1
        Y_new = spca.transform(X_new)
        assert Y_new.shape == (small_feature_matrix.shape[0], 2)

    def test_inverse_transform(self, small_feature_matrix):
        spca = SparsePCAReducer(n_components=4, random_state=42)
        Y = spca.fit_transform(small_feature_matrix)
        X_reconstructed = spca.inverse_transform(Y)
        assert X_reconstructed.shape == small_feature_matrix.shape

    def test_dataframe_feature_names(self, small_feature_df):
        spca = SparsePCAReducer(n_components=2, random_state=42)
        spca.fit(small_feature_df)
        assert hasattr(spca, "feature_names_in_")
        assert list(spca.feature_names_in_) == list(small_feature_df.columns)
        names_out = spca.get_feature_names_out()
        assert list(names_out) == ["sparse_pca_0", "sparse_pca_1"]

    def test_dataframe_column_order_mismatch(self, small_feature_df):
        spca = SparsePCAReducer(n_components=2, random_state=42)
        spca.fit(small_feature_df)
        with pytest.raises(ValueError, match="same order"):
            spca.transform(small_feature_df.iloc[:, ::-1])

    def test_reduction_artifacts(self, small_feature_df):
        spca = SparsePCAReducer(n_components=2, random_state=42)
        spca.fit(small_feature_df)
        artifacts = spca.get_reduction_artifacts()
        assert artifacts is not None
        assert list(artifacts.feature_names) == list(small_feature_df.columns)
        importances = artifacts.feature_importances
        assert importances is not None
        assert "importance" in importances.columns
        assert importances.shape == (small_feature_df.shape[1], 3)
