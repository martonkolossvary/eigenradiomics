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


class TestTransformPaired:
    """Tests for BaseReducer.transform_paired."""

    def test_transform_paired_numpy(self, small_feature_matrix):
        pca = PCAReducer(n_components=2).fit(small_feature_matrix)
        X_pre = small_feature_matrix
        X_post = small_feature_matrix + 0.5

        paired = pca.transform_paired(X_pre, X_post, relative=True, compute_distance=True)
        np.testing.assert_allclose(paired.delta, paired.followup - paired.baseline)
        assert paired.baseline.shape == (40, 2)
        assert paired.followup.shape == (40, 2)
        assert paired.delta.shape == (40, 2)
        assert paired.relative_change.shape == (40, 2)
        assert paired.trajectory_distance.shape == (40,)
        assert paired.effect_size.shape == (2,)

    def test_transform_paired_dataframe_aligned(self, small_feature_df):
        pca = PCAReducer(n_components=2).fit(small_feature_df)
        X_pre = small_feature_df
        X_post = small_feature_df + 1.0

        paired = pca.transform_paired(X_pre, X_post, relative=True, compute_distance=True)
        assert isinstance(paired.baseline, pd.DataFrame)
        assert isinstance(paired.followup, pd.DataFrame)
        assert isinstance(paired.delta, pd.DataFrame)
        assert isinstance(paired.relative_change, pd.DataFrame)
        assert isinstance(paired.trajectory_distance, pd.Series)
        assert isinstance(paired.effect_size, pd.Series)
        assert list(paired.baseline.index) == list(small_feature_df.index)
        assert list(paired.baseline.columns) == ["pca_0", "pca_1"]

    def test_transform_paired_dataframe_mismatched_index(self, small_feature_df):
        pca = PCAReducer(n_components=2).fit(small_feature_df)
        X_pre = small_feature_df.iloc[:30].copy()
        X_pre.index = [f"patient_{i}" for i in range(30)]
        X_post = small_feature_df.iloc[10:].copy()
        X_post.index = [f"patient_{i}" for i in range(10, 40)]

        with pytest.warns(UserWarning, match="Index alignment dropped"):
            paired = pca.transform_paired(X_pre, X_post, align_indices=True)

        assert len(paired.baseline) == 20
        assert paired.baseline.index[0] == "patient_10"
        assert paired.baseline.index[-1] == "patient_29"

    def test_transform_paired_to_dataframe(self, small_feature_df):
        pca = PCAReducer(n_components=2).fit(small_feature_df)
        paired = pca.transform_paired(
            small_feature_df, small_feature_df + 0.2, relative=True, compute_distance=True
        )

        df_suffix = paired.to_dataframe(prefix_mode="suffix")
        assert "pca_0_baseline" in df_suffix.columns
        assert "pca_0_followup" in df_suffix.columns
        assert "pca_0_delta" in df_suffix.columns
        assert "pca_0_relative_change" in df_suffix.columns
        assert "trajectory_distance" in df_suffix.columns

        df_prefix = paired.to_dataframe(prefix_mode="prefix")
        assert "baseline_pca_0" in df_prefix.columns
        assert "followup_pca_0" in df_prefix.columns
        assert "delta_pca_0" in df_prefix.columns

    def test_transform_paired_transductive_raises(self, small_feature_matrix):
        from eigenradiomics.reducers import TSNEReducer

        tsne = TSNEReducer(n_components=2)
        tsne.fit(small_feature_matrix)
        with pytest.raises(NotImplementedError, match="does not support transform_paired"):
            tsne.transform_paired(small_feature_matrix, small_feature_matrix)

    def test_transform_paired_mismatched_sample_counts_raises(self, small_feature_matrix):
        pca = PCAReducer(n_components=2).fit(small_feature_matrix)
        with pytest.raises(ValueError, match="Sample count mismatch"):
            pca.transform_paired(small_feature_matrix[:20], small_feature_matrix[:30])

    def test_transform_paired_disjoint_indices_raises(self, small_feature_df):
        pca = PCAReducer(n_components=2).fit(small_feature_df)
        df1 = small_feature_df.iloc[:5].copy()
        df1.index = [f"p_{i}" for i in range(5)]
        df2 = small_feature_df.iloc[:5].copy()
        df2.index = [f"other_{i}" for i in range(5)]
        with pytest.raises(ValueError, match="No common index entries"):
            pca.transform_paired(df1, df2, align_indices=True)

    def test_transform_paired_unaligned_dataframe_branches(self, small_feature_df):
        pca = PCAReducer(n_components=2).fit(small_feature_df)
        df1 = small_feature_df.iloc[:10]
        df2 = small_feature_df.iloc[:10]
        df3 = small_feature_df.iloc[:15]

        # align_indices=False with matched lengths
        paired = pca.transform_paired(df1, df2, align_indices=False)
        assert len(paired.baseline) == 10

        # align_indices=False with mismatched lengths
        with pytest.raises(ValueError, match="Sample count mismatch"):
            pca.transform_paired(df1, df3, align_indices=False)

    def test_transform_paired_mixed_types_raises(self, small_feature_df, small_feature_matrix):
        pca = PCAReducer(n_components=2).fit(small_feature_df)
        with pytest.raises(TypeError, match="must both be DataFrames or both be ndarrays"):
            pca.transform_paired(small_feature_df, small_feature_matrix)
        with pytest.raises(TypeError, match="must both be DataFrames or both be ndarrays"):
            pca.transform_paired(small_feature_matrix, small_feature_df)
