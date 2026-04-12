"""Shared test fixtures for eigenradiomics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def rng():
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture()
def small_feature_matrix(rng) -> np.ndarray:
    """Synthetic (50, 200) matrix with 5 groups of 40 correlated features.

    Each group shares a latent factor plus noise, so features within a group
    are strongly correlated.
    """
    n_samples, n_groups, features_per_group = 50, 5, 40
    n_features = n_groups * features_per_group

    X = np.empty((n_samples, n_features))
    for g in range(n_groups):
        latent = rng.standard_normal(n_samples)
        start = g * features_per_group
        for j in range(features_per_group):
            X[:, start + j] = latent + rng.standard_normal(n_samples) * 0.3
    return X


@pytest.fixture()
def small_feature_df(small_feature_matrix) -> pd.DataFrame:
    """Same data as ``small_feature_matrix``, wrapped in a DataFrame."""
    n_features = small_feature_matrix.shape[1]
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(small_feature_matrix, columns=cols)


@pytest.fixture()
def wide_feature_matrix(rng) -> np.ndarray:
    """(30, 1000) matrix for scalability checks."""
    n_samples, n_groups, features_per_group = 30, 10, 100
    n_features = n_groups * features_per_group

    X = np.empty((n_samples, n_features))
    for g in range(n_groups):
        latent = rng.standard_normal(n_samples)
        start = g * features_per_group
        for j in range(features_per_group):
            X[:, start + j] = latent + rng.standard_normal(n_samples) * 0.25
    return X


@pytest.fixture()
def matrix_with_constant_cols(rng) -> np.ndarray:
    """(40, 50) matrix where columns 0, 10, 20 are constant."""
    X: np.ndarray = rng.standard_normal((40, 50))
    X[:, 0] = 5.0
    X[:, 10] = -2.0
    X[:, 20] = 0.0
    return X
