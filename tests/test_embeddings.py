"""Tests for all embedding/manifold learning reducers."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from eigenradiomics.reducers import (
    IsomapReducer,
    LLEReducer,
    MDSReducer,
    PaCMAPReducer,
    SpectralReducer,
    TriMAPReducer,
    TSNEReducer,
    UMAPReducer,
)


@pytest.fixture()
def dummy_data():
    """Create a simple 2D dataset for embedding testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.standard_normal((15, 5)), columns=[f"f_{i}" for i in range(5)])


def test_tsne_reducer(dummy_data):
    reducer = TSNEReducer(n_components=2, perplexity=5.0, random_state=42)
    assert reducer._reducer_prefix == "tsne"

    Y = reducer.fit_transform(dummy_data)
    assert Y.shape == (15, 2)
    assert reducer.n_components_ == 2

    with pytest.raises(NotImplementedError, match="transductive-only"):
        reducer.transform(dummy_data)


def test_mds_reducer(dummy_data):
    reducer = MDSReducer(n_components=2, random_state=42)
    assert reducer._reducer_prefix == "mds"

    Y = reducer.fit_transform(dummy_data)
    assert Y.shape == (15, 2)
    assert reducer.n_components_ == 2

    with pytest.raises(NotImplementedError, match="transductive-only"):
        reducer.transform(dummy_data)


def test_spectral_reducer(dummy_data):
    reducer = SpectralReducer(n_components=2, random_state=42, affinity="nearest_neighbors")
    assert reducer._reducer_prefix == "spectral"

    Y = reducer.fit_transform(dummy_data)
    assert Y.shape == (15, 2)
    assert reducer.n_components_ == 2

    with pytest.raises(NotImplementedError, match="transductive-only"):
        reducer.transform(dummy_data)


def test_isomap_reducer(dummy_data):
    reducer = IsomapReducer(n_components=2, n_neighbors=3)
    assert reducer._reducer_prefix == "isomap"

    reducer.fit(dummy_data)
    assert reducer.n_components_ == 2

    Y = reducer.transform(dummy_data)
    assert Y.shape == (15, 2)


def test_lle_reducer(dummy_data):
    reducer = LLEReducer(n_components=2, n_neighbors=3, random_state=42)
    assert reducer._reducer_prefix == "lle"

    reducer.fit(dummy_data)
    assert reducer.n_components_ == 2

    Y = reducer.transform(dummy_data)
    assert Y.shape == (15, 2)


def test_optional_reducers_missing_packages(dummy_data):
    # Test that UMAP, PaCMAP, TriMAP raise ImportError when they are not installed
    with pytest.raises(ImportError, match="umap-learn"):
        UMAPReducer().fit(dummy_data)

    with pytest.raises(ImportError, match="pacmap"):
        PaCMAPReducer().fit(dummy_data)

    with pytest.raises(ImportError, match="trimap"):
        TriMAPReducer().fit(dummy_data)


def test_umap_reducer_mocked(dummy_data):
    mock_umap = MagicMock()
    mock_estimator = MagicMock()
    mock_umap.UMAP.return_value = mock_estimator
    mock_estimator.transform.return_value = np.zeros((15, 2))

    with patch.dict(sys.modules, {"umap": mock_umap}):
        reducer = UMAPReducer(n_components=2, densmap=True, random_state=42)
        assert reducer._reducer_prefix == "umap"

        reducer.fit(dummy_data)
        mock_umap.UMAP.assert_called_once_with(
            n_components=2, densmap=True, random_state=42
        )
        mock_estimator.fit.assert_called_once()

        Y = reducer.transform(dummy_data)
        assert Y.shape == (15, 2)
        mock_estimator.transform.assert_called_once()


def test_pacmap_reducer_mocked(dummy_data):
    mock_pacmap = MagicMock()
    mock_estimator = MagicMock()
    mock_pacmap.PaCMAP.return_value = mock_estimator
    mock_estimator.transform.return_value = np.zeros((15, 2))

    with patch.dict(sys.modules, {"pacmap": mock_pacmap}):
        reducer = PaCMAPReducer(n_components=2, random_state=42)
        assert reducer._reducer_prefix == "pacmap"

        reducer.fit(dummy_data)
        mock_pacmap.PaCMAP.assert_called_once_with(
            n_dims=2, random_state=42
        )
        mock_estimator.fit.assert_called_once()

        Y = reducer.transform(dummy_data)
        assert Y.shape == (15, 2)
        mock_estimator.transform.assert_called_once()


def test_trimap_reducer_mocked(dummy_data):
    mock_trimap = MagicMock()
    mock_estimator = MagicMock()
    mock_trimap.TRIMAP.return_value = mock_estimator
    mock_estimator.transform.return_value = np.zeros((15, 2))

    # Mock presence of fit and transform
    mock_estimator.fit = MagicMock()
    mock_estimator.transform = MagicMock(return_value=np.zeros((15, 2)))

    with patch.dict(sys.modules, {"trimap": mock_trimap}):
        reducer = TriMAPReducer(n_components=2)
        assert reducer._reducer_prefix == "trimap"

        reducer.fit(dummy_data)
        mock_trimap.TRIMAP.assert_called_once_with(n_dims=2)
        mock_estimator.fit.assert_called_once()

        Y = reducer.transform(dummy_data)
        assert Y.shape == (15, 2)
        mock_estimator.transform.assert_called_once()


def test_trimap_fallback_fit_transform(dummy_data):
    mock_trimap = MagicMock()
    mock_estimator = MagicMock()
    del mock_estimator.fit  # remove fit to trigger fallback
    mock_estimator.fit_transform.return_value = np.zeros((15, 2))
    mock_trimap.TRIMAP.return_value = mock_estimator

    with patch.dict(sys.modules, {"trimap": mock_trimap}):
        reducer = TriMAPReducer(n_components=2)
        reducer.fit(dummy_data)
        mock_estimator.fit_transform.assert_called_once()
