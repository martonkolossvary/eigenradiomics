"""Tests for ReductionArtifacts and the BaseReducer artifact protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from eigenradiomics import ReductionArtifacts
from eigenradiomics.reducers import BaseReducer


class _MinimalReducer(BaseReducer):
    """A reducer that overrides no artifact hooks (base defaults -> all None)."""

    _reducer_prefix = "min"

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        self.feature_names_in_ = np.asarray(["a", "b", "c"])
        self.n_components_ = 3
        return self

    def transform(self, X):  # noqa: ANN001, ANN201
        return np.asarray(X, dtype=float)


def test_available_lists_only_populated():
    art = ReductionArtifacts(
        feature_names=np.array(["a", "b"]),
        similarity=pd.DataFrame([[1.0, 0.5], [0.5, 1.0]]),
        cluster_labels=pd.Series(["m1", "m1"]),
    )
    assert set(art.available()) == {"similarity", "cluster_labels"}
    assert "feature_names" not in art.available()
    assert "n_features=2" in repr(art)


def test_base_reducer_defaults_are_none():
    art = _MinimalReducer().fit(None).get_reduction_artifacts()
    assert list(art.feature_names) == ["a", "b", "c"]
    assert art.available() == []
    assert art.similarity is None
    assert art.linkage is None
    assert art.cluster_labels is None
    assert art.feature_order is None
    assert art.feature_importances is None


def test_get_reduction_artifacts_requires_fit():
    with pytest.raises(NotFittedError):
        _MinimalReducer().get_reduction_artifacts()
