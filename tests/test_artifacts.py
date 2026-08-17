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


def test_artifacts_get_catalog_enrichment():
    art_no_labels = ReductionArtifacts(feature_names=np.array(["a", "b"]))
    with pytest.raises(ValueError, match="no cluster_labels"):
        art_no_labels.get_catalog_enrichment(
            pd.DataFrame({"feature": ["a", "b"], "group": ["g1", "g2"]})
        )

    art_with_labels = ReductionArtifacts(
        feature_names=np.array(["a", "b", "c", "d"]),
        cluster_labels=pd.Series(["m1", "m1", "m2", "m2"], index=["a", "b", "c", "d"]),
    )
    catalog = pd.DataFrame(
        {
            "feature": ["a", "b", "c", "d"],
            "family": ["f1", "f1", "f2", "f2"],
        }
    )
    res = art_with_labels.get_catalog_enrichment(catalog)
    assert isinstance(res, pd.DataFrame)
    assert len(res) > 0


def test_paired_reduction_result():
    from eigenradiomics.artifacts import PairedReductionResult

    base = np.array([[1.0, 2.0], [3.0, 4.0]])
    post = np.array([[1.5, 2.5], [3.5, 4.5]])
    delta = post - base
    rel = delta / np.abs(base)
    dist = np.linalg.norm(delta, axis=1)

    result = PairedReductionResult(
        baseline=base,
        followup=post,
        delta=delta,
        relative_change=rel,
        trajectory_distance=dist,
        feature_names_out=np.array(["c0", "c1"]),
    )
    df = result.to_dataframe(prefix_mode="suffix")
    assert "c0_baseline" in df.columns
    assert "c1_followup" in df.columns
    assert "c0_delta" in df.columns
    assert "c0_relative_change" in df.columns
    assert "trajectory_distance" in df.columns

    with pytest.raises(ValueError, match="prefix_mode"):
        result.to_dataframe(prefix_mode="invalid")
