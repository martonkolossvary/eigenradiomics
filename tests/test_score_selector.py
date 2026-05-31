"""Tests for FeatureScoreSelector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from eigenradiomics import FeatureScoreSelector, RadiomicsPrepTransformer


def _X() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.standard_normal((12, 4)), columns=["f0", "f1", "f2", "f3"]
    )


def test_keep_above_from_dataframe_scores():
    icc = pd.DataFrame({"feature": ["f0", "f1", "f2", "f3"], "icc_2_1": [0.95, 0.5, 0.85, 0.2]})
    sel = FeatureScoreSelector(icc, threshold=0.80, score_column="icc_2_1").fit(_X())
    assert list(sel.get_feature_names_out()) == ["f0", "f2"]
    assert list(sel.transform(_X()).columns) == ["f0", "f2"]


def test_keep_below_for_effect_size():
    eta = pd.Series({"f0": 0.02, "f1": 0.4, "f2": 0.01, "f3": 0.5})  # keep low batch effect
    sel = FeatureScoreSelector(eta, threshold=0.10, keep="below").fit(_X())
    assert list(sel.get_feature_names_out()) == ["f0", "f2"]


def test_scores_as_mapping():
    sel = FeatureScoreSelector({"f0": 0.9, "f1": 0.1}, threshold=0.5, on_missing="drop").fit(_X())
    assert list(sel.get_feature_names_out()) == ["f0"]  # f2/f3 unscored -> dropped


def test_on_missing_keep_and_nan_score():
    scores = pd.Series({"f0": 0.9, "f1": np.nan})  # f1 NaN, f2/f3 absent
    sel = FeatureScoreSelector(scores, threshold=0.5, on_missing="keep").fit(_X())
    assert list(sel.get_feature_names_out()) == ["f0", "f1", "f2", "f3"]


def test_pipeline_composes_with_prep():
    icc = pd.Series({"f0": 0.95, "f1": 0.3, "f2": 0.9, "f3": 0.1})
    pipe = Pipeline(
        [
            ("prep", RadiomicsPrepTransformer().set_output(transform="pandas")),
            ("select", FeatureScoreSelector(icc, threshold=0.8)),
        ]
    )
    out = pipe.fit_transform(_X())
    assert list(out.columns) == ["f0", "f2"]


def test_fit_requires_dataframe():
    sel = FeatureScoreSelector({"f0": 1.0}, threshold=0.5)
    with pytest.raises(ValueError, match="requires a pandas DataFrame"):
        sel.fit(np.zeros((4, 2)))


def test_transform_requires_dataframe():
    sel = FeatureScoreSelector({"f0": 1.0}, threshold=0.5).fit(_X())
    with pytest.raises(ValueError, match="requires a pandas DataFrame"):
        sel.transform(np.zeros((4, 4)))


def test_transform_missing_kept_feature_raises():
    sel = FeatureScoreSelector({"f0": 0.9}, threshold=0.5).fit(_X())
    with pytest.raises(ValueError, match="missing .* kept feature"):
        sel.transform(_X().drop(columns=["f0"]))


def test_dataframe_scores_without_score_column_raises():
    icc = pd.DataFrame({"feature": ["f0"], "icc_2_1": [0.9]})
    with pytest.raises(ValueError, match="score_column must be set"):
        FeatureScoreSelector(icc, threshold=0.8).fit(_X())


def test_invalid_keep_and_on_missing_raise():
    with pytest.raises(ValueError, match="keep must be"):
        FeatureScoreSelector({"f0": 1.0}, threshold=0.5, keep="sideways").fit(_X())
    with pytest.raises(ValueError, match="on_missing must be"):
        FeatureScoreSelector({"f0": 1.0}, threshold=0.5, on_missing="maybe").fit(_X())
