"""Score-based feature selection transformer.

Bridges QC outputs (reproducibility ICC, batch-effect statistics) into a
scikit-learn ``Pipeline``: the per-feature scores are computed once (QC needs
multiple reader datasets / a batch label, so it cannot run inside a single-``X``
fit), passed to the selector, which then drops features that fail the threshold.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

ScoreInput: TypeAlias = pd.DataFrame | pd.Series | Mapping[str, float]


class FeatureScoreSelector(TransformerMixin, BaseEstimator):
    """Keep features whose precomputed QC score passes a threshold.

    Parameters
    ----------
    scores : DataFrame, Series, or mapping
        Per-feature scores. A Series or mapping maps feature name -> score; a
        DataFrame must carry feature names in *feature_column* and scores in
        *score_column* (e.g. ``compute_reproducibility(...)["ICC"]`` with
        ``score_column="icc_2_1"``).
    threshold : float
        Cut-off applied to the score.
    keep : {"above", "below"}
        Keep features scoring ``>= threshold`` ("above", e.g. ICC reliability) or
        ``<= threshold`` ("below", e.g. a batch-effect size).
    feature_column, score_column : str
        Columns to read when *scores* is a DataFrame (``feature_column`` defaults
        to ``"feature"``; ``score_column`` is required for a DataFrame).
    on_missing : {"keep", "drop"}
        How to treat features absent from *scores* (or with a NaN score).
    """

    def __init__(
        self,
        scores: ScoreInput,
        *,
        threshold: float,
        keep: str = "above",
        feature_column: str = "feature",
        score_column: str | None = None,
        on_missing: str = "keep",
    ) -> None:
        self.scores = scores
        self.threshold = threshold
        self.keep = keep
        self.feature_column = feature_column
        self.score_column = score_column
        self.on_missing = on_missing

    def _score_map(self) -> dict[str, float]:
        scores = self.scores
        if isinstance(scores, pd.DataFrame):
            if self.score_column is None:
                raise ValueError("score_column must be set when scores is a DataFrame.")
            keys = scores[self.feature_column].astype(str)
            values = scores[self.score_column]
            return {str(k): float(v) for k, v in zip(keys, values, strict=False)}
        items = scores.items() if isinstance(scores, pd.Series) else dict(scores).items()
        return {str(k): float(v) for k, v in items}

    def fit(self, X: pd.DataFrame, y: None = None) -> FeatureScoreSelector:
        """Resolve which feature columns to keep from the scores and threshold."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "FeatureScoreSelector requires a pandas DataFrame with feature names "
                "(it selects columns by name)."
            )
        if self.keep not in ("above", "below"):
            raise ValueError(f"keep must be 'above' or 'below', got {self.keep!r}.")
        if self.on_missing not in ("keep", "drop"):
            raise ValueError(f"on_missing must be 'keep' or 'drop', got {self.on_missing!r}.")

        score_map = self._score_map()
        self.feature_names_in_ = np.asarray([str(col) for col in X.columns], dtype=str)
        self.n_features_in_ = X.shape[1]

        kept: list[str] = []
        for col in self.feature_names_in_:
            value: float | None = score_map.get(col)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                keep_it = self.on_missing == "keep"  # unscored / NaN score
            elif self.keep == "above":
                keep_it = value >= self.threshold
            else:
                keep_it = value <= self.threshold
            if keep_it:
                kept.append(col)
        self.kept_features_ = np.asarray(kept, dtype=str)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return *X* with only the kept feature columns."""
        check_is_fitted(self, "kept_features_")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("FeatureScoreSelector.transform requires a pandas DataFrame.")
        missing = [col for col in self.kept_features_ if col not in X.columns]
        if missing:
            raise ValueError(
                f"transform input is missing {len(missing)} kept feature(s): "
                f"{', '.join(map(str, missing[:5]))}."
            )
        return X.loc[:, list(self.kept_features_)].copy()

    def get_feature_names_out(self, input_features: Any = None) -> NDArray:
        """Return the names of the kept features."""
        check_is_fitted(self, "kept_features_")
        out: NDArray = np.asarray(self.kept_features_, dtype=str).copy()
        return out

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
