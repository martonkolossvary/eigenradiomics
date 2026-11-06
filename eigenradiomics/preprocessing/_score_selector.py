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
        ``score_column="icc_2_1"``). Feature keys are compared as text. Two keys
        that give the same text (for example ``1`` and ``"1"``) raise an error.
    threshold : float
        Cut-off applied to the score (inclusive).
    keep : {"above", "below"}
        Keep features scoring ``>= threshold`` ("above", e.g. ICC reliability) or
        ``<= threshold`` ("below", e.g. a batch-effect size).
    feature_column, score_column : str
        Columns to read when *scores* is a DataFrame (``feature_column`` defaults
        to ``"feature"``; ``score_column`` is required for a DataFrame).
    on_missing : {"keep", "drop", "raise"}
        How to treat features that are absent from *scores* or have a NaN or
        infinite score: keep them (the legacy default), drop them, or raise.

    Attributes
    ----------
    decisions_ : DataFrame
        One row per input feature, in input order, with ``feature``, ``score``,
        ``kept`` and ``reason`` (``passed``, ``below_threshold``,
        ``above_threshold``, ``unscored`` or ``nonfinite_score``).
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
            keys = scores[self.feature_column].tolist()
            values = scores[self.score_column].tolist()
        elif isinstance(scores, pd.Series):
            keys, values = scores.index.tolist(), scores.tolist()
        else:
            keys, values = list(scores.keys()), list(scores.values())
        score_map: dict[str, float] = {}
        for key, value in zip(keys, values, strict=True):
            name = str(key)
            if name in score_map:
                raise ValueError(f"scores has more than one entry for feature {name!r}.")
            score_map[name] = float(value)
        return score_map

    @staticmethod
    def _column_names(X: pd.DataFrame) -> list[str]:
        names = [str(col) for col in X.columns]
        if len(set(names)) != len(names):
            raise ValueError("X has duplicate column names (compared as text).")
        return names

    def fit(self, X: pd.DataFrame, y: None = None) -> FeatureScoreSelector:
        """Resolve which feature columns to keep from the scores and threshold."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "FeatureScoreSelector requires a pandas DataFrame with feature names "
                "(it selects columns by name)."
            )
        if self.keep not in ("above", "below"):
            raise ValueError(f"keep must be 'above' or 'below', got {self.keep!r}.")
        if self.on_missing not in ("keep", "drop", "raise"):
            raise ValueError(
                f"on_missing must be 'keep', 'drop' or 'raise', got {self.on_missing!r}."
            )
        if not np.isfinite(self.threshold):
            raise ValueError(f"threshold must be finite, got {self.threshold!r}.")

        score_map = self._score_map()
        names = self._column_names(X)
        self.feature_names_in_ = np.asarray(names, dtype=str)
        self.n_features_in_ = X.shape[1]

        records = []
        for col in names:
            value = score_map.get(col, np.nan)
            if not np.isfinite(value):
                reason = "unscored" if col not in score_map else "nonfinite_score"
                keep_it = self.on_missing == "keep"
            elif self.keep == "above":
                keep_it = value >= self.threshold
                reason = "passed" if keep_it else "below_threshold"
            else:
                keep_it = value <= self.threshold
                reason = "passed" if keep_it else "above_threshold"
            records.append({"feature": col, "score": value, "kept": keep_it, "reason": reason})
        decisions = pd.DataFrame(records, columns=["feature", "score", "kept", "reason"])
        if self.on_missing == "raise":
            bad = decisions.loc[decisions["reason"].isin(["unscored", "nonfinite_score"])]
            if not bad.empty:
                raise ValueError(
                    f"{len(bad)} feature(s) have no finite score: "
                    f"{', '.join(bad['feature'].head(5))}."
                )
        self.decisions_ = decisions
        self.kept_features_ = np.asarray(decisions.loc[decisions["kept"], "feature"], dtype=str)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return *X* with only the kept feature columns."""
        check_is_fitted(self, "kept_features_")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("FeatureScoreSelector.transform requires a pandas DataFrame.")
        by_name = dict(zip(self._column_names(X), X.columns, strict=True))
        missing = [col for col in self.kept_features_ if col not in by_name]
        if missing:
            raise ValueError(
                f"transform input is missing {len(missing)} kept feature(s): "
                f"{', '.join(map(str, missing[:5]))}."
            )
        return X.loc[:, [by_name[col] for col in self.kept_features_]].copy()

    def get_feature_names_out(self, input_features: Any = None) -> NDArray:
        """Return the names of the kept features."""
        check_is_fitted(self, "kept_features_")
        out: NDArray = np.asarray(self.kept_features_, dtype=str).copy()
        return out

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
