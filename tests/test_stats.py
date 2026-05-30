"""Edge-case unit tests for the shared statistics in eigenradiomics._stats."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eigenradiomics._stats import (
    _bootstrap_icc_ci,
    _fdr_correct,
    _fisher_ci,
    _icc_2_1_estimate,
    anova_effect,
    kruskal_effect,
    levene_test,
    permanova_euclidean,
)


class TestFdrCorrect:
    def test_empty(self) -> None:
        out = _fdr_correct(np.array([]))
        assert len(out) == 0

    def test_all_nan(self) -> None:
        out = _fdr_correct(np.array([np.nan, np.nan]))
        assert np.isnan(out).all()

    def test_monotone(self) -> None:
        out = _fdr_correct(np.array([0.01, 0.02, np.nan, 0.5]))
        assert np.isnan(out[2])
        assert (out[[0, 1, 3]] >= np.array([0.01, 0.02, 0.5])).all()


class TestIcc:
    def test_too_small_returns_nan(self) -> None:
        assert np.isnan(_icc_2_1_estimate(np.array([[1.0]]))["icc"])  # n=1, k=1

    def test_degenerate_denominator_nan(self) -> None:
        # n=2, k=2 constant -> denominator == 0 -> ICC is NaN.
        assert np.isnan(_icc_2_1_estimate(np.ones((2, 2)))["icc"])

    def test_valid_icc(self) -> None:
        Y = np.array([[9.0, 8.0, 9.0], [7.0, 7.0, 6.0], [5.0, 4.0, 5.0], [3.0, 3.0, 2.0]])
        assert 0.5 < _icc_2_1_estimate(Y)["icc"] < 1.0

    def test_bootstrap_iterations_zero(self) -> None:
        Y = np.array([[1.0, 1.1], [2.0, 2.1], [3.0, 3.2], [4.0, 3.9]])
        assert np.isnan(_bootstrap_icc_ci(Y, "f", iterations=0)[0])

    def test_bootstrap_all_nan_estimates(self) -> None:
        # A fully-missing observer column makes every bootstrap ICC NaN.
        Y = np.column_stack([np.arange(5.0), np.full(5, np.nan)])
        lo, hi = _bootstrap_icc_ci(Y, "f", iterations=50)
        assert np.isnan(lo) and np.isnan(hi)


class TestFisherCi:
    def test_small_n(self) -> None:
        assert np.isnan(_fisher_ci(0.5, 3)[0])

    def test_nan_r(self) -> None:
        assert np.isnan(_fisher_ci(np.nan, 50)[0])

    def test_valid(self) -> None:
        lo, hi = _fisher_ci(0.8, 50)
        assert lo < 0.8 < hi


class TestGroupTests:
    def test_anova_single_group(self) -> None:
        assert np.isnan(anova_effect([np.array([1.0, 2.0, 3.0])])[0])

    def test_anova_constant(self) -> None:
        f, p, eta2 = anova_effect([np.array([1.0, 1.0]), np.array([1.0, 1.0])])
        assert np.isnan(f)

    def test_kruskal_single_group(self) -> None:
        assert np.isnan(kruskal_effect([np.array([1.0, 2.0, 3.0])])[0])

    def test_kruskal_insufficient(self) -> None:
        # len(values) <= len(groups)
        assert np.isnan(kruskal_effect([np.array([1.0]), np.array([2.0])])[0])

    def test_levene_single_group(self) -> None:
        assert np.isnan(levene_test([np.array([1.0, 2.0, 3.0])])[0])

    def test_levene_valid(self) -> None:
        stat, p = levene_test([np.array([1.0, 2.0, 3.0]), np.array([4.0, 9.0, 1.0])])
        assert not np.isnan(stat)


class TestPermanova:
    def _scores(self, vals: list[list[float]]) -> pd.DataFrame:
        return pd.DataFrame(vals, columns=["PC1", "PC2"])

    def test_single_batch(self) -> None:
        scores = self._scores([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]])
        batch = pd.Series(["A", "A", "A"])
        assert np.isnan(permanova_euclidean(scores, batch)[0])

    def test_zero_total_variance(self) -> None:
        scores = self._scores([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        batch = pd.Series(["A", "A", "B", "B"])
        assert np.isnan(permanova_euclidean(scores, batch)[0])

    def test_zero_permutations(self) -> None:
        scores = self._scores([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]])
        batch = pd.Series(["A", "A", "B", "B"])
        f, r2, p = permanova_euclidean(scores, batch, permutations=0)
        assert np.isfinite(f) and np.isnan(p)

    def test_infinite_pseudo_f(self) -> None:
        # Perfect within-group agreement -> ss_within == 0 -> f = inf, p = nan.
        scores = self._scores([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
        batch = pd.Series(["A", "A", "B", "B"])
        f, r2, p = permanova_euclidean(scores, batch, permutations=10)
        assert np.isinf(f) and np.isnan(p)

    def test_significant_separation(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((10, 2))
        b = rng.standard_normal((10, 2)) + 6
        scores = pd.DataFrame(np.vstack([a, b]), columns=["PC1", "PC2"])
        batch = pd.Series(["A"] * 10 + ["B"] * 10)
        f, r2, p = permanova_euclidean(scores, batch, permutations=199, random_state=0)
        assert f > 1 and 0 < p <= 0.05
