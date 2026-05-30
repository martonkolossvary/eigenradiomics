"""Unit tests for WGCNA internals that do not require the PyWGCNA backend."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from eigenradiomics.reducers import WGCNAReducer
from eigenradiomics.reducers._wgcna_utils import _wgcna_compute_eigengene, _wgcna_fit_single


def test_compute_eigengene_multicomponent_degenerate() -> None:
    # Rank-1 matrix: the 2nd component has ~zero variance, exercising the
    # sign-alignment skip in the multi-component branch.
    X = np.outer(np.linspace(-1, 1, 12), np.ones(5))
    eig, load = _wgcna_compute_eigengene(X, n_components=2)
    assert eig.shape == (12, 2)
    assert load.shape == (5, 2)


def test_fit_single_zero_variance_warns() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4))
    X[:, 2] = 7.0  # constant feature
    with pytest.warns(UserWarning, match="zero-variance"):
        mod, _centers, scales, _loadings = _wgcna_fit_single("blue", X, np.array([0, 1, 2, 3]))
    assert mod == "blue"
    assert (scales > 0).all()  # zero scale replaced with 1.0


def test_merge_close_modules_merges_and_handles_constant() -> None:
    rng = np.random.default_rng(1)
    n = 40
    latent = rng.standard_normal(n)
    cols = [latent + rng.standard_normal(n) * 0.05 for _ in range(4)]
    cols.append(np.full(n, 3.0))  # constant feature -> zero-variance branch in merge
    cols += [latent + rng.standard_normal(n) * 0.05 for _ in range(5)]  # "red" shares latent
    X = np.column_stack(cols)
    colors = ["blue"] * 5 + ["red"] * 5

    reducer = WGCNAReducer(me_diss_threshold=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        merged = reducer._merge_close_modules(X, colors)
    # blue and red share a latent, so their eigengenes collapse to one module.
    assert len(set(merged)) == 1


def test_merge_close_modules_protect_unassigned() -> None:
    rng = np.random.default_rng(2)
    n = 30
    latent = rng.standard_normal(n)
    cols = [latent + rng.standard_normal(n) * 0.05 for _ in range(6)]
    cols += [rng.standard_normal(n) for _ in range(4)]  # unassigned noise
    X = np.column_stack(cols)
    colors = ["blue"] * 6 + ["grey"] * 4

    reducer = WGCNAReducer(me_diss_threshold=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        merged = reducer._merge_close_modules(X, colors, protect="grey")
    # The protected ("grey") colour is never merged away.
    assert "grey" in set(merged)
