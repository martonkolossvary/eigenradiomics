"""Tests for the feature-vs-clinical correlation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eigenradiomics import (
    RadiomicsDataset,
    compute_clinical_correlations,
    encode_clinical_series,
)

# ---- encode_clinical_series ----------------------------------------------


def test_encode_numeric_passthrough():
    s = pd.Series([1.0, 2.0, np.nan, 4.0])
    out = encode_clinical_series(s)
    assert out.tolist()[:2] == [1.0, 2.0]
    assert np.isnan(out.iloc[2])


def test_encode_numeric_strings():
    out = encode_clinical_series(pd.Series(["3", "1", "2"]))
    assert out.tolist() == [3.0, 1.0, 2.0]


def test_encode_binary_and_ordinal_tokens():
    out = encode_clinical_series(pd.Series(["yes", "no", "Male", "FEMALE", "III"]))
    assert out.tolist() == [1.0, 0.0, 0.0, 1.0, 3.0]


def test_encode_unknown_categories_ordinal_alphabetical():
    out = encode_clinical_series(pd.Series(["beta", "alpha", "gamma", "alpha"]))
    # alphabetical -> alpha=1, beta=2, gamma=3
    assert out.tolist() == [2.0, 1.0, 3.0, 1.0]


def test_encode_mostly_categorical_with_one_numeric_token():
    # A lone parseable token must not flip the column to (mostly-NaN) numeric;
    # it should ordinal-encode every value instead.
    out = encode_clinical_series(pd.Series(["mild", "moderate", "3", "severe"]))
    assert out.notna().all()
    assert out.nunique() == 4


# ---- compute_clinical_correlations ---------------------------------------


def _toy(n: int = 60):
    """Features driven by two latents; clinical vars tied to those latents."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    idx = [f"S{i}" for i in range(n)]
    X = pd.DataFrame(
        {
            "original__f0": a + 0.2 * rng.standard_normal(n),
            "original__f1": a + 0.2 * rng.standard_normal(n),
            "original__f2": b + 0.2 * rng.standard_normal(n),
        },
        index=idx,
    )
    clinical = pd.DataFrame(
        {
            "drives_a": a + 0.1 * rng.standard_normal(n),
            "sex": rng.choice(["male", "female"], n),
            "constant": ["x"] * n,  # no variance -> dropped
        },
        index=idx,
    )
    return X, clinical


def test_dataframe_inputs_drop_constant_variable():
    X, clinical = _toy()
    corr = compute_clinical_correlations(X, clinical, method="spearman", min_pairs=20)
    assert list(corr.index) == list(X.columns)
    assert "constant" not in corr.columns  # zero-variance variable dropped
    assert set(corr.columns) == {"drives_a", "sex"}
    # f0/f1 (driven by a) correlate with drives_a more than f2 (driven by b)
    assert corr.loc["original__f0", "drives_a"] > corr.loc["original__f2", "drives_a"]


def test_radiomics_dataset_with_column_names():
    X, clinical = _toy()
    data = pd.concat([X, clinical], axis=1)
    ds = RadiomicsDataset(data, feature_columns=list(X.columns))
    corr = compute_clinical_correlations(ds, ["drives_a", "sex"], min_pairs=20)
    assert list(corr.index) == list(X.columns)
    assert list(corr.columns) == ["drives_a", "sex"]


def test_radiomics_dataset_with_clinical_frame():
    X, clinical = _toy()
    ds = RadiomicsDataset(X, feature_columns=list(X.columns))
    corr = compute_clinical_correlations(ds, clinical[["drives_a"]], min_pairs=20)
    assert list(corr.columns) == ["drives_a"]


def test_no_usable_variable_raises():
    X, _ = _toy()
    clinical = pd.DataFrame({"flat": ["x"] * len(X)}, index=X.index)
    with pytest.raises(ValueError, match="non-missing, varying"):
        compute_clinical_correlations(X, clinical, min_pairs=20)


def test_clinical_must_be_frame_without_dataset():
    X, _ = _toy()
    with pytest.raises(TypeError, match="DataFrame when features"):
        compute_clinical_correlations(X, ["drives_a"], min_pairs=20)


def test_invalid_method_raises():
    X, clinical = _toy()
    with pytest.raises(ValueError, match="spearman.*pearson.*kendall"):
        compute_clinical_correlations(X, clinical, method="cosine")


def test_disjoint_index_raises_distinct_error():
    # A pure index mismatch must report itself, not masquerade as "not enough data".
    X, clinical = _toy()
    clinical = clinical.copy()
    clinical.index = [f"OTHER{i}" for i in range(len(clinical))]
    with pytest.raises(ValueError, match="no common index"):
        compute_clinical_correlations(X, clinical, min_pairs=20)
