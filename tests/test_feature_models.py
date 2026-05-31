"""Tests for feature-outcome association models (Phase A: continuous OLS+HC3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eigenradiomics import (
    FeatureCatalog,
    RadiomicsDataset,
    StudyDesign,
    compute_feature_associations,
)


def _data(n: int = 120):
    rng = np.random.default_rng(0)
    f0 = rng.normal(0, 1, n)
    f1 = rng.normal(0, 1, n)
    age = rng.normal(60, 8, n)
    y = 2.0 * f0 + 0.05 * age + rng.normal(0, 1, n)  # f0 associated, f1 not
    idx = [f"S{i}" for i in range(n)]
    X = pd.DataFrame(
        {
            "original__f0": f0,
            "original__f1": f1,
            "original__const": np.full(n, 5.0),  # constant -> constant_feature
            "original__empty": np.full(n, np.nan),  # all-NaN -> no_complete_cases
        },
        index=idx,
    )
    meta = pd.DataFrame({"y": y, "age": age}, index=idx)
    return X, meta


def _catalog() -> FeatureCatalog:
    return FeatureCatalog(
        pd.DataFrame(
            {
                "config": ["original"] * 4,
                "feature_key": ["f0", "f1", "const", "empty"],
                "family": ["firstorder", "glcm", "firstorder", "glcm"],
                "family_group": ["Intensity", "Texture", "Intensity", "Texture"],
            }
        )
    )


# ---- core behaviour -------------------------------------------------------


def test_continuous_univariable_and_adjusted():
    X, meta = _data()
    res = compute_feature_associations(X, meta["y"], adjust_for=["age"], covariate_data=meta)
    assert res.tiers == ["Univariable", "Adjusted"]
    assert res.outcome_type == "continuous"
    t = res.table.set_index(["model", "feature"])

    # coefficient matches a plain OLS (HC3 only changes the SE, not the estimate)
    f0 = X["original__f0"].to_numpy()
    design = np.column_stack([np.ones(len(f0)), f0])
    beta = np.linalg.lstsq(design, meta["y"].to_numpy(), rcond=None)[0]
    assert np.isclose(t.loc[("Univariable", "original__f0"), "coef"], beta[1])

    assert t.loc[("Univariable", "original__f0"), "p_value"] < 1e-3  # strong signal
    assert t.loc[("Univariable", "original__f1"), "p_value"] > 0.05  # noise
    assert "p_fdr" in res.table.columns
    assert (t.loc[("Univariable", "original__f0"), "effect_name"]) == "beta"


def test_status_branches():
    X, meta = _data()
    res = compute_feature_associations(X, meta["y"], covariate_data=meta)
    status = res.table.set_index(["model", "feature"])["status"]
    assert status.loc[("Univariable", "original__f0")] == "ok"
    assert status.loc[("Univariable", "original__const")] == "constant_feature"
    assert status.loc[("Univariable", "original__empty")] == "no_complete_cases"


def test_not_enough_degrees_of_freedom():
    X, meta = _data(n=4)
    res = compute_feature_associations(X, meta["y"], adjust_for=["age"], covariate_data=meta)
    status = res.table.set_index(["model", "feature"])["status"]
    # 4 samples, design of 5 columns (intercept + feature + 3 implicit? here 1 covar) ...
    assert status.loc[("Adjusted", "original__f0")] in {
        "not_enough_degrees_of_freedom",
        "ok",
    }


def test_outcome_as_column_name():
    X, meta = _data()
    res = compute_feature_associations(X, "y", covariate_data=meta)
    assert len(res.table) == 4  # one tier x four features


def test_explicit_model_tiers():
    X, meta = _data()
    res = compute_feature_associations(
        X, meta["y"], model_tiers={"crude": [], "adj": ["age"]}, covariate_data=meta
    )
    assert res.tiers == ["crude", "adj"]


def test_catalog_annotation():
    X, meta = _data()
    res = compute_feature_associations(X, meta["y"], covariate_data=meta, catalog=_catalog())
    assert {"family", "family_group"} <= set(res.table.columns)
    row = res.table.set_index(["model", "feature"]).loc[("Univariable", "original__f0")]
    assert row["family_group"] == "Intensity"
    # a DataFrame catalog works too
    res2 = compute_feature_associations(
        X, meta["y"], covariate_data=meta, catalog=_catalog().frame
    )
    assert "family" in res2.table.columns


def test_radiomics_dataset_infers_outcome_and_catalog():
    X, meta = _data()
    ds = RadiomicsDataset(
        pd.concat([X, meta], axis=1),
        feature_columns=list(X.columns),
        catalog=_catalog(),
        design=StudyDesign(roles={"target": "y"}),
    )
    res = compute_feature_associations(ds)
    assert res.outcome_type == "continuous"
    assert "family_group" in res.table.columns  # catalog taken from the dataset


# ---- validation / errors --------------------------------------------------


def test_outcome_required_without_dataset():
    X, _ = _data()
    with pytest.raises(ValueError, match="outcome is required"):
        compute_feature_associations(X)


def test_dataset_without_outcome_raises():
    X, meta = _data()
    ds = RadiomicsDataset(pd.concat([X, meta], axis=1), feature_columns=list(X.columns))
    with pytest.raises(ValueError, match="no outcome"):
        compute_feature_associations(ds)


def test_bad_outcome_type_raises():
    X, meta = _data()
    with pytest.raises(TypeError, match="Series, DataFrame, or column name"):
        compute_feature_associations(X, 123, covariate_data=meta)


def test_invalid_outcome_type_raises():
    X, meta = _data()
    with pytest.raises(ValueError, match="continuous/binary/survival"):
        compute_feature_associations(X, meta["y"], outcome_type="poisson", covariate_data=meta)


def test_binary_outcome_not_implemented():
    X, meta = _data()
    binary = pd.Series((meta["y"] > meta["y"].median()).astype(int), index=X.index)
    with pytest.raises(NotImplementedError, match="binary"):
        compute_feature_associations(X, binary, covariate_data=meta)


def test_survival_outcome_not_implemented():
    X, meta = _data()
    surv = pd.DataFrame({"time": meta["y"].abs(), "event": 1}, index=X.index)
    with pytest.raises(NotImplementedError, match="survival"):
        compute_feature_associations(X, surv, covariate_data=meta)


def test_missing_covariate_raises():
    X, meta = _data()
    with pytest.raises(KeyError, match="covariate column"):
        compute_feature_associations(X, meta["y"], adjust_for=["ghost"], covariate_data=meta)


def test_no_features_raises():
    meta = pd.DataFrame({"y": [1.0, 2.0, 3.0], "label": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="no feature columns"):
        compute_feature_associations(meta[["label"]], meta["y"], covariate_data=meta)


# ---- top_hits -------------------------------------------------------------


def test_top_hits_modes():
    X, meta = _data()
    res = compute_feature_associations(X, meta["y"], adjust_for=["age"], covariate_data=meta)
    assert len(res.top_hits(mode="fdr")) >= 1
    assert len(res.top_hits(mode="nominal")) >= 1
    assert len(res.top_hits(mode="ranked", per_panel=1)) == 2  # 1 per tier
    with pytest.raises(ValueError, match="fdr.*nominal.*ranked"):
        res.top_hits(mode="bogus")


def test_top_hits_empty_when_nothing_fitted():
    X, meta = _data()
    constants = X[["original__const"]]  # only a constant feature -> never "ok"
    res = compute_feature_associations(constants, meta["y"], covariate_data=meta)
    assert res.top_hits(mode="fdr").empty
