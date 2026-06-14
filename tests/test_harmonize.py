"""Tests for ComBatHarmonizer.

The harmonizer reuses inmoose's internal estimation helpers. Since CI does not
install the optional ``inmoose`` dependency, the estimation is covered with a
shape-faithful fake (identity correction), while correctness against the real
``pycombat_norm`` is validated under ``skipif`` when inmoose is present. All
validation and transform branches are covered without inmoose.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from eigenradiomics import ComBatHarmonizer


def _has_inmoose() -> bool:
    try:
        import inmoose  # noqa: F401

        return True
    except Exception:
        return False


HAS_INMOOSE = _has_inmoose()


def _data(n: int = 24, g: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    arr = rng.normal(0, 1, (n, g))
    arr[: n // 2] += 2.5  # additive batch effect on the first half
    batch = np.array([0] * (n // 2) + [1] * (n - n // 2))
    X = pd.DataFrame(arr, columns=[f"f{i}" for i in range(g)], index=[f"S{i}" for i in range(n)])
    return X, batch


# ---- shape-faithful fake inmoose (identity correction) -------------------


def _fake_make_design_matrix(counts, batch, covar_mod=None, ref_batch=None, na_cov_action="raise"):
    counts = np.asarray(counts)
    batch = list(batch)
    cats = list(dict.fromkeys(batch))
    n_samples = len(batch)
    n_covar = 0 if covar_mod is None else np.asarray(covar_mod).shape[1]
    design = np.zeros((n_samples, len(cats) + n_covar))
    for j, c in enumerate(cats):
        design[[i for i, b in enumerate(batch) if b == c], j] = 1.0
    if n_covar:
        design[:, len(cats):] = np.asarray(covar_mod, dtype=float)
    vci = types.SimpleNamespace(
        counts=counts,
        design=design,
        n_batch=len(cats),
        batch_composition={c: [i for i, b in enumerate(batch) if b == c] for c in cats},
        batch=types.SimpleNamespace(categories=cats),
        ref_batch_idx=(cats.index(ref_batch) if ref_batch is not None else None),
        list_samples=list(range(n_samples)),
        list_genes=list(range(counts.shape[0])),
    )
    return vci


def _fake_calc_mean_var(design, batches, ref, dat, n_batches, n_batch, n_array):
    dat = np.asarray(dat)
    b_hat = np.zeros((np.asarray(design).shape[0], dat.shape[0]))
    return b_hat, dat.mean(axis=1), dat.var(axis=1) + 1.0


def _fake_calc_stand_mean(grand_mean, n_array, design, n_batch, b_hat):
    return np.outer(np.asarray(grand_mean), np.ones(n_array))


def _fake_standardise(dat, stand_mean, var_pooled, n_array):
    return (np.asarray(dat) - np.asarray(stand_mean)) / np.sqrt(np.asarray(var_pooled))[:, None]


def _fake_fit_model(design, n_batch, s_data, batches, mean_only, par_prior, precision, ref):
    n_genes = np.asarray(s_data).shape[0]
    return np.zeros((n_batch, n_genes)), np.ones((n_batch, n_genes)), None


@pytest.fixture()
def fake_inmoose(monkeypatch):
    cov = types.ModuleType("inmoose.pycombat.covariates")
    cov.make_design_matrix = _fake_make_design_matrix
    pcn = types.ModuleType("inmoose.pycombat.pycombat_norm")
    pcn.calculate_mean_var = _fake_calc_mean_var
    pcn.calculate_stand_mean = _fake_calc_stand_mean
    pcn.standardise_data = _fake_standardise
    pcn.fit_model = _fake_fit_model
    monkeypatch.setitem(sys.modules, "inmoose", types.ModuleType("inmoose"))
    monkeypatch.setitem(sys.modules, "inmoose.pycombat", types.ModuleType("inmoose.pycombat"))
    monkeypatch.setitem(sys.modules, "inmoose.pycombat.covariates", cov)
    monkeypatch.setitem(sys.modules, "inmoose.pycombat.pycombat_norm", pcn)


# ---- correctness against the real pycombat_norm (needs inmoose) ----------


@pytest.mark.skipif(not HAS_INMOOSE, reason="inmoose not installed")
def test_fit_transform_matches_pycombat_norm_no_covariates():
    from inmoose.pycombat import pycombat_norm

    X, batch = _data()
    ref = np.asarray(pycombat_norm(X.to_numpy().T, list(batch))).T
    out = ComBatHarmonizer().fit_transform(X, batch=batch)
    assert np.allclose(out.to_numpy(), ref, atol=1e-6)


@pytest.mark.skipif(not HAS_INMOOSE, reason="inmoose not installed")
def test_fit_transform_matches_pycombat_norm_with_covariates():
    from inmoose.pycombat import pycombat_norm

    X, batch = _data()
    covar = pd.DataFrame({"sex": np.array([0.0, 1.0] * (len(X) // 2))}, index=X.index)
    ref = np.asarray(pycombat_norm(X.to_numpy().T, list(batch), covar_mod=covar)).T
    out = ComBatHarmonizer().fit_transform(X, batch=batch, covariates=covar)
    assert np.allclose(out.to_numpy(), ref, atol=1e-6)


@pytest.mark.skipif(not HAS_INMOOSE, reason="inmoose not installed")
def test_leakage_safe_train_test():
    X, batch = _data(n=30)
    h = ComBatHarmonizer().fit(X.iloc[:20], batch=batch[:20])
    out = h.transform(X.iloc[20:], batch=batch[20:])  # held-out, seen batches
    assert out.shape == (10, X.shape[1])
    assert np.isfinite(out.to_numpy()).all()


# ---- estimation + transform branches via the fake (no real inmoose) ------


def test_fit_transform_identity_with_fake(fake_inmoose):
    X, batch = _data()
    out = ComBatHarmonizer().fit_transform(X, batch=batch)
    assert isinstance(out, pd.DataFrame)
    assert np.allclose(out.to_numpy(), X.to_numpy())  # fake = identity correction


def test_pipeline_metadata_routing_with_fake(fake_inmoose):
    import sklearn
    from sklearn.pipeline import Pipeline

    X, batch = _data()
    sklearn.set_config(enable_metadata_routing=True)
    try:
        harmonizer = ComBatHarmonizer()
        harmonizer.set_fit_request(batch=True).set_transform_request(batch=True)
        pipe = Pipeline([("harmonize", harmonizer)])
        out = pipe.fit(X, batch=batch).transform(X, batch=batch)  # batch routed through both
    finally:
        sklearn.set_config(enable_metadata_routing=False)
    assert out.shape == X.shape


def test_transform_array_input_with_fake(fake_inmoose):
    X, batch = _data()
    h = ComBatHarmonizer().fit(X, batch=batch)
    out = h.transform(X.to_numpy(), batch=batch)
    assert isinstance(out, np.ndarray)


def test_covariates_roundtrip_with_fake(fake_inmoose):
    X, batch = _data()
    covar = pd.DataFrame({"a": np.arange(len(X), dtype=float), "b": 1.0}, index=X.index)
    h = ComBatHarmonizer().fit(X, batch=batch, covariates=covar)
    assert h.covariate_columns_ == ["a", "b"]
    # a transform covariate frame missing a column is reindexed (fill 0), not an error
    out = h.transform(X, batch=batch, covariates=covar[["a"]])
    assert out.shape == X.shape


def test_reference_batch_passthrough_with_fake(fake_inmoose):
    X, batch = _data()
    out = ComBatHarmonizer(reference_batch=0).fit_transform(X, batch=batch)
    ref_rows = batch == 0
    assert np.allclose(out.to_numpy()[ref_rows], X.to_numpy()[ref_rows])  # reference unchanged


def test_constant_feature_passthrough_with_fake(fake_inmoose):
    X, batch = _data()
    X = X.copy()
    X["f0"] = 5.0  # constant column -> excluded from ComBat, passed through
    h = ComBatHarmonizer().fit(X, batch=batch)
    assert h.constant_mask_[0]
    out = h.transform(X, batch=batch)
    assert np.allclose(out["f0"].to_numpy(), 5.0)


def test_unseen_batch_passthrough_warns_with_fake(fake_inmoose):
    X, batch = _data()
    h = ComBatHarmonizer().fit(X, batch=batch)
    with pytest.warns(UserWarning, match="not seen during fit"):
        out = h.transform(X, batch=np.array([9] * len(X)))
    assert np.allclose(out.to_numpy(), X.to_numpy())  # uncorrected passthrough


def test_unseen_batch_error_with_fake(fake_inmoose):
    X, batch = _data()
    h = ComBatHarmonizer(on_unseen_batch="error").fit(X, batch=batch)
    with pytest.raises(ValueError, match="not seen during fit"):
        h.transform(X, batch=np.array([9] * len(X)))


def test_transform_covariate_row_mismatch_with_fake(fake_inmoose):
    X, batch = _data()
    covar = pd.DataFrame({"a": np.arange(len(X), dtype=float)}, index=X.index)
    h = ComBatHarmonizer().fit(X, batch=batch, covariates=covar)
    with pytest.raises(ValueError, match="rows but X has"):
        h.transform(X, batch=batch, covariates=covar.iloc[:3])


def test_transform_covariate_consistency_with_fake(fake_inmoose):
    X, batch = _data()
    covar = pd.DataFrame({"a": np.arange(len(X), dtype=float)}, index=X.index)
    # fitted with covariates -> transform requires them
    h = ComBatHarmonizer().fit(X, batch=batch, covariates=covar)
    with pytest.raises(ValueError, match="requires the same covariates"):
        h.transform(X, batch=batch)
    # fitted without covariates -> transform must not pass them
    h2 = ComBatHarmonizer().fit(X, batch=batch)
    with pytest.raises(ValueError, match="without covariates"):
        h2.transform(X, batch=batch, covariates=covar)


# ---- validation (raises before the inmoose call -> no inmoose needed) -----


def test_invalid_on_unseen_raises():
    X, batch = _data()
    with pytest.raises(ValueError, match="on_unseen_batch must be"):
        ComBatHarmonizer(on_unseen_batch="maybe").fit(X, batch=batch)


def test_batch_required():
    X, _ = _data()
    with pytest.raises(ValueError, match="requires `batch`"):
        ComBatHarmonizer().fit(X)


def test_batch_length_mismatch_raises():
    X, _ = _data()
    with pytest.raises(ValueError, match="labels but X has"):
        ComBatHarmonizer().fit(X, batch=np.array([0, 1, 0]))


def test_single_batch_raises():
    X, _ = _data()
    with pytest.raises(ValueError, match="at least 2 batches"):
        ComBatHarmonizer().fit(X, batch=np.zeros(len(X)))


def test_bad_reference_batch_raises():
    X, batch = _data()
    with pytest.raises(ValueError, match="not among the batches"):
        ComBatHarmonizer(reference_batch="ghost").fit(X, batch=batch)


def test_all_constant_raises():
    X, batch = _data()
    X = pd.DataFrame(np.ones((len(X), 4)), columns=["a", "b", "c", "d"], index=X.index)
    with pytest.raises(ValueError, match="all features are constant"):
        ComBatHarmonizer().fit(X, batch=batch)


def test_non_numeric_covariates_raise():
    X, batch = _data()
    covar = pd.DataFrame({"grade": ["A", "B"] * (len(X) // 2)}, index=X.index)
    with pytest.raises(ValueError, match="numeric model matrix"):
        ComBatHarmonizer().fit(X, batch=batch, covariates=covar)


def test_covariate_row_mismatch_raises():
    X, batch = _data()
    covar = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="rows but X has"):
        ComBatHarmonizer().fit(X, batch=batch, covariates=covar)


def test_import_error_without_inmoose(monkeypatch):
    monkeypatch.setitem(sys.modules, "inmoose", None)
    monkeypatch.setitem(sys.modules, "inmoose.pycombat", None)
    monkeypatch.setitem(sys.modules, "inmoose.pycombat.covariates", None)
    monkeypatch.setitem(sys.modules, "inmoose.pycombat.pycombat_norm", None)
    X, batch = _data()
    with pytest.raises(ImportError, match="requires the optional 'inmoose'"):
        ComBatHarmonizer().fit(X, batch=batch)


def test_constant_column_warning_with_fake(fake_inmoose):
    X, batch = _data()
    # Make one column constant
    X["f0"] = 1.0
    h = ComBatHarmonizer()
    with pytest.warns(UserWarning, match="have near-zero variance"):
        h.fit(X, batch=batch)
    assert h.constant_mask_[0]
    assert not h.constant_mask_[1]
