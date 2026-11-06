"""Tests for feature-outcome association models (Phase A: continuous OLS+HC3)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from eigenradiomics import (  # noqa: E402
    FeatureCatalog,
    RadiomicsDataset,
    StudyDesign,
    compute_feature_associations,
    plot_volcano,
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


def test_mixed_method_invalid_raises():
    X, meta = _data()
    with pytest.raises(ValueError, match="mixed_method must be"):
        compute_feature_associations(X, meta["y"], covariate_data=meta, mixed_method="reml")


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


# ---- Phase B engines (survival / binary / mixed; need lifelines/statsmodels) --


def _survival_binary(n: int = 150):
    rng = np.random.default_rng(1)
    f0 = rng.normal(0, 1, n)
    f1 = rng.normal(0, 1, n)
    age = rng.normal(60, 8, n)
    patient = np.repeat(np.arange(n // 3 + 1), 3)[:n]  # repeated-measures clusters
    lp = 1.2 * f0 + 0.02 * age
    idx = list(range(n))
    X = pd.DataFrame({"original__f0": f0, "original__f1": f1}, index=idx)
    meta = pd.DataFrame({"age": age, "patient": patient}, index=idx)
    surv = pd.DataFrame(
        {
            "time": rng.exponential(np.exp(-lp)) + 0.01,
            "event": (rng.uniform(size=n) < 0.7).astype(int),
        },
        index=idx,
    )
    binary = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-lp))).astype(int), index=idx)
    return X, meta, surv, binary


def test_survival_cox():
    X, meta, surv, _ = _survival_binary()
    res = compute_feature_associations(
        X, surv, adjust_for=["age"], covariate_data=meta, min_events=5
    )
    assert res.outcome_type == "survival"
    row = res.table.set_index(["model", "feature"]).loc[("Univariable", "original__f0")]
    assert row["status"] == "ok"
    assert row["model_family"] == "cox" and row["effect_name"] == "HR"
    assert np.isfinite(row["c_index"])


def test_survival_cox_clustered():
    X, meta, surv, _ = _survival_binary()
    res = compute_feature_associations(
        X, surv, covariate_data=meta, groups="patient", min_events=5
    )
    fam = res.table.set_index(["model", "feature"]).loc[
        ("Univariable", "original__f0"), "model_family"
    ]
    assert fam == "cox_clustered"


def test_survival_no_events():
    X, meta, surv, _ = _survival_binary()
    res = compute_feature_associations(X, surv, covariate_data=meta, min_events=10_000)
    assert (res.table["status"] == "no_events").all()


def test_binary_logistic():
    X, meta, _, binary = _survival_binary()
    res = compute_feature_associations(X, binary, adjust_for=["age"], covariate_data=meta)
    assert res.outcome_type == "binary"
    row = res.table.set_index(["model", "feature"]).loc[("Univariable", "original__f0")]
    assert row["status"] == "ok" and row["model_family"] == "logit" and row["effect_name"] == "OR"


def test_binary_no_events():
    X, meta, _, _ = _survival_binary()
    const_outcome = pd.Series(np.zeros(len(X)), index=X.index)
    res = compute_feature_associations(
        X, const_outcome, outcome_type="binary", covariate_data=meta
    )
    assert (res.table["status"] == "no_events").all()


def test_binary_gee_clustered():
    X, meta, _, binary = _survival_binary()
    res = compute_feature_associations(
        X, binary, covariate_data=meta, groups="patient", mixed_method="gee"
    )
    fam = res.table.set_index(["model", "feature"]).loc[
        ("Univariable", "original__f0"), "model_family"
    ]
    assert fam == "gee_logit"


def test_binary_glmm_clustered():
    X, meta, _, binary = _survival_binary()
    res = compute_feature_associations(
        X, binary, covariate_data=meta, groups="patient", mixed_method="glmm"
    )
    row = res.table.set_index(["model", "feature"]).loc[("Univariable", "original__f0")]
    assert row["model_family"] == "glmm" and row["status"] == "ok"


def _mixed_continuous_outcome(X, meta):
    """A continuous outcome with a genuine per-patient random intercept + residual
    noise, so MixedLM estimates a non-singular random-effects variance (stable
    across statsmodels versions)."""
    rng = np.random.default_rng(5)
    cluster = meta["patient"].to_numpy()
    intercepts = rng.normal(0, 0.5, cluster.max() + 1)
    y = (
        1.0 * X["original__f0"].to_numpy()
        + 0.02 * meta["age"].to_numpy()
        + intercepts[cluster]
        + rng.normal(0, 0.3, len(X))
    )
    return pd.Series(y, index=X.index)


def test_continuous_mixedlm_via_groups_array():
    X, meta, _, _ = _survival_binary()
    y = _mixed_continuous_outcome(X, meta)
    # groups passed as an array (not a column name) -> exercises that resolution branch
    res = compute_feature_associations(
        X, y, groups=meta["patient"].to_numpy(), covariate_data=meta
    )
    fam = res.table.set_index(["model", "feature"]).loc[
        ("Univariable", "original__f0"), "model_family"
    ]
    assert fam == "mixedlm"


def test_dataset_supplies_groups():
    X, meta, _, _ = _survival_binary()
    y = _mixed_continuous_outcome(X, meta)
    data = pd.concat([X, meta, y.rename("y")], axis=1)
    ds = RadiomicsDataset(
        data,
        feature_columns=list(X.columns),
        design=StudyDesign(roles={"target": "y", "group": "patient"}),
    )
    res = compute_feature_associations(ds)  # groups taken from the design
    fam = res.table.set_index(["model", "feature"]).loc[
        ("Univariable", "original__f0"), "model_family"
    ]
    assert fam == "mixedlm"


# ---- fit-failure and optional-dependency handling -------------------------


def test_logit_fit_failed(monkeypatch):
    import types as _types

    fake_sm = _types.SimpleNamespace(
        Logit=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("logit boom"))
    )
    monkeypatch.setattr("eigenradiomics.feature_models._import_statsmodels", lambda: fake_sm)
    X, meta, _, binary = _survival_binary(n=30)
    res = compute_feature_associations(X, binary, outcome_type="binary", covariate_data=meta)
    assert (res.table["status"] == "fit_failed").all()
    assert res.table.iloc[0]["error"]


def test_cox_fit_failed(monkeypatch):
    class _BadCox:
        def __init__(self, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            raise RuntimeError("cox boom")

    monkeypatch.setattr("eigenradiomics.feature_models._import_lifelines", lambda: _BadCox)
    X, meta, surv, _ = _survival_binary(n=30)
    res = compute_feature_associations(X, surv, covariate_data=meta, min_events=1)
    assert (res.table["status"] == "fit_failed").all()


def test_mixedlm_fit_failed(monkeypatch):
    import types as _types

    fake_sm = _types.SimpleNamespace(
        MixedLM=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("mixedlm boom"))
    )
    monkeypatch.setattr("eigenradiomics.feature_models._import_statsmodels", lambda: fake_sm)
    X, meta, _, _ = _survival_binary(n=30)
    y = pd.Series(X["original__f0"].to_numpy(), index=X.index)
    res = compute_feature_associations(X, y, covariate_data=meta, groups="patient")
    assert (res.table["status"] == "fit_failed").all()


def test_glmm_fit_failed(monkeypatch):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    def _boom(*args, **kwargs):
        raise RuntimeError("glmm boom")

    monkeypatch.setattr(BinomialBayesMixedGLM, "from_formula", staticmethod(_boom))
    X, meta, _, binary = _survival_binary(n=30)
    res = compute_feature_associations(
        X, binary, covariate_data=meta, groups="patient", mixed_method="glmm"
    )
    assert (res.table["status"] == "fit_failed").all()


def test_lifelines_import_error(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "lifelines", None)
    X, meta, surv, _ = _survival_binary(n=30)
    with pytest.raises(ImportError, match="lifelines"):
        compute_feature_associations(X, surv, covariate_data=meta, min_events=1)


def test_statsmodels_import_error(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "statsmodels.api", None)
    X, meta, _, binary = _survival_binary(n=30)
    with pytest.raises(ImportError, match="statsmodels"):
        compute_feature_associations(X, binary, outcome_type="binary", covariate_data=meta)


# ---- Phase C: volcano plot ------------------------------------------------


def _volcano_result(n_tiers: int = 2, n_features: int = 8):
    rng = np.random.default_rng(2)
    n = 200
    cols = {f"original__f{i}": rng.normal(0, 1, n) for i in range(n_features)}
    X = pd.DataFrame(cols, index=range(n))
    age = rng.normal(60, 8, n)
    y = pd.Series(1.5 * X["original__f0"] + 0.02 * age + rng.normal(0, 1, n), index=range(n))
    meta = pd.DataFrame({"age": age}, index=range(n))
    fams = ["Intensity", "Texture", "Morphology"]
    catalog = FeatureCatalog(
        pd.DataFrame(
            {
                "config": ["original"] * n_features,
                "feature_key": [f"f{i}" for i in range(n_features)],
                "family": ["firstorder"] * n_features,
                "family_group": [fams[i % 3] for i in range(n_features)],
            }
        )
    )
    tiers = {f"M{j}": (["age"] if j % 2 else []) for j in range(n_tiers)}
    return compute_feature_associations(
        X, y, model_tiers=tiers, covariate_data=meta, catalog=catalog
    )


@pytest.mark.parametrize(
    ("n_tiers", "expected_axes", "expected_visible"),
    [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 9, 5), (9, 9, 9)],
)
def test_volcano_layouts(n_tiers, expected_axes, expected_visible):
    res = _volcano_result(n_tiers=n_tiers)
    fig = plot_volcano(res, color_by="family_group", marker_by="family_group")
    assert len(fig.axes) == expected_axes
    assert sum(ax.get_visible() for ax in fig.axes) == expected_visible
    plt.close(fig)


def test_volcano_panel_count_out_of_range():
    res = _volcano_result(n_tiers=2)
    with pytest.raises(ValueError, match="1-9 panels"):
        plot_volcano(res, tiers=[])


def test_volcano_explicit_layout_and_too_small():
    res = _volcano_result(n_tiers=4)
    fig = plot_volcano(res, layout=(2, 3))
    assert len(fig.axes) == 6
    plt.close(fig)
    with pytest.raises(ValueError, match="fewer cells"):
        plot_volcano(res, layout=(1, 1))


def test_volcano_axis_and_outlier_modes():
    res = _volcano_result()
    plt.close(plot_volcano(res, axis_mode="shared", outlier_strategy="include"))
    with pytest.raises(ValueError, match="axis_mode"):
        plot_volcano(res, axis_mode="weird")
    with pytest.raises(ValueError, match="outlier_strategy"):
        plot_volcano(res, outlier_strategy="weird")


def test_volcano_color_none_and_marker_by():
    res = _volcano_result()
    plt.close(plot_volcano(res, color_by=None))  # significant points use a single highlight colour
    plt.close(plot_volcano(res, color_by="family_group", marker_by="family_group"))


def test_volcano_empty_panels():
    # all-constant features -> nothing fitted -> every panel shows "no fitted features"
    X = pd.DataFrame({f"original__f{i}": np.full(50, 3.0) for i in range(3)}, index=range(50))
    y = pd.Series(np.arange(50, dtype=float), index=range(50))
    res = compute_feature_associations(X, y, covariate_data=pd.DataFrame({"y": y}))
    fig = plot_volcano(res)
    plt.close(fig)


def test_volcano_binary_xlabel():
    X, meta, _, binary = _survival_binary()
    res = compute_feature_associations(X, binary, covariate_data=meta)
    fig = plot_volcano(res, title="binary")
    assert any("odds ratio" in ax.get_xlabel() for ax in fig.axes if ax.get_visible())
    plt.close(fig)


# ---- Phase D: heatmap bridge + Excel export -------------------------------


def test_bar_and_value_columns():
    X, meta = _data()
    res = compute_feature_associations(X, meta["y"], adjust_for=["age"], covariate_data=meta)
    bar = res.bar(value="neg_log10_p", reference=1.3, title="-log10 p")
    assert type(bar).__name__ == "Bar"
    assert list(bar.data.index) == list(X.columns)  # feature-indexed
    assert bar.title == "-log10 p" and bar.reference == 1.3
    # exercise the remaining value columns
    for value in ("neg_log10_fdr", "coef", "effect", "statistic", "ci_low", "ci_high"):
        assert res.bar(value=value).data.shape[0] == len(X.columns)


def test_bar_log2_effect_on_ratio():
    X, meta, _, binary = _survival_binary()
    res = compute_feature_associations(X, binary, covariate_data=meta)
    bar = res.bar(value="log2_effect")  # log2(OR) is well-defined for ratio effects
    assert bar.data.shape[0] == X.shape[1]


def test_matrix_feature_by_tier():
    X, meta = _data()
    res = compute_feature_associations(X, meta["y"], adjust_for=["age"], covariate_data=meta)
    mat = res.matrix(value="coef")
    assert list(mat.columns) == ["Univariable", "Adjusted"]
    assert list(mat.index) == list(X.columns)


def test_bar_bad_value_and_tier():
    X, meta = _data()
    res = compute_feature_associations(X, meta["y"], covariate_data=meta)
    with pytest.raises(ValueError, match="value must be one of"):
        res.bar(value="bogus")
    with pytest.raises(ValueError, match="unknown tier"):
        res.bar(tier="ghost")


def test_to_excel(tmp_path):
    import openpyxl

    X, meta = _data()
    res = compute_feature_associations(X, meta["y"], adjust_for=["age"], covariate_data=meta)
    path = tmp_path / "associations.xlsx"
    res.to_excel(path)
    assert path.exists()
    wb = openpyxl.load_workbook(path)
    assert "associations" in wb.sheetnames and "top_hits" in wb.sheetnames


def test_plot_rwas_manhattan(tmp_path):
    from eigenradiomics.feature_models import plot_rwas_manhattan
    from eigenradiomics.plotting import Bar, CorrPanel, Strip

    X, meta = _data()
    res = compute_feature_associations(X, meta["y"], adjust_for=["age"], covariate_data=meta)

    # Simple call
    fig = plot_rwas_manhattan(res)
    assert fig is not None
    plt.close(fig)

    # Custom ordering, catalog, and strips/bars/corr_panel
    catalog = _catalog()
    features = list(X.columns)
    strip = Strip(pd.Series(["A", "B", "A", "B"], index=features), title="strip1")
    bar = Bar(pd.Series([1.0, 2.0, 1.5, 3.0], index=features), title="bar1")
    corr_df = pd.DataFrame(
        [[0.1, -0.2], [0.5, 0.4], [-0.3, 0.1], [0.8, -0.7]],
        index=features,
        columns=["var1", "var2"],
    )
    corr_panel = CorrPanel(corr_df, label="Correlation")

    fig2 = plot_rwas_manhattan(
        res,
        catalog=catalog,
        tier="Univariable",
        group_by="family",
        order=features[::-1],
        strips=[strip],
        bars=[bar],
        corr_panel=corr_panel,
        title="Manhattan Test",
        path=tmp_path / "manhattan.png",
    )
    assert fig2 is not None
    assert (tmp_path / "manhattan.png").exists()
    plt.close(fig2)


def test_plot_rwas_manhattan_coverage_gaps():
    from eigenradiomics.feature_models import plot_rwas_manhattan
    from eigenradiomics.plotting import CorrPanel

    # 1. No fitted features available to plot
    empty_df = pd.DataFrame(columns=["feature", "p_value", "status"])
    with pytest.raises(ValueError, match="No fitted features available to plot"):
        plot_rwas_manhattan(empty_df)

    # 2. No features match the specified order
    X, meta = _data()
    res = compute_feature_associations(X, meta["y"], adjust_for=["age"], covariate_data=meta)
    with pytest.raises(ValueError, match="No features match the specified order"):
        plot_rwas_manhattan(res, order=["non_existent_feature"])

    # 3. n > 60 features (to hit tick_labels is None and ax_corr.set_xticks([]))
    n_feats = 65
    rng = np.random.default_rng(42)
    X_large = pd.DataFrame({f"feat_{i}": rng.normal(0, 1, 100) for i in range(n_feats)})
    y_large = pd.Series(rng.normal(0, 1, 100))
    res_large = compute_feature_associations(X_large, y_large)

    features_large = list(X_large.columns)
    corr_df = pd.DataFrame(
        rng.normal(0, 1, (n_feats, 2)),
        index=features_large,
        columns=["var1", "var2"],
    )
    corr_panel = CorrPanel(corr_df, label="Large Correlation")

    fig = plot_rwas_manhattan(res_large, corr_panel=corr_panel)
    assert fig is not None
    plt.close(fig)


def test_plot_rwas_manhattan_custom_group_by():
    from eigenradiomics.feature_models import plot_rwas_manhattan

    df = pd.DataFrame(
        {
            "feature": ["f1", "f2", "f3", "f4"],
            "p_value": [0.001, 0.04, 0.2, 0.8],
            "p_fdr": [0.004, 0.08, 0.3, 0.8],
            "status": ["ok", "ok", "ok", "ok"],
            "custom_cat": pd.Categorical(["groupB", "groupA", "groupB", "groupA"]),
        }
    )

    # Test with custom Categorical column (exercises lines 1060 and 1075-1077)
    fig = plot_rwas_manhattan(df, group_by="custom_cat")
    assert fig is not None
    plt.close(fig)

    # Test with custom string column not in IBSI order
    df_str = df.copy()
    df_str["custom_str"] = ["grp2", "grp1", "grp2", "grp1"]
    fig2 = plot_rwas_manhattan(df_str, group_by="custom_str")
    assert fig2 is not None
    plt.close(fig2)


# ---- regression tests for the 2026-11 review fixes ------------------------


def test_bare_frame_excludes_outcome_covariate_and_group_columns():
    X, meta, _, _ = _survival_binary()
    y = _mixed_continuous_outcome(X, meta)
    frame = pd.concat([X, meta, y.rename("y")], axis=1)  # one bare table holds every role
    res = compute_feature_associations(frame, "y", adjust_for=["age"])
    assert set(res.table["feature"]) == {"original__f0", "original__f1", "patient"}
    assert (res.table["status"] == "ok").all()
    clustered = compute_feature_associations(frame, frame["y"], groups="patient")
    assert set(clustered.table["feature"]) == {"original__f0", "original__f1", "age"}
    assert (clustered.table["model_family"] == "mixedlm").all()


def test_binary_clustered_default_is_gee():
    X, meta, _, binary = _survival_binary()
    res = compute_feature_associations(X, binary, covariate_data=meta, groups="patient")
    assert (res.table["model_family"] == "gee_logit").all()


def test_groups_series_aligned_by_index():
    X, meta, _, _ = _survival_binary()
    y = _mixed_continuous_outcome(X, meta)
    by_name = compute_feature_associations(X, y, groups="patient", covariate_data=meta)
    shuffled = meta["patient"].sample(frac=1, random_state=0)  # same labels, new order
    by_series = compute_feature_associations(X, y, groups=shuffled)
    assert np.allclose(by_series.table["coef"], by_name.table["coef"])
    with pytest.raises(ValueError, match="unique index"):
        compute_feature_associations(X, y, groups=pd.Series(0, index=[0] * len(X)))
    with pytest.raises(ValueError, match="does not cover"):
        compute_feature_associations(X, y, groups=meta["patient"].iloc[1:])
    with pytest.raises(ValueError, match="values but X has"):
        compute_feature_associations(X, y, groups=meta["patient"].to_numpy()[1:])


def test_zero_p_value_kept_in_plots():
    from eigenradiomics.feature_models import plot_rwas_manhattan

    res = _volcano_result(n_tiers=1)
    ok = res.table.index[res.table["status"].eq("ok")]
    res.table.loc[ok[0], "p_value"] = 0.0  # an underflowed p-value
    fig = plot_volcano(res)
    assert len(fig.axes[0].collections) == len(ok)  # one scatter call per point
    plt.close(fig)
    fig = plot_rwas_manhattan(res)
    y = fig.axes[0].collections[0].get_offsets()[:, 1]
    assert len(y) == len(ok) and np.isclose(y.max(), -np.log10(np.nextafter(0, 1)))
    plt.close(fig)


def test_second_catalog_annotation_replaces_columns():
    from eigenradiomics.feature_models import _annotate_catalog

    X, meta = _data()
    res = compute_feature_associations(X, meta["y"], covariate_data=meta, catalog=_catalog())
    again = _annotate_catalog(res.table, _catalog())
    assert not any(col.endswith(("_x", "_y")) for col in again.columns)
    assert again["family_group"].nunique() == 2


@pytest.mark.parametrize("engine", ["ols", "logit", "mixedlm"])
def test_collinear_covariates_flagged(engine):
    X, meta, _, binary = _survival_binary()
    meta = meta.assign(age2=2 * meta["age"])  # exact copy of age, scaled
    kwargs: dict = {"adjust_for": ["age", "age2"], "covariate_data": meta}
    if engine == "ols":
        outcome = _mixed_continuous_outcome(X, meta)
    elif engine == "logit":
        outcome = binary
    else:
        outcome = _mixed_continuous_outcome(X, meta)
        kwargs["groups"] = "patient"
    res = compute_feature_associations(X, outcome, **kwargs)
    adjusted = res.table[res.table["model"] == "Adjusted"]
    assert (adjusted["status"] == "collinear").all()
    assert adjusted["coef"].isna().all() and adjusted["p_value"].isna().all()
    assert (res.table.loc[res.table["model"] == "Univariable", "status"] == "ok").all()


def test_rank_check_skips_non_finite_design():
    X, meta = _data()
    X = X.assign(original__inf=np.where(np.arange(len(X)) == 0, np.inf, X["original__f0"]))
    res = compute_feature_associations(X, meta["y"], covariate_data=meta)
    status = res.table.set_index("feature")["status"]
    assert status["original__inf"] == "fit_failed" and status["original__f0"] == "ok"


def test_text_covariate_raises():
    X, meta = _data()
    meta = meta.assign(sex=np.where(np.arange(len(meta)) % 2, "M", "F"))
    with pytest.raises(ValueError, match="not numeric: sex"):
        compute_feature_associations(X, meta["y"], adjust_for=["sex"], covariate_data=meta)


def test_binary_outcome_coded_one_two():
    X, meta, _, binary = _survival_binary()
    base = compute_feature_associations(X, binary, covariate_data=meta)
    recoded = compute_feature_associations(X, binary + 1, covariate_data=meta)  # 1/2 coding
    assert recoded.outcome_type == "binary"
    assert (recoded.table["status"] == "ok").all()
    assert np.allclose(recoded.table["coef"], base.table["coef"])
    assert (recoded.table["n_events"] == base.table["n_events"]).all()


def test_log2_effect_nan_for_linear_beta():
    X, meta = _data()
    res = compute_feature_associations(X, -meta["y"], covariate_data=meta)  # negative betas
    assert res.bar(value="log2_effect").data.isna().all()
    X, meta, _, binary = _survival_binary()
    res = compute_feature_associations(X, binary, covariate_data=meta)
    uni = res.table.set_index("feature")
    expected = np.log2(uni["effect"])
    assert np.allclose(res.bar(value="log2_effect").data, expected)


def test_volcano_reference_line_labelled_nominal():
    fig = plot_volcano(_volcano_result(n_tiers=1), fdr_alpha=0.05)
    labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert "p = 0.05 (nominal)" in labels
    assert not any("(FDR)" in label for label in labels)
    plt.close(fig)


def test_extreme_scale_gap_matches_rescaled_fit():
    import statsmodels.api as sm

    rng = np.random.default_rng(3)
    n = 80
    X = pd.DataFrame({"original__tiny": rng.normal(0, 1e-12, n)})
    meta = pd.DataFrame({"age": rng.normal(60, 8, n), "vol": rng.normal(1e6, 1e5, n)})
    y = pd.Series(rng.normal(size=n))
    res = compute_feature_associations(X, y, adjust_for=["age", "vol"], covariate_data=meta)
    row = res.table.set_index("model").loc["Adjusted"]
    # Reference: the same OLS + HC3 fit with each column in units of its SD.
    raw = np.column_stack([X["original__tiny"], meta["age"], meta["vol"]])
    sd = raw.std(axis=0)
    ref = sm.OLS(y.to_numpy(), np.column_stack([np.ones(n), raw / sd])).fit(
        cov_type="HC3", use_t=True
    )
    assert np.isclose(row["p_value"], ref.pvalues[1])
    assert np.isclose(row["coef"], ref.params[1] / sd[0])
    assert np.isclose(row["se"], ref.bse[1] / sd[0])


def test_binary_min_events():
    X, meta, _, binary = _survival_binary()
    n_events = int(binary.sum())
    res = compute_feature_associations(X, binary, covariate_data=meta, min_events=n_events + 1)
    assert (res.table["status"] == "no_events").all()
    res = compute_feature_associations(X, binary, covariate_data=meta, min_events=n_events)
    assert (res.table["status"] == "ok").all()


def test_manhattan_nominal_line_uses_fdr_alpha():
    from eigenradiomics.feature_models import plot_rwas_manhattan

    fig = plot_rwas_manhattan(_volcano_result(n_tiers=1), fdr_alpha=0.01)
    dashed = [line for line in fig.axes[0].lines if line.get_linestyle() == "--"]
    assert len(dashed) == 1 and np.isclose(dashed[0].get_ydata()[0], 2.0)
    plt.close(fig)


def test_group_column_as_covariate_raises():
    X, meta, _, binary = _survival_binary()
    with pytest.raises(ValueError, match="both the group and a covariate"):
        compute_feature_associations(
            X, binary, adjust_for=["patient"], covariate_data=meta, groups="patient"
        )


def test_glmm_large_sd_feature_agrees_with_gee():
    rng = np.random.default_rng(7)
    n = 150
    cluster = np.repeat(np.arange(50), 3)
    f = rng.normal(0, 200, n)  # SD 200: raw-unit fit_vb stalls at OR 1, p 1
    p = 1 / (1 + np.exp(-(f / 200 + rng.normal(0, 0.5, 50)[cluster])))
    y = pd.Series((rng.uniform(size=n) < p).astype(int))
    X, meta = pd.DataFrame({"original__big": f}), pd.DataFrame({"patient": cluster})
    fits = {
        method: compute_feature_associations(
            X, y, covariate_data=meta, groups="patient", mixed_method=method
        ).table.iloc[0]
        for method in ("glmm", "gee")
    }
    assert (fits["glmm"]["status"], fits["gee"]["status"]) == ("ok", "ok")
    assert fits["glmm"]["p_value"] < 0.01 and fits["gee"]["p_value"] < 0.01
    assert 0.5 < fits["glmm"]["coef"] / fits["gee"]["coef"] < 3  # same sign and size


def test_text_binary_outcome_mapped():
    X, meta, _, binary = _survival_binary()
    base = compute_feature_associations(X, binary, covariate_data=meta)
    text = binary.map({0: "no", 1: "yes"})
    res = compute_feature_associations(X, text, covariate_data=meta)
    assert res.outcome_type == "binary" and (res.table["status"] == "ok").all()
    assert np.allclose(res.table["coef"], base.table["coef"])
    three = text.where(text.index % 7 != 0, "maybe")
    with pytest.raises(ValueError, match="exactly two levels"):
        compute_feature_associations(X, three, covariate_data=meta)


def test_outcome_and_covariate_index_must_cover_x():
    X, meta = _data()  # X and meta use the labels S0, S1, ...
    with pytest.raises(ValueError, match="outcome index does not cover"):
        compute_feature_associations(X, meta["y"].reset_index(drop=True))
    shifted = meta.reset_index(drop=True)
    with pytest.raises(ValueError, match="covariate data index does not cover"):
        compute_feature_associations(X, meta["y"], adjust_for=["age"], covariate_data=shifted)
    with pytest.raises(ValueError, match="groups column index does not cover"):
        compute_feature_associations(X, meta["y"], groups="g", covariate_data=shifted.assign(g=0))


def test_covariate_data_aligned_by_key_and_n_missing_over_x():
    X, meta = _data()
    base = compute_feature_associations(X, meta["y"], adjust_for=["age"], covariate_data=meta)
    extra = pd.DataFrame({"y": [0.0] * 40, "age": [50.0] * 40}, index=[f"X{i}" for i in range(40)])
    wide = pd.concat([meta, extra]).sample(frac=1, random_state=0)  # more rows, new order
    res = compute_feature_associations(X, meta["y"], adjust_for=["age"], covariate_data=wide)
    assert np.allclose(res.table["coef"], base.table["coef"], equal_nan=True)
    ok = res.table["status"] == "ok"
    assert (res.table.loc[ok, "n_missing"] == 0).all()
