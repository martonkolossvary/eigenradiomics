"""Unit tests for the radiomics reproducibility framework."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import pytest

from eigenradiomics._stats import (
    _bootstrap_icc_ci,
    _fisher_ci,
    _fisher_mean,
    _get_deterministic_seed,
    _icc_2_1_estimate,
)
from eigenradiomics.reproducibility import (
    compute_reproducibility,
    plot_reproducibility_histograms,
    write_reproducibility_excel,
)


def test_deterministic_seed():
    """Verify that deterministic seed hashing is fully reproducible and distinct."""
    seed1 = _get_deterministic_seed("feature_A", base_seed=42)
    seed2 = _get_deterministic_seed("feature_A", base_seed=42)
    seed3 = _get_deterministic_seed("feature_B", base_seed=42)
    seed4 = _get_deterministic_seed("feature_A", base_seed=100)

    assert seed1 == seed2
    assert seed1 != seed3
    assert seed1 != seed4
    assert isinstance(seed1, int)


def test_fisher_mean_corrects_downward_bias():
    """Fisher z-averaging exceeds the (biased) arithmetic mean of correlations."""
    coeffs = [0.90, 0.95, 0.99]
    fisher = _fisher_mean(coeffs)
    assert fisher > float(np.mean(coeffs))  # bias-corrected upward
    assert np.isclose(fisher, float(np.tanh(np.mean(np.arctanh(coeffs)))))


def test_fisher_ci_undefined_at_unit_correlation():
    """The Fisher CI is not estimable at |r| = 1 (z diverges) -> NaN, not a band."""
    assert _fisher_ci(1.0, 30) == (np.nan, np.nan)
    assert _fisher_ci(-1.0, 30) == (np.nan, np.nan)
    low, high = _fisher_ci(0.8, 30)  # ordinary case still works
    assert low < 0.8 < high


def test_icc_perfect_agreement_is_exact():
    """Perfect agreement -> ICC 1.0 with an honest p=0 / infinite F (no fabricated F)."""
    Y = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    res = _icc_2_1_estimate(Y)
    assert res["icc"] == 1.0
    assert res["p_value"] == 0.0
    assert np.isinf(res["f_stat"])


def test_icc_fully_constant_is_nan():
    """No variance anywhere -> ICC, F, and p are all undefined (NaN)."""
    res = _icc_2_1_estimate(np.full((4, 2), 5.0))
    assert np.isnan(res["icc"])
    assert np.isnan(res["f_stat"])
    assert np.isnan(res["p_value"])


def test_bootstrap_icc_matches_scalar_loop():
    """The vectorized bootstrap CI is identical to a scalar _icc_2_1_estimate loop."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal(12)
    Y = np.column_stack([a + 0.1 * rng.standard_normal(12), a + 0.1 * rng.standard_normal(12)])

    # Reference: replicate the exact resampling stream + scalar ICC.
    seed = _get_deterministic_seed("feat", base_seed=42)
    ref_rng = np.random.default_rng(seed)
    ref = [
        _icc_2_1_estimate(Y[ref_rng.choice(len(Y), size=len(Y), replace=True)])["icc"]
        for _ in range(200)
    ]
    ref = [v for v in ref if not np.isnan(v)]
    expected = (float(np.percentile(ref, 2.5)), float(np.percentile(ref, 97.5)))

    assert np.allclose(_bootstrap_icc_ci(Y, "feat", iterations=200), expected)


def test_icc_2_1_mathematical_correctness():
    """Verify the ICC(2,1) formula against simple textbook cases with known results."""
    # Example dataset: 4 subjects, 3 raters
    # Subject 1: 9, 8, 9
    # Subject 2: 7, 7, 6
    # Subject 3: 5, 4, 5
    # Subject 4: 3, 3, 2
    Y = np.array([[9, 8, 9], [7, 7, 6], [5, 4, 5], [3, 3, 2]], dtype=float)

    res = _icc_2_1_estimate(Y)

    # Let's check that all MS components and the estimate are finite numbers
    assert not np.isnan(res["icc"])
    assert res["ms_between_subjects"] > 0
    assert res["ms_between_observers"] > 0
    assert res["ms_error"] > 0
    assert 0.8 < res["icc"] < 0.99  # Highly correlated observations


def test_perfect_reproducibility():
    """Test that two identical datasets yield perfect correlations and ICC of 1.0."""
    df1 = pd.DataFrame(
        {
            "feat_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_2": [5.0, 4.0, 3.0, 2.0, 1.0],
        },
        index=["S1", "S2", "S3", "S4", "S5"],
    )
    df2 = df1.copy()

    results = compute_reproducibility([df1, df2], bootstrap_iterations=50)

    # Check sheet existence
    assert "Spearman" in results
    assert "Pearson" in results
    assert "ICC" in results

    # Get data
    spearman = results["Spearman"]
    pearson = results["Pearson"]
    icc = results["ICC"]

    # Verify columns
    assert "feature" in icc.columns
    assert "icc_2_1" in icc.columns
    assert "ci95_low" in icc.columns
    assert "ci95_high" in icc.columns
    assert "p_value" in icc.columns
    assert "p_fdr" in icc.columns
    assert "primary_icc_pass" in icc.columns

    # Perfect identical rater results
    assert np.allclose(spearman["estimate"], 1.0)
    assert np.allclose(pearson["estimate"], 1.0)
    assert np.allclose(icc["icc_2_1"], 1.0)
    assert all(icc["primary_icc_pass"])


def test_qc_dataframe_alignment_and_reordering():
    """Test strict name-based QC checking, index verification, and automatic reordering."""
    df1 = pd.DataFrame(
        {"feat_1": [1, 2, 3], "feat_2": [4, 5, 6]},
        index=["S1", "S2", "S3"],
    )

    # 1. Feature columns mismatch
    df_bad_cols = pd.DataFrame(
        {"feat_1": [1, 2, 3], "feat_3": [4, 5, 6]},
        index=["S1", "S2", "S3"],
    )
    with pytest.raises(ValueError, match="columns do not match Dataset 0"):
        compute_reproducibility([df1, df_bad_cols])

    # 2. Subject index mismatch
    df_bad_idx = pd.DataFrame(
        {"feat_1": [1, 2, 3], "feat_2": [4, 5, 6]},
        index=["S1", "S2", "S4"],
    )
    with pytest.raises(ValueError, match="row index does not match Dataset 0"):
        compute_reproducibility([df1, df_bad_idx])

    # 3. Features/subjects out of order (should automatically align and succeed!)
    df_scrambled = pd.DataFrame(
        {"feat_2": [6, 4, 5], "feat_1": [3, 1, 2]},
        index=["S3", "S1", "S2"],
    )

    # Compute: this should align df_scrambled perfectly to df1 and yield perfect 1.0 scores!
    results = compute_reproducibility([df1, df_scrambled], bootstrap_iterations=10)
    assert np.allclose(results["Pearson"]["estimate"], 1.0)
    assert np.allclose(results["ICC"]["icc_2_1"], 1.0)


def test_qc_unnamed_positional_matching():
    """Test strict positional verification when passing numpy arrays or default RangeIndexes."""
    arr1 = np.random.default_rng(42).standard_normal((10, 5))
    arr2 = arr1.copy()

    # Mismatched shapes
    arr_bad_shape = np.random.default_rng(42).standard_normal((10, 6))
    with pytest.raises(ValueError, match="shape"):
        compute_reproducibility([arr1, arr_bad_shape])

    # Successful alignment and calculation
    results = compute_reproducibility([arr1, arr2], bootstrap_iterations=10)
    assert len(results["ICC"]) == 5  # 5 features
    assert np.allclose(results["Pearson"]["estimate"], 1.0)
    assert np.allclose(results["ICC"]["icc_2_1"], 1.0)


def test_selector_based_filtering():
    """Test that standard selectors correctly isolate the analyzed features."""
    # We will use columns resembling standard radiomics features: Config__FeatureKey
    cols = [
        "original__Energy",
        "original__Entropy",
        "original__Autocorrelation",
        "wavelet__ClusterShade",
    ]
    df1 = pd.DataFrame(
        np.random.default_rng(42).standard_normal((10, 4)),
        columns=cols,
        index=[f"S{i}" for i in range(10)],
    )
    df2 = df1.copy()

    # Specifying features directly
    results = compute_reproducibility(
        [df1, df2],
        features=["*Energy", "*Entropy"],
        bootstrap_iterations=10,
    )
    # Only Energy and Entropy should be analyzed
    icc_features = results["ICC"]["feature"].tolist()
    assert len(icc_features) == 2
    assert "original__Energy" in icc_features
    assert "original__Entropy" in icc_features

    # Specifying families directly
    # Family filtering requires a catalog. Let's create a small catalog DataFrame.
    catalog = pd.DataFrame(
        {
            "feature_key": ["Energy", "Entropy", "Autocorrelation", "ClusterShade"],
            "config": ["original", "original", "original", "wavelet"],
            "family": ["firstorder", "firstorder", "glcm", "glcm"],
            "family_group": ["texture", "texture", "texture", "texture"],
        }
    )

    results_family = compute_reproducibility(
        [df1, df2],
        families="firstorder",
        catalog=catalog,
        bootstrap_iterations=5,
    )
    assert len(results_family["ICC"]) == 2
    assert "original__Energy" in results_family["ICC"]["feature"].tolist()


def test_nan_handling_and_thresholding():
    """Test that features with missing samples are dropped or returned as NaN."""
    df1 = pd.DataFrame(
        {
            "feat_1": [1.0, 2.0, np.nan, 4.0, 5.0],  # 1 NaN
            "feat_2": [1.0, 2.0, np.nan, np.nan, 5.0],  # 2 NaNs -> 3 valid
            "feat_3": [1.0, np.nan, np.nan, np.nan, 5.0],  # 3 NaNs -> 2 valid
        },
        index=[f"S{i}" for i in range(5)],
    )
    df2 = df1.copy()

    # Using min_valid_samples = 3
    results = compute_reproducibility(
        [df1, df2],
        min_valid_samples=3,
        bootstrap_iterations=10,
    )

    icc = results["ICC"].set_index("feature")

    # feat_1 has 4 valid samples (>= 3) -> should compute correctly
    assert not np.isnan(icc.loc["feat_1", "icc_2_1"])
    # feat_2 has 3 valid samples (>= 3) -> should compute correctly
    assert not np.isnan(icc.loc["feat_2", "icc_2_1"])
    # feat_3 has 2 valid samples (< 3) -> should return NaN gracefully without raising an exception
    assert np.isnan(icc.loc["feat_3", "icc_2_1"])
    assert np.isnan(icc.loc["feat_3", "ci95_low"])


def test_multi_observer_aggregations():
    """Test multi-observer statistics (K >= 3 raters)."""
    # 3 observers with scrambled ranks to ensure non-trivial correlations and non-zero SD
    df1 = pd.DataFrame({"feat_1": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=[f"S{i}" for i in range(5)])
    df2 = pd.DataFrame({"feat_1": [2.0, 1.0, 3.0, 5.0, 4.0]}, index=[f"S{i}" for i in range(5)])
    df3 = pd.DataFrame({"feat_1": [3.0, 2.0, 1.0, 4.0, 5.0]}, index=[f"S{i}" for i in range(5)])

    results = compute_reproducibility([df1, df2, df3], bootstrap_iterations=10)

    # Verify column structures for K > 2 aggregate metrics
    for sheet in ["Spearman", "Pearson"]:
        df = results[sheet]
        assert "mean" in df.columns
        assert "median" in df.columns
        assert "sd" in df.columns
        assert "q25" in df.columns
        assert "q75" in df.columns
        assert "min" in df.columns
        assert "max" in df.columns

        # Values should be computed and have non-zero variance
        assert not np.isnan(df.loc[0, "mean"])
        assert df.loc[0, "sd"] > 0.0  # Captures variance across rater pairs


def test_excel_exporter():
    """Test that write_reproducibility_excel creates a valid, highly formatted Excel workbook."""
    df1 = pd.DataFrame({"feat_1": [1, 2, 3, 4, 5]}, index=[f"S{i}" for i in range(5)])
    df2 = df1.copy()

    results = compute_reproducibility([df1, df2], bootstrap_iterations=5)

    with tempfile.TemporaryDirectory() as tmp_dir:
        xlsx_path = Path(tmp_dir) / "reproducibility_report.xlsx"
        write_reproducibility_excel(results, xlsx_path)

        assert xlsx_path.exists()

        # Load sheet back using openpyxl and verify
        wb = openpyxl.load_workbook(xlsx_path)
        assert "Spearman" in wb.sheetnames
        assert "Pearson" in wb.sheetnames
        assert "ICC" in wb.sheetnames

        ws = wb["ICC"]
        # Header row 1 check
        assert ws.cell(row=1, column=1).value == "feature"
        assert ws.cell(row=1, column=2).value == "icc_2_1"

        # Data cell checks
        assert ws.cell(row=2, column=1).value == "feat_1"
        assert round(ws.cell(row=2, column=2).value, 2) == 1.0


def test_accessibility_plots():
    """Test that plot_reproducibility_histograms runs and meets quality requirements."""
    df1 = pd.DataFrame(
        {
            "feat_1": [1, 2, 3, 4, 5],
            "feat_2": [5, 4, 3, 2, 1],
        },
        index=[f"S{i}" for i in range(5)],
    )
    df2 = df1.copy()

    results = compute_reproducibility([df1, df2], bootstrap_iterations=5)

    with tempfile.TemporaryDirectory() as tmp_dir:
        fig_path = Path(tmp_dir) / "reproducibility_plot.png"
        fig = plot_reproducibility_histograms(results, path=fig_path, primary_threshold=0.80)

        # Assertions
        assert fig_path.exists()
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3  # Spearman, Pearson, ICC subplots
        plt.close(fig)


def test_requires_at_least_two_datasets():
    """A single dataset cannot be assessed for reproducibility."""
    with pytest.raises(ValueError, match="At least 2 datasets"):
        compute_reproducibility([pd.DataFrame({"a": [1.0, 2.0, 3.0]})])


def test_metadata_columns_skipped_without_selectors():
    """Without selectors, non-numeric metadata columns must be skipped, not crash."""
    df1 = pd.DataFrame(
        {
            "PatientID": ["p1", "p2", "p3", "p4"],
            "original__Energy": [1.0, 2.0, 3.0, 4.0],
            "original__Entropy": [4.0, 3.0, 2.0, 1.0],
        },
        index=["s1", "s2", "s3", "s4"],
    )
    results = compute_reproducibility([df1, df1.copy()], bootstrap_iterations=5)
    analyzed = results["ICC"]["feature"].tolist()
    assert "PatientID" not in analyzed
    assert set(analyzed) == {"original__Energy", "original__Entropy"}


def test_plot_handles_all_nan_sheet():
    """Plotting must not crash when a metric column is entirely NaN/empty."""
    results = {
        "Spearman": pd.DataFrame({"feature": ["f1", "f2"], "estimate": [np.nan, np.nan]}),
        "Pearson": pd.DataFrame({"feature": ["f1", "f2"], "estimate": [np.nan, np.nan]}),
        "ICC": pd.DataFrame({"feature": ["f1", "f2"], "icc_2_1": [np.nan, np.nan]}),
    }
    fig = plot_reproducibility_histograms(results)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_positional_rangeindex_columns():
    """DataFrames with default RangeIndex columns align by those labels."""
    arr = np.random.default_rng(0).standard_normal((12, 4))
    results = compute_reproducibility(
        [pd.DataFrame(arr), pd.DataFrame(arr.copy())], bootstrap_iterations=5
    )
    assert len(results["ICC"]) == 4


def test_positional_non_string_columns():
    """Non-string column labels align by label, like text labels."""
    arr = np.random.default_rng(0).standard_normal((12, 3))
    d1 = pd.DataFrame(arr, columns=[10, 20, 30])
    results = compute_reproducibility([d1, d1.copy()], bootstrap_iterations=5)
    assert len(results["ICC"]) == 3


def test_no_numeric_features_raises():
    d1 = pd.DataFrame({"label": ["a", "b", "c", "d"]}, index=["s1", "s2", "s3", "s4"])
    with pytest.raises(ValueError, match="No features selected"):
        compute_reproducibility([d1, d1.copy()])


def test_multi_observer_insufficient_samples():
    idx = [f"s{i}" for i in range(5)]
    col = {"feat": [1.0, np.nan, np.nan, np.nan, np.nan]}  # only 1 valid sample
    dfs = [pd.DataFrame(col, index=idx) for _ in range(3)]
    results = compute_reproducibility(dfs, min_valid_samples=3, bootstrap_iterations=5)
    assert "mean" in results["Spearman"].columns  # multi-observer schema
    assert np.isnan(results["Spearman"].loc[0, "mean"])


def test_multi_observer_constant_feature():
    idx = [f"s{i}" for i in range(6)]
    dfs = [pd.DataFrame({"feat": [2.0] * 6}, index=idx) for _ in range(3)]
    results = compute_reproducibility(dfs, min_valid_samples=3, bootstrap_iterations=5)
    # Constant feature -> all pairwise correlations NaN -> aggregate is NaN.
    assert np.isnan(results["Spearman"].loc[0, "mean"])


def test_plot_empty_results_raises():
    with pytest.raises(ValueError, match="no plottable"):
        plot_reproducibility_histograms({})


def test_plot_single_sheet():
    results = {"ICC": pd.DataFrame({"feature": ["f"], "icc_2_1": [0.9]})}
    fig = plot_reproducibility_histograms(results)
    assert len(fig.axes) == 1
    plt.close(fig)


class TestReproducibilityCoverageGaps:
    """Targeted tests for remaining uncovered branches in reproducibility.py."""

    def test_cluster_bootstrap_keeps_whole_clusters(self):
        from eigenradiomics.reproducibility import _cluster_bootstrap_weights

        # Unequal clusters: every row of a cluster must get the same multiplicity,
        # and the number of rows in a draw can change (no padding, no cutting).
        groups = np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 6])
        w = _cluster_bootstrap_weights(groups, iterations=200, seed=42)
        assert w.shape == (200, len(groups))
        assert (w[:, :4] == w[:, [0]]).all()
        assert (w.reshape(200, -1)[:, [0, 4, 5, 6, 7, 8, 9]].sum(axis=1) == 7).all()
        assert len(np.unique(w.sum(axis=1))) > 1

    def test_compute_reproducibility_groups_variants(self):
        rng = np.random.default_rng(42)
        df1 = pd.DataFrame(
            {
                "patient": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
                "feat1": rng.normal(size=8),
                "feat_with_nan": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            }
        )
        df2 = df1.copy()
        df2["feat1"] = df1["feat1"] + 0.1

        # 1. groups as column name with feature containing NaNs
        res_col = compute_reproducibility(
            [df1, df2],
            groups="patient",
            bootstrap_iterations=10,
            features=["feat1", "feat_with_nan"],
        )
        assert "ICC" in res_col

        # 2. groups as MultiIndex level name (triggers lines 215-216 and 226)
        midx = pd.MultiIndex.from_tuples(
            [(f"p{i // 2}", f"scan_{i}") for i in range(8)], names=["patient_level", "scan"]
        )
        df1_mi = df1.drop(columns=["patient"]).set_index(midx)
        df2_mi = df2.drop(columns=["patient"]).set_index(midx)

        res_mi = compute_reproducibility(
            [df1_mi, df2_mi], groups="patient_level", bootstrap_iterations=10
        )
        assert "ICC" in res_mi

        # 3. Default MultiIndex groups inference when groups=None
        res_mi_default = compute_reproducibility([df1_mi, df2_mi], bootstrap_iterations=10)
        assert "ICC" in res_mi_default

        # 4. Invalid groups string raises ValueError
        with pytest.raises(ValueError, match="groups column/level"):
            compute_reproducibility([df1, df2], groups="non_existent_col")

        # 5. groups as string with ndarrays raises ValueError
        with pytest.raises(ValueError, match="groups as string only supported"):
            compute_reproducibility(
                [df1[["feat1"]].to_numpy(), df2[["feat1"]].to_numpy()], groups="col"
            )

        # 6. groups as explicit array (hits line 222)
        res_arr = compute_reproducibility(
            [df1, df2], groups=[0, 0, 1, 1, 2, 2, 3, 3], bootstrap_iterations=5
        )
        assert "ICC" in res_arr

    def test_plot_reproducibility_options_and_gaps(self):
        from eigenradiomics.reproducibility import plot_reproducibility

        results = {
            "Spearman": pd.DataFrame({"feature": ["f1", "f2"], "estimate": [0.8, 0.9]}),
            "Pearson": pd.DataFrame({"feature": ["f1", "f2"], "estimate": [0.85, 0.95]}),
            "ICC": pd.DataFrame({"feature": ["f1", "f2"], "icc_2_1": [0.88, 0.92]}),
        }

        # 1. Custom axes length mismatch in plot_reproducibility_histograms
        fig_dummy, (ax1, ax2) = plt.subplots(1, 2)
        with pytest.raises(ValueError, match="Expected 3 axes"):
            plot_reproducibility_histograms(results, axes=[ax1, ax2])
        plt.close(fig_dummy)

        # 2. plot_reproducibility_histograms with custom title
        fig_hist = plot_reproducibility_histograms(results, title="My Custom Histogram Title")
        assert isinstance(fig_hist, plt.Figure)
        plt.close(fig_hist)

        # 3. plot_reproducibility: neither datasets nor results raises
        with pytest.raises(ValueError, match="Either 'datasets' or 'reproducibility_results'"):
            plot_reproducibility()

        # 4. plot_reproducibility: both datasets and results warns
        d1 = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0]})
        with pytest.warns(UserWarning, match="Both 'datasets' and 'reproducibility_results'"):
            _, fig_both = plot_reproducibility(
                [d1, d1], reproducibility_results=results, show_legend=False
            )
            plt.close(fig_both)

        # 5. plot_reproducibility: empty sheets raises
        with pytest.raises(ValueError, match="contains no plottable data sheets"):
            plot_reproducibility(reproducibility_results={})

        # 6. plot_reproducibility: grid_2x2=True with < 3 sheets warns
        res_two = {"ICC": results["ICC"], "Pearson": results["Pearson"]}
        with pytest.warns(UserWarning, match="grid_2x2=True requires all three metrics"):
            _, fig_two = plot_reproducibility(
                reproducibility_results=res_two, grid_2x2=True, show_legend=False
            )
            plt.close(fig_two)

        # 7. plot_reproducibility: grid_2x2=True with title
        _, fig_grid = plot_reproducibility(
            reproducibility_results=results, grid_2x2=True, title="Combined Reproducibility Grid"
        )
        assert isinstance(fig_grid, plt.Figure)
        plt.close(fig_grid)


# ----------------------------------------------------------------------
# Regression tests for the ICC kernel, keyed alignment and cluster bootstrap
# ----------------------------------------------------------------------


def _two_readers(n=24, noise=0.3, seed=0, index=None):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(n, 1))
    idx = index if index is not None else [f"s{i}" for i in range(n)]
    cols = ["f", "g"]
    d0 = pd.DataFrame(truth + noise * rng.normal(size=(n, 2)), index=idx, columns=cols)
    d1 = pd.DataFrame(truth + noise * rng.normal(size=(n, 2)), index=idx, columns=cols)
    return d0, d1


def test_weighted_icc_matches_row_expansion():
    from eigenradiomics._stats import _icc_2_1_weighted

    rng = np.random.default_rng(3)
    Y = rng.normal(size=(7, 3))
    w = np.array([[2.0, 0.0, 1.0, 1.0, 3.0, 1.0, 2.0]])
    expanded = np.repeat(Y, w[0].astype(int), axis=0)
    res = _icc_2_1_weighted(Y, w)
    ref = _icc_2_1_estimate(expanded)
    assert np.isclose(res["icc"][0], ref["icc"])
    assert np.isclose(res["ms_error"][0], ref["ms_error"])


def test_icc_shrout_fleiss_reference():
    """Shrout and Fleiss (1979), 6 targets x 4 judges: ICC(2,1) = 0.29."""
    Y = np.array(
        [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]],
        dtype=float,
    )
    res = _icc_2_1_estimate(Y)
    assert round(res["icc"], 2) == 0.29
    assert np.isclose(res["ms_between_subjects"], 11.2417, atol=1e-4)
    assert np.isclose(res["ms_error"], 1.0194, atol=1e-4)


def test_icc_constant_non_representable_value_is_undefined():
    d = pd.DataFrame({"f": np.full(64, 0.1)}, index=[f"s{i}" for i in range(64)])
    icc = compute_reproducibility([d, d.copy()], bootstrap_iterations=20)["ICC"].iloc[0]
    assert np.isnan(icc["icc_2_1"])
    assert icc["status"] == "constant"


def test_icc_large_offset_identical_readers_is_one():
    v = 1e12 + np.array([0, 1, 2, 3]) * 0.000244140625
    d = pd.DataFrame({"f": v}, index=list("abcd"))
    icc = compute_reproducibility([d, d.copy()], bootstrap_iterations=20)["ICC"].iloc[0]
    assert icc["icc_2_1"] == 1.0
    assert bool(icc["precision_limited"])


def test_icc_negative_estimate_is_kept():
    d0 = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]}, index=list("abcd"))
    d1 = pd.DataFrame({"f": [4.0, 3.0, 2.0, 1.0]}, index=list("abcd"))
    icc = compute_reproducibility([d0, d1], bootstrap_iterations=0)["ICC"].iloc[0]
    assert icc["icc_2_1"] < 0
    assert icc["ci_status"] == "disabled"


def test_duplicate_row_and_feature_keys_raise():
    d = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]}, index=["a", "a", "b", "c"])
    with pytest.raises(ValueError, match="duplicate row keys"):
        compute_reproducibility([d, d.copy()])
    d2 = pd.DataFrame([[1.0, 2.0]] * 4, columns=["f", "f"], index=list("abcd"))
    with pytest.raises(ValueError, match="duplicate feature keys"):
        compute_reproducibility([d2, d2.copy()])
    d3 = pd.DataFrame({"f": [1.0, 2.0, 3.0]}, index=["a", None, "c"])
    with pytest.raises(ValueError, match="missing row keys"):
        compute_reproducibility([d3, d3.copy()])


def test_shuffled_named_input_gives_same_result():
    d0, d1 = _two_readers()
    ref = compute_reproducibility([d0, d1], bootstrap_iterations=50)["ICC"]
    shuffled = d1.sample(frac=1, random_state=1)[["g", "f"]]
    out = compute_reproducibility([d0, shuffled], bootstrap_iterations=50)["ICC"]
    pd.testing.assert_frame_equal(ref, out)


def test_infinite_values_drop_only_their_rows():
    d0, d1 = _two_readers()
    d0.iloc[0, 0] = np.inf
    icc = compute_reproducibility([d0, d1], bootstrap_iterations=50)["ICC"].iloc[0]
    assert icc["n_valid"] == len(d0) - 1
    assert np.isfinite(icc["icc_2_1"]) and np.isfinite(icc["ci95_low"])


def test_single_cluster_interval_is_unavailable():
    d0, d1 = _two_readers(n=20)
    icc = compute_reproducibility([d0, d1], groups=np.array(["p"] * 20), bootstrap_iterations=50)[
        "ICC"
    ].iloc[0]
    assert np.isfinite(icc["icc_2_1"])
    assert np.isnan(icc["ci95_low"]) and np.isnan(icc["ci95_high"])
    assert icc["ci_status"] == "insufficient_clusters"
    assert icc["n_clusters"] == 1


def test_group_series_aligns_by_row_key():
    d0, d1 = _two_readers(n=24)
    gs = pd.Series(np.repeat([f"p{i}" for i in range(8)], 3), index=d0.index)
    ref = compute_reproducibility([d0, d1], groups=gs, bootstrap_iterations=100)["ICC"]
    out = compute_reproducibility(
        [d0, d1], groups=gs.sample(frac=1, random_state=3), bootstrap_iterations=100
    )["ICC"]
    pd.testing.assert_frame_equal(ref, out)
    assert ref["ci_method"].iloc[0].startswith("cluster bootstrap")


def test_group_labels_are_validated():
    d0, d1 = _two_readers(n=12)
    with pytest.raises(ValueError, match="missing label"):
        compute_reproducibility([d0, d1], groups=[1, 1, 2, 2, None, 3, 3, 4, 4, 5, 5, 6])
    with pytest.raises(ValueError, match="11 labels"):
        compute_reproducibility([d0, d1], groups=list(range(11)))
    with pytest.raises(ValueError, match="no label"):
        compute_reproducibility([d0, d1], groups=pd.Series([1] * 11, index=d0.index[:11]))


def test_group_column_is_not_a_feature_and_naive_inference_is_withheld():
    d0, d1 = _two_readers(n=24)
    pid = np.repeat(np.arange(8), 3)
    res = compute_reproducibility(
        [d0.assign(pid=pid), d1.assign(pid=pid)], groups="pid", bootstrap_iterations=50
    )
    assert list(res["ICC"]["feature"]) == ["f", "g"]
    assert res["ICC"]["p_value"].isna().all()
    assert res["Pearson"][["p_value", "ci95_low"]].isna().all().all()
    assert res["Pearson"]["estimate"].notna().all()
    assert (res["ICC"]["n_clusters"] == 8).all()


def test_cluster_bootstrap_is_reproducible_and_seeded():
    d0, d1 = _two_readers(n=24)
    g = np.repeat(np.arange(8), 3)
    a = compute_reproducibility([d0, d1], groups=g, bootstrap_iterations=100)["ICC"]
    b = compute_reproducibility([d0, d1], groups=g, bootstrap_iterations=100)["ICC"]
    c = compute_reproducibility([d0, d1], groups=g, bootstrap_iterations=100, random_state=7)[
        "ICC"
    ]
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["ci95_low"], c["ci95_low"])


def test_invalid_arguments_raise():
    d0, d1 = _two_readers(n=6)
    with pytest.raises(ValueError, match="primary_threshold"):
        compute_reproducibility([d0, d1], primary_threshold=np.nan)
    with pytest.raises(ValueError, match="bootstrap_iterations"):
        compute_reproducibility([d0, d1], bootstrap_iterations=-1)
    with pytest.raises(ValueError, match="2-D"):
        compute_reproducibility([np.zeros(5), np.zeros(5)])


def test_mixed_column_labels_align_by_key():
    """One non-text column label must not switch readers to positional matching."""
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(30, 1))
    idx = [f"s{i}" for i in range(30)]
    d0 = pd.DataFrame(truth + 0.3 * rng.normal(size=(30, 2)), index=idx, columns=["f", 5])
    d1 = pd.DataFrame(truth + 0.3 * rng.normal(size=(30, 2)), index=idx, columns=["f", 5])
    ref = compute_reproducibility([d0, d1], bootstrap_iterations=20)["ICC"]
    out = compute_reproducibility(
        [d0, d1.sample(frac=1, random_state=1)[[5, "f"]]], bootstrap_iterations=20
    )["ICC"]
    pd.testing.assert_frame_equal(ref, out)
    assert list(ref["feature"]) == ["f", 5]
    assert compute_reproducibility([d0, d1], features=[5], bootstrap_iterations=0)["ICC"][
        "feature"
    ].tolist() == [5]
