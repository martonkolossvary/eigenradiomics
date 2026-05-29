"""Unit tests for the batch effect diagnostics framework."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import pytest

from eigenradiomics.batch_effects import (
    anova_effect,
    compute_batch_effects,
    kruskal_effect,
    levene_test,
    permanova_euclidean,
    plot_batch_effects,
    write_batch_effects_excel,
)
from eigenradiomics.preprocessing import RadiomicsPrepTransformer


def test_radiomics_prep_transformer():
    """Verify that RadiomicsPrepTransformer clips, transforms, scales, and carries NaNs."""
    df = pd.DataFrame(
        {
            "feat_1": [1.0, 2.0, np.nan, 4.0, 100.0],  # Outlier 100.0
            "feat_2": [10.0, 10.0, 10.0, 10.0, 10.0],  # Constant column
        }
    )

    transformer = RadiomicsPrepTransformer(
        winsor_lower=0.1, winsor_upper=0.9, skip_yeo_johnson=False
    )
    transformer.fit(df)

    # 1. Inspect fitted attributes
    assert transformer.n_features_in_ == 2
    assert len(transformer.winsor_bounds_) == 2

    # Check winsor bounds (1st feature winsorized at 10% and 90% -> roughly 1.3 and 70.8)
    low, high = transformer.winsor_bounds_[0]
    assert low < 2.0
    assert high < 100.0

    # Constant column power transformer should be None to avoid errors
    assert transformer.power_transformers_[1] is None

    # 2. Inspect transform output
    df_trans = transformer.transform(df)
    assert isinstance(df_trans, pd.DataFrame)
    assert np.isnan(df_trans.loc[2, "feat_1"])  # NaN must be preserved!
    assert df_trans.loc[4, "feat_1"] < 2.0  # Clipped/winsorized outlier should be small

    # Constant column should be returned as raw or zero-centered safely
    assert np.allclose(df_trans["feat_2"].dropna(), 10.0)


def test_batch_effects_anova_kruskal_levene():
    """Verify the feature-level ANOVA, Kruskal-Wallis, and Levene calculators."""
    g1 = np.array([1.0, 2.0, 3.0])
    g2 = np.array([4.0, 5.0, 6.0])

    f, p, eta2 = anova_effect([g1, g2])
    assert f > 0
    assert 0 < p < 1
    assert 0.5 < eta2 <= 1.0

    h, p_k, eps2 = kruskal_effect([g1, g2])
    assert h > 0
    assert 0 < p_k < 1
    assert eps2 > 0

    l_stat, p_l = levene_test([g1, g2])
    assert not np.isnan(l_stat)
    assert 0 < p_l <= 1.0


def test_permanova_pseudo_f():
    """Verify the PERMANOVA pseudo-F permutation calculation."""
    df = pd.DataFrame(
        {
            "PC1": [1.0, 1.1, 1.2, 5.0, 5.1, 5.2],
            "PC2": [2.0, 2.1, 2.2, 8.0, 8.1, 8.2],
        }
    )
    batch = pd.Series(["A", "A", "A", "B", "B", "B"])

    f, r2, p = permanova_euclidean(df, batch, permutations=99, random_state=42)
    assert f > 5.0
    assert r2 > 0.8
    assert 0 < p <= 0.1  # Significant separation


def test_compute_batch_effects_qc_and_alignment():
    """Test strict index QC alignment, reordering, and shape consistency validations."""
    X = pd.DataFrame(
        {"feat_1": [1, 2, 3], "feat_2": [4, 5, 6]},
        index=["S1", "S2", "S3"],
    )
    pd.Series(["A", "B", "A"], index=["S1", "S2", "S3"])

    # 1. Index mismatch should raise ValueError
    batch_bad = pd.Series(["A", "B", "A"], index=["S1", "S2", "S4"])
    with pytest.raises(ValueError, match="Indices of X and batch do not match"):
        compute_batch_effects(X, batch_bad)

    # 2. Scrambled index should automatically reorder and succeed
    batch_scrambled = pd.Series(["B", "A", "A"], index=["S2", "S1", "S3"])
    results = compute_batch_effects(
        X,
        batch_scrambled,
        permutations=10,
        min_valid_samples=3,
        min_valid_per_batch=1,
        min_batches_per_feature=2,
        no_combat=True,
    )
    assert "feature_stats" in results

    # 3. Shape mismatch on arrays
    arr = X.to_numpy()
    batch_arr_bad = np.array(["A", "B"])
    with pytest.raises(ValueError, match="Length of batch"):
        compute_batch_effects(arr, batch_arr_bad)


def test_compute_batch_effects_selectors_and_combat():
    """Test feature selector filtering, custom preprocessors, and ComBat sensitivity."""
    # Build standard wide radiomics features columns config__feature_key
    cols = [
        "original__Energy",
        "original__Entropy",
        "original__Autocorrelation",
        "wavelet__ClusterShade",
    ]
    df = pd.DataFrame(
        np.random.default_rng(42).standard_normal((30, 4)),
        columns=cols,
        index=[f"S{i}" for i in range(30)],
    )
    # 3 batches
    batch = pd.Series(["A", "B", "C"] * 10, index=df.index)

    # Pre-select feature catalog
    catalog = pd.DataFrame(
        {
            "feature_key": ["Energy", "Entropy", "Autocorrelation", "ClusterShade"],
            "config": ["original", "original", "original", "wavelet"],
            "family": ["firstorder", "firstorder", "glcm", "glcm"],
            "family_group": ["texture", "texture", "texture", "texture"],
        }
    )

    # Execute with selectors and combat sensitivity
    results = compute_batch_effects(
        df,
        batch,
        features=["*Energy", "*Entropy"],
        catalog=catalog,
        permutations=10,
        no_combat=False,
    )

    # Check results
    assert "dataset_summary" in results
    assert "batch_counts" in results
    assert "global_diagnostics" in results
    assert "feature_stats" in results

    # Only Energy and Entropy should be analyzed
    feature_stats = results["feature_stats"]
    assert len(feature_stats) == 2
    assert "original__Energy" in feature_stats["feature"].tolist()
    assert "original__Entropy" in feature_stats["feature"].tolist()

    # ComBat sensitivity sheets should be present (inmoose is installed in the env)
    assert "combat_feature_stats" in results
    assert "combat_adjustment_notes" in results


def test_excel_and_plot_batch_effects():
    """Test Excel writing and high-contrast accessibility figures."""
    cols = ["original__Energy", "original__Entropy"]
    df = pd.DataFrame(
        np.random.default_rng(42).standard_normal((30, 2)),
        columns=cols,
        index=[f"S{i}" for i in range(30)],
    )
    batch = pd.Series(["A", "B", "C"] * 10, index=df.index)

    results = compute_batch_effects(df, batch, permutations=10, no_combat=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Excel Workbook export check
        xlsx_path = Path(tmp_dir) / "batch_report.xlsx"
        write_batch_effects_excel(results, xlsx_path)

        assert xlsx_path.exists()
        wb = openpyxl.load_workbook(xlsx_path)
        assert "dataset_summary" in wb.sheetnames
        assert "global_diagnostics" in wb.sheetnames
        assert "feature_stats" in wb.sheetnames

        ws = wb["feature_stats"]
        assert ws.cell(row=1, column=1).value == "feature"
        assert ws.cell(row=2, column=1).value == "original__Energy"

        # 2. Plotting accessibility visuals check
        fig_path = Path(tmp_dir) / "batch_effect_plot.png"
        fig = plot_batch_effects(results, path=fig_path, primary_alpha=0.05)

        assert fig_path.exists()
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3  # PCA, PCA ComBat, Histogram panels
        plt.close(fig)


def test_batch_effects_skips_non_numeric_metadata_without_selectors():
    """Without selectors, non-numeric metadata columns must be skipped, not crash."""
    X = pd.DataFrame(
        {
            "PatientID": [f"p{i}" for i in range(6)],
            "original__Energy": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "original__Entropy": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    batch = pd.Series(["A", "B"] * 3, index=X.index)

    results = compute_batch_effects(
        X,
        batch,
        permutations=5,
        no_combat=True,
        min_valid_samples=3,
        min_valid_per_batch=1,
        min_batches_per_feature=1,
    )
    analyzed = results["feature_stats"]["feature"].tolist()
    assert "PatientID" not in analyzed
    assert set(analyzed) == {"original__Energy", "original__Entropy"}
