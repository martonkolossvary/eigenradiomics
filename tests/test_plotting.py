"""Tests for plot_clustered_heatmap."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy.cluster.hierarchy import leaves_list, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

from eigenradiomics import (  # noqa: E402
    Bar,
    CorrPanel,
    ReductionArtifacts,
    Strip,
    plot_clustered_heatmap,
)


def _values(n: int) -> np.ndarray:
    """Deterministic numeric values for bottom-bar tests."""
    return np.linspace(0.0, 3.0, n)


def _blocks(n_per: int = 10, n_blocks: int = 3):
    """Block-structured similarity matrix, its linkage, and per-feature labels."""
    n = n_per * n_blocks
    sim = np.full((n, n), 0.05)
    labels = []
    for b in range(n_blocks):
        start = b * n_per
        sim[start : start + n_per, start : start + n_per] = 0.85
        labels += [f"block{b}"] * n_per
    np.fill_diagonal(sim, 1.0)
    sim = (sim + sim.T) / 2
    names = [f"f{i}" for i in range(n)]
    df = pd.DataFrame(sim, index=names, columns=names)
    z = linkage(squareform(1.0 - sim, checks=False), method="average")
    return df, z, pd.Series(labels, index=names)


# ---- core (MVP) behaviour ------------------------------------------------


def test_raw_array_minimal():
    df, _, _ = _blocks()
    fig = plot_clustered_heatmap(df.to_numpy())  # identity order, no dendro/strip/legend
    assert len(fig.axes) == 2  # heatmap + colorbar
    plt.close(fig)


def test_dataframe_with_linkage_and_labels():
    df, z, lab = _blocks()
    fig = plot_clustered_heatmap(
        df,
        linkage=z,
        cluster_labels=list(lab),
        cluster_colors={"block0": "#d62728", "block1": "#2ca02c", "block2": "#1f77b4"},
        below_cutoff_color="black",
        vmin=0.0,
        vmax=1.0,
        labels=list(df.index),
        title="t",
    )
    assert len(fig.axes) == 5  # heat + cbar + dendro + cluster strip + Module legend
    plt.close(fig)


def test_from_artifacts():
    df, z, lab = _blocks()
    order = [df.index[i] for i in leaves_list(z)]
    art = ReductionArtifacts(
        feature_names=np.asarray(df.index),
        similarity=df,
        linkage=z,
        cluster_labels=lab,
        feature_order=np.asarray(order),
    )
    fig = plot_clustered_heatmap(art)  # everything filled from the artifacts
    assert len(fig.axes) == 5
    plt.close(fig)


def test_order_from_cluster_labels_only():
    df, _, lab = _blocks()
    fig = plot_clustered_heatmap(df, cluster_labels=lab)  # Series, no linkage/order
    assert len(fig.axes) == 4  # heat + cbar + cluster strip + Module legend
    plt.close(fig)


def test_labels_none_and_dendrogram_disabled():
    df, z, lab = _blocks()
    fig = plot_clustered_heatmap(
        df, linkage=z, cluster_labels=lab, labels=None, show_dendrogram=False
    )
    assert len(fig.axes) == 4  # heat + cbar + cluster strip + Module legend
    plt.close(fig)


def test_artifacts_without_similarity_raises():
    art = ReductionArtifacts(feature_names=np.asarray(["a", "b"]))
    with pytest.raises(ValueError, match="no .similarity. matrix"):
        plot_clustered_heatmap(art)


def test_non_square_raises():
    with pytest.raises(ValueError, match="square"):
        plot_clustered_heatmap(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="square"):
        plot_clustered_heatmap(pd.DataFrame(np.zeros((3, 4))))


def test_unknown_order_name_raises():
    df, _, _ = _blocks()
    with pytest.raises(ValueError, match="absent from the similarity index"):
        plot_clustered_heatmap(df, order=[*list(df.index)[:-1], "ghost"])


def test_all_nan_similarity_raises():
    df, _, _ = _blocks()
    df.loc[:, :] = np.nan
    with pytest.raises(ValueError, match="no finite values"):
        plot_clustered_heatmap(df)


# ---- top annotation strips + legends -------------------------------------


def test_top_strips_and_legends():
    df, z, lab = _blocks(n_per=10, n_blocks=3)
    names = list(df.index)
    families = ["Intensity", "Texture", "Morphology"]
    family = pd.Series([families[i % 3] for i in range(len(names))], index=names, name="Family")
    region = pd.Series(["total" if i % 2 else "calcium" for i in range(len(names))], index=names)
    fig = plot_clustered_heatmap(
        df,
        linkage=z,
        cluster_labels=lab,
        top=[family, Strip(region, title="Region")],  # Series (auto) + Strip (spec)
        title="cornerstone",
    )
    # heat + cbar + dendro + cluster strip + 2 top strips + 3 legend blocks
    assert len(fig.axes) == 9
    plt.close(fig)


def test_show_legend_false():
    df, _, lab = _blocks()
    names = list(df.index)
    family = pd.Series(["a", "b"] * (len(names) // 2), index=names, name="Family")
    fig = plot_clustered_heatmap(df, cluster_labels=lab, top=[family], show_legend=False)
    assert len(fig.axes) == 4  # heat + cbar + cluster strip + 1 top strip (no legends)
    plt.close(fig)


def test_strip_explicit_colors_and_missing_features():
    df, _, lab = _blocks(n_per=10, n_blocks=3)
    names = list(df.index)
    subset = names[:20]  # strip covers only some features -> rest get the fallback colour
    region = pd.Series(["total" if i % 2 else "calcium" for i in range(20)], index=subset)
    fig = plot_clustered_heatmap(
        df,
        cluster_labels=lab,
        top=[Strip(region, title="Region", colors={"total": "#0072B2", "calcium": "#D55E00"})],
    )
    assert len(fig.axes) == 6  # heat + cbar + cluster strip + 1 top + Module + Region legends
    plt.close(fig)


def test_many_categories_use_extended_palette():
    df, _, lab = _blocks(n_per=10, n_blocks=3)
    names = list(df.index)
    many = pd.Series([f"c{i % 9}" for i in range(len(names))], index=names, name="Many")  # > 8
    fig = plot_clustered_heatmap(df, cluster_labels=lab, top=[many])
    assert len(fig.axes) == 6
    plt.close(fig)


def test_strip_title_fallback():
    df, _, lab = _blocks()
    names = list(df.index)
    unnamed = pd.Series(["x", "y"] * (len(names) // 2), index=names)  # name is None
    fig = plot_clustered_heatmap(df, cluster_labels=lab, top=[unnamed])
    assert len(fig.axes) == 6
    plt.close(fig)


def test_invalid_top_type_raises():
    df, _, _ = _blocks()
    with pytest.raises(TypeError, match="Series or Strip"):
        plot_clustered_heatmap(df, top=[123])


# ---- bottom annotation bars ----------------------------------------------


def test_bottom_bars_by_module_with_reference_and_labels():
    df, z, lab = _blocks(n_per=10, n_blocks=3)
    names = list(df.index)
    family = pd.Series(["A", "B"] * (len(names) // 2), index=names, name="Family")
    pvals = pd.Series(_values(len(names)), index=names, name="-log10 p")  # Series -> Bar
    effect = Bar(pd.Series(_values(len(names)), index=names), title="Effect", reference=1.3)
    fig = plot_clustered_heatmap(
        df,
        linkage=z,
        cluster_labels=lab,  # by_module colouring + reference line
        top=[family],
        bottom=[pvals, effect],
        labels=list(df.index),  # x-tick labels move to the bottom-most bar
    )
    # heat + cbar + dendro + cluster strip + 1 top + 2 bottom + 2 legends (Module + Family)
    assert len(fig.axes) == 9
    plt.close(fig)


def test_bottom_bar_fixed_color_no_modules():
    df, _, _ = _blocks()
    names = list(df.index)
    bar = Bar(pd.Series(_values(len(names)), index=names), color="steelblue")  # title=None
    fig = plot_clustered_heatmap(df, bottom=[bar], labels=None)
    assert len(fig.axes) == 3  # heat + cbar + 1 bottom bar (no dendro/strip/legend)
    plt.close(fig)


def test_bottom_bar_by_module_fallback_when_no_modules():
    df, _, _ = _blocks()
    names = list(df.index)
    # default color="by_module" but no cluster_labels -> single fallback colour
    fig = plot_clustered_heatmap(df, bottom=[pd.Series(_values(len(names)), index=names)])
    assert len(fig.axes) == 3
    plt.close(fig)


def test_invalid_bottom_type_raises():
    df, _, _ = _blocks()
    with pytest.raises(TypeError, match="Series or Bar"):
        plot_clustered_heatmap(df, bottom=[123])


# ---- right correlation panel ---------------------------------------------


def _corr_frame(names, n_vars: int = 4) -> pd.DataFrame:
    """A features x variables correlation matrix in [-1, 1]."""
    grid = np.linspace(-0.9, 0.9, len(names) * n_vars).reshape(len(names), n_vars)
    return pd.DataFrame(grid, index=names, columns=[f"var{j}" for j in range(n_vars)])


def test_right_panel_minimal():
    df, _, _ = _blocks()
    fig = plot_clustered_heatmap(df, right=_corr_frame(list(df.index)))
    assert len(fig.axes) == 4  # heat + cbar + corr panel + corr colorbar
    plt.close(fig)


def test_right_panel_spec_custom_cmap():
    df, _, _ = _blocks()
    panel = CorrPanel(
        _corr_frame(list(df.index)), cmap="coolwarm", vmin=-0.5, vmax=0.5, label="Spearman r"
    )
    fig = plot_clustered_heatmap(df, right=panel)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_right_panel_full_cornerstone():
    df, z, lab = _blocks(n_per=10, n_blocks=3)
    names = list(df.index)
    family = pd.Series(["A", "B"] * (len(names) // 2), index=names, name="Family")
    pvals = pd.Series(_values(len(names)), index=names, name="-log10 p")
    fig = plot_clustered_heatmap(
        df,
        linkage=z,
        cluster_labels=lab,
        top=[family],
        bottom=[pvals],
        right=_corr_frame(names, n_vars=5),
        labels=names,
    )
    # heat + cbar + dendro + strip + 1 top + 1 bottom + corr + corr cbar + 2 legends
    assert len(fig.axes) == 10
    plt.close(fig)


def test_invalid_right_type_raises():
    df, _, _ = _blocks()
    with pytest.raises(TypeError, match="DataFrame or CorrPanel"):
        plot_clustered_heatmap(df, right=123)


def test_plot_reproducibility_synteny(tmp_path):
    from eigenradiomics.plotting import plot_reproducibility_synteny

    features = [f"f{i}" for i in range(10)]
    df_repro = pd.DataFrame(
        {
            "feature": features,
            "icc": [0.95, 0.45, 0.82, 0.73, 0.91, 0.33, 0.61, 0.88, 0.79, 0.52],
            "correlation": [0.96, 0.50, 0.85, 0.75, 0.92, 0.38, 0.65, 0.90, 0.82, 0.55],
        }
    )

    catalog = pd.DataFrame({"feature": features, "family": ["firstorder"] * 5 + ["glcm"] * 5})

    fig1 = plot_reproducibility_synteny(df_repro, catalog=catalog)
    assert fig1 is not None
    plt.close(fig1)

    fig2 = plot_reproducibility_synteny(
        df_repro,
        catalog=catalog,
        metric="correlation",
        order=features[::-1],
        title="Synteny test",
        path=tmp_path / "synteny.png",
    )
    assert fig2 is not None
    assert (tmp_path / "synteny.png").exists()
    plt.close(fig2)


def test_plot_reproducibility_synteny_coverage_gaps(tmp_path):
    from eigenradiomics.plotting import plot_reproducibility_synteny

    features = [f"f{i}" for i in range(10)]
    df_repro = pd.DataFrame(
        {
            "feature": features,
            "icc": [0.95, np.nan, 0.82, 0.73, 0.91, 0.33, 0.61, 0.88, 0.79, 0.52],
            "correlation": [0.96, 0.50, 0.85, 0.75, 0.92, 0.38, 0.65, 0.90, 0.82, 0.55],
        }
    )

    # 1. reproducibility_results as dict
    fig = plot_reproducibility_synteny({"ICC": df_repro})
    assert fig is not None
    plt.close(fig)

    fig = plot_reproducibility_synteny({"Spearman": df_repro})
    assert fig is not None
    plt.close(fig)

    fig = plot_reproducibility_synteny({"Pearson": df_repro}, metric="pearson")
    assert fig is not None
    plt.close(fig)

    fig = plot_reproducibility_synteny({"ICC": df_repro, "Pearson": df_repro}, metric="pearson")
    assert fig is not None
    plt.close(fig)

    fig = plot_reproducibility_synteny({"ICC": df_repro, "Spearman": df_repro}, metric="spearman")
    assert fig is not None
    plt.close(fig)

    # DataFrame with custom metric column
    df_custom = pd.DataFrame(
        {
            "feature": features,
            "custom_val": [0.9] * 10,
        }
    )
    fig = plot_reproducibility_synteny({"Pearson": df_custom}, metric="custom_val")
    assert fig is not None
    plt.close(fig)

    # 2. DataFrame without "icc" or "correlation" (inferred_metric = df.columns[1])
    fig = plot_reproducibility_synteny(df_custom)
    assert fig is not None
    plt.close(fig)

    # 2.b DataFrame with "correlation" only (to hit line 1004)
    df_corr_only = pd.DataFrame(
        {
            "feature": features,
            "correlation": [0.9] * 10,
        }
    )
    fig = plot_reproducibility_synteny(df_corr_only)
    assert fig is not None
    plt.close(fig)

    # 3. raise ValueError: metric not found
    with pytest.raises(ValueError, match="Metric column 'missing' not found"):
        plot_reproducibility_synteny(df_repro, metric="missing")

    # 4. raise ValueError: "feature" column not found
    df_no_feat = df_repro.drop(columns=["feature"])
    with pytest.raises(ValueError, match="must contain a 'feature' column"):
        plot_reproducibility_synteny(df_no_feat)

    # 5. group_by not in df columns (fallback to "All Features")
    fig = plot_reproducibility_synteny(df_repro, group_by="missing_family")
    assert fig is not None
    plt.close(fig)

    # 6. Empty dataframe after filtering (n == 0)
    empty_df = pd.DataFrame(columns=["feature", "icc"])
    with pytest.raises(ValueError, match="No features available to plot"):
        plot_reproducibility_synteny(empty_df)

    # 7. n > 50 features (to hit tick_labels is None, etc.)
    features_large = [f"f{i}" for i in range(55)]
    df_large = pd.DataFrame(
        {
            "feature": features_large,
            "icc": [0.9] * 55,
        }
    )
    fig = plot_reproducibility_synteny(df_large)
    assert fig is not None
    plt.close(fig)


def test_plot_reproducibility(tmp_path):
    import numpy as np
    import pandas as pd

    from eigenradiomics.reproducibility import plot_reproducibility

    # Create dummy replicate datasets
    features = [f"f{i}" for i in range(10)]
    df1 = pd.DataFrame(np.random.rand(10, 10), columns=features)
    df2 = pd.DataFrame(df1.values + 0.05 * np.random.rand(10, 10), columns=features)

    # 1. Run with datasets directly
    excel_file = tmp_path / "report.xlsx"
    plot_file = tmp_path / "combined_repro.png"
    csv_dir = tmp_path / "csv_out"

    results, fig = plot_reproducibility(
        datasets=[df1, df2],
        path=plot_file,
        excel_path=excel_file,
        csv_dir=csv_dir,
        compute_kws={"bootstrap_iterations": 5},
        save_pdf=True,
        save_tiff=True,
    )

    assert results is not None
    assert "ICC" in results
    assert "Spearman" in results
    assert "Pearson" in results
    assert fig is not None
    assert plot_file.exists()
    assert excel_file.exists()
    assert csv_dir.exists()
    assert (csv_dir / "ICC.csv").exists()
    assert (csv_dir / "Spearman.csv").exists()
    assert (csv_dir / "Pearson.csv").exists()

    plt.close(fig)

    # 2. Run with precomputed results
    fig2 = plot_reproducibility(
        reproducibility_results=results,
        path=tmp_path / "precomputed_repro.png",
        show_subplot_titles=False,
        title=None,
    )[1]
    assert fig2 is not None
    plt.close(fig2)


def test_plot_reproducibility_options(tmp_path):
    import numpy as np
    import pandas as pd

    from eigenradiomics.reproducibility import plot_reproducibility

    features = [f"f{i}" for i in range(10)]
    df1 = pd.DataFrame(np.random.rand(10, 10), columns=features)
    df2 = pd.DataFrame(df1.values + 0.05 * np.random.rand(10, 10), columns=features)

    # 1. Run 2x2 grid option with legends
    results, fig_grid = plot_reproducibility(
        datasets=[df1, df2],
        path=tmp_path / "grid_2x2_legend.png",
        compute_kws={"bootstrap_iterations": 5},
        grid_2x2=True,
        show_legend=True,
    )
    assert fig_grid is not None
    assert (tmp_path / "grid_2x2_legend.png").exists()
    plt.close(fig_grid)

    # 2. Run 2x2 grid option without legends
    results, fig_grid_noleg = plot_reproducibility(
        reproducibility_results=results,
        path=tmp_path / "grid_2x2_noleg.png",
        grid_2x2=True,
        show_legend=False,
    )
    assert fig_grid_noleg is not None
    assert (tmp_path / "grid_2x2_noleg.png").exists()
    plt.close(fig_grid_noleg)

    results, fig_noleg = plot_reproducibility(
        reproducibility_results=results,
        path=tmp_path / "default_noleg.png",
        grid_2x2=False,
        show_legend=False,
    )
    assert fig_noleg is not None
    assert (tmp_path / "default_noleg.png").exists()
    plt.close(fig_noleg)


class TestSyntenyPlottingCoverageGaps:
    """Targeted tests for remaining uncovered branches in plotting.py."""

    def test_synteny_metric_selection_and_inferences(self):
        from eigenradiomics.plotting import plot_reproducibility_synteny

        features = [f"f{i}" for i in range(10)]

        # 1. Dict with 'ICC' and 'CustomSheet'
        results_custom_sheet = {
            "CustomSheet": pd.DataFrame(
                {
                    "feature": features,
                    "estimate": [0.85] * 10,
                }
            )
        }
        fig1 = plot_reproducibility_synteny(results_custom_sheet, show_legend=False)
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        # 2. Dict with 'Spearman' having 'mean' column
        results_mean = {
            "Spearman": pd.DataFrame(
                {
                    "feature": features,
                    "mean": [0.80] * 10,
                }
            )
        }
        fig2 = plot_reproducibility_synteny(results_mean, metric="spearman")
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

        # 3. Direct DataFrame with 'estimate' and 'mean'
        df_est = pd.DataFrame({"feature": features, "estimate": [0.75] * 10})
        fig3 = plot_reproducibility_synteny(df_est)
        plt.close(fig3)

        df_mean = pd.DataFrame({"feature": features, "mean": [0.70] * 10})
        fig4 = plot_reproducibility_synteny(df_mean)
        plt.close(fig4)

        # 4. Metric 'icc' resolving to 'icc_2_1' or 'icc'
        df_icc = pd.DataFrame({"feature": features, "icc_2_1": [0.9] * 10})
        fig5 = plot_reproducibility_synteny(df_icc, metric="icc")
        plt.close(fig5)

    def test_synteny_discretisation_ribbons_counts(self):
        from eigenradiomics.plotting import plot_reproducibility_synteny

        features = [f"f{i}" for i in range(10)]

        # 1 param
        df1 = pd.DataFrame(
            {
                "feature": features,
                "icc": [0.85] * 10,
                "is_discretised": [True] * 10,
                "discretisation_param": [32] * 10,
            }
        )
        fig1 = plot_reproducibility_synteny(df1, show_discretisation_ribbon=True)
        plt.close(fig1)

        # 2 params
        df2 = pd.DataFrame(
            {
                "feature": features,
                "icc": [0.85] * 10,
                "is_discretised": [True, False] * 5,
                "discretisation_param": [16, np.nan] * 5,
            }
        )
        fig2 = plot_reproducibility_synteny(df2, show_discretisation_ribbon=True)
        plt.close(fig2)

        # 3 params
        df3 = pd.DataFrame(
            {
                "feature": features,
                "icc": [0.85] * 10,
                "is_discretised": [True, True, True, False] * 2 + [True, False],
                "discretisation_param": [16, 32, 64, np.nan] * 2 + [16, np.nan],
            }
        )
        fig3 = plot_reproducibility_synteny(df3, show_discretisation_ribbon=True)
        plt.close(fig3)

        # 4+ params
        df4 = pd.DataFrame(
            {
                "feature": features,
                "icc": [0.85] * 10,
                "is_discretised": [True] * 10,
                "discretisation_param": [8, 16, 32, 64, 128] * 2,
            }
        )
        fig4 = plot_reproducibility_synteny(df4, show_discretisation_ribbon=True)
        plt.close(fig4)

    def test_synteny_legend_and_axes_validation(self):
        from eigenradiomics.plotting import plot_reproducibility_synteny

        features = [f"f{i}" for i in range(10)]
        df = pd.DataFrame({"feature": features, "icc": [0.85] * 10})

        # Custom ax with show_legend=True and invalid legend_axes
        fig_main, ax_main = plt.subplots()
        with pytest.raises(ValueError, match="Must provide 3 legend_axes"):
            plot_reproducibility_synteny(df, ax=ax_main, show_legend=True, legend_axes=None)

        fig_dummy, leg_axes = plt.subplots(1, 3)
        fig_custom = plot_reproducibility_synteny(
            df, ax=ax_main, show_legend=True, legend_axes=leg_axes
        )
        assert fig_custom is not None
        plt.close(fig_main)
        plt.close(fig_dummy)

        # Single category cutoff handle
        fig_single = plot_reproducibility_synteny(df, thresholds=[])
        plt.close(fig_single)

        # > 4 thresholds raises (hits line 1184)
        with pytest.raises(ValueError, match="At most 4 thresholds"):
            plot_reproducibility_synteny(df, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5])

        # Dict with 'ICC' and metric='icc' (hits line 1087)
        fig_dict_icc = plot_reproducibility_synteny({"ICC": df}, metric="icc")
        plt.close(fig_dict_icc)

        # Catalog with disc_cols (hits lines 1161-1162)
        from eigenradiomics.catalog import FeatureCatalog

        cat_df = pd.DataFrame(
            {
                "feature": features,
                "family": ["glcm"] * 10,
                "is_discretised": [True] * 10,
                "discretisation_param": [32] * 10,
            }
        )
        fig_cat = plot_reproducibility_synteny(df, catalog=FeatureCatalog(cat_df))
        plt.close(fig_cat)

        # 2 unique non-null discretization parameters (hits lines 1291-1293)
        df_2params = pd.DataFrame(
            {
                "feature": features,
                "icc": [0.85] * 10,
                "is_discretised": [True] * 10,
                "discretisation_param": [16, 32] * 5,
            }
        )
        fig_2p = plot_reproducibility_synteny(df_2params, show_discretisation_ribbon=True)
        plt.close(fig_2p)

        # is_discretised=True with NaN param (hits line 1312)
        df_nan_p = pd.DataFrame(
            {
                "feature": features,
                "icc": [0.85] * 10,
                "is_discretised": [True] * 10,
                "discretisation_param": [np.nan] * 10,
            }
        )
        fig_np = plot_reproducibility_synteny(df_nan_p, show_discretisation_ribbon=True)
        plt.close(fig_np)
