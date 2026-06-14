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
    panel = CorrPanel(_corr_frame(list(df.index)), cmap="coolwarm", vmin=-0.5, vmax=0.5,
                      label="Spearman r")
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


def test_plot_observer_synteny(tmp_path):
    from eigenradiomics.plotting import plot_observer_synteny

    features = [f"f{i}" for i in range(10)]
    df_repro = pd.DataFrame(
        {
            "feature": features,
            "icc": [0.95, 0.45, 0.82, 0.73, 0.91, 0.33, 0.61, 0.88, 0.79, 0.52],
            "correlation": [0.96, 0.50, 0.85, 0.75, 0.92, 0.38, 0.65, 0.90, 0.82, 0.55],
        }
    )

    catalog = pd.DataFrame({"feature": features, "family": ["firstorder"] * 5 + ["glcm"] * 5})

    fig1 = plot_observer_synteny(df_repro, catalog=catalog)
    assert fig1 is not None
    plt.close(fig1)

    fig2 = plot_observer_synteny(
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


def test_plot_observer_synteny_coverage_gaps(tmp_path):
    from eigenradiomics.plotting import plot_observer_synteny

    features = [f"f{i}" for i in range(10)]
    df_repro = pd.DataFrame(
        {
            "feature": features,
            "icc": [0.95, np.nan, 0.82, 0.73, 0.91, 0.33, 0.61, 0.88, 0.79, 0.52],
            "correlation": [0.96, 0.50, 0.85, 0.75, 0.92, 0.38, 0.65, 0.90, 0.82, 0.55],
        }
    )

    # 1. reproducibility_results as dict
    fig = plot_observer_synteny({"ICC": df_repro})
    assert fig is not None
    plt.close(fig)

    fig = plot_observer_synteny({"Spearman": df_repro})
    assert fig is not None
    plt.close(fig)

    # DataFrame with custom metric column
    df_custom = pd.DataFrame(
        {
            "feature": features,
            "custom_val": [0.9] * 10,
        }
    )
    fig = plot_observer_synteny({"Pearson": df_custom}, metric="custom_val")
    assert fig is not None
    plt.close(fig)

    # 2. DataFrame without "icc" or "correlation" (inferred_metric = df.columns[1])
    fig = plot_observer_synteny(df_custom)
    assert fig is not None
    plt.close(fig)

    # 2.b DataFrame with "correlation" only (to hit line 1004)
    df_corr_only = pd.DataFrame(
        {
            "feature": features,
            "correlation": [0.9] * 10,
        }
    )
    fig = plot_observer_synteny(df_corr_only)
    assert fig is not None
    plt.close(fig)

    # 3. raise ValueError: metric not found
    with pytest.raises(ValueError, match="Metric column 'missing' not found"):
        plot_observer_synteny(df_repro, metric="missing")

    # 4. raise ValueError: "feature" column not found
    df_no_feat = df_repro.drop(columns=["feature"])
    with pytest.raises(ValueError, match="must contain a 'feature' column"):
        plot_observer_synteny(df_no_feat)

    # 5. group_by not in df columns (fallback to "All Features")
    fig = plot_observer_synteny(df_repro, group_by="missing_family")
    assert fig is not None
    plt.close(fig)

    # 6. Empty dataframe after filtering (n == 0)
    empty_df = pd.DataFrame(columns=["feature", "icc"])
    with pytest.raises(ValueError, match="No features available to plot"):
        plot_observer_synteny(empty_df)

    # 7. n > 50 features (to hit tick_labels is None, etc.)
    features_large = [f"f{i}" for i in range(55)]
    df_large = pd.DataFrame(
        {
            "feature": features_large,
            "icc": [0.9] * 55,
        }
    )
    fig = plot_observer_synteny(df_large)
    assert fig is not None
    plt.close(fig)

