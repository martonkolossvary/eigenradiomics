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

from eigenradiomics import Bar, ReductionArtifacts, Strip, plot_clustered_heatmap  # noqa: E402


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
