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

from eigenradiomics import ReductionArtifacts, plot_clustered_heatmap  # noqa: E402


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


def test_raw_array_minimal():
    df, _, _ = _blocks()
    fig = plot_clustered_heatmap(df.to_numpy())  # identity order, no dendro/strip
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
    assert len(fig.axes) == 4  # heatmap + colorbar + dendrogram + strip
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
    assert len(fig.axes) == 4
    plt.close(fig)


def test_order_from_cluster_labels_only():
    df, _, lab = _blocks()
    fig = plot_clustered_heatmap(df, cluster_labels=lab)  # Series, no linkage/order
    assert len(fig.axes) == 3  # heatmap + colorbar + strip (no dendrogram)
    plt.close(fig)


def test_labels_none_and_dendrogram_disabled():
    df, z, lab = _blocks()
    fig = plot_clustered_heatmap(
        df, linkage=z, cluster_labels=lab, labels=None, show_dendrogram=False
    )
    assert len(fig.axes) == 3  # heatmap + colorbar + strip
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
