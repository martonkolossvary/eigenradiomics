"""Public plotting utilities for eigenradiomics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from numpy.typing import NDArray
from scipy.cluster.hierarchy import dendrogram, leaves_list

from eigenradiomics.artifacts import ReductionArtifacts


def _as_similarity_frame(similarity: NDArray | pd.DataFrame) -> pd.DataFrame:
    """Coerce *similarity* to a square feature-by-feature DataFrame."""
    if isinstance(similarity, pd.DataFrame):
        if similarity.shape[0] != similarity.shape[1]:
            raise ValueError(f"similarity must be square, got shape {similarity.shape}.")
        return similarity
    arr = np.asarray(similarity, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"similarity must be a square 2-D matrix, got shape {arr.shape}.")
    names = [f"feature_{i}" for i in range(arr.shape[0])]
    return pd.DataFrame(arr, index=names, columns=names)


def plot_clustered_heatmap(
    similarity: NDArray | pd.DataFrame | ReductionArtifacts,
    *,
    cluster_labels: pd.Series | Sequence[Any] | None = None,
    linkage: NDArray | None = None,
    order: Sequence[Any] | NDArray | None = None,
    cluster_colors: Mapping[Any, Any] | None = None,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    below_cutoff_color: str | None = None,
    show_dendrogram: bool = True,
    show_cluster_strip: bool = True,
    labels: Any = "auto",
    colorbar_label: str = "Similarity",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot a clustered feature-by-feature similarity heatmap.

    "Bring your own" inputs: pass any symmetric ``similarity`` matrix plus,
    optionally, a per-feature ``cluster_labels`` assignment, a SciPy ``linkage``
    matrix (for the left dendrogram), and/or an explicit feature ``order``. A
    :class:`~eigenradiomics.ReductionArtifacts` may be passed directly as
    ``similarity``; its ``similarity`` / ``cluster_labels`` / ``linkage`` /
    ``feature_order`` fill any argument left unset.

    Parameters
    ----------
    similarity : ndarray, DataFrame, or ReductionArtifacts
        A symmetric ``(n_features, n_features)`` similarity (e.g. WGCNA TOM), or
        a fitted reducer's artifacts.
    cluster_labels : Series or sequence, optional
        Per-feature cluster / module label; drives the left colour strip and the
        default ordering when no ``linkage``/``order`` is given.
    linkage : ndarray, optional
        SciPy linkage matrix; drives the left dendrogram and the leaf ordering.
    order : sequence, optional
        Explicit feature ordering (feature names); overrides the linkage leaves.
    cluster_colors : mapping, optional
        ``{label: colour}``; otherwise a categorical palette is assigned.
    cmap : str
        Colormap for the similarity heatmap.
    vmin, vmax : float, optional
        Colour limits; default to the data range.
    below_cutoff_color : str, optional
        Colour for values below ``vmin`` (``set_under``).
    show_dendrogram, show_cluster_strip : bool
        Toggle the left dendrogram / cluster colour strip.
    labels : "auto", sequence, None, or False
        Tick labels. ``"auto"`` shows feature names only for small matrices.
    colorbar_label : str
        Label for the similarity colorbar.
    figsize : tuple, optional
        Figure size; auto-sized from the feature count otherwise.
    title : str, optional
        Heatmap title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # 1. Resolve inputs (a ReductionArtifacts fills any unset argument).
    if isinstance(similarity, ReductionArtifacts):
        artifacts = similarity
        if artifacts.similarity is None:
            raise ValueError(
                "ReductionArtifacts has no `similarity` matrix; fit the reducer "
                "with store_tom=True (WGCNA) or pass a similarity matrix directly."
            )
        sim = artifacts.similarity
        if cluster_labels is None:
            cluster_labels = artifacts.cluster_labels
        if linkage is None:
            linkage = artifacts.linkage
        if order is None:
            order = artifacts.feature_order
    else:
        sim = _as_similarity_frame(similarity)

    feature_names = list(sim.index)
    n = len(feature_names)

    labels_series: pd.Series | None = None
    if cluster_labels is not None:
        if isinstance(cluster_labels, pd.Series):
            labels_series = cluster_labels.reindex(feature_names)
        else:
            labels_series = pd.Series(list(cluster_labels), index=feature_names)

    # 2. Determine the display order (feature names).
    if order is not None:
        order_names = list(order)
    elif linkage is not None:
        order_names = [feature_names[i] for i in leaves_list(np.asarray(linkage))]
    elif labels_series is not None:
        order_names = list(labels_series.sort_values(kind="stable").index)
    else:
        order_names = feature_names

    ordered = sim.loc[order_names, order_names]
    if vmin is None:
        vmin = float(np.nanmin(ordered.to_numpy()))
    if vmax is None:
        vmax = float(np.nanmax(ordered.to_numpy()))

    if isinstance(labels, str) and labels == "auto":
        tick_labels = order_names if n <= 60 else None
    elif labels is None or labels is False:
        tick_labels = None
    else:
        tick_labels = list(labels)

    # 3. Lay out the panels.
    draw_dendro = show_dendrogram and linkage is not None
    draw_strip = show_cluster_strip and labels_series is not None

    if figsize is None:
        side = float(np.clip(n / 12.0, 6.0, 14.0))
        figsize = (side + 2.0, side)
    fig = plt.figure(figsize=figsize)

    width_ratios: list[float] = []
    col: dict[str, int] = {}
    if draw_dendro:
        col["dendro"] = len(width_ratios)
        width_ratios.append(1.6)
    if draw_strip:
        col["strip"] = len(width_ratios)
        width_ratios.append(0.30)
    col["heat"] = len(width_ratios)
    width_ratios.append(10.0)

    grid = fig.add_gridspec(
        2,
        len(width_ratios),
        width_ratios=width_ratios,
        height_ratios=[20.0, 1.0],
        wspace=0.02,
        hspace=0.05,
    )

    # 4. Main heatmap (+ colorbar below it).
    ax_heat = fig.add_subplot(grid[0, col["heat"]])
    cmap_obj = plt.get_cmap(cmap).copy()
    if below_cutoff_color is not None:
        cmap_obj.set_under(below_cutoff_color)
    image = ax_heat.imshow(
        ordered.to_numpy(), aspect="auto", cmap=cmap_obj,
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    if tick_labels is not None:
        ax_heat.set_xticks(range(n))
        ax_heat.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        ax_heat.set_yticks(range(n))
        ax_heat.set_yticklabels(tick_labels, fontsize=6)
    else:
        ax_heat.set_xticks([])
        ax_heat.set_yticks([])
    if title is not None:
        ax_heat.set_title(title, weight="bold")

    ax_cbar = fig.add_subplot(grid[1, col["heat"]])
    fig.colorbar(image, cax=ax_cbar, orientation="horizontal", label=colorbar_label)

    # 5. Left dendrogram, aligned row-for-row with the heatmap.
    if draw_dendro:
        ax_dendro = fig.add_subplot(grid[0, col["dendro"]])
        dendrogram(
            np.asarray(linkage), orientation="left", no_labels=True,
            ax=ax_dendro, color_threshold=0, above_threshold_color="#555555",
        )
        # imshow uses origin="upper" (row 0 at top); force the dendrogram's leaf
        # axis to span [0, 10n] inverted so leaf position j lands on heatmap row j.
        ax_dendro.set_ylim(10.0 * n, 0.0)
        ax_dendro.set_xticks([])
        ax_dendro.set_yticks([])
        for spine in ax_dendro.spines.values():
            spine.set_visible(False)

    # 6. Left cluster colour strip.
    if draw_strip:
        assert labels_series is not None  # draw_strip implies labels_series is set
        ordered_labels = labels_series.loc[order_names]
        unique_labels = list(dict.fromkeys(ordered_labels))
        if cluster_colors is None:
            palette = plt.get_cmap("tab20")
            color_map = {label: palette(i % 20) for i, label in enumerate(unique_labels)}
        else:
            color_map = {label: cluster_colors.get(label, "lightgrey") for label in unique_labels}
        strip = np.array([to_rgba(color_map[label]) for label in ordered_labels]).reshape(n, 1, 4)
        ax_strip = fig.add_subplot(grid[0, col["strip"]])
        ax_strip.imshow(strip, aspect="auto", interpolation="nearest")
        ax_strip.set_xticks([])
        ax_strip.set_yticks([])

    return fig
