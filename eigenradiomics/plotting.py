"""Public plotting utilities for eigenradiomics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from scipy.cluster.hierarchy import dendrogram, leaves_list

from eigenradiomics.artifacts import ReductionArtifacts

#: Okabe-Ito colourblind-safe qualitative palette (default for categorical strips).
OKABE_ITO: list[str] = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

#: Default bar colour when "by_module" is requested but no modules are available.
_DEFAULT_BAR_COLOR = "#4477AA"


@dataclass
class Strip:
    """A categorical annotation strip drawn above the heatmap (a top track).

    Parameters
    ----------
    data : pandas.Series
        Per-feature category, indexed by feature name.
    title : str, optional
        Strip name shown at its left edge and as the legend title (defaults to
        the Series name).
    colors : mapping, optional
        ``{category: colour}``; otherwise a colourblind-safe palette is assigned.
    """

    data: pd.Series
    title: str | None = None
    colors: Mapping[Any, Any] | None = None


@dataclass
class Bar:
    """A numeric annotation bar track drawn below the heatmap (a bottom track).

    Parameters
    ----------
    data : pandas.Series
        Per-feature numeric value, indexed by feature name.
    title : str, optional
        Track name shown at its left edge (defaults to the Series name).
    color : str
        ``"by_module"`` (default) colours each bar by its feature's module;
        otherwise any Matplotlib colour applied to all bars.
    reference : float, optional
        If set, draw a dashed horizontal reference line (e.g. ``-log10(0.05)``).
    """

    data: pd.Series
    title: str | None = None
    color: str = "by_module"
    reference: float | None = None


@dataclass
class CorrPanel:
    """A feature-by-variable correlation panel drawn to the right of the heatmap.

    Parameters
    ----------
    data : pandas.DataFrame
        ``(n_features, n_variables)`` correlations, indexed by feature name (e.g.
        from :func:`~eigenradiomics.compute_clinical_correlations`).
    cmap : str
        Diverging colourmap (``"RdBu_r"`` by default).
    vmin, vmax : float
        Colour limits, symmetric about zero by default (``-1`` / ``1``).
    label : str
        Label for the panel's colourbar.
    """

    data: pd.DataFrame
    cmap: str = "RdBu_r"
    vmin: float = -1.0
    vmax: float = 1.0
    label: str = "Correlation"


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


def _assign_colors(
    categories: Sequence[Any],
    explicit: Mapping[Any, Any] | None,
) -> dict[Any, Any]:
    """Map each category to a colour (explicit overrides; else a palette)."""
    if explicit is not None:
        return {category: explicit.get(category, "lightgrey") for category in categories}
    if len(categories) <= len(OKABE_ITO):
        palette: list[Any] = list(OKABE_ITO)
    else:
        tab = plt.get_cmap("tab20")
        palette = [tab(i % 20) for i in range(len(categories))]
    return {category: palette[i] for i, category in enumerate(categories)}


def _as_strip(item: pd.Series | Strip) -> Strip:
    """Normalize a top-track input to a :class:`Strip`."""
    if isinstance(item, Strip):
        return item
    if isinstance(item, pd.Series):
        return Strip(data=item, title=item.name)
    raise TypeError(f"top tracks must be a pandas Series or Strip, got {type(item).__name__}.")


def _as_bar(item: pd.Series | Bar) -> Bar:
    """Normalize a bottom-track input to a :class:`Bar`."""
    if isinstance(item, Bar):
        return item
    if isinstance(item, pd.Series):
        return Bar(data=item, title=item.name)
    raise TypeError(f"bottom tracks must be a pandas Series or Bar, got {type(item).__name__}.")


def _as_corr_panel(item: pd.DataFrame | CorrPanel) -> CorrPanel:
    """Normalize a right-panel input to a :class:`CorrPanel`."""
    if isinstance(item, CorrPanel):
        return item
    if isinstance(item, pd.DataFrame):
        return CorrPanel(data=item)
    raise TypeError(f"right must be a pandas DataFrame or CorrPanel, got {type(item).__name__}.")


def plot_clustered_heatmap(
    similarity: NDArray | pd.DataFrame | ReductionArtifacts,
    *,
    cluster_labels: pd.Series | Sequence[Any] | None = None,
    linkage: NDArray | None = None,
    order: Sequence[Any] | NDArray | None = None,
    cluster_colors: Mapping[Any, Any] | None = None,
    top: Sequence[pd.Series | Strip] | None = None,
    bottom: Sequence[pd.Series | Bar] | None = None,
    right: pd.DataFrame | CorrPanel | None = None,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    below_cutoff_color: str | None = None,
    show_dendrogram: bool = True,
    show_cluster_strip: bool = True,
    show_legend: bool = True,
    labels: Any = "auto",
    colorbar_label: str = "Similarity",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot a clustered feature-by-feature similarity heatmap.

    "Bring your own" inputs: pass any symmetric ``similarity`` matrix plus,
    optionally, a per-feature ``cluster_labels`` assignment, a SciPy ``linkage``
    matrix (left dendrogram), an explicit feature ``order``, categorical
    annotation strips via ``top``, and numeric annotation bars via ``bottom``. A
    :class:`~eigenradiomics.ReductionArtifacts` may be passed directly as
    ``similarity``; its fields fill any argument left unset.

    Parameters
    ----------
    similarity : ndarray, DataFrame, or ReductionArtifacts
        Symmetric ``(n_features, n_features)`` similarity, or a reducer's artifacts.
    cluster_labels : Series or sequence, optional
        Per-feature cluster / module label; drives the left colour strip, the
        default ordering, and ``"by_module"`` bar colours.
    linkage : ndarray, optional
        SciPy linkage matrix; drives the left dendrogram and leaf ordering.
    order : sequence, optional
        Explicit feature ordering (overrides the linkage leaves).
    cluster_colors : mapping, optional
        ``{label: colour}`` for the module strip/legend/bars; otherwise a palette.
    top : sequence of Series or Strip, optional
        Categorical annotation strips drawn above the heatmap.
    bottom : sequence of Series or Bar, optional
        Numeric annotation bar tracks drawn below the heatmap.
    right : DataFrame or CorrPanel, optional
        Feature-by-variable correlation panel (e.g. features vs clinical
        variables) drawn to the right with its own diverging colourbar.
    cmap, vmin, vmax, below_cutoff_color :
        Similarity colour scaling (``below_cutoff_color`` is ``set_under``).
    show_dendrogram, show_cluster_strip, show_legend : bool
        Toggle the left dendrogram, the left cluster strip, and the legend column.
    labels : "auto", sequence, None, or False
        Feature tick labels. ``"auto"`` shows names only for small matrices.
    colorbar_label : str
        Label for the similarity colourbar.
    figsize : tuple, optional
        Figure size; auto-sized from the feature count otherwise.
    title : str, optional
        Figure title.

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

    strips = [_as_strip(item) for item in top] if top is not None else []
    bars = [_as_bar(item) for item in bottom] if bottom is not None else []
    corr_panel = _as_corr_panel(right) if right is not None else None

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

    # 3. Colours for categorical tracks + legend blocks.
    draw_dendro = show_dendrogram and linkage is not None
    draw_strip = show_cluster_strip and labels_series is not None
    n_top = len(strips)
    n_bottom = len(bars)

    module_color_map: dict[Any, Any] | None = None
    legend_blocks: list[tuple[str, dict[Any, Any]]] = []
    if draw_strip:
        assert labels_series is not None
        module_cats = list(dict.fromkeys(labels_series.loc[order_names].dropna()))
        module_color_map = _assign_colors(module_cats, cluster_colors)
        legend_blocks.append(("Module", module_color_map))
    strip_color_maps: list[dict[Any, Any]] = []
    for index, strip in enumerate(strips):
        ordered_cats = strip.data.reindex(order_names)
        unique_cats = list(dict.fromkeys(ordered_cats.dropna()))
        color_map = _assign_colors(unique_cats, strip.colors)
        strip_color_maps.append(color_map)
        legend_blocks.append((strip.title or f"strip {index + 1}", color_map))

    has_legend = show_legend and bool(legend_blocks)

    # 4. Lay out the panels (rows: top strips, heatmap, bottom bars, colorbar;
    #    cols: dendro, cluster strip, heatmap, correlation panel, legend).
    corr_ratio = (
        float(np.clip(0.5 * corr_panel.data.shape[1], 1.4, 4.5))
        if corr_panel is not None
        else 0.0
    )
    if figsize is None:
        side = float(np.clip(n / 12.0, 6.0, 14.0))
        width = (
            side + 2.0
            + (2.6 if has_legend else 0.0)
            + (0.1 * side * corr_ratio + 0.6 if corr_panel is not None else 0.0)
        )
        height = side + 1.0 + 0.4 * n_top + 0.6 * n_bottom
        figsize = (width, height)
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
    if corr_panel is not None:
        col["corr"] = len(width_ratios)
        width_ratios.append(corr_ratio)
    if has_legend:
        col["legend"] = len(width_ratios)
        width_ratios.append(2.6)

    height_ratios = [0.32] * n_top + [10.0] + [1.1] * n_bottom + [1.0]
    heat_row = n_top
    cbar_row = n_top + 1 + n_bottom

    grid = fig.add_gridspec(
        len(height_ratios),
        len(width_ratios),
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=0.02,
        hspace=0.08,
    )

    # 5. Main heatmap (+ colorbar at the bottom).
    ax_heat = fig.add_subplot(grid[heat_row, col["heat"]])
    cmap_obj = plt.get_cmap(cmap).copy()
    if below_cutoff_color is not None:
        cmap_obj.set_under(below_cutoff_color)
    image = ax_heat.imshow(
        ordered.to_numpy(), aspect="auto", cmap=cmap_obj,
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    if tick_labels is not None:
        ax_heat.set_yticks(range(n))
        ax_heat.set_yticklabels(tick_labels, fontsize=6)
        if n_bottom == 0:  # x labels move to the bottom-most bar when bars exist
            ax_heat.set_xticks(range(n))
            ax_heat.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        else:
            ax_heat.set_xticks([])
    else:
        ax_heat.set_xticks([])
        ax_heat.set_yticks([])
    if title is not None:
        fig.suptitle(title, weight="bold")

    # Compact colourbar tucked into the bottom-left corner under the heatmap.
    cbar_host = grid[cbar_row, col["heat"]].subgridspec(1, 2, width_ratios=[0.5, 0.5], wspace=0.0)
    ax_cbar = fig.add_subplot(cbar_host[0, 0])
    fig.colorbar(image, cax=ax_cbar, orientation="horizontal", label=colorbar_label)

    # 6. Top categorical strips (aligned with the heatmap columns).
    for index, strip in enumerate(strips):
        ordered_cats = strip.data.reindex(order_names)
        color_map = strip_color_maps[index]
        rgba = np.array(
            [to_rgba(color_map.get(category, "lightgrey")) for category in ordered_cats]
        ).reshape(1, n, 4)
        ax_top = fig.add_subplot(grid[index, col["heat"]])
        ax_top.imshow(rgba, aspect="auto", interpolation="nearest")
        ax_top.set_xticks([])
        ax_top.set_yticks([0])
        ax_top.set_yticklabels([strip.title or f"strip {index + 1}"], fontsize=7)

    # 7. Bottom numeric bars (aligned with the heatmap columns).
    for index, bar in enumerate(bars):
        ax_bar = fig.add_subplot(grid[heat_row + 1 + index, col["heat"]])
        values = bar.data.reindex(order_names).to_numpy(dtype=float)
        if bar.color == "by_module" and labels_series is not None and module_color_map is not None:
            bar_color: Any = [
                module_color_map.get(label, "lightgrey")
                for label in labels_series.loc[order_names]
            ]
        else:
            bar_color = bar.color if bar.color != "by_module" else _DEFAULT_BAR_COLOR
        ax_bar.bar(np.arange(n), values, width=1.0, color=bar_color, linewidth=0)
        ax_bar.set_xlim(-0.5, n - 0.5)
        if bar.reference is not None:
            ax_bar.axhline(bar.reference, ls="--", lw=0.8, color="0.4")
        ax_bar.set_ylabel(
            bar.title or f"bar {index + 1}", rotation=0, ha="right", va="center", fontsize=7
        )
        ax_bar.yaxis.set_major_locator(MaxNLocator(nbins=2))
        ax_bar.tick_params(labelsize=6)
        if index == n_bottom - 1 and tick_labels is not None:
            ax_bar.set_xticks(range(n))
            ax_bar.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        else:
            ax_bar.set_xticks([])

    # 7b. Right correlation panel (shares the heatmap row order; own colourbar).
    if corr_panel is not None:
        panel = corr_panel.data.reindex(order_names)
        ax_corr = fig.add_subplot(grid[heat_row, col["corr"]])
        corr_image = ax_corr.imshow(
            panel.to_numpy(dtype=float), aspect="auto", cmap=corr_panel.cmap,
            vmin=corr_panel.vmin, vmax=corr_panel.vmax,
            interpolation="nearest", origin="upper",
        )
        ax_corr.set_yticks([])
        ax_corr.set_xticks(range(panel.shape[1]))
        ax_corr.set_xticklabels(list(panel.columns), rotation=90, fontsize=7)
        # Compact colourbar tucked into the bottom-right corner under the panel.
        corr_cbar_host = grid[cbar_row, col["corr"]].subgridspec(
            1, 2, width_ratios=[0.2, 0.8], wspace=0.0
        )
        ax_corr_cbar = fig.add_subplot(corr_cbar_host[0, 1])
        fig.colorbar(
            corr_image, cax=ax_corr_cbar, orientation="horizontal", label=corr_panel.label
        )

    # 8. Left dendrogram, aligned row-for-row with the heatmap.
    if draw_dendro:
        ax_dendro = fig.add_subplot(grid[heat_row, col["dendro"]])
        dendrogram(
            np.asarray(linkage), orientation="left", no_labels=True,
            ax=ax_dendro, color_threshold=0, above_threshold_color="#555555",
        )
        ax_dendro.set_ylim(10.0 * n, 0.0)
        ax_dendro.set_xticks([])
        ax_dendro.set_yticks([])
        for spine in ax_dendro.spines.values():
            spine.set_visible(False)

    # 9. Left cluster colour strip.
    if draw_strip:
        assert labels_series is not None and module_color_map is not None
        ordered_labels = labels_series.loc[order_names]
        strip_rgba = np.array(
            [to_rgba(module_color_map.get(label, "lightgrey")) for label in ordered_labels]
        ).reshape(n, 1, 4)
        ax_strip = fig.add_subplot(grid[heat_row, col["strip"]])
        ax_strip.imshow(strip_rgba, aspect="auto", interpolation="nearest")
        ax_strip.set_xticks([])
        ax_strip.set_yticks([])

    # 10. Right-edge stacked legend column (one titled block per categorical track).
    if has_legend:
        legend_gs = grid[:, col["legend"]].subgridspec(
            len(legend_blocks), 1, height_ratios=[len(cmap) + 1.5 for _, cmap in legend_blocks]
        )
        for block_index, (block_title, color_map) in enumerate(legend_blocks):
            ax_legend = fig.add_subplot(legend_gs[block_index])
            handles = [
                Patch(facecolor=color, edgecolor="0.3", label=str(category))
                for category, color in color_map.items()
            ]
            ax_legend.legend(
                handles=handles, title=block_title, loc="center left",
                frameon=False, fontsize=7, title_fontsize=8, handlelength=1.0,
            )
            ax_legend.axis("off")

    return fig
