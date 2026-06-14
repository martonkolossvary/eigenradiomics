"""Public plotting utilities for eigenradiomics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
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
        unknown = [name for name in order_names if name not in set(feature_names)]
        if unknown:
            raise ValueError(
                f"`order` contains {len(unknown)} name(s) absent from the similarity "
                f"index: {', '.join(map(str, unknown[:5]))}."
            )
    elif linkage is not None:
        order_names = [feature_names[i] for i in leaves_list(np.asarray(linkage))]
    elif labels_series is not None:
        order_names = list(labels_series.sort_values(kind="stable").index)
    else:
        order_names = feature_names

    ordered = sim.loc[order_names, order_names]
    ordered_values = ordered.to_numpy()
    if not np.isfinite(ordered_values).any():
        raise ValueError("similarity matrix has no finite values to plot.")
    if vmin is None:
        vmin = float(np.nanmin(ordered_values))
    if vmax is None:
        vmax = float(np.nanmax(ordered_values))

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
            len(legend_blocks), 1,
            height_ratios=[len(block_map) + 1.5 for _, block_map in legend_blocks],
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


def plot_hub_significance(
    k_me_df: pd.DataFrame,
    feature_significance: pd.Series,
    cluster_labels: pd.Series | dict[str, str | int],
    target_cluster: str | int,
    *,
    top_n_labels: int = 5,
    title: str | None = None,
    path: str | Path | None = None,
) -> plt.Figure:
    """Plot Module Membership (k_ME) vs. Feature Significance (GS/GS-Trait).

    X-axis represents feature correlation with eigengene (membership strength),
    while Y-axis represents feature significance to a trait or outcome.
    Highly accessible rendering: target cluster uses solid circles, secondary
    features use open triangles.

    Parameters
    ----------
    k_me_df : pandas.DataFrame
        Module membership table (features x components) e.g., from compute_module_membership.
    feature_significance : pandas.Series
        Association metrics indexed by feature name.
    cluster_labels : pd.Series or dict
        Cluster assignment per feature.
    target_cluster : str or int
        Target cluster/color.
    top_n_labels : int, default=5
        Number of top hub features to label.
    title : str, optional
        Figure title.
    path : str or Path, optional
        Destination path.
    """
    from scipy import stats

    from eigenradiomics._plotting import apply_science_style
    apply_science_style()

    # Align inputs
    if str(target_cluster) in k_me_df.columns:
        k_me_series = k_me_df[str(target_cluster)]
    else:
        k_me_series = k_me_df.iloc[:, 0]

    common_idx = k_me_series.index.intersection(feature_significance.index)
    k_me_series = k_me_series.loc[common_idx]
    sig_series = feature_significance.loc[common_idx]

    lbl_series = (
        pd.Series(cluster_labels) if isinstance(cluster_labels, dict) else cluster_labels
    )
    lbl_series = lbl_series.reindex(common_idx)

    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")

    # Split into target and secondary
    in_target = lbl_series == target_cluster

    # Target cluster: solid circles, colorblind-safe Vermillion or specific hex
    ax.scatter(
        k_me_series[in_target],
        sig_series[in_target],
        color="#D55E00",  # Vermillion
        marker="o",
        s=45,
        alpha=0.85,
        label=f"In Module ({target_cluster})",
        edgecolors="#7f2e00",
        linewidths=0.5
    )

    # Outside target: open triangles, dark slate
    ax.scatter(
        k_me_series[~in_target],
        sig_series[~in_target],
        color="none",
        marker="^",
        s=35,
        alpha=0.6,
        label="Out of Module",
        edgecolors="#222222",
        linewidths=0.8
    )

    # Label top hubs
    target_kme = k_me_series[in_target]
    sorted_target = target_kme.abs().sort_values(ascending=False)
    top_hubs = sorted_target.index[:top_n_labels]

    for _idx, feat in enumerate(top_hubs):
        x = k_me_series.loc[feat]
        y = sig_series.loc[feat]
        short_name = feat.split("__")[-1] if "__" in feat else feat
        ax.annotate(
            short_name,
            xy=(x, y),
            xytext=(x + 0.02 * np.sign(x), y + 0.02 * np.sign(y)),
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="0.7"),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=0.6)
        )

    # Calculate correlation metric for the module
    if len(k_me_series[in_target]) > 1:
        r, p = stats.spearmanr(k_me_series[in_target], sig_series[in_target])
        ax.text(
            0.05, 0.95,
            f"Module Hub-Significance:\np = {r:.2f} (p = {p:.1e})",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="square", fc="white", alpha=0.9, ec="0.8")
        )

    ax.set_xlabel(f"Module Membership ($k_{{ME}}$) for {target_cluster}", fontsize=11)
    ax.set_ylabel("Feature-Trait Significance", fontsize=11)

    if title:
        ax.set_title(title, fontsize=12, pad=10)
    else:
        # Avoid long line E501
        ax.set_title(
            f"Hub Features & Significance in Module {target_cluster}",
            fontsize=12,
            pad=10,
        )

    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="0.8")
    ax.grid(True, linestyle=":", alpha=0.5)

    if path:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    return fig


def plot_eigengene_profiles(
    eigengenes: pd.DataFrame | NDArray,
    trait: pd.Series | NDArray,
    *,
    trait_name: str = "Trait",
    component_idx: int = 0,
    title: str | None = None,
    path: str | Path | None = None,
) -> plt.Figure:
    """Plot eigengene profiles/trajectories grouped by a clinical variable.

    Automatically handles both categorical grouped boxplots (with distinct hatches
    for high accessibility) and continuous scatter plots with linear regression lines.

    Parameters
    ----------
    eigengenes : array-like of shape (n_samples, n_components)
        Exposed module eigengenes or component scores.
    trait : array-like of shape (n_samples,)
        Clinical trait/outcome to partition or regress across.
    trait_name : str, default="Trait"
        Display label for the trait.
    component_idx : int, default=0
        Index of the component/eigengene to plot.
    """
    from scipy import stats

    from eigenradiomics._plotting import apply_science_style
    apply_science_style()

    # Align data
    if isinstance(eigengenes, pd.DataFrame):
        y_vals = eigengenes.iloc[:, component_idx].to_numpy()
        comp_name = eigengenes.columns[component_idx]
    else:
        y_vals = np.asarray(eigengenes)[:, component_idx]
        comp_name = f"Component {component_idx}"

    x_vals = trait.to_numpy() if isinstance(trait, pd.Series) else np.asarray(trait)

    # Detect if trait is discrete/categorical vs continuous
    unique_vals = np.unique(x_vals[~pd.isna(x_vals)])
    is_categorical = (
        isinstance(unique_vals[0], (str, bool))
        or len(unique_vals) <= 6
    )

    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")

    if is_categorical:
        categories = sorted(list(unique_vals))
        groups_data = [y_vals[x_vals == cat] for cat in categories]

        colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2"][:len(categories)]
        hatches = ["", "//", "\\\\", "xx", ".."][:len(categories)]

        box = ax.boxplot(
            groups_data,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            boxprops=dict(linewidth=1.0)
        )

        for b_patch, color, hatch in zip(box["boxes"], colors, hatches, strict=False):
            b_patch.set_facecolor(to_rgba(color, alpha=0.7))
            b_patch.set_edgecolor("black")
            b_patch.set_hatch(hatch)

        for i, grp in enumerate(groups_data):
            jitter = np.random.default_rng(i).uniform(-0.15, 0.15, size=len(grp))
            ax.scatter(
                np.full_like(grp, i + 1) + jitter,
                grp,
                color="#333333",
                alpha=0.4,
                s=15,
                edgecolors="none"
            )

        ax.set_xticks(range(1, len(categories) + 1))
        ax.set_xticklabels([str(cat) for cat in categories], fontsize=10)
        ax.set_xlabel(trait_name, fontsize=11)
    else:
        valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
        x_clean = x_vals[valid_mask].astype(float)
        y_clean = y_vals[valid_mask].astype(float)

        ax.scatter(
            x_clean,
            y_clean,
            color="#56B4E9",
            marker="o",
            s=25,
            edgecolors="#004477",
            alpha=0.7,
            label="Patients"
        )

        slope, intercept, r_val, p_val, _ = stats.linregress(x_clean, y_clean)
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        y_line = slope * x_line + intercept

        ax.plot(
            x_line,
            y_line,
            color="#D55E00",
            linestyle="--",
            linewidth=1.5,
            label="Linear Fit"
        )

        ax.text(
            0.05, 0.95,
            f"Fit Summary:\nr = {r_val:.2f}\np = {p_val:.1e}",
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9, ec="0.8")
        )
        ax.set_xlabel(trait_name, fontsize=11)
        ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="0.8")

    ax.set_ylabel(f"{comp_name} Score", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)

    if title:
        ax.set_title(title, fontsize=12, pad=10)
    else:
        ax.set_title(f"Profile: {comp_name} across {trait_name}", fontsize=12, pad=10)

    if path:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    return fig


def plot_batch_distributions(
    X_before: pd.DataFrame | NDArray,
    X_after: pd.DataFrame | NDArray,
    batch: pd.Series | NDArray,
    feature_name: str | int,
    *,
    batch_name: str = "Batch",
    title: str | None = None,
    path: str | Path | None = None,
) -> plt.Figure:
    """Plot probability density of a feature before and after harmonization.

    Useful for evaluating ComBat or other normalization effects side-by-side.
    Accessible design relies on distinct linestyles and Okabe-Ito colors.

    Parameters
    ----------
    X_before, X_after : array-like
        Feature matrices before and after correction.
    batch : array-like
        Batch/center identifier per sample.
    feature_name : str or int
        Column name or index of the feature to plot.
    """
    from scipy import stats

    from eigenradiomics._plotting import apply_science_style
    apply_science_style()

    if isinstance(X_before, pd.DataFrame):
        vals_before = X_before[feature_name].to_numpy()
        feat_label = str(feature_name)
    else:
        vals_before = np.asarray(X_before)[:, int(feature_name)]
        feat_label = f"Feature {feature_name}"

    if isinstance(X_after, pd.DataFrame):
        vals_after = X_after[feature_name].to_numpy()
    else:
        vals_after = np.asarray(X_after)[:, int(feature_name)]

    batches = batch.to_numpy() if isinstance(batch, pd.Series) else np.asarray(batch)

    unique_batches = np.unique(batches[~pd.isna(batches)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True, layout="constrained")

    linestyles = ["-", "--", ":", "-."]
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2"]

    for idx, b_id in enumerate(unique_batches):
        mask = batches == b_id

        ls = linestyles[idx % len(linestyles)]
        color = colors[idx % len(colors)]

        b_vals_before = vals_before[mask]
        b_clean_before = b_vals_before[~np.isnan(b_vals_before)]

        b_vals_after = vals_after[mask]
        b_clean_after = b_vals_after[~np.isnan(b_vals_after)]

        # Before
        if len(b_clean_before) > 1 and np.std(b_clean_before) > 1e-12:
            kde_before = stats.gaussian_kde(b_clean_before)
            grid_before = np.linspace(b_clean_before.min(), b_clean_before.max(), 150)
            ax1.plot(
                grid_before,
                kde_before(grid_before),
                linestyle=ls,
                color=color,
                linewidth=2.0,
                label=f"{batch_name} {b_id}",
            )
        else:
            ax1.hist(
                b_clean_before,
                alpha=0.3,
                color=color,
                label=f"{batch_name} {b_id}",
                density=True,
            )

        # After
        if len(b_clean_after) > 1 and np.std(b_clean_after) > 1e-12:
            kde_after = stats.gaussian_kde(b_clean_after)
            grid_after = np.linspace(b_clean_after.min(), b_clean_after.max(), 150)
            ax2.plot(
                grid_after,
                kde_after(grid_after),
                linestyle=ls,
                color=color,
                linewidth=2.0,
                label=f"{batch_name} {b_id}",
            )
        else:
            ax2.hist(
                b_clean_after,
                alpha=0.3,
                color=color,
                label=f"{batch_name} {b_id}",
                density=True,
            )

    ax1.set_title("Before Harmonization", fontsize=11)
    ax1.set_xlabel(f"Value of {feat_label}", fontsize=10)
    ax1.set_ylabel("Probability Density", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.set_title("After Harmonization", fontsize=11)
    ax2.set_xlabel(f"Value of {feat_label}", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)

    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="0.8", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    else:
        fig.suptitle(
            f"Batch Effect Harmonization Diagnostics for {feat_label}",
            fontsize=12,
            fontweight="bold",
        )

    if path:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    return fig
