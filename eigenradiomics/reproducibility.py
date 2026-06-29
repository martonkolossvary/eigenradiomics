"""Inter- and intra-observer reproducibility analysis framework for radiomics."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.typing import NDArray

from eigenradiomics._excel import write_styled_workbook
from eigenradiomics._features import resolve_analysis_features
from eigenradiomics._plotting import apply_science_style
from eigenradiomics._stats import (
    _bootstrap_icc_ci,
    _fdr_correct,
    _fisher_ci,
    _fisher_mean,
    _icc_2_1_estimate,
)
from eigenradiomics._utils import _save_figure


def _generate_cluster_bootstrap_indices(
    groups: NDArray,
    iterations: int = 1000,
    seed: int = 42,
) -> NDArray:
    """Generate a (iterations, n_samples) matrix of bootstrap indices clustered by groups."""
    rng = np.random.default_rng(seed)
    n_samples = len(groups)
    unique_groups, group_inverse = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)

    # Pre-group the row indices for each unique group
    group_to_indices = [np.where(group_inverse == i)[0] for i in range(n_groups)]

    bootstrap_indices = np.zeros((iterations, n_samples), dtype=int)
    for i in range(iterations):
        # Sample unique groups with replacement
        sampled_group_indices = rng.choice(n_groups, size=n_groups, replace=True)
        # Gather all row indices belonging to these groups
        sampled_indices = []
        for g_idx in sampled_group_indices:
            sampled_indices.extend(group_to_indices[g_idx])

        sampled_indices = np.array(sampled_indices)
        L = len(sampled_indices)
        if n_samples == L:
            boot_idx = sampled_indices
        elif n_samples > L:
            # Pad by sampling with replacement from the gathered indices
            padding = rng.choice(sampled_indices, size=n_samples - L, replace=True)
            boot_idx = np.concatenate([sampled_indices, padding])
        else:
            # Truncate: randomly select n_samples from the gathered indices (no replacement)
            boot_idx = rng.choice(sampled_indices, size=n_samples, replace=False)

        bootstrap_indices[i] = boot_idx

    return bootstrap_indices


def compute_reproducibility(
    datasets: Sequence[pd.DataFrame | NDArray],
    *,
    features: Any = None,
    configs: Any = None,
    families: Any = None,
    family_groups: Any = None,
    catalog: Any = None,
    min_valid_samples: int = 3,
    bootstrap_iterations: int = 1000,
    primary_threshold: float = 0.80,
    groups: Any = None,
) -> dict[str, pd.DataFrame]:
    """Evaluate feature-by-feature reproducibility across multiple reader datasets.

    Parameters
    ----------
    datasets : sequence of pd.DataFrame or ndarray
        Replicate datasets (e.g. Reader 1, Reader 2, ...). Minimum length is 2.
    features, configs, families, family_groups, catalog : selectors
        Standard Pictologics selectors used to isolate target features.
    min_valid_samples : int
        Minimum number of complete paired subject measurements required.
    bootstrap_iterations : int
        Number of bootstrap iterations for ICC 95% CI estimation.
    primary_threshold : float
        Threshold used to populate the `retained` flag.

    Returns
    -------
    results : dict[str, pd.DataFrame]
        A dictionary containing three sheets: "Spearman", "Pearson",
        and "ICC" with detailed statistics.

    Notes
    -----
    The ICC ``p_value`` tests whether subjects are distinguishable (the F-test
    ``H0: MS_between_subjects = MS_error``); it is *not* a test that the ICC
    exceeds a clinically useful threshold, and is almost always significant for
    real data. Judge reliability from the ICC estimate and its CI versus
    ``primary_threshold``, not from the p-value. For three or more observers the
    Spearman/Pearson ``mean`` is pooled in Fisher z-space; the ICC is the
    principled multi-rater statistic.
    """
    if len(datasets) < 2:
        raise ValueError("At least 2 datasets must be provided for reproducibility analysis.")

    # 1. Quality Control validation and alignment
    is_named = True
    for df in datasets:
        if not isinstance(df, pd.DataFrame):
            is_named = False
            break
        if isinstance(df.columns, pd.RangeIndex):
            is_named = False
            break
        if not all(isinstance(c, str) for c in df.columns):
            is_named = False
            break

    aligned_datasets: list[pd.DataFrame] = []

    if is_named:
        # Strict Name-Based Verification and Automatic Reordering
        ref_df = cast(pd.DataFrame, datasets[0])
        ref_columns = ref_df.columns
        ref_index = ref_df.index

        for idx, df in enumerate(datasets):
            df = cast(pd.DataFrame, df)
            # Verify feature column names
            if set(df.columns) != set(ref_columns):
                missing = list(set(ref_columns) - set(df.columns))
                unexpected = list(set(df.columns) - set(ref_columns))
                raise ValueError(
                    f"Dataset {idx} columns do not match Dataset 0. "
                    f"Missing features: {missing[:5]}. Unexpected features: {unexpected[:5]}."
                )
            # Verify row index (subjects)
            if set(df.index) != set(ref_index):
                missing = list(set(ref_index) - set(df.index))
                unexpected = list(set(df.index) - set(ref_index))
                raise ValueError(
                    f"Dataset {idx} row index does not match Dataset 0. "
                    f"Missing subjects: {missing[:5]}. Unexpected subjects: {unexpected[:5]}."
                )
            # Auto-align columns and index to reference
            aligned_df = df.loc[ref_index, ref_columns]
            aligned_datasets.append(aligned_df)
    else:
        # Strict Positional Verification
        ref_shape = datasets[0].shape
        for idx, ds in enumerate(datasets):
            if ds.shape != ref_shape:
                raise ValueError(
                    f"Dataset {idx} shape {ds.shape} does not match reference shape {ref_shape}."
                )

        # Standardize unnamed/array datasets into labeled pandas DataFrames
        n_samples, n_features = ref_shape
        ref_columns = pd.Index([f"feature_{col_idx}" for col_idx in range(n_features)])
        ref_index = pd.Index([f"subject_{row_idx}" for row_idx in range(n_samples)])

        for ds in datasets:
            if isinstance(ds, pd.DataFrame):
                df = pd.DataFrame(
                    ds.to_numpy(dtype=float), index=ref_index, columns=ref_columns
                )
            else:
                df = pd.DataFrame(
                    np.asarray(ds, dtype=float), index=ref_index, columns=ref_columns
                )
            aligned_datasets.append(df)

    # 2. Resolve features to analyze (selectors, else every numeric column).
    features_to_analyze = resolve_analysis_features(
        aligned_datasets[0],
        features=features,
        configs=configs,
        families=families,
        family_groups=family_groups,
        catalog=catalog,
    )

    if len(features_to_analyze) == 0:
        raise ValueError("No features selected for reproducibility analysis.")

    # Resolve groups parameter for cluster-robust bootstrapping
    groups_arr = None
    if groups is not None:
        if isinstance(groups, str):
            ref_df = datasets[0]
            if isinstance(ref_df, pd.DataFrame):
                if groups in ref_df.columns:
                    groups_arr = ref_df[groups].to_numpy()
                elif ref_df.index.names and groups in ref_df.index.names:
                    groups_arr = ref_df.index.get_level_values(groups).to_numpy()
                else:
                    raise ValueError(f"groups column/level {groups!r} not found in dataset.")
            else:
                raise ValueError("groups as string only supported when datasets are DataFrames.")
        else:
            groups_arr = np.asarray(groups)
    else:
        ref_df = datasets[0]
        if isinstance(ref_df, pd.DataFrame) and isinstance(ref_df.index, pd.MultiIndex):
            groups_arr = ref_df.index.get_level_values(0).to_numpy()

    # Precompute global bootstrap indices if possible
    global_bootstrap_indices = None
    if groups_arr is not None:
        global_bootstrap_indices = _generate_cluster_bootstrap_indices(
            groups_arr, iterations=bootstrap_iterations, seed=42
        )

    # 3. Core Reproducibility Calculations
    n_features = len(features_to_analyze)
    k_observers = len(aligned_datasets)

    # Setup empty result arrays/lists
    icc_rows: list[dict[str, Any]] = []
    spearman_rows: list[dict[str, Any]] = []
    pearson_rows: list[dict[str, Any]] = []

    for f_name in features_to_analyze:
        # Construct subject-by-observer matrix (N, K)
        Y = np.column_stack([df[f_name].to_numpy() for df in aligned_datasets])

        # Filter out NaN rows to get paired samples
        valid_mask = ~np.isnan(Y).any(axis=1)
        Y_clean = Y[valid_mask]
        n_valid = len(Y_clean)

        if n_valid < min_valid_samples:
            # Handle insufficient valid samples gracefully
            icc_rows.append(
                {
                    "feature": f_name,
                    "icc_2_1": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "p_value": np.nan,
                    "ms_between_subjects": np.nan,
                    "ms_between_observers": np.nan,
                    "ms_error": np.nan,
                }
            )

            if k_observers == 2:
                for row_list in [spearman_rows, pearson_rows]:
                    row_list.append(
                        {
                            "feature": f_name,
                            "estimate": np.nan,
                            "ci95_low": np.nan,
                            "ci95_high": np.nan,
                            "p_value": np.nan,
                        }
                    )
            else:
                for row_list in [spearman_rows, pearson_rows]:
                    row_list.append(
                        {
                            "feature": f_name,
                            "mean": np.nan,
                            "median": np.nan,
                            "sd": np.nan,
                            "q25": np.nan,
                            "q75": np.nan,
                            "min": np.nan,
                            "max": np.nan,
                        }
                    )
            continue

        # A. Calculate ICC(2,1)
        icc_est = _icc_2_1_estimate(Y_clean)
        if groups_arr is not None:
            if n_valid == len(groups_arr):
                boot_indices = global_bootstrap_indices
            else:
                groups_clean = groups_arr[valid_mask]
                boot_indices = _generate_cluster_bootstrap_indices(
                    groups_clean, iterations=bootstrap_iterations, seed=42
                )
            ci_low, ci_high = _bootstrap_icc_ci(
                Y_clean,
                f_name,
                iterations=bootstrap_iterations,
                base_seed=42,
                bootstrap_indices=boot_indices,
            )
        else:
            ci_low, ci_high = _bootstrap_icc_ci(
                Y_clean, f_name, iterations=bootstrap_iterations, base_seed=42
            )

        icc_rows.append(
            {
                "feature": f_name,
                "icc_2_1": icc_est["icc"],
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "p_value": icc_est["p_value"],
                "ms_between_subjects": icc_est["ms_between_subjects"],
                "ms_between_observers": icc_est["ms_between_observers"],
                "ms_error": icc_est["ms_error"],
            }
        )

        # B. Calculate Spearman & Pearson
        if k_observers == 2:
            # Complete two-reader detailed metrics
            with warnings.catch_warnings():
                # Constant/near-constant features make scipy emit warnings; the
                # resulting NaN correlations are handled downstream.
                warnings.simplefilter("ignore")
                s_rho, s_p = stats.spearmanr(Y_clean[:, 0], Y_clean[:, 1])
                p_r, p_p = stats.pearsonr(Y_clean[:, 0], Y_clean[:, 1])

            s_low, s_high = _fisher_ci(s_rho, n_valid, is_spearman=True)
            p_low, p_high = _fisher_ci(p_r, n_valid, is_spearman=False)

            spearman_rows.append(
                {
                    "feature": f_name,
                    "estimate": s_rho,
                    "ci95_low": s_low,
                    "ci95_high": s_high,
                    "p_value": s_p,
                }
            )

            pearson_rows.append(
                {
                    "feature": f_name,
                    "estimate": p_r,
                    "ci95_low": p_low,
                    "ci95_high": p_high,
                    "p_value": p_p,
                }
            )
        else:
            # Multi-reader aggregate metrics
            s_coeffs = []
            p_coeffs = []

            with warnings.catch_warnings():
                # Constant/near-constant features make scipy emit warnings; the
                # resulting NaN correlations are filtered out below.
                warnings.simplefilter("ignore")
                for i in range(k_observers):
                    for j in range(i + 1, k_observers):
                        s_val, _ = stats.spearmanr(Y_clean[:, i], Y_clean[:, j])
                        p_val, _ = stats.pearsonr(Y_clean[:, i], Y_clean[:, j])
                        if not np.isnan(s_val):
                            s_coeffs.append(s_val)
                        if not np.isnan(p_val):
                            p_coeffs.append(p_val)

            for coeffs, row_list in [(s_coeffs, spearman_rows), (p_coeffs, pearson_rows)]:
                if len(coeffs) == 0:
                    row_list.append(
                        {
                            "feature": f_name,
                            "mean": np.nan,
                            "median": np.nan,
                            "sd": np.nan,
                            "q25": np.nan,
                            "q75": np.nan,
                            "min": np.nan,
                            "max": np.nan,
                        }
                    )
                else:
                    # `mean` pools the pairwise coefficients in Fisher z-space (an
                    # arithmetic mean of correlations is downward-biased). The
                    # spread columns (sd/quartiles/min/max) describe the raw pairwise
                    # distribution; note the pairs are not independent, so they
                    # understate the true sampling uncertainty.
                    sd_val = float(np.std(coeffs, ddof=1)) if len(coeffs) > 1 else 0.0
                    row_list.append(
                        {
                            "feature": f_name,
                            "mean": _fisher_mean(coeffs),
                            "median": float(np.median(coeffs)),
                            "sd": sd_val,
                            "q25": float(np.percentile(coeffs, 25)),
                            "q75": float(np.percentile(coeffs, 75)),
                            "min": float(np.min(coeffs)),
                            "max": float(np.max(coeffs)),
                        }
                    )

    # 4. Construct DataFrames and Apply FDR Corrections
    icc_df = pd.DataFrame(icc_rows)
    spearman_df = pd.DataFrame(spearman_rows)
    pearson_df = pd.DataFrame(pearson_rows)

    # FDR correction for ICC p-values
    if "p_value" in icc_df.columns:
        icc_df.insert(5, "p_fdr", _fdr_correct(icc_df["p_value"].to_numpy()))

    # FDR correction for two-reader correlation p-values
    if k_observers == 2:
        for df in [spearman_df, pearson_df]:
            if "p_value" in df.columns:
                df["p_fdr"] = _fdr_correct(df["p_value"].to_numpy())

    # Flag features whose ICC reaches the primary threshold (a single, honestly
    # named column — the threshold is a parameter, so a hardcoded "_0_80" name
    # would lie when primary_threshold != 0.80).
    icc_df["primary_icc_pass"] = icc_df["icc_2_1"] >= primary_threshold

    return {
        "Spearman": spearman_df,
        "Pearson": pearson_df,
        "ICC": icc_df,
    }


def _reproducibility_number_format(col_name: str, val: float) -> str | None:
    """Excel number format for a finite numeric reproducibility cell."""
    if "p_value" in col_name or "p_fdr" in col_name:
        return "0.00E+00" if val < 0.0001 else "0.0000"
    if "ms_" in col_name or "f_stat" in col_name:
        return "0.00"
    return "0.000"


def write_reproducibility_excel(results: dict[str, pd.DataFrame], path: str | Path) -> None:
    """Export reproducibility analysis results to a highly polished,
    formatted multi-sheet Excel file.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        The results dictionary returned by compute_reproducibility.
    path : str or Path
        Target file path for the Excel sheet.
    """
    sheets = {name: results[name] for name in ["Spearman", "Pearson", "ICC"] if name in results}
    write_styled_workbook(sheets, path, _reproducibility_number_format)


def plot_reproducibility_histograms(
    results: dict[str, pd.DataFrame],
    path: str | Path | None = None,
    primary_threshold: float = 0.80,
    title: str | None = None,
    dpi: int = 300,
    save_pdf: bool = False,
    save_tiff: bool = False,
    axes: Sequence[plt.Axes] | None = None,
    show_subplot_titles: bool = True,
    show_legend: bool = True,
) -> plt.Figure:
    """Generate accessible, high-contrast scientific histograms for reproducibility metrics.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        The results dictionary returned by compute_reproducibility.
    path : str or Path, optional
        If provided, saves the figure to the specified path.
    primary_threshold : float
        The cutoff line to display on the ICC histogram.
    title : str, optional
        Figure title.
    dpi : int, default=300
        The resolution in dots per inch (DPI) for saving the image.
    save_pdf : bool, default=False
        Whether to also save a PDF copy of the plot. Enabled globally by the
        ``SAVE_PDF`` environment variable.
    save_tiff : bool, default=False
        Whether to also save a TIFF copy of the plot. Enabled globally by the
        ``SAVE_TIFF`` environment variable. DPI is set by the ``TIFF_DPI`` environment
        variable (falling back to ``dpi``).
    axes : sequence of Axes, optional
        Custom axes to plot into.
    show_subplot_titles : bool, default=True
        Whether to show subplot titles.
    show_legend : bool, default=True
        Whether to show the legend of correlation types.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The created figure object.
    """
    # 1. Apply science plots formatting rules with sans-serif fonts and clean
    #    styling. Skip when drawing into caller-supplied axes (the caller has
    #    already applied the style) to avoid redundant global rcParam churn.
    if axes is None:
        apply_science_style(figure_titlesize=13)

    # 2. Check which sheets exist and extract data
    sheets_to_plot = []
    data_to_plot = []
    colors = []
    titles = []

    # Map sheets, colors, and extract arrays
    if "Spearman" in results:
        df = results["Spearman"]
        metric_col = "estimate" if "estimate" in df.columns else "mean"
        val = df[metric_col].dropna().to_numpy()
        sheets_to_plot.append("Spearman")
        data_to_plot.append(val)
        colors.append("#4F46E5")  # Vibrant Indigo (Brand Signature)
        titles.append("Spearman Correlation")

    if "Pearson" in results:
        df = results["Pearson"]
        metric_col = "estimate" if "estimate" in df.columns else "mean"
        val = df[metric_col].dropna().to_numpy()
        sheets_to_plot.append("Pearson")
        data_to_plot.append(val)
        colors.append("#F43F5E")  # Vibrant Rose/Coral (Brand Signature)
        titles.append("Pearson Correlation")

    if "ICC" in results:
        df = results["ICC"]
        val = df["icc_2_1"].dropna().to_numpy()
        sheets_to_plot.append("ICC")
        data_to_plot.append(val)
        colors.append("#0D9488")  # Vibrant Teal (Brand Signature)
        titles.append("Intraclass Correlation (ICC(2,1))")

    n_plots = len(sheets_to_plot)
    if n_plots == 0:
        raise ValueError("Results dict contains no plottable data sheets.")

    # Create figure subplots
    is_custom_axes = axes is not None
    if not is_custom_axes:
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), sharey=False)
        if n_plots == 1:
            axes = [axes]
    else:
        fig = axes[0].figure
        if len(axes) != n_plots:
            raise ValueError(f"Expected {n_plots} axes, but got {len(axes)}")

    # Helper for bounding boxes
    text_bbox = dict(
        facecolor="white",
        edgecolor="0.8",
        boxstyle="round,pad=0.3",
        alpha=0.9,
    )

    for idx, ax in enumerate(axes):
        val = data_to_plot[idx]
        color = colors[idx]
        t_val = titles[idx]
        sheet = sheets_to_plot[idx]

        # Enforce 1:1 physical aspect ratio for the plot box
        ax.set_box_aspect(1.0)

        # Guard against empty / all-NaN sheets (np.min has no identity on empty).
        vmin = float(np.min(val)) if val.size else 0.0

        # Draw histogram with high-contrast distinct boundaries
        ax.hist(
            val,
            bins=np.arange(-1.0, 1.05, 0.1) if vmin < 0 else np.arange(0.0, 1.05, 0.05),
            color=color,
            edgecolor="0.25",
            linewidth=0.8,
            alpha=0.85,
        )

        if show_subplot_titles:
            ax.set_title(t_val, weight="bold", pad=12)
        ax.set_xlabel("Value", labelpad=6)
        if idx == 0:
            ax.set_ylabel("Feature Count", labelpad=6)

        # Set clean limits and grids
        ax.set_xlim(min(-0.1, vmin - 0.1), 1.05)
        ax.grid(True, linestyle=":", alpha=0.5, color="0.7")

        # Summary box calculations
        mean_val = np.mean(val) if len(val) > 0 else np.nan
        median_val = np.median(val) if len(val) > 0 else np.nan
        n_total = len(val)

        if sheet == "ICC":
            n_pass = np.sum(val >= primary_threshold) if n_total > 0 else 0
            pass_rate = (n_pass / n_total) * 100 if n_total > 0 else np.nan
            stats_str = (
                f"Mean: {mean_val:.3f}\n"
                f"Median: {median_val:.3f}\n"
                f"Pass Rate: {pass_rate:.1f}% ({n_pass}/{n_total})"
            )
        else:
            stats_str = f"Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nFeatures: {n_total}"

        # Display summary box in top-left
        ax.text(
            0.05,
            0.95,
            stats_str,
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment="top",
            bbox=text_bbox,
        )

        # Annotate reference threshold line on the ICC plot (or correlation plots if desired)
        if sheet == "ICC":
            ax.axvline(primary_threshold, color="#CC3311", linestyle="--", linewidth=1.5)
            # Find a nice vertical height for the label to avoid overlap
            ylim = ax.get_ylim()
            ax.text(
                primary_threshold - 0.03,
                ylim[1] * 0.5,
                f"Threshold ({primary_threshold:.2f})",
                color="#CC3311",
                weight="bold",
                fontsize=9,
                horizontalalignment="right",
                bbox=dict(
                    facecolor="white",
                    edgecolor="#CC3311",
                    boxstyle="round,pad=0.2",
                    alpha=0.9,
                ),
            )

    # Render a legend of correlation types on the first axis if requested
    if show_legend:
        from matplotlib.patches import Patch
        legend_handles = []
        if "Spearman" in results:
            legend_handles.append(Patch(facecolor="#4F46E5", edgecolor="0.25", label="Spearman"))
        if "Pearson" in results:
            legend_handles.append(Patch(facecolor="#F43F5E", edgecolor="0.25", label="Pearson"))
        if "ICC" in results:
            legend_handles.append(Patch(facecolor="#0D9488", edgecolor="0.25", label="ICC(2,1)"))

        if legend_handles and len(axes) > 0:
            axes[0].legend(
                handles=legend_handles, loc="upper right", frameon=True,
                fontsize=8.5, framealpha=0.9,
            )

    if not is_custom_axes:
        if title is not None:
            fig.suptitle(title, weight="bold", fontsize=12)
        fig.tight_layout()
        _save_figure(fig, path, dpi, save_pdf, save_tiff)

    return fig


def plot_reproducibility(
    datasets: Sequence[pd.DataFrame | NDArray] | None = None,
    *,
    reproducibility_results: dict[str, pd.DataFrame] | None = None,
    catalog: Any = None,
    primary_threshold: float = 0.80,
    compute_kws: dict[str, Any] | None = None,
    synteny_kws: dict[str, Any] | None = None,
    grid_2x2: bool = False,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    show_legend: bool = True,
    show_subplot_titles: bool = True,
    path: str | Path | None = None,
    excel_path: str | Path | None = None,
    csv_dir: str | Path | None = None,
    dpi: int = 300,
    save_pdf: bool = False,
    save_tiff: bool = False,
) -> tuple[dict[str, pd.DataFrame], plt.Figure]:
    """Unified wrapper to compute reproducibility and plot combined histograms + synteny.

    The call runs a four-stage pipeline and each parameter belongs to one stage:

    1. **Input** -- supply *either* ``datasets`` (raw replicates) *or*
       ``reproducibility_results`` (precomputed); ``catalog`` is shared.
    2. **Compute** (only when ``datasets`` is given) -- forwarded to
       :func:`compute_reproducibility`.
    3. **Plot** -- histogram panels, the synteny panel, and overall layout.
    4. **Export / save** -- tabular reports and the figure image.

    Parameters
    ----------
    datasets : sequence of pd.DataFrame or ndarray, optional
        [input] Replicate datasets (e.g. Reader 1, Reader 2). If None,
        ``reproducibility_results`` must be provided. If both are given,
        ``datasets`` takes precedence.
    reproducibility_results : dict of pd.DataFrame, optional
        [input] Precomputed results dictionary. Ignored when ``datasets`` is given.
    catalog : FeatureCatalog or pd.DataFrame, optional
        [input] Resolves feature groups/families and discretization params; shared
        by the compute and synteny stages.
    primary_threshold : float, default=0.80
        [compute + plot] Cutoff for reproducibility retention; also drawn on the
        ICC histogram.
    compute_kws : dict, optional
        [compute] Extra keyword arguments forwarded to
        :func:`compute_reproducibility` (e.g. ``features``, ``configs``,
        ``families``, ``family_groups``, ``groups``, ``min_valid_samples``,
        ``bootstrap_iterations``). Ignored when ``datasets`` is None.
    synteny_kws : dict, optional
        [synteny] Extra keyword arguments forwarded to
        :func:`~eigenradiomics.plotting.plot_reproducibility_synteny` (e.g.
        ``metric``, ``order``, ``group_by``, ``thresholds``,
        ``show_family_ribbon``, ``show_discretisation_ribbon``,
        ``observer_labels``).
    grid_2x2 : bool, default=False
        [layout] Arrange panels in a 2x2 grid. Requires all three metrics
        (Spearman, Pearson, ICC); otherwise falls back to the stacked layout.
    figsize : tuple, optional
        [layout] Figure size.
    title : str, optional
        [layout] Overall figure title.
    show_legend : bool, default=True
        [layout] Show the legends for histograms and synteny plots.
    show_subplot_titles : bool, default=True
        [layout] Show histogram subplot titles.
    path : str or Path, optional
        [output] Save path for the combined plot image.
    excel_path : str or Path, optional
        [output] Save path for the Excel workbook report.
    csv_dir : str or Path, optional
        [output] Save directory for CSV sheets.
    dpi : int, default=300
        [output] Save resolution in DPI.
    save_pdf : bool, default=False
        [output] Also save a PDF copy.
    save_tiff : bool, default=False
        [output] Also save a TIFF copy.

    Returns
    -------
    results : dict[str, pd.DataFrame]
        The results dictionary containing Spearman, Pearson, and ICC metrics.
    fig : matplotlib.pyplot.Figure
        The created combined figure object.
    """
    if datasets is None and reproducibility_results is None:
        raise ValueError("Either 'datasets' or 'reproducibility_results' must be provided.")
    if datasets is not None and reproducibility_results is not None:
        warnings.warn(
            "Both 'datasets' and 'reproducibility_results' were provided; "
            "'datasets' takes precedence and 'reproducibility_results' is ignored.",
            stacklevel=2,
        )

    # 1. Compute reproducibility if datasets are provided
    if datasets is not None:
        results = compute_reproducibility(
            datasets,
            catalog=catalog,
            primary_threshold=primary_threshold,
            **(compute_kws or {}),
        )
    else:
        results = reproducibility_results

    # 2. Generate combined figure
    apply_science_style()

    # Count sheets to plot for histograms
    sheets_to_plot = [name for name in ["Spearman", "Pearson", "ICC"] if name in results]
    n_plots = len(sheets_to_plot)
    if n_plots == 0:
        raise ValueError("Results dict contains no plottable data sheets.")

    # A 2x2 grid only fits cleanly with all three histograms + synteny; with
    # fewer metrics it leaves an empty cell, so fall back to the stacked layout.
    use_grid_2x2 = grid_2x2 and n_plots == 3
    if grid_2x2 and not use_grid_2x2:
        warnings.warn(
            "grid_2x2=True requires all three metrics (Spearman, Pearson, ICC); "
            f"only {n_plots} present. Falling back to the stacked layout.",
            stacklevel=2,
        )

    if figsize is None:
        figsize = (10.0, 10.0) if use_grid_2x2 else (11.0, 7.5)

    fig = plt.figure(figsize=figsize, layout="constrained")

    active_axes = []

    if use_grid_2x2:
        # Create a 2x2 GridSpec
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        hist_axes = []

        if "Spearman" in results:
            spearman_ax = fig.add_subplot(gs[0, 0])
            hist_axes.append(spearman_ax)
            active_axes.append(spearman_ax)
        else:
            spearman_ax = None

        if "Pearson" in results:
            pearson_ax = fig.add_subplot(gs[0, 1])
            hist_axes.append(pearson_ax)
            active_axes.append(pearson_ax)
        else:
            pearson_ax = None

        if "ICC" in results:
            icc_ax = fig.add_subplot(gs[1, 0])
            hist_axes.append(icc_ax)
            active_axes.append(icc_ax)
        else:
            icc_ax = None

        # Synteny cell is gs[1, 1]
        if show_legend:
            gs_synteny = gs[1, 1].subgridspec(2, 1, height_ratios=[2.0, 0.8], hspace=0.2)
            synteny_ax = fig.add_subplot(gs_synteny[0, 0])
            gs_leg = gs_synteny[1, 0].subgridspec(1, 3)
            legend_axes = [
                fig.add_subplot(gs_leg[0, 0]),
                fig.add_subplot(gs_leg[0, 1]),
                fig.add_subplot(gs_leg[0, 2]),
            ]
        else:
            synteny_ax = fig.add_subplot(gs[1, 1])
            legend_axes = None

        synteny_ax.set_box_aspect(1.0)
        active_axes.append(synteny_ax)

    else:
        # Stacked layout: the histogram row on top, a full-width synteny panel at
        # half the histogram-row height below, and an optional legend strip beneath.
        # The legend row needs a healthy share (0.9) so the family legend fits
        # without the constrained-layout solver collapsing it at high feature counts.
        if show_legend:
            gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.0, 0.9], hspace=0.08)
        else:
            gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.08)

        gs_hist = gs[0, 0].subgridspec(1, n_plots, wspace=0.25)
        hist_axes = [fig.add_subplot(gs_hist[0, i]) for i in range(n_plots)]
        active_axes.extend(hist_axes)

        synteny_ax = fig.add_subplot(gs[1, 0])  # full width, no side insets
        active_axes.append(synteny_ax)

        if show_legend:
            gs_leg = gs[2, 0].subgridspec(1, 3)
            legend_axes = [fig.add_subplot(gs_leg[0, i]) for i in range(3)]
        else:
            legend_axes = None

    # Draw Histograms
    plot_reproducibility_histograms(
        results,
        primary_threshold=primary_threshold,
        axes=hist_axes,
        show_subplot_titles=show_subplot_titles,
        show_legend=show_legend,
    )

    # Draw Synteny Plot
    from eigenradiomics.plotting import plot_reproducibility_synteny
    plot_reproducibility_synteny(
        results,
        catalog=catalog,
        ax=synteny_ax,
        legend_axes=legend_axes,
        show_legend=show_legend,
        **(synteny_kws or {}),
    )

    # Label subfigures (A, B, C, D...). Histograms hang the label in their left
    # margin (ha="right") so it clears the centred subplot title. The full-width
    # synteny panel has no y-axis margin and sits flush against the figure's left
    # edge, so its label is anchored inside the top-left corner (ha="left") to
    # avoid being pushed off-canvas.
    letter_idx = 0
    for ax in active_axes:
        if ax is None:
            continue
        is_synteny = ax is synteny_ax
        ax.annotate(
            chr(ord("A") + letter_idx),
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(0.0, 6.0) if is_synteny else (-2.0, 6.0),
            textcoords="offset points",
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="left" if is_synteny else "right",
            annotation_clip=False,
            in_layout=False,
        )
        letter_idx += 1

    # Overall title
    if title:
        fig.suptitle(title, weight="bold", fontsize=14)

    # Draw a single black border tight around the legend block (stacked layout
    # only). Finalise the constrained-layout positions first (so the box matches
    # the rendered legend extents, including any suptitle reflow), then frame the
    # union of the legends.
    if show_legend and legend_axes and not use_grid_2x2:
        from matplotlib.patches import Rectangle

        fig.draw_without_rendering()
        legends = [ax.get_legend() for ax in legend_axes]
        legends = [lg for lg in legends if lg is not None]
        if legends:
            inv = fig.transFigure.inverted()
            corners = []
            for lg in legends:
                ext = lg.get_window_extent()
                corners.append(inv.transform((ext.x0, ext.y0)))
                corners.append(inv.transform((ext.x1, ext.y1)))
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            pad = 0.012
            fig.add_artist(
                Rectangle(
                    (min(xs) - pad, min(ys) - pad),
                    (max(xs) - min(xs)) + 2 * pad,
                    (max(ys) - min(ys)) + 2 * pad,
                    transform=fig.transFigure,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.0,
                    zorder=5,
                    in_layout=False,
                )
            )

    # Save figure in desired formats. The figure uses the constrained layout
    # engine, so disable bbox_inches="tight" (mixing the two shifts margins and
    # can clip the out-of-axes subfigure letters).
    _save_figure(fig, path, dpi, save_pdf, save_tiff, bbox_inches=None)

    # Export tabular reports only after the figure is built and saved, so a
    # plotting failure does not leave partial Excel/CSV outputs on disk.
    if excel_path is not None:
        write_reproducibility_excel(results, excel_path)

    if csv_dir is not None:
        csv_dir_path = Path(csv_dir)
        csv_dir_path.mkdir(parents=True, exist_ok=True)
        for sheet_name, df in results.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(csv_dir_path / f"{sheet_name}.csv", index=False)

    return results, fig
