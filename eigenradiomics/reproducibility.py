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
    _icc_2_1_estimate,
)


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
                    sd_val = float(np.std(coeffs, ddof=1)) if len(coeffs) > 1 else 0.0
                    row_list.append(
                        {
                            "feature": f_name,
                            "mean": float(np.mean(coeffs)),
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

    # Add retention flags to ICC sheet
    icc_df["retained_ge_0_80"] = icc_df["icc_2_1"] >= primary_threshold
    icc_df["primary_icc_pass"] = icc_df["retained_ge_0_80"]

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

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The created figure object.
    """
    # 1. Apply science plots formatting rules with sans-serif fonts and clean styling
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
        colors.append("#4682B4")  # Steel Blue
        titles.append("Spearman Correlation")

    if "Pearson" in results:
        df = results["Pearson"]
        metric_col = "estimate" if "estimate" in df.columns else "mean"
        val = df[metric_col].dropna().to_numpy()
        sheets_to_plot.append("Pearson")
        data_to_plot.append(val)
        colors.append("#CD5C5C")  # Warm Indian Red
        titles.append("Pearson Correlation")

    if "ICC" in results:
        df = results["ICC"]
        val = df["icc_2_1"].dropna().to_numpy()
        sheets_to_plot.append("ICC")
        data_to_plot.append(val)
        colors.append("#008080")  # Muted Teal
        titles.append("Intraclass Correlation (ICC(2,1))")

    n_plots = len(sheets_to_plot)
    if n_plots == 0:
        raise ValueError("Results dict contains no plottable data sheets.")

    # Create figure subplots
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), sharey=False)
    if n_plots == 1:
        axes = [axes]

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
        title = titles[idx]
        sheet = sheets_to_plot[idx]

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

        ax.set_title(title, weight="bold", pad=12)
        ax.set_xlabel("Value", labelpad=6)
        if idx == 0:
            ax.set_ylabel("Feature Count", labelpad=6)

        # Set clean limits and grids
        ax.set_xlim(min(-0.1, vmin - 0.1), 1.05)
        ax.grid(True, linestyle=":", alpha=0.5, color="0.7")

        # Summary box calculations
        mean_val = np.mean(val) if len(val) > 0 else np.nan
        median_val = np.median(val) if len(val) > 0 else np.nan

        if sheet == "ICC":
            pass_rate = np.mean(val >= primary_threshold) * 100 if len(val) > 0 else np.nan
            stats_str = (
                f"Mean: {mean_val:.3f}\n"
                f"Median: {median_val:.3f}\n"
                f"Pass Rate: {pass_rate:.1f}%"
            )
        else:
            stats_str = f"Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nFeatures: {len(val)}"

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
            ax.axvline(primary_threshold, color="#D32F2F", linestyle="--", linewidth=1.5)
            # Find a nice vertical height for the label to avoid overlap
            ylim = ax.get_ylim()
            ax.text(
                primary_threshold - 0.03,
                ylim[1] * 0.5,
                f"Threshold ({primary_threshold:.2f})",
                color="#D32F2F",
                weight="bold",
                fontsize=9,
                horizontalalignment="right",
                bbox=dict(
                    facecolor="white",
                    edgecolor="#D32F2F",
                    boxstyle="round,pad=0.2",
                    alpha=0.9,
                ),
            )

    fig.tight_layout()

    if path is not None:
        # Create output directories if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig
