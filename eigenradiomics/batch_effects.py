"""Batch effect diagnostics and estimation framework for radiomics."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from eigenradiomics._excel import write_styled_workbook
from eigenradiomics._features import resolve_analysis_features
from eigenradiomics._plotting import apply_science_style
from eigenradiomics._stats import (
    _fdr_correct,
    anova_effect,
    kruskal_effect,
    levene_test,
    permanova_euclidean,
)
from eigenradiomics.preprocessing._feature_remover import _load_catalog
from eigenradiomics.preprocessing._prep import RadiomicsPrepTransformer


def _feature_qc(
    values: pd.DataFrame,
    batch: pd.Series,
    max_missing_fraction: float,
    min_valid_samples: int,
    min_valid_per_batch: int,
    min_batches_per_feature: int,
    min_unique_values: int,
) -> pd.DataFrame:
    """Calculate feature QC flags and missing statistics."""
    rows = []
    batches = sorted(batch.astype(str).unique())

    for feature in values.columns:
        series = pd.to_numeric(values[feature], errors="coerce")
        valid = series.notna()
        per_batch = series.groupby(batch.astype(str)).apply(lambda s: int(s.notna().sum()))
        valid_batches = int((per_batch >= min_valid_per_batch).sum())

        row = {
            "feature": feature,
            "n_samples": int(len(series)),
            "n_valid": int(valid.sum()),
            "missing_fraction": float(1 - valid.mean()),
            "n_unique": int(series.nunique(dropna=True)),
            "valid_batches": valid_batches,
            "min_valid_per_batch": int(per_batch.min()) if not per_batch.empty else 0,
            "keep": bool(
                valid.sum() >= min_valid_samples
                and (1 - valid.mean()) <= max_missing_fraction
                and series.nunique(dropna=True) >= min_unique_values
                and valid_batches >= min_batches_per_feature
            ),
        }
        for b in batches:
            row[f"n_batch_{b}"] = int(per_batch.get(b, 0))
        rows.append(row)

    return pd.DataFrame(rows)


def _compute_feature_level_stats(
    raw_df: pd.DataFrame,
    trans_df: pd.DataFrame,
    batch_series: pd.Series,
) -> pd.DataFrame:
    """Run ANOVA, Kruskal-Wallis, and Levene tests on a set of features.

    ANOVA (and its ``eta_squared``) is computed on the *transformed* matrix, which
    is closer to the test's normality assumption; the rank-based Kruskal-Wallis
    (``epsilon_squared``) and the Levene variance test use the *raw* matrix. So
    ``eta_squared`` and ``epsilon_squared`` in one row summarize different inputs.
    """
    batches = sorted(batch_series.astype(str).unique())
    features = raw_df.columns.tolist()
    rows = []

    for feature in features:
        raw_series = pd.to_numeric(raw_df[feature], errors="coerce")
        trans_series = pd.to_numeric(trans_df[feature], errors="coerce")

        model_groups = [
            trans_series.loc[batch_series.astype(str) == b].dropna().to_numpy(float)
            for b in batches
        ]
        raw_groups = [
            raw_series.loc[batch_series.astype(str) == b].dropna().to_numpy(float)
            for b in batches
        ]

        f_stat, anova_p, eta2 = anova_effect(model_groups)
        h_stat, kruskal_p, epsilon2 = kruskal_effect(raw_groups)
        levene_stat, levene_p = levene_test(raw_groups)

        row = {
            "feature": feature,
            "n_valid": int(raw_series.notna().sum()),
            "n_batches": int(sum(len(g) > 0 for g in raw_groups)),
            "anova_f": f_stat,
            "anova_p": anova_p,
            "eta_squared": eta2,
            "kruskal_h": h_stat,
            "kruskal_p": kruskal_p,
            "epsilon_squared": epsilon2,
            "levene_stat": levene_stat,
            "levene_p": levene_p,
        }
        for b in batches:
            vals = raw_series.loc[batch_series.astype(str) == b]
            row[f"center_{b}_n"] = int(vals.notna().sum())
            row[f"center_{b}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"center_{b}_median"] = float(vals.median()) if vals.notna().any() else np.nan
        rows.append(row)

    table = pd.DataFrame(rows)
    for p_col, q_col in [
        ("anova_p", "anova_q"),
        ("kruskal_p", "kruskal_q"),
        ("levene_p", "levene_q"),
    ]:
        if p_col in table.columns:
            table[q_col] = _fdr_correct(table[p_col].to_numpy())

    return table


def _compute_global_diagnostics(
    X_global: pd.DataFrame,
    batch_series: pd.Series,
    matrix_label: str,
    permanova_components: int,
    permutations: int,
) -> tuple[dict[str, Any], pd.DataFrame, PCA]:
    """Calculate PCA explained variances, Silhouette scores, and permutation pseudo-F PERMANOVA."""
    n_samples, n_features = X_global.shape
    max_components = min(permanova_components, n_samples - 1, n_features)

    pca = PCA(n_components=max_components, random_state=42)
    scores_arr = pca.fit_transform(X_global)
    scores = pd.DataFrame(
        scores_arr,
        index=X_global.index,
        columns=[f"PC{i + 1}" for i in range(max_components)],
    )

    batch_subset = batch_series.loc[X_global.index]
    f_stat, r2, permanova_p = permanova_euclidean(
        scores,
        batch_subset,
        permutations=permutations,
        random_state=42,
    )

    # PC1 and PC2 one-way ANOVA p-values
    groups_pc1 = [
        scores["PC1"].loc[batch_subset.astype(str) == b].dropna().to_numpy(float)
        for b in sorted(batch_subset.astype(str).unique())
    ]
    _, pc1_anova_p, _ = anova_effect(groups_pc1)

    pc2_anova_p = np.nan
    if "PC2" in scores.columns:
        groups_pc2 = [
            scores["PC2"].loc[batch_subset.astype(str) == b].dropna().to_numpy(float)
            for b in sorted(batch_subset.astype(str).unique())
        ]
        _, pc2_anova_p, _ = anova_effect(groups_pc2)

    silhouette = np.nan
    if batch_subset.nunique() > 1 and len(batch_subset) > batch_subset.nunique():
        try:
            silhouette = float(
                silhouette_score(scores.iloc[:, : min(10, scores.shape[1])], batch_subset)
            )
        except ValueError:  # pragma: no cover - only on degenerate label sets QC prevents
            silhouette = np.nan

    diag_dict = {
        "matrix": matrix_label,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_batches": int(batch_subset.nunique()),
        "pc_components_used": int(scores.shape[1]),
        "pc1_variance": float(pca.explained_variance_ratio_[0])
        if len(pca.explained_variance_ratio_)
        else np.nan,
        "pc2_variance": float(pca.explained_variance_ratio_[1])
        if len(pca.explained_variance_ratio_) > 1
        else np.nan,
        "pc1_anova_p": pc1_anova_p,
        "pc2_anova_p": pc2_anova_p,
        "permanova_f": f_stat,
        "permanova_r2": r2,
        "permanova_p": permanova_p,
        "silhouette_pc": silhouette,
    }

    return diag_dict, scores, pca


def _annotate_features(table: pd.DataFrame, catalog: pd.DataFrame | None) -> pd.DataFrame:
    """Merge catalog descriptors into a feature statistics table if available."""
    if catalog is None or catalog.empty:
        return table

    # Match based on feature name keys
    has_legacy = "feature" in catalog.columns
    if not has_legacy and {"config", "feature_key"}.issubset(catalog.columns):
        catalog = catalog.copy()
        catalog["feature"] = (
            catalog["config"].astype(str) + "__" + catalog["feature_key"].astype(str)
        )

    annotated = table.merge(
        catalog,
        left_on="feature",
        right_on="feature",
        how="left",
        suffixes=("", "_catalog"),
    )
    leading = [
        "feature",
        "config",
        "feature_key",
        "feature_name",
        "family",
        "family_group",
    ]
    leading = [col for col in leading if col in annotated.columns]
    other = [col for col in annotated.columns if col not in leading]
    return annotated[leading + other]


def compute_batch_effects(
    X: pd.DataFrame | NDArray,
    batch: Sequence[Any] | NDArray,
    *,
    features: Any = None,
    configs: Any = None,
    families: Any = None,
    family_groups: Any = None,
    catalog: Any = None,
    pipeline: TransformerMixin | None = None,
    max_missing_fraction: float = 0.50,
    min_valid_samples: int = 20,
    min_valid_per_batch: int = 5,
    min_batches_per_feature: int = 3,
    min_unique_values: int = 3,
    global_missing_strategy: str = "complete-features",
    permanova_components: int = 20,
    permutations: int = 999,
    no_combat: bool = False,
    combat_mean_only: bool = False,
    combat_nonparametric: bool = False,
    combat_reference_batch: Any = None,
    combat_covariates: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Evaluate scanner/center-associated batch effects across a radiomics dataset.

    Parameters
    ----------
    X : pd.DataFrame or ndarray of shape (n_samples, n_features)
        Wide-format feature matrix.
    batch : sequence of shape (n_samples,)
        Center/batch identifiers corresponding to each sample.
    features, configs, families, family_groups, catalog : selectors
        Standard Pictologics selectors used to isolate target features.
    pipeline : TransformerMixin, default=RadiomicsPrepTransformer()
        Fitted preprocessing pipeline. Safe cloned before fit.
    max_missing_fraction : float, default=0.50
        Maximum permitted fraction of missing samples per feature.
    min_valid_samples : int, default=20
        Minimum overall non-NaN samples required per feature.
    min_valid_per_batch : int, default=5
        Minimum valid non-NaN samples required inside each batch to count as valid.
    min_batches_per_feature : int, default=3
        Minimum number of valid batches required per feature.
    min_unique_values : int, default=3
        Minimum unique values required per feature.
    global_missing_strategy : {"complete-features", "complete-cases", "median-impute"}
        How to form the complete global matrix required for PCA/PERMANOVA and ComBat.
    permanova_components : int, default=20
        Number of principal components used for PERMANOVA testing.
    permutations : int, default=999
        Number of permutation steps for PERMANOVA p-value calculation.
    no_combat : bool, default=False
        If True, skips the ComBat sensitivity diagnostic.
    combat_mean_only : bool, default=False
        If True, performs location-only ComBat.
    combat_nonparametric : bool, default=False
        If True, estimates ComBat priors non-parametrically.
    combat_reference_batch : Any, default=None
        Optionally specifies a batch to use as reference.
    combat_covariates : pd.DataFrame, default=None
        Categorical covariates to preserve during ComBat.

    Returns
    -------
    results : dict[str, pd.DataFrame]
        Multi-sheet statistics tables containing batch-effect diagnostics.
    """
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    batch_series = pd.Series(batch)

    # 1. Quality Control validation and index matching
    if isinstance(X, pd.DataFrame):
        if set(X.index) != set(batch_series.index):
            missing = list(set(X.index) - set(batch_series.index))
            unexpected = list(set(batch_series.index) - set(X.index))
            raise ValueError(
                "Indices of X and batch do not match. "
                f"Missing in batch: {missing[:5]}. Unexpected in batch: {unexpected[:5]}."
            )
        # Reorder batch to align perfectly with X.index
        batch_series = batch_series.reindex(index=X.index)
    else:
        if X_df.shape[0] != len(batch_series):
            raise ValueError(
                f"Length of batch ({len(batch_series)}) must match "
                f"number of samples in X ({X_df.shape[0]})."
            )
        batch_series.index = X_df.index

    # Load catalog (DataFrame, FeatureCatalog, or CSV path) — same handling as
    # compute_reproducibility, so dataset.catalog can be passed to either.
    catalog_df = _load_catalog(catalog)

    # 2. Resolve features to analyze (selectors, else every numeric column).
    features_to_analyze = resolve_analysis_features(
        X_df,
        features=features,
        configs=configs,
        families=families,
        family_groups=family_groups,
        catalog=catalog_df,
    )

    if len(features_to_analyze) == 0:
        raise ValueError("No features selected for batch effects analysis.")

    # 3. Fit and Apply Preprocessing Pipeline
    X_subset = X_df[features_to_analyze].copy()
    actual_pipeline = pipeline if pipeline is not None else RadiomicsPrepTransformer()
    clf_pipeline = clone(actual_pipeline)
    X_trans_raw = clf_pipeline.fit_transform(X_subset)

    # Standardize back to DataFrame format if pipeline returns ndarray
    if not isinstance(X_trans_raw, pd.DataFrame):
        X_trans = pd.DataFrame(X_trans_raw, index=X_subset.index, columns=X_subset.columns)
    else:
        X_trans = X_trans_raw

    # 4. Feature QC Checks and Flagging
    qc_table = _feature_qc(
        X_subset,
        batch_series,
        max_missing_fraction,
        min_valid_samples,
        min_valid_per_batch,
        min_batches_per_feature,
        min_unique_values,
    )

    kept_features = qc_table.loc[qc_table["keep"], "feature"].tolist()
    if not kept_features:
        raise ValueError("No features passed QC filtering checks.")

    X_qc_raw = X_subset[kept_features]
    X_qc_trans = X_trans[kept_features]

    # 5. Core Feature-Level Batch effect Statistics
    feature_stats = _compute_feature_level_stats(X_qc_raw, X_qc_trans, batch_series)
    feature_stats_annotated = _annotate_features(feature_stats, catalog_df)

    # 6. Form complete global matrix for PCA and PERMANOVA
    X_clean = X_qc_trans.replace([np.inf, -np.inf], np.nan)
    if global_missing_strategy == "complete-features":
        X_global = X_clean.dropna(axis=1, how="any")
    elif global_missing_strategy == "complete-cases":
        X_global = X_clean.dropna(axis=0, how="any")
    elif global_missing_strategy == "median-impute":
        X_global = X_clean.apply(lambda col: col.fillna(col.median()), axis=0).dropna(
            axis=1, how="any"
        )
    else:
        raise ValueError(f"Unsupported global missing strategy: {global_missing_strategy}")

    # 7. Global Multivariate diagnostics
    global_rows = []
    pca_results = {}
    if X_global.shape[0] >= 3 and X_global.shape[1] >= 2:
        diag, scores, pca_model = _compute_global_diagnostics(
            X_global, batch_series, "raw_transformed", permanova_components, permutations
        )
        global_rows.append(diag)
        pca_results[("raw_transformed", "scores")] = scores
        pca_results[("raw_transformed", "pca")] = pca_model
    else:
        diag = {
            "matrix": "raw_transformed",
            "n_samples": int(X_global.shape[0]),
            "n_features": int(X_global.shape[1]),
            "n_batches": int(batch_series.loc[X_global.index].nunique())
            if len(X_global)
            else 0,
            "pc_components_used": 0,
            "pc1_variance": np.nan,
            "pc2_variance": np.nan,
            "pc1_anova_p": np.nan,
            "pc2_anova_p": np.nan,
            "permanova_f": np.nan,
            "permanova_r2": np.nan,
            "permanova_p": np.nan,
            "silhouette_pc": np.nan,
        }
        global_rows.append(diag)

    # 8. ComBat Sensitivity Analysis
    combat_stats_annotated = pd.DataFrame()
    combat_replacements = pd.DataFrame()
    run_combat_sensitivity = not no_combat and (X_global.shape[0] >= 3 and X_global.shape[1] >= 2)

    if run_combat_sensitivity:
        try:
            from inmoose.pycombat import pycombat_norm

            inmoose_installed = True
        except ImportError:
            inmoose_installed = False
            warnings.warn(
                "ComBat sensitivity diagnostics skipped: `inmoose` library not found.",
                UserWarning,
                stacklevel=2,
            )

        if inmoose_installed:
            # Check variance in complete matrix
            finite_variance = X_global.std(axis=0) > 1e-12
            combat_features = finite_variance[finite_variance].index.tolist()
            skipped_features = int((~finite_variance).sum())

            if combat_features:
                covar_mod = None
                if combat_covariates is not None:
                    missing = X_global.index.difference(combat_covariates.index)
                    if len(missing) > 0:
                        raise ValueError(
                            f"combat_covariates is missing {len(missing)} sample(s) present "
                            f"in X (e.g. {list(missing[:3])}); its index must cover X's index."
                        )
                    covar_mod = combat_covariates.loc[X_global.index].astype("category")

                # Transpose complete matrix for inmoose (features x samples)
                combat_input = X_global[combat_features].T
                corrected_trans = pycombat_norm(
                    combat_input,
                    batch=batch_series.loc[X_global.index].tolist(),
                    covar_mod=covar_mod,
                    par_prior=not combat_nonparametric,
                    mean_only=combat_mean_only,
                    ref_batch=combat_reference_batch,
                )

                if isinstance(corrected_trans, pd.DataFrame):
                    corrected_df = corrected_trans.T
                else:
                    corrected_df = pd.DataFrame(
                        np.asarray(corrected_trans).T,
                        index=X_global.index,
                        columns=combat_features,
                    )

                corrected_df = corrected_df.replace([np.inf, -np.inf], np.nan)
                nonfinite_mask = corrected_df.isna()
                n_vals_replaced = 0
                n_feats_replaced = 0

                if nonfinite_mask.any().any():
                    n_vals_replaced = int(nonfinite_mask.to_numpy().sum())
                    n_feats_replaced = int(nonfinite_mask.any(axis=0).sum())
                    corrected_df = corrected_df.mask(nonfinite_mask, X_global[combat_features])
                    # Defensive median fill: unreachable in practice because the
                    # complete matrix used for the mask has no missing values.
                    residual_na = corrected_df.columns[corrected_df.isna().any(axis=0)]
                    for f in residual_na:  # pragma: no cover
                        fill_value = corrected_df[f].median(skipna=True)
                        if not np.isfinite(fill_value):
                            fill_value = 0.0
                        corrected_df[f] = corrected_df[f].fillna(float(fill_value))

                combat_replacements = pd.DataFrame(
                    [
                        {
                            "nonfinite_values_replaced": n_vals_replaced,
                            "nonfinite_features_replaced": n_feats_replaced,
                            "zero_variance_features_skipped": skipped_features,
                        }
                    ]
                )

                # Re-calculate statistics after ComBat
                combat_feature_stats = _compute_feature_level_stats(
                    X_qc_raw.loc[X_global.index, combat_features],
                    corrected_df,
                    batch_series,
                )
                combat_stats_annotated = _annotate_features(combat_feature_stats, catalog_df)

                # Re-calculate PCA/PERMANOVA on ComBat corrected matrix
                diag_c, scores_c, pca_c = _compute_global_diagnostics(
                    corrected_df, batch_series, "combat", permanova_components, permutations
                )
                global_rows.append(diag_c)
                pca_results[("combat", "scores")] = scores_c
                pca_results[("combat", "pca")] = pca_c

    # 9. Form high-level summary tables
    dataset_summary = pd.DataFrame(
        [
            {
                "n_samples": X_df.shape[0],
                "n_features_raw": X_df.shape[1],
                "n_features_selected": len(features_to_analyze),
                "n_features_kept_qc": len(kept_features),
                "global_missing_strategy": global_missing_strategy,
                "n_samples_global": X_global.shape[0],
                "n_features_global": X_global.shape[1],
                "features_excluded_from_global": len(kept_features) - X_global.shape[1],
                "samples_excluded_from_global": X_df.shape[0] - X_global.shape[0],
                "n_batches": batch_series.nunique(),
                "median_missing_fraction": float(qc_table["missing_fraction"].median()),
                "features_removed_qc": int((~qc_table["keep"]).sum()),
            }
        ]
    )

    batch_counts = batch_series.value_counts().sort_index().reset_index(name="n_samples")
    # Rename positionally: reset_index names the label column after the Series
    # (not "index") whenever the batch Series carries a name, which is the norm.
    batch_counts.columns = ["Batch", "n_samples"]
    global_summary = pd.DataFrame(global_rows)

    feature_summaries = []
    # Raw feature level summary
    feature_summaries.append(
        {
            "matrix": "raw_transformed",
            "n_features": len(feature_stats),
            "median_eta_squared": float(feature_stats["eta_squared"].median()),
            "p90_eta_squared": float(feature_stats["eta_squared"].quantile(0.90)),
            "median_epsilon_squared": float(feature_stats["epsilon_squared"].median()),
            "p90_epsilon_squared": float(feature_stats["epsilon_squared"].quantile(0.90)),
            "anova_q_le_0_05": int((feature_stats["anova_q"] <= 0.05).sum()),
            "kruskal_q_le_0_05": int((feature_stats["kruskal_q"] <= 0.05).sum()),
            "levene_q_le_0_05": int((feature_stats["levene_q"] <= 0.05).sum()),
        }
    )

    if not combat_stats_annotated.empty:
        feature_summaries.append(
            {
                "matrix": "combat",
                "n_features": len(combat_stats_annotated),
                "median_eta_squared": float(combat_stats_annotated["eta_squared"].median()),
                "p90_eta_squared": float(combat_stats_annotated["eta_squared"].quantile(0.90)),
                "median_epsilon_squared": float(
                    combat_stats_annotated["epsilon_squared"].median()
                ),
                "p90_epsilon_squared": float(
                    combat_stats_annotated["epsilon_squared"].quantile(0.90)
                ),
                "anova_q_le_0_05": int((combat_stats_annotated["anova_q"] <= 0.05).sum()),
                "kruskal_q_le_0_05": int((combat_stats_annotated["kruskal_q"] <= 0.05).sum()),
                "levene_q_le_0_05": int((combat_stats_annotated["levene_q"] <= 0.05).sum()),
            }
        )
    feature_summary = pd.DataFrame(feature_summaries)

    results_dict = {
        "dataset_summary": dataset_summary,
        "batch_counts": batch_counts,
        "global_diagnostics": global_summary,
        "feature_summary": feature_summary,
        "feature_qc": qc_table,
        "feature_stats": feature_stats_annotated,
    }

    if not combat_stats_annotated.empty:
        results_dict["combat_feature_stats"] = combat_stats_annotated
        results_dict["combat_adjustment_notes"] = combat_replacements

    # Save internal PCA coordinates for plotting referencing
    results_dict["_pca_results"] = pca_results  # Hidden key for plotting reference
    results_dict["_batch_series"] = batch_series  # Hidden key for plotting reference
    return results_dict


def _batch_effects_number_format(col_name: str, val: float) -> str | None:
    """Excel number format for a finite numeric batch-effects cell."""
    if "_p" in col_name or "_q" in col_name:
        return "0.00E+00" if val < 0.0001 else "0.0000"
    if "variance" in col_name or "missing" in col_name or "eta" in col_name:
        return "0.000"
    return "0.00"


def write_batch_effects_excel(results: dict[str, pd.DataFrame], path: str | Path) -> None:
    """Export batch effect analysis results to a highly formatted Excel workbook.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        The results dictionary returned by compute_batch_effects.
    path : str or Path
        Target file path for the Excel workbook.
    """
    # Skip internal tracking entries (prefixed with "_").
    sheets = {name: df for name, df in results.items() if not name.startswith("_")}
    write_styled_workbook(sheets, path, _batch_effects_number_format)


def plot_batch_effects(
    results: dict[str, pd.DataFrame],
    path: str | Path | None = None,
    primary_alpha: float = 0.05,
) -> plt.Figure:
    """Generate accessible, OUP-compliant scientific PCA scatterplots and statistic histograms.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        The results dictionary returned by compute_batch_effects.
    path : str or Path, optional
        If provided, saves the figure to the specified path.
    primary_alpha : float
        FDR alpha cutoff displayed on the ANOVA q-value histogram.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The created figure object.
    """
    apply_science_style(figure_titlesize=14)

    pca_results = results.get("_pca_results", {})
    batch_series = results.get("_batch_series", pd.Series(dtype=float))
    feature_stats = results["feature_stats"]

    has_combat = ("combat", "scores") in pca_results
    has_pca = ("raw_transformed", "scores") in pca_results

    n_cols = 1 + int(has_combat) + 1  # PCA panels + ANOVA q-value panel
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharey=False)
    if n_cols == 1:  # pragma: no cover - always >= 2 panels (PCA + histogram)
        axes = [axes]

    ax_idx = 0
    text_bbox = dict(facecolor="white", edgecolor="0.8", boxstyle="round,pad=0.2", alpha=0.9)

    # 1. PCA Before ComBat
    if has_pca:
        ax = axes[ax_idx]
        scores = pca_results[("raw_transformed", "scores")]
        pca_model = pca_results[("raw_transformed", "pca")]
        batches_subset = batch_series.loc[scores.index].astype(str)

        # Plot PCA scatter
        unique_batches = sorted(batches_subset.unique())
        plt.colormaps.get_cmap("tab10")

        for _idx, b in enumerate(unique_batches):
            mask = batches_subset == b
            ax.scatter(
                scores.loc[mask, "PC1"],
                scores.loc[mask, "PC2"],
                label=f"Batch {b}",
                alpha=0.85,
                edgecolor="0.25",
                linewidth=0.5,
                s=30,
            )

        ax.set_title("PCA Before ComBat", weight="bold", pad=12)
        ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", fontsize=8, edgecolor="0.8")
        ax_idx += 1

    # 2. PCA After ComBat
    if has_combat:
        ax = axes[ax_idx]
        scores_c = pca_results[("combat", "scores")]
        pca_model_c = pca_results[("combat", "pca")]
        batches_subset = batch_series.loc[scores_c.index].astype(str)

        unique_batches = sorted(batches_subset.unique())
        for _idx, b in enumerate(unique_batches):
            mask = batches_subset == b
            ax.scatter(
                scores_c.loc[mask, "PC1"],
                scores_c.loc[mask, "PC2"],
                label=f"Batch {b}",
                alpha=0.85,
                edgecolor="0.25",
                linewidth=0.5,
                s=30,
            )

        ax.set_title("PCA After ComBat", weight="bold", pad=12)
        ax.set_xlabel(f"PC1 ({pca_model_c.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca_model_c.explained_variance_ratio_[1]*100:.1f}%)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax_idx += 1

    # 3. ANOVA q-value Histogram
    ax = axes[ax_idx]
    q_vals = feature_stats["anova_q"].dropna().to_numpy()

    ax.hist(
        q_vals,
        bins=np.arange(0.0, 1.05, 0.05),
        color="#CD5C5C",  # Warm Indian Red
        edgecolor="0.25",
        linewidth=0.8,
        alpha=0.85,
    )

    ax.set_title("ANOVA FDR q-values", weight="bold", pad=12)
    ax.set_xlabel("FDR q-value", labelpad=6)
    ax.set_ylabel("Feature Count", labelpad=6)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Summary box
    significant_count = int((q_vals <= primary_alpha).sum())
    percent_sig = (significant_count / len(q_vals)) * 100 if len(q_vals) > 0 else 0.0
    summary_str = (
        f"Features: {len(q_vals)}\n"
        f"q <= {primary_alpha}: {significant_count} ({percent_sig:.1f}%)"
    )

    ax.text(0.95, 0.95, summary_str, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right", bbox=text_bbox)

    # Reference cut line
    ax.axvline(primary_alpha, color="#D32F2F", linestyle="--", linewidth=1.5)
    ylim = ax.get_ylim()
    ax.text(
        primary_alpha + 0.03,
        ylim[1] * 0.5,
        f"q = {primary_alpha}",
        color="#D32F2F",
        weight="bold",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#D32F2F", boxstyle="round,pad=0.2", alpha=0.9),
    )

    fig.tight_layout()

    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig
