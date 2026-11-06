"""Post-reduction functional analysis, hub identification, and group enrichment.

Provides transcriptomics-inspired metrics adapted for radiomics features:
- **Module/Component Membership (k_ME)**: Correlation between raw features and eigengenes.
- **Hub Feature Identification**: Extracting physical features representing each module.
- **Group/Family Over-Representation Analysis (ORA)**: Hypergeometric / Fisher's Exact tests
  to identify significant enrichment of feature groups within modules.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from eigenradiomics._stats import _fdr_correct


def compute_module_membership(
    X: pd.DataFrame | NDArray,
    eigengenes: pd.DataFrame | NDArray | None = None,
    *,
    reducer: Any = None,
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute Module Membership (k_ME) for features across eigengenes/components.

    Module Membership (k_ME) is defined as the correlation coefficient between each
    individual feature and the module/component eigengene. This indicates how strongly
    each feature aligns with the holistic representation of each cluster/module.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Original feature matrix. Can be a pandas DataFrame or numpy array.
    eigengenes : array-like of shape (n_samples, n_components), optional
        Eigengene/reduced coordinate matrix. If None, `reducer` must be provided.
    reducer : BaseReducer, optional
        Fitted reducer used to compute eigengenes on the fly if `eigengenes` is None.
    method : {"spearman", "pearson", "kendall"}, default="spearman"
        Correlation method. Spearman is recommended for robustness to monotonic non-linearities.

    Returns
    -------
    k_ME : pandas.DataFrame of shape (n_features, n_components)
        Correlation values, indexed by feature name, with columns naming the components.
    """
    if eigengenes is None and reducer is None:
        raise ValueError("Either 'eigengenes' or a fitted 'reducer' must be provided.")
    if method not in {"spearman", "pearson", "kendall"}:
        raise ValueError(f"method must be 'spearman', 'pearson', or 'kendall', got {method!r}.")

    feature_names: Sequence[str] | None = None
    if isinstance(X, pd.DataFrame):
        X_arr = X.to_numpy()
        feature_names = X.columns
    else:
        X_arr = np.asarray(X)

    if reducer is not None:
        if not hasattr(reducer, "transform"):
            raise TypeError("Provided 'reducer' is not a valid scikit-learn estimator.")
        Y = reducer.transform(X)
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
        elif hasattr(reducer, "feature_names_in_"):
            feature_names = reducer.feature_names_in_

        # Determine column names for components
        if hasattr(reducer, "module_names_") and hasattr(reducer, "module_loadings_"):
            # WGCNAReducer: name each column by its module. A module can have
            # fewer components than requested, so count the stored loadings.
            comp_names = []
            for mod in reducer.module_names_:
                loadings = np.asarray(reducer.module_loadings_[mod])
                n_comp = loadings.shape[1] if loadings.ndim == 2 else 1
                if reducer.n_module_components == 1:
                    comp_names.append(mod)
                else:
                    comp_names.extend(f"{mod}_{comp}" for comp in range(n_comp))
        else:
            comp_names = list(reducer.get_feature_names_out())
    else:
        Y = np.asarray(eigengenes)
        if isinstance(eigengenes, pd.DataFrame):
            comp_names = list(eigengenes.columns)
            if isinstance(X, pd.DataFrame) and len(eigengenes) == len(X):
                # Pair samples by index label, not by row position.
                if not (eigengenes.index.is_unique and X.index.is_unique) or set(
                    eigengenes.index
                ) != set(X.index):
                    raise ValueError(
                        "X and eigengenes must have the same unique index labels; "
                        "pass NumPy arrays for positional pairing."
                    )
                Y = eigengenes.loc[X.index].to_numpy()
        else:
            comp_names = [f"component_{i}" for i in range(Y.shape[1])]

    Y_arr = np.asarray(Y)

    if X_arr.shape[0] != Y_arr.shape[0]:
        raise ValueError(
            f"Row count mismatch: X has {X_arr.shape[0]} samples, "
            f"but eigengenes have {Y_arr.shape[0]} samples."
        )

    n_features = X_arr.shape[1]
    n_components = Y_arr.shape[1]

    k_ME_matrix = np.zeros((n_features, n_components))

    for f_idx in range(n_features):
        x_vals = X_arr[:, f_idx]
        for c_idx in range(n_components):
            y_vals = Y_arr[:, c_idx]
            if method == "pearson":
                corr, _ = stats.pearsonr(x_vals, y_vals)
            elif method == "kendall":
                corr, _ = stats.kendalltau(x_vals, y_vals)
            else:
                corr, _ = stats.spearmanr(x_vals, y_vals)
            k_ME_matrix[f_idx, c_idx] = corr

    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(n_features)]

    return pd.DataFrame(k_ME_matrix, index=feature_names, columns=comp_names)


def identify_hub_features(
    X: pd.DataFrame | NDArray,
    cluster_labels: pd.Series | dict[str, str | int] | Sequence[Any],
    eigengenes: pd.DataFrame | NDArray | None = None,
    *,
    reducer: Any = None,
    top_n: int = 1,
    method: str = "spearman",
) -> pd.DataFrame:
    """Identify the top hub features for each module/cluster.

    Hub features are defined as the individual features within each cluster
    that have the highest absolute correlation (k_ME) with their assigned
    cluster's eigengene.

    Each cluster label must name an eigengene column: the label itself, or
    ``"<label>_0"`` (the first component of a multi-component WGCNA module).
    A cluster without a matching column is skipped with a warning; the function
    never maps a label to a column by position.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Original feature matrix.
    cluster_labels : pd.Series, dict, or sequence of shape (n_features,)
        Module / cluster assignment per feature.
    eigengenes : array-like of shape (n_samples, n_components), optional
        Eigengene coordinates. If None, `reducer` must be provided.
    reducer : BaseReducer, optional
        Fitted reducer used to compute eigengenes on the fly if `eigengenes` is None.
    top_n : int, default=1
        Number of top hub features to return per cluster.
    method : {"spearman", "pearson", "kendall"}, default="spearman"
        Correlation method for module membership estimation.

    Returns
    -------
    hubs : pandas.DataFrame
        DataFrame listing the identified hub features.
        Columns: `cluster`, `feature`, `k_ME`, `rank`.
    """
    k_me = compute_module_membership(X, eigengenes, reducer=reducer, method=method)

    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
    elif reducer is not None and hasattr(reducer, "feature_names_in_"):
        feature_names = list(reducer.feature_names_in_)
    else:
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]

    if isinstance(cluster_labels, pd.Series):
        labels_series = cluster_labels
    else:
        if isinstance(cluster_labels, dict):
            labels_series = pd.Series(cluster_labels)
        else:
            labels_series = pd.Series(cluster_labels, index=feature_names)

    labels_series = labels_series.reindex(k_me.index)

    unique_clusters = sorted(labels_series.dropna().unique())
    hub_records = []
    skipped: list[Any] = []

    for cluster in unique_clusters:
        cluster_mask = labels_series == cluster
        cluster_features = k_me.index[cluster_mask]

        col_name = next(
            (c for c in (cluster, str(cluster), f"{cluster}_0") if c in k_me.columns), None
        )
        if col_name is None:
            skipped.append(cluster)
            continue

        cluster_kme = k_me.loc[cluster_features, col_name]
        sorted_kme = cluster_kme.abs().sort_values(ascending=False)

        top_features = sorted_kme.index[:top_n]
        for rank_idx, feat in enumerate(top_features):
            original_val = cluster_kme.loc[feat]
            hub_records.append(
                {
                    "cluster": cluster,
                    "feature": feat,
                    "k_ME": original_val,
                    "rank": rank_idx + 1,
                }
            )

    if skipped:
        warnings.warn(
            f"{len(skipped)} cluster label(s) have no matching eigengene column and were "
            f"skipped: {', '.join(map(str, skipped[:5]))}.",
            UserWarning,
            stacklevel=2,
        )
    return pd.DataFrame(hub_records, columns=["cluster", "feature", "k_ME", "rank"])


def _single_column_enrichment(
    c_series: pd.Series,
    g_series: pd.Series,
    method: str,
    alternative: str,
) -> pd.DataFrame:
    """Run over-representation analysis between a cluster Series and a group Series."""
    common_idx = c_series.index.intersection(g_series.index)
    if len(common_idx) == 0:
        if len(c_series) == 0 or len(g_series) == 0:
            return pd.DataFrame(
                columns=[
                    "cluster",
                    "group",
                    "n_overlap",
                    "cluster_size",
                    "group_size",
                    "total_features",
                    "p_value",
                    "fdr_q_value",
                    "odds_ratio",
                ]
            )
        raise ValueError(
            "No overlapping feature names between cluster labels and group assignments."
        )

    c_sub = c_series.loc[common_idx]
    g_sub = g_series.loc[common_idx]
    # A feature without a cluster or group label is not part of the universe.
    labelled = c_sub.notna() & g_sub.notna()
    c_sub, g_sub = c_sub[labelled], g_sub[labelled]

    unique_clusters = sorted(c_sub.unique())
    unique_groups = sorted(g_sub.unique())
    total_n = int(labelled.sum())

    records = []

    for cluster in unique_clusters:
        for group in unique_groups:
            cluster_mask = c_sub == cluster
            group_mask = g_sub == group

            a = int((cluster_mask & group_mask).sum())
            b = int((cluster_mask & ~group_mask).sum())
            c = int((~cluster_mask & group_mask).sum())
            d = int((~cluster_mask & ~group_mask).sum())

            table = np.array([[a, b], [c, d]])

            if method == "fisher":
                res = stats.fisher_exact(table, alternative=alternative)
                odds_ratio = float(res.statistic)
                p_val = float(res.pvalue)
            elif method == "chi2":
                odds_ratio = (
                    float((a * d) / (b * c))
                    if (b * c) > 0
                    else (float(np.inf) if a * d > 0 else float(np.nan))
                )
                try:
                    chi2_res = stats.chi2_contingency(table, correction=True)
                    if alternative == "two-sided":
                        p_val = float(chi2_res.pvalue)
                    else:
                        # One-sided test: signed root of the 1-df statistic.
                        z = np.sign(a - chi2_res.expected_freq[0, 0]) * np.sqrt(chi2_res.statistic)
                        p_val = float(
                            stats.norm.sf(z) if alternative == "greater" else stats.norm.cdf(z)
                        )
                except ValueError:
                    p_val = float(np.nan)
            elif method == "hypergeometric":
                odds_ratio = (
                    float((a * d) / (b * c))
                    if (b * c) > 0
                    else (float(np.inf) if a * d > 0 else float(np.nan))
                )
                M = total_n
                n_pop = a + c
                n_draw = a + b
                if alternative == "greater":
                    p_val = float(stats.hypergeom.sf(a - 1, M, n_pop, n_draw))
                elif alternative == "less":
                    p_val = float(stats.hypergeom.cdf(a, M, n_pop, n_draw))
                else:  # two-sided
                    res = stats.fisher_exact(table, alternative="two-sided")
                    p_val = float(res.pvalue)
            else:
                raise ValueError(
                    f"method must be 'fisher', 'chi2', or 'hypergeometric', got {method!r}."
                )

            records.append(
                {
                    "cluster": cluster,
                    "group": group,
                    "n_overlap": a,
                    "cluster_size": a + b,
                    "group_size": a + c,
                    "total_features": total_n,
                    "p_value": p_val,
                    "odds_ratio": odds_ratio,
                }
            )

    return pd.DataFrame(records)


def compute_group_enrichment(
    cluster_labels: pd.Series | dict[str, str | int] | Sequence[Any],
    group_assignments: Any,
    *,
    feature_names: Sequence[str] | None = None,
    columns: str | Sequence[str] | None = None,
    method: str = "fisher",
    alternative: str = "greater",
) -> pd.DataFrame:
    """Perform Over-Representation Analysis (ORA) of groups within clusters.

    Tests whether features belonging to a particular functional group or metadata
    category (e.g., radiomics feature families, filter classes, compartments) are
    statistically enriched within specific modules.

    Parameters
    ----------
    cluster_labels : pd.Series, dict, or sequence
        Module / cluster assignment per feature, or mapping from feature to cluster.
    group_assignments : FeatureCatalog, pd.DataFrame, pd.Series, dict, or sequence
        Group assignments per feature or a catalog table.
    feature_names : sequence of str, optional
        Names of features, used to align inputs if passed as simple sequences.
    columns : str or sequence of str, optional
        When group_assignments is a FeatureCatalog or DataFrame, columns to test
        (e.g., ``["family", "family_group"]``). If None, tests all eligible string columns.
    method : {"fisher", "chi2", "hypergeometric"}, default="fisher"
        Statistical test to evaluate over-representation.
    alternative : {"greater", "less", "two-sided"}, default="greater"
        Alternative hypothesis direction. For ``"chi2"`` a one-sided p-value
        comes from the signed square root of the Yates-corrected statistic.

    Returns
    -------
    enrichment : pandas.DataFrame
        Detailed over-representation results table. Features without a cluster
        or a group label are not part of the background (``total_features``).
        Columns: `cluster`, `group`, `n_overlap`, `cluster_size`, `group_size`,
                 `total_features`, `p_value`, `fdr_q_value`, `odds_ratio`
                 (and `catalog_column` when multi-column/catalog input is used).
    """
    valid_methods = {"fisher", "chi2", "hypergeometric"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method!r}.")
    valid_alts = {"greater", "less", "two-sided"}
    if alternative not in valid_alts:
        raise ValueError(f"alternative must be one of {valid_alts}, got {alternative!r}.")

    # Coerce cluster_labels to Series
    if isinstance(cluster_labels, pd.Series):
        c_series = cluster_labels
    elif isinstance(cluster_labels, dict):
        c_series = pd.Series(cluster_labels)
    else:
        if feature_names is None:
            raise ValueError("feature_names is required when cluster_labels is a sequence.")
        c_series = pd.Series(cluster_labels, index=feature_names)

    # Check if group_assignments is a FeatureCatalog or DataFrame
    cat_frame = None
    if hasattr(group_assignments, "frame") and isinstance(group_assignments.frame, pd.DataFrame):
        cat_frame = group_assignments.frame.set_index("feature")
    elif isinstance(group_assignments, pd.DataFrame):
        if "feature" in group_assignments.columns:
            cat_frame = group_assignments.set_index("feature")
        else:
            cat_frame = group_assignments

    if cat_frame is not None:
        if columns is not None:
            cols_to_run = [columns] if isinstance(columns, str) else list(columns)
        else:
            cols_to_run = [
                c
                for c in cat_frame.columns
                if cat_frame[c].dtype == "object"
                or isinstance(cat_frame[c].dtype, pd.CategoricalDtype)
                or cat_frame[c].dtype == "string"
            ]
            if not cols_to_run:
                cols_to_run = list(cat_frame.columns)

        missing_cols = [c for c in cols_to_run if c not in cat_frame.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in catalog: {missing_cols}")

        all_results = []
        for col in cols_to_run:
            g_series = cat_frame[col].dropna()
            col_df = _single_column_enrichment(c_series, g_series, method, alternative)
            if len(col_df) > 0:
                col_df.insert(0, "catalog_column", col)
                col_df["fdr_q_value"] = _fdr_correct(col_df["p_value"].to_numpy())
                all_results.append(col_df)

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame(
            columns=[
                "catalog_column",
                "cluster",
                "group",
                "n_overlap",
                "cluster_size",
                "group_size",
                "total_features",
                "p_value",
                "fdr_q_value",
                "odds_ratio",
            ]
        )

    # Single Series / dict / sequence
    if isinstance(group_assignments, pd.Series):
        g_series = group_assignments
    elif isinstance(group_assignments, dict):
        g_series = pd.Series(group_assignments)
    else:
        if feature_names is None:
            raise ValueError("feature_names is required when group_assignments is a sequence.")
        g_series = pd.Series(group_assignments, index=feature_names)

    df = _single_column_enrichment(c_series, g_series, method, alternative)
    if len(df) > 0:
        df["fdr_q_value"] = _fdr_correct(df["p_value"].to_numpy())
    else:
        df["fdr_q_value"] = np.nan

    return df
