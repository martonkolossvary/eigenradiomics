"""Post-reduction functional analysis, hub identification, and group enrichment.

Provides transcriptomics-inspired metrics adapted for radiomics features:
- **Module/Component Membership (k_ME)**: Correlation between raw features and eigengenes.
- **Hub Feature Identification**: Extracting physical features representing each module.
- **Group/Family Over-Representation Analysis (ORA)**: Hypergeometric tests (Fisher's Exact)
  to identify significant enrichment of feature groups within modules.
"""

from __future__ import annotations

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
        if hasattr(reducer, "module_names_") and hasattr(reducer, "n_module_components"):
            # WGCNAReducer column naming mapping
            comp_names = []
            for mod in reducer.module_names_:
                if reducer.n_module_components == 1:
                    comp_names.append(mod)
                else:
                    for comp in range(reducer.n_module_components):
                        comp_names.append(f"{mod}_{comp}")
        else:
            comp_names = list(reducer.get_feature_names_out())
    else:
        Y = np.asarray(eigengenes)
        if isinstance(eigengenes, pd.DataFrame):
            comp_names = list(eigengenes.columns)
        else:
            comp_names = [f"component_{i}" for i in range(Y.shape[1])]

    if X_arr.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Row count mismatch: X has {X_arr.shape[0]} samples, "
            f"but eigengenes have {Y.shape[0]} samples."
        )

    n_features = X_arr.shape[1]
    n_components = Y.shape[1]

    k_ME_matrix = np.zeros((n_features, n_components))

    for f_idx in range(n_features):
        x_vals = X_arr[:, f_idx]
        for c_idx in range(n_components):
            y_vals = Y[:, c_idx]
            # Handle standard correlation checks
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

    # Coerce cluster_labels to pd.Series indexed by feature names
    if isinstance(cluster_labels, pd.Series):
        labels_series = cluster_labels
    else:  # pragma: no cover
        if isinstance(cluster_labels, dict):
            labels_series = pd.Series(cluster_labels)
        else:
            labels_series = pd.Series(cluster_labels, index=feature_names)

    # Align labels with the k_me index
    labels_series = labels_series.reindex(k_me.index)  # pragma: no cover

    unique_clusters = sorted(labels_series.dropna().unique())
    hub_records = []

    for cluster in unique_clusters:
        # Features belonging to this cluster
        cluster_mask = labels_series == cluster
        cluster_features = k_me.index[cluster_mask]

        if len(cluster_features) == 0:
            continue  # pragma: no cover

        # Find the column in k_me matching this cluster name.
        # Fallback: if names match, use the matched column. Otherwise, match by position.
        col_name = None
        if str(cluster) in k_me.columns:
            col_name = str(cluster)
        else:
            # Fallback mapping: find unique sorted clusters in labels
            all_sorted_labels = sorted(labels_series.dropna().unique())
            try:
                col_idx = all_sorted_labels.index(cluster)
                if col_idx < len(k_me.columns):
                    col_name = k_me.columns[col_idx]
            except ValueError:  # pragma: no cover
                pass

        if col_name is None:
            # Final fallback, just use the first column or skip
            col_name = k_me.columns[0] if len(k_me.columns) > 0 else None  # pragma: no cover

        if col_name is None:  # pragma: no cover
            continue

        # Extract k_ME values for features in this cluster
        cluster_kme = k_me.loc[cluster_features, col_name]
        # Sort by absolute correlation descending
        sorted_kme = cluster_kme.abs().sort_values(ascending=False)

        # Get top N feature names and their original signed k_ME values
        top_features = sorted_kme.index[:top_n]
        for rank_idx, feat in enumerate(top_features):
            original_val = cluster_kme.loc[feat]
            hub_records.append({
                "cluster": cluster,
                "feature": feat,
                "k_ME": original_val,
                "rank": rank_idx + 1,
            })

    return pd.DataFrame(hub_records)


def compute_group_enrichment(
    cluster_labels: pd.Series | dict[str, str | int] | Sequence[Any],
    group_assignments: pd.Series | dict[str, str | int] | Sequence[Any],
    *,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Perform Over-Representation Analysis (hypergeometric test) of groups within clusters.

    Tests whether features belonging to a particular functional group (e.g., radiomics
    feature families, scanner protocols, or observers) are statistically enriched within
    specific co-expression or dimensionality-reduction modules.

    Parameters
    ----------
    cluster_labels : pd.Series, dict, or sequence
        Module / cluster assignment per feature, or mapping from feature to cluster.
    group_assignments : pd.Series, dict, or sequence
        Group assignments per feature, or mapping from feature to functional group.
    feature_names : sequence of str, optional
        Names of features, used to align inputs if passed as simple sequences.

    Returns
    -------
    enrichment : pandas.DataFrame
        Detailed over-representation results table.
        Columns: `cluster`, `group`, `n_overlap`, `cluster_size`, `group_size`,
                 `total_features`, `p_value`, `fdr_q_value`, `odds_ratio`.
    """
    # Coerce both to Series
    if isinstance(cluster_labels, pd.Series):
        c_series = cluster_labels
    elif isinstance(cluster_labels, dict):
        c_series = pd.Series(cluster_labels)
    else:
        if feature_names is None:
            raise ValueError("feature_names is required when cluster_labels is a sequence.")
        c_series = pd.Series(cluster_labels, index=feature_names)

    if isinstance(group_assignments, pd.Series):
        g_series = group_assignments
    elif isinstance(group_assignments, dict):
        g_series = pd.Series(group_assignments)
    else:
        if feature_names is None:
            raise ValueError("feature_names is required when group_assignments is a sequence.")
        g_series = pd.Series(group_assignments, index=feature_names)

    # Align indices
    common_idx = c_series.index.intersection(g_series.index)
    if len(common_idx) == 0:
        if len(c_series) == 0 or len(g_series) == 0:
            # Handle empty inputs gracefully without raising overlap error
            return pd.DataFrame(
                columns=[
                    "cluster", "group", "n_overlap", "cluster_size",
                    "group_size", "total_features", "p_value", "fdr_q_value", "odds_ratio"
                ]
            )
        raise ValueError(
            "No overlapping feature names between cluster labels and group assignments."
        )

    c_series = c_series.loc[common_idx]
    g_series = g_series.loc[common_idx]

    unique_clusters = sorted(c_series.dropna().unique())
    unique_groups = sorted(g_series.dropna().unique())
    total_n = len(common_idx)

    records = []

    for cluster in unique_clusters:
        for group in unique_groups:
            # Construct 2x2 contingency table:
            #              In Group    Not In Group
            # In Cluster      a            b
            # Not In Cluster  c            d

            cluster_mask = c_series == cluster
            group_mask = g_series == group

            a = int((cluster_mask & group_mask).sum())
            b = int((cluster_mask & ~group_mask).sum())
            c = int((~cluster_mask & group_mask).sum())
            d = int((~cluster_mask & ~group_mask).sum())

            # Contingency table
            table = np.array([[a, b], [c, d]])

            # Fisher's Exact test for over-representation (enrichment)
            res = stats.fisher_exact(table, alternative="greater")
            odds_ratio = res.statistic
            p_val = res.pvalue

            records.append({
                "cluster": cluster,
                "group": group,
                "n_overlap": a,
                "cluster_size": a + b,
                "group_size": a + c,
                "total_features": total_n,
                "p_value": p_val,
                "odds_ratio": odds_ratio,
            })

    df = pd.DataFrame(records)
    if len(df) > 0:
        # Correct p-values
        df["fdr_q_value"] = _fdr_correct(df["p_value"].to_numpy())
    else:  # pragma: no cover
        df["fdr_q_value"] = np.nan

    return df
