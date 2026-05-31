"""Trait-association helpers.

Two related analyses connect radiomics output to clinical traits:

* :func:`compute_clinical_correlations` correlates raw **features** with clinical
  variables — it sources the right correlation panel of
  :func:`~eigenradiomics.plotting.plot_clustered_heatmap`.
* :func:`compute_module_trait_associations` correlates module **eigengenes**
  (the reducer's output) with traits and reports p-values and FDR — the standard
  WGCNA "module-trait relationship" used for downstream statistical reporting.

Clinical variables are mixed (continuous, ordinal, binary, categorical), so they
are first coerced to a common numeric scale by :func:`encode_clinical_series`
and then correlated by rank (Spearman by default).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from eigenradiomics._stats import _fdr_correct
from eigenradiomics.dataset import RadiomicsDataset

#: Correlation routines exposing a p-value, keyed by method name.
_CORRELATION_FUNCS = {
    "spearman": stats.spearmanr,
    "pearson": stats.pearsonr,
    "kendall": stats.kendalltau,
}

#: Recognized binary / ordinal string tokens (lower-cased) -> numeric code.
_BINARY_MAP: dict[str, float] = {
    "0": 0, "1": 1, "false": 0, "true": 1, "f": 0, "t": 1,
    "no": 0, "yes": 1, "n": 0, "y": 1, "male": 0, "female": 1,
    "i": 1, "ii": 2, "iii": 3, "iv": 4,
}


def encode_clinical_series(series: pd.Series) -> pd.Series:
    """Coerce a clinical variable to a numeric Series for correlation.

    Applied in order: (1) numeric parse; (2) common binary/ordinal string tokens
    (``yes``/``no``, ``true``/``false``, ``male``/``female``, ``I``–``IV``);
    (3) alphabetical ordinal encoding of the remaining categories.

    Parameters
    ----------
    series : pandas.Series
        Raw clinical column (numeric, string, or mixed).

    Returns
    -------
    encoded : pandas.Series
        Float-valued Series aligned to ``series.index`` (unmappable entries NaN).
    """
    numeric = pd.to_numeric(series, errors="coerce")
    n_present = int(series.notna().sum())
    # Treat the column as numeric only when most non-missing values parse as
    # numbers; a lone numeric token in an otherwise categorical column must not
    # blank out the rest (which would silently drop it later).
    if n_present and numeric.notna().sum() >= 0.5 * n_present:
        return numeric
    cleaned = series.astype("string").str.strip().str.lower()
    mapped = cleaned.map(_BINARY_MAP)
    if mapped.notna().any():
        return mapped.astype(float)
    categories = sorted(cleaned.dropna().unique())
    category_map = {category: index + 1 for index, category in enumerate(categories)}
    return cleaned.map(category_map).astype(float)


def _resolve_inputs(
    features: pd.DataFrame | RadiomicsDataset,
    clinical: pd.DataFrame | Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve ``(feature_matrix, clinical_frame)`` from the accepted input forms."""
    if isinstance(features, RadiomicsDataset):
        feature_matrix = features.features
        if isinstance(clinical, pd.DataFrame):
            clinical_frame = clinical
        else:
            clinical_frame = features.data[list(clinical)]
    else:
        feature_matrix = features
        if not isinstance(clinical, pd.DataFrame):
            raise TypeError(
                "clinical must be a DataFrame when features is not a RadiomicsDataset."
            )
        clinical_frame = clinical
    return feature_matrix, clinical_frame


def compute_clinical_correlations(
    features: pd.DataFrame | RadiomicsDataset,
    clinical: pd.DataFrame | Sequence[str],
    *,
    method: str = "spearman",
    min_pairs: int = 10,
) -> pd.DataFrame:
    """Correlate each feature with each clinical variable.

    Parameters
    ----------
    features : DataFrame or RadiomicsDataset
        Feature matrix (samples x features). For a :class:`RadiomicsDataset` its
        feature matrix is used and *clinical* may name metadata columns.
    clinical : DataFrame or sequence of str
        Clinical variables (samples x variables), or — when *features* is a
        :class:`RadiomicsDataset` — a list of its metadata column names.
        Variables are numerically encoded via :func:`encode_clinical_series`.
    method : {"spearman", "pearson", "kendall"}
        Correlation method (``"spearman"`` by default, robust to mixed scales).
    min_pairs : int
        Minimum non-missing overlapping observations required per correlation;
        variables with fewer valid values or no variance are dropped.

    Returns
    -------
    correlations : pandas.DataFrame
        ``(n_features, n_usable_variables)`` correlation matrix, feature-indexed,
        ready to pass to ``plot_clustered_heatmap(..., right=...)``.
    """
    if method not in {"spearman", "pearson", "kendall"}:
        raise ValueError(f"method must be 'spearman', 'pearson', or 'kendall', got {method!r}.")
    feature_matrix, clinical_frame = _resolve_inputs(features, clinical)
    if feature_matrix.index.intersection(clinical_frame.index).empty:
        raise ValueError(
            "features and clinical share no common index labels; align their indexes "
            "before correlating (or pass a RadiomicsDataset so they stay aligned)."
        )
    encoded = pd.DataFrame(
        {column: encode_clinical_series(clinical_frame[column]) for column in clinical_frame},
        index=clinical_frame.index,
    ).reindex(feature_matrix.index)
    usable = [
        column
        for column in encoded.columns
        if encoded[column].notna().sum() >= min_pairs
        and encoded[column].nunique(dropna=True) > 1
    ]
    if not usable:
        raise ValueError(
            "No clinical variable has enough non-missing, varying values "
            f"(need >= {min_pairs} pairs with >1 unique value)."
        )
    combined = pd.concat([feature_matrix, encoded[usable]], axis=1)
    correlations = combined.corr(method=method, min_periods=min_pairs)
    return correlations.loc[list(feature_matrix.columns), usable]


def compute_module_trait_associations(
    eigengenes: pd.DataFrame,
    traits: pd.DataFrame | RadiomicsDataset,
    columns: Sequence[str] | None = None,
    *,
    method: str = "spearman",
    min_pairs: int = 10,
) -> dict[str, pd.DataFrame]:
    """Correlate module eigengenes with clinical traits (the WGCNA module-trait relationship).

    For each (module, trait) pair, computes the correlation coefficient and its
    p-value over pairwise-complete samples, then applies a Benjamini-Hochberg FDR
    across the whole table. This is the standard downstream summary linking a
    reduced feature space to outcomes.

    Parameters
    ----------
    eigengenes : DataFrame
        Module eigengenes, ``(n_samples, n_modules)`` — e.g.
        ``reducer.set_output(transform="pandas").transform(X)``.
    traits : DataFrame or RadiomicsDataset
        Clinical traits, ``(n_samples, n_traits)``; for a :class:`RadiomicsDataset`
        name the metadata columns via *columns*. Encoded with
        :func:`encode_clinical_series`.
    columns : sequence of str, optional
        Metadata column names when *traits* is a :class:`RadiomicsDataset`.
    method : {"spearman", "pearson", "kendall"}
        Correlation method (``"spearman"`` by default).
    min_pairs : int
        Minimum non-missing overlapping observations required per correlation;
        traits with fewer valid or no varying values are dropped.

    Returns
    -------
    dict of DataFrame
        ``{"r", "p", "p_fdr"}``, each ``(n_modules, n_traits)`` and module-indexed.
    """
    if method not in _CORRELATION_FUNCS:
        raise ValueError(f"method must be 'spearman', 'pearson', or 'kendall', got {method!r}.")
    if isinstance(traits, RadiomicsDataset):
        traits = traits.data[list(columns)] if columns is not None else traits.metadata
    if eigengenes.index.intersection(traits.index).empty:
        raise ValueError(
            "eigengenes and traits share no common index labels; align their indexes "
            "before correlating."
        )
    encoded = pd.DataFrame(
        {column: encode_clinical_series(traits[column]) for column in traits},
        index=traits.index,
    ).reindex(eigengenes.index)
    usable = [
        column
        for column in encoded.columns
        if encoded[column].notna().sum() >= min_pairs
        and encoded[column].nunique(dropna=True) > 1
    ]
    if not usable:
        raise ValueError(
            "No trait has enough non-missing, varying values "
            f"(need >= {min_pairs} pairs with >1 unique value)."
        )

    corr_func = _CORRELATION_FUNCS[method]
    modules = list(eigengenes.columns)
    r = pd.DataFrame(np.nan, index=modules, columns=usable, dtype=float)
    p = pd.DataFrame(np.nan, index=modules, columns=usable, dtype=float)
    for module in modules:
        for trait in usable:
            pair = pd.concat([eigengenes[module], encoded[trait]], axis=1).dropna()
            if (
                len(pair) < min_pairs
                or pair.iloc[:, 0].nunique() < 2
                or pair.iloc[:, 1].nunique() < 2
            ):
                continue
            coef, pval = corr_func(pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy())
            r.loc[module, trait] = float(coef)
            p.loc[module, trait] = float(pval)

    p_fdr = pd.DataFrame(
        _fdr_correct(p.to_numpy().ravel()).reshape(p.shape),
        index=modules,
        columns=usable,
    )
    return {"r": r, "p": p, "p_fdr": p_fdr}
