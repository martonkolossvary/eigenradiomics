"""Feature-vs-clinical correlation helpers.

These source the **right correlation panel** of
:func:`~eigenradiomics.plotting.plot_clustered_heatmap`: a features-by-clinical
matrix of associations. Clinical variables are mixed (continuous, ordinal,
binary, categorical), so they are first coerced to a common numeric scale by
:func:`encode_clinical_series` and then correlated by rank (Spearman) — the same
convention used across WGCNA radiomics studies.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from eigenradiomics.dataset import RadiomicsDataset

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
    if numeric.notna().any():
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
