"""Shared feature-column resolution for the analysis modules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from eigenradiomics.preprocessing._feature_remover import RadiomicsFeatureRemover


def resolve_analysis_features(
    df: pd.DataFrame,
    *,
    features: Any = None,
    configs: Any = None,
    families: Any = None,
    family_groups: Any = None,
    catalog: Any = None,
) -> NDArray:
    """Resolve which feature columns of *df* should be analyzed.

    Shared by :func:`compute_reproducibility` and :func:`compute_batch_effects`.

    - If ``features`` is an explicit list of column names that are all present in
      *df* (and no catalog-driven selector is given), those columns are used
      directly. This lets generic, non-Pictologics DataFrames be subset by name
      without the Pictologics name-pattern remover.
    - If any catalog-driven selector (``configs``/``families``/``family_groups``),
      or a ``features`` pattern that is not a literal set of columns, is given,
      the matched columns are resolved with :class:`RadiomicsFeatureRemover`
      (where "removed" == "selected for analysis"). This path requires
      Pictologics-style column names / a catalog.
    - Otherwise every **numeric** column is returned; non-numeric metadata
      columns (e.g. ``PatientID``) are skipped so they do not break downstream
      preprocessing or statistics.

    Returns
    -------
    ndarray of str
        Feature-column names to analyze.
    """
    catalog_selectors = any(
        selector is not None for selector in (configs, families, family_groups)
    )

    # Fast path: an explicit list of column names present in df is used as-is, so
    # generic DataFrames work without the Pictologics-name-aware remover.
    if features is not None and not catalog_selectors:
        feat_list = [features] if isinstance(features, str) else list(features)
        feat_list = [str(f) for f in feat_list]
        if feat_list and all(f in df.columns for f in feat_list):
            return np.asarray(feat_list)

    has_selectors = features is not None or catalog_selectors

    if has_selectors:
        remover = RadiomicsFeatureRemover(
            features=features,
            configs=configs,
            families=families,
            family_groups=family_groups,
            catalog=catalog,
            metadata_columns="auto",
            allow_missing=True,
        )
        remover.fit(df)
        return np.asarray(remover.removed_feature_names_)

    # No selectors: analyse every numeric column directly. Selecting by dtype
    # (rather than by stringified catalog names) keeps int-labelled,
    # ndarray-derived columns working too.
    return np.asarray(df.select_dtypes(include="number").columns.tolist())
