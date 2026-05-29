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

    - If any Pictologics-style selector (``features``/``configs``/``families``/
      ``family_groups``) is given, the matched columns are returned (resolved
      with :class:`RadiomicsFeatureRemover`, where "removed" == "selected for
      analysis").
    - Otherwise every **numeric** column is returned; non-numeric metadata
      columns (e.g. ``PatientID``) are skipped so they do not break downstream
      preprocessing or statistics.

    Returns
    -------
    ndarray of str
        Feature-column names to analyze.
    """
    has_selectors = any(
        selector is not None for selector in (features, configs, families, family_groups)
    )

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

    remover = RadiomicsFeatureRemover(metadata_columns="auto")
    remover.fit(df)
    return np.asarray(
        [col for col in remover.kept_feature_names_ if pd.api.types.is_numeric_dtype(df[col])]
    )
