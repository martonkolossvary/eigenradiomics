"""Standardized structured outputs that reducers expose for downstream use.

Different reducers can produce different subsets of these elements — one might
only estimate a feature-feature similarity, another might also assign clusters —
but they all package what they have into the same :class:`ReductionArtifacts`
container. Downstream utilities (clustered heatmaps, analysis, exports) then
consume whichever elements are present without caring which reducer produced
them.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReductionArtifacts:
    """A reducer's standardized outputs.

    Beyond ``feature_names`` every field is optional; a reducer populates the
    ones it can produce and leaves the rest as ``None``.

    Attributes
    ----------
    feature_names : ndarray of str
        Names of the input features (always present).
    similarity : pandas.DataFrame, optional
        Symmetric feature-by-feature similarity (e.g. WGCNA TOM, ``|corr|``),
        indexed and columned by feature name.
    linkage : ndarray, optional
        SciPy hierarchical-clustering linkage matrix over the features.
    cluster_labels : pandas.Series, optional
        Per-feature cluster / module label, indexed by feature name.
    feature_order : ndarray of str, optional
        Feature names in a meaningful display order (e.g. dendrogram leaves).
    feature_importances : pandas.DataFrame, optional
        Per-feature contribution to the reduction (columns vary by reducer).
    """

    feature_names: NDArray
    similarity: pd.DataFrame | None = None
    linkage: NDArray | None = None
    cluster_labels: pd.Series | None = None
    feature_order: NDArray | None = None
    feature_importances: pd.DataFrame | None = None

    def available(self) -> list[str]:
        """Return the names of the optional elements that are populated."""
        return [
            field.name
            for field in fields(self)
            if field.name != "feature_names" and getattr(self, field.name) is not None
        ]

    def __repr__(self) -> str:
        return (
            f"ReductionArtifacts(n_features={len(self.feature_names)}, "
            f"available={self.available()})"
        )
