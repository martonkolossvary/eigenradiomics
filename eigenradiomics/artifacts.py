"""Standardized structured outputs that reducers expose for downstream use.

Different reducers can produce different subsets of these elements — one might
only estimate a feature-feature similarity, another might also assign clusters —
but they all package what they have into the same :class:`ReductionArtifacts`
container. Downstream utilities (clustered heatmaps, analysis, exports) then
consume whichever elements are present without caring which reducer produced
them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
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

    def get_catalog_enrichment(
        self,
        catalog: Any,
        columns: str | Sequence[str] | None = None,
        *,
        method: str = "fisher",
        alternative: str = "greater",
    ) -> pd.DataFrame:
        """Compute over-representation analysis of catalog metadata within discovered clusters.

        Parameters
        ----------
        catalog : FeatureCatalog, DataFrame, or CSV path
            Catalog describing features.
        columns : str or sequence of str, optional
            Catalog columns to test (e.g. ``["family", "family_group"]``). If None,
            tests all eligible string/categorical columns in the catalog.
        method : {"fisher", "chi2", "hypergeometric"}, default="fisher"
            Statistical test for over-representation.
        alternative : {"greater", "less", "two-sided"}, default="greater"
            Alternative hypothesis for the test.

        Returns
        -------
        enrichment : pandas.DataFrame
            Detailed over-representation results table.
        """
        if self.cluster_labels is None:
            raise ValueError(
                "ReductionArtifacts has no cluster_labels; catalog enrichment cannot be computed."
            )
        from eigenradiomics.analysis import compute_group_enrichment

        return compute_group_enrichment(
            self.cluster_labels,
            catalog,
            columns=columns,
            method=method,
            alternative=alternative,
        )

    def __repr__(self) -> str:
        return (
            f"ReductionArtifacts(n_features={len(self.feature_names)}, "
            f"available={self.available()})"
        )


@dataclass(frozen=True)
class PairedReductionResult:
    """Container for paired/longitudinal reduction outputs.

    Attributes
    ----------
    baseline : pandas.DataFrame or ndarray
        Baseline latent coordinates (ME_pre).
    followup : pandas.DataFrame or ndarray
        Follow-up latent coordinates (ME_post).
    delta : pandas.DataFrame or ndarray
        Latent change (ME_post - ME_pre).
    relative_change : pandas.DataFrame or ndarray, optional
        Relative change delta / |baseline|. The value is NaN where
        |baseline| < epsilon.
    trajectory_distance : pandas.Series or ndarray, optional
        Euclidean trajectory distance ||ME_post - ME_pre||_2 per sample.
    effect_size : pandas.Series or ndarray, optional
        Standardized Cohen's d effect size per component (mean(delta) / std(delta)).
    feature_names_out : ndarray, optional
        Output latent feature names.
    """

    baseline: pd.DataFrame | NDArray
    followup: pd.DataFrame | NDArray
    delta: pd.DataFrame | NDArray
    relative_change: pd.DataFrame | NDArray | None = None
    trajectory_distance: pd.Series | NDArray | None = None
    effect_size: pd.Series | NDArray | None = None
    feature_names_out: NDArray | None = None

    def to_dataframe(self, prefix_mode: str = "suffix") -> pd.DataFrame:
        """Merge baseline, followup, and delta into a single wide DataFrame.

        Parameters
        ----------
        prefix_mode : {"suffix", "prefix"}
            Whether to add state tag as a suffix (e.g., ``wgcna_0_baseline``) or
            prefix (e.g., ``baseline_wgcna_0``).

        Returns
        -------
        df : pandas.DataFrame
            Merged DataFrame.
        """
        if prefix_mode not in ("suffix", "prefix"):
            raise ValueError(f"prefix_mode must be 'suffix' or 'prefix', got {prefix_mode!r}.")

        if isinstance(self.baseline, pd.DataFrame):
            base_df = self.baseline.copy()
            post_df = (
                self.followup.copy()
                if isinstance(self.followup, pd.DataFrame)
                else pd.DataFrame(self.followup, index=base_df.index, columns=base_df.columns)
            )
            delta_df = (
                self.delta.copy()
                if isinstance(self.delta, pd.DataFrame)
                else pd.DataFrame(self.delta, index=base_df.index, columns=base_df.columns)
            )
            cols = list(base_df.columns)
            idx = base_df.index
        else:
            cols = (
                list(self.feature_names_out)
                if self.feature_names_out is not None
                else [f"component_{i}" for i in range(np.asarray(self.baseline).shape[1])]
            )
            idx = pd.RangeIndex(np.asarray(self.baseline).shape[0])
            base_df = pd.DataFrame(self.baseline, index=idx, columns=cols)
            post_df = pd.DataFrame(self.followup, index=idx, columns=cols)
            delta_df = pd.DataFrame(self.delta, index=idx, columns=cols)

        if prefix_mode == "suffix":
            base_df.columns = [f"{c}_baseline" for c in cols]
            post_df.columns = [f"{c}_followup" for c in cols]
            delta_df.columns = [f"{c}_delta" for c in cols]
        else:
            base_df.columns = [f"baseline_{c}" for c in cols]
            post_df.columns = [f"followup_{c}" for c in cols]
            delta_df.columns = [f"delta_{c}" for c in cols]

        merged = pd.concat([base_df, post_df, delta_df], axis=1)

        if self.relative_change is not None:
            rel_df = (
                self.relative_change.copy()
                if isinstance(self.relative_change, pd.DataFrame)
                else pd.DataFrame(self.relative_change, index=idx, columns=cols)
            )
            if prefix_mode == "suffix":
                rel_df.columns = [f"{c}_relative_change" for c in cols]
            else:
                rel_df.columns = [f"relative_change_{c}" for c in cols]
            merged = pd.concat([merged, rel_df], axis=1)

        if self.trajectory_distance is not None:
            dist_s = (
                self.trajectory_distance.copy()
                if isinstance(self.trajectory_distance, pd.Series)
                else pd.Series(self.trajectory_distance, index=idx, name="trajectory_distance")
            )
            merged["trajectory_distance"] = dist_s

        return merged
