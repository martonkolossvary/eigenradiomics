"""Container that carries features, metadata, catalog, and study design together.

A radiomics analysis table mixes feature columns with metadata (identifiers,
batch/center, clinical covariates, survival endpoints). scikit-learn estimators
operate on the feature matrix only, while splitters and downstream models need
the metadata. :class:`RadiomicsDataset` keeps them together and hands a clean
feature matrix (plus optional target / groups) to a pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from numpy.typing import NDArray

from eigenradiomics.catalog import FeatureCatalog


@dataclass(frozen=True)
class StudyDesign:
    """Mapping of study-design roles to dataset columns.

    ``roles`` maps a role name to a column name, e.g.
    ``{"group": "PatientID", "batch": "Center", "time": "fu_years",
    "event": "death", "observer": "Reader", "phase": "Phase",
    "timepoint": "Visit", "mask": "Vessel"}``. The common roles
    (``group``, ``batch``, ``time``, ``event``, ``target``) are exposed as
    properties; any other role is still available via ``roles``.
    """

    roles: Mapping[str, str] = field(default_factory=dict)

    @property
    def group(self) -> str | None:
        """Patient/subject column for grouped (leakage-safe) splitting."""
        return self.roles.get("group")

    @property
    def batch(self) -> str | None:
        """Center/scanner column for batch-effect analysis."""
        return self.roles.get("batch")

    @property
    def time(self) -> str | None:
        """Survival time/duration column."""
        return self.roles.get("time")

    @property
    def event(self) -> str | None:
        """Survival event indicator column."""
        return self.roles.get("event")

    @property
    def target(self) -> str | None:
        """Generic (non-survival) outcome column."""
        return self.roles.get("target")

    def columns(self) -> list[str]:
        """Return the unique columns referenced by any role."""
        return list(dict.fromkeys(self.roles.values()))


class RadiomicsDataset:
    """A wide radiomics table with explicit feature/metadata roles.

    Parameters
    ----------
    data : pandas.DataFrame
        Wide table: one row per sample, columns are features and metadata.
    feature_columns : iterable of str, optional
        Feature columns. If None, inferred from the catalog (when given) or as
        columns containing the Pictologics ``__`` separator.
    catalog : FeatureCatalog, optional
        Feature catalog used for inference and annotation.
    metadata_columns : iterable of str, optional
        Non-feature columns. If None, all columns not in ``feature_columns``.
    design : StudyDesign, optional
        Roles of metadata columns (group/batch/time/event/...).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        feature_columns: Iterable[str] | None = None,
        catalog: FeatureCatalog | None = None,
        metadata_columns: Iterable[str] | None = None,
        design: StudyDesign | None = None,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("RadiomicsDataset requires a pandas DataFrame.")

        self.data = data
        self.catalog = catalog
        self.design = design if design is not None else StudyDesign()

        if feature_columns is None:
            feature_columns = self._infer_feature_columns(data, catalog)
        resolved_features = list(feature_columns)
        missing = [col for col in resolved_features if col not in data.columns]
        if missing:
            raise ValueError(f"feature_columns not present in data: {missing[:5]}")
        self.feature_columns = resolved_features

        if metadata_columns is None:
            feature_set = set(resolved_features)
            metadata_columns = [col for col in data.columns if col not in feature_set]
        self.metadata_columns = list(metadata_columns)

        for role, column in self.design.roles.items():
            if column not in data.columns:
                raise ValueError(f"design role {role!r} maps to missing column {column!r}.")

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_feature_columns(
        data: pd.DataFrame,
        catalog: FeatureCatalog | None,
    ) -> list[str]:
        if catalog is not None:
            known = set(catalog.feature_names)
            from_catalog = [col for col in data.columns if col in known]
            if from_catalog:
                return from_catalog
        return [col for col in data.columns if isinstance(col, str) and "__" in col]

    @classmethod
    def from_wide(
        cls,
        data: pd.DataFrame,
        *,
        catalog: FeatureCatalog | pd.DataFrame | str | Path | None = None,
        feature_columns: Iterable[str] | None = None,
        group: str | None = None,
        batch: str | None = None,
        time: str | None = None,
        event: str | None = None,
        target: str | None = None,
        roles: Mapping[str, str] | None = None,
    ) -> RadiomicsDataset:
        """Build a dataset from a wide table, declaring design roles inline.

        ``catalog`` may be a :class:`FeatureCatalog`, a DataFrame, or a CSV path.
        Common roles are passed as keyword arguments; extra roles (``observer``,
        ``phase``, ``timepoint``, ``mask``, ...) go through ``roles``.
        """
        merged_roles: dict[str, str] = dict(roles) if roles else {}
        for name, column in (
            ("group", group),
            ("batch", batch),
            ("time", time),
            ("event", event),
            ("target", target),
        ):
            if column is not None:
                merged_roles[name] = column

        resolved_catalog: FeatureCatalog | None
        if catalog is None or isinstance(catalog, FeatureCatalog):
            resolved_catalog = catalog
        elif isinstance(catalog, pd.DataFrame):
            resolved_catalog = FeatureCatalog(catalog)
        else:
            resolved_catalog = FeatureCatalog.from_csv(catalog)

        return cls(
            data,
            feature_columns=feature_columns,
            catalog=resolved_catalog,
            design=StudyDesign(roles=merged_roles),
        )

    # ------------------------------------------------------------------
    # access
    # ------------------------------------------------------------------

    @property
    def features(self) -> pd.DataFrame:
        """The feature matrix (X) as a DataFrame."""
        return self.data[self.feature_columns]

    @property
    def metadata(self) -> pd.DataFrame:
        """The non-feature columns."""
        return self.data[self.metadata_columns]

    @property
    def n_samples(self) -> int:
        return len(self.data)

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_samples, self.n_features)

    @property
    def groups(self) -> NDArray | None:
        """Group labels (e.g. patient IDs) for grouped CV, or None."""
        column = self.design.group
        if column is None:
            return None
        result: NDArray = self.data[column].to_numpy()
        return result

    def y(self) -> pd.DataFrame | pd.Series | None:
        """Return the target.

        A two-column ``[time, event]`` frame for survival designs, a Series for
        a single ``target`` column, or None when no outcome role is set.
        """
        if self.design.time is not None and self.design.event is not None:
            return self.data[[self.design.time, self.design.event]]
        if self.design.target is not None:
            return self.data[self.design.target]
        return None

    def to_pipeline_input(self) -> tuple[pd.DataFrame, pd.DataFrame | pd.Series | None]:
        """Return ``(X, y)`` for a scikit-learn pipeline.

        ``X`` is the feature DataFrame; ``y`` follows :meth:`y`. Use
        :attr:`groups` for group-aware splitters.
        """
        return self.features, self.y()

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return (
            f"RadiomicsDataset(n_samples={self.n_samples}, "
            f"n_features={self.n_features}, "
            f"n_metadata={len(self.metadata_columns)}, "
            f"roles={dict(self.design.roles)})"
        )
