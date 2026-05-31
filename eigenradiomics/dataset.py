"""Container that carries features, metadata, catalog, and study design together.

A radiomics analysis table mixes feature columns with metadata (identifiers,
batch/center, clinical covariates, survival endpoints). scikit-learn estimators
operate on the feature matrix only, while splitters and downstream models need
the metadata. :class:`RadiomicsDataset` keeps them together and hands a clean
feature matrix (plus optional target / groups) to a pipeline.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from numpy.typing import NDArray

from eigenradiomics.catalog import FeatureCatalog
from eigenradiomics.pictologics import _split_pictologics_name


def _resolve_catalog(
    catalog: FeatureCatalog | pd.DataFrame | str | Path | None,
) -> FeatureCatalog | None:
    """Coerce a catalog argument (object / DataFrame / CSV path) to a FeatureCatalog."""
    if catalog is None or isinstance(catalog, FeatureCatalog):
        return catalog
    if isinstance(catalog, pd.DataFrame):
        return FeatureCatalog(catalog)
    return FeatureCatalog.from_csv(catalog)


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
            warnings.warn(
                "a catalog was provided but none of its features match the table's "
                "columns; falling back to the '__' name heuristic. Check for an "
                "observer/config prefix mismatch between the table and the catalog.",
                stacklevel=2,
            )
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

        return cls(
            data,
            feature_columns=feature_columns,
            catalog=_resolve_catalog(catalog),
            design=StudyDesign(roles=merged_roles),
        )

    @classmethod
    def from_pictologics(
        cls,
        table: pd.DataFrame | str | Path,
        catalog: FeatureCatalog | pd.DataFrame | str | Path | None = None,
        *,
        drop_subject_id: bool = True,
        group: str | None = None,
        batch: str | None = None,
        time: str | None = None,
        event: str | None = None,
        target: str | None = None,
        roles: Mapping[str, str] | None = None,
    ) -> RadiomicsDataset:
        """Build a dataset from a (single-observer) Pictologics wide export.

        Reads ``table`` (a DataFrame or CSV path), drops the ``subject_id`` leak
        columns Pictologics emits, detects ``config__feature_key`` feature columns,
        and validates them against ``catalog``. When ``catalog`` is ``None`` and
        ``table`` is a path, the sidecar ``features_catalog.csv`` next to it is used
        if present. Design roles are declared inline as in :meth:`from_wide`.

        For observer-paired reproducibility tables, use
        :func:`~eigenradiomics.split_observer_tables` (and ``observer_prefixes`` on
        :class:`~eigenradiomics.RadiomicsFeatureRemover`).
        """
        table_path: Path | None = None
        if isinstance(table, (str, Path)):
            table_path = Path(table)
            data = pd.read_csv(table_path)
        else:
            data = table.copy()

        if catalog is None and table_path is not None:
            sidecar = table_path.parent / "features_catalog.csv"
            if sidecar.exists():
                catalog = sidecar
        resolved_catalog = _resolve_catalog(catalog)

        if drop_subject_id:
            drop = [
                col
                for col in data.columns
                if isinstance(col, str)
                and (_split_pictologics_name(col)[2] == "subject_id" or col == "subject_id")
            ]
            data = data.drop(columns=drop)

        feature_like = [
            col
            for col in data.columns
            if isinstance(col, str) and _split_pictologics_name(col)[1] is not None
        ]
        if resolved_catalog is not None:
            feature_columns = [col for col in feature_like if col in resolved_catalog]
            unmatched = [col for col in feature_like if col not in resolved_catalog]
            if unmatched:
                warnings.warn(
                    f"{len(unmatched)} feature-like column(s) are not in the catalog and were "
                    f"treated as metadata: {', '.join(unmatched[:5])}"
                    f"{' ...' if len(unmatched) > 5 else ''}. Check for a config/prefix mismatch.",
                    stacklevel=2,
                )
        else:
            feature_columns = feature_like

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
        values = self.data[column]
        if values.isna().any():
            raise ValueError(
                f"group column {column!r} has missing values; grouped (leakage-safe) "
                "cross-validation requires a non-null group label for every sample."
            )
        result: NDArray = values.to_numpy()
        return result

    def y(self) -> pd.DataFrame | pd.Series | None:
        """Return the target.

        A two-column ``[time, event]`` frame for survival designs, a Series for
        a single ``target`` column, or None when no outcome role is set.
        """
        time, event = self.design.time, self.design.event
        if (time is None) != (event is None):
            raise ValueError(
                "a survival design needs both 'time' and 'event' roles; got "
                f"time={time!r}, event={event!r}."
            )
        if time is not None and event is not None:
            return self.data[[time, event]]
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
