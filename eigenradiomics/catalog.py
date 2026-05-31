"""First-class radiomics feature catalog.

A *feature catalog* describes the columns of a wide radiomics table. Pictologics
tables name feature columns ``{config}__{feature_key}`` and ship a sidecar
catalog with one row per feature carrying metadata such as ``family`` and
``family_group``. :class:`FeatureCatalog` wraps that table (or a legacy single
``feature`` column) and provides the operations shared across selection,
reproducibility summaries, and reporting: annotation, validation, and grouping.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from eigenradiomics.preprocessing._feature_remover import _load_catalog, _normalize_catalog

# Columns shown first by :meth:`FeatureCatalog.annotate` when present.
_LEADING_COLUMNS = ["feature", "config", "feature_key", "feature_name", "family", "family_group"]


class FeatureCatalog:
    """Normalized radiomics feature catalog keyed by ``config__feature_key``.

    Parameters
    ----------
    frame : pandas.DataFrame
        A catalog table containing either ``config`` + ``feature_key`` columns
        or a legacy ``feature`` column holding ``config__feature_key`` values.
        Any additional columns (``family``, ``family_group``, ``ibsi_code``,
        ``feature_name``, ...) are preserved.

    Attributes
    ----------
    frame : pandas.DataFrame
        The normalized catalog with a canonical ``feature`` column.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        normalized = _normalize_catalog(frame).copy()
        normalized["feature"] = normalized["_column_name"].astype(str)
        helper_columns = [col for col in normalized.columns if col.startswith("_")]
        self._frame = normalized.drop(columns=helper_columns).reset_index(drop=True)
        duplicates = self._frame["feature"][self._frame["feature"].duplicated()].unique()
        if len(duplicates) > 0:
            preview = ", ".join(map(str, duplicates[:5]))
            suffix = " ..." if len(duplicates) > 5 else ""
            raise ValueError(
                f"catalog has {len(duplicates)} duplicate feature key(s); "
                f"config__feature_key must be unique: {preview}{suffix}"
            )
        self._feature_set = set(self._frame["feature"])

    @classmethod
    def from_csv(cls, path: str | Path) -> FeatureCatalog:
        """Load a catalog from a CSV file."""
        loaded = _load_catalog(path)
        if loaded is None:  # pragma: no cover - _load_catalog returns None only for None input
            raise FileNotFoundError(f"Could not load feature catalog from {path!r}.")
        return cls(loaded)

    # ------------------------------------------------------------------
    # basic access
    # ------------------------------------------------------------------

    @property
    def frame(self) -> pd.DataFrame:
        """Return a copy of the normalized catalog table."""
        return self._frame.copy()

    @property
    def feature_names(self) -> list[str]:
        """Return the canonical ``config__feature_key`` feature names."""
        return [str(name) for name in self._frame["feature"]]

    def families(self) -> list[str]:
        """Return the sorted unique ``family`` values (empty if no such column)."""
        if "family" not in self._frame.columns:
            return []
        return sorted(self._frame["family"].dropna().astype(str).unique().tolist())

    def family_groups(self) -> list[str]:
        """Return the sorted unique ``family_group`` values (empty if absent)."""
        if "family_group" not in self._frame.columns:
            return []
        return sorted(self._frame["family_group"].dropna().astype(str).unique().tolist())

    def __len__(self) -> int:
        return len(self._frame)

    def __contains__(self, feature: object) -> bool:
        return feature in self._feature_set

    def __repr__(self) -> str:
        return (
            f"FeatureCatalog(n_features={len(self)}, "
            f"n_families={len(self.families())}, "
            f"n_family_groups={len(self.family_groups())})"
        )

    # ------------------------------------------------------------------
    # operations
    # ------------------------------------------------------------------

    def validate(
        self,
        feature_columns: Iterable[str],
        *,
        allow_missing: bool = False,
    ) -> list[str]:
        """Check that *feature_columns* are present in the catalog.

        Parameters
        ----------
        feature_columns : iterable of str
            Feature-column names to validate.
        allow_missing : bool
            If False (default), raise when any column is missing from the
            catalog. If True, return the missing names instead.

        Returns
        -------
        missing : list of str
            Feature columns not found in the catalog.
        """
        missing = [str(col) for col in feature_columns if col not in self._feature_set]
        if missing and not allow_missing:
            preview = ", ".join(missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise ValueError(
                f"{len(missing)} feature column(s) not found in the catalog: {preview}{suffix}"
            )
        return missing

    def annotate(
        self,
        table: pd.DataFrame,
        *,
        on: str = "feature",
        columns: list[str] | None = None,
        lead: bool = True,
    ) -> pd.DataFrame:
        """Left-join catalog metadata onto a feature-keyed *table*.

        Parameters
        ----------
        table : pandas.DataFrame
            A table with a column (``on``) of feature names to annotate.
        on : str
            Column in *table* holding ``config__feature_key`` feature names.
        columns : list of str, optional
            Restrict the annotation to these catalog columns (``feature`` is
            always included). If None, all catalog columns are merged.
        lead : bool
            If True, reorder so identifier/metadata columns appear first.

        Returns
        -------
        annotated : pandas.DataFrame
        """
        if on not in table.columns:
            raise ValueError(f"Column {on!r} not found in the table to annotate.")

        meta = self._frame
        if columns is not None:
            keep = ["feature", *[c for c in columns if c in meta.columns and c != "feature"]]
            meta = meta[keep]

        merged = table.merge(
            meta,
            left_on=on,
            right_on="feature",
            how="left",
            suffixes=("", "_catalog"),
        )
        # Drop the duplicate join key the merge introduces when on != "feature".
        if on != "feature" and "feature" in merged.columns:
            merged = merged.drop(columns="feature")

        if lead:
            leading = [col for col in _LEADING_COLUMNS if col in merged.columns]
            other = [col for col in merged.columns if col not in leading]
            merged = merged[leading + other]
        return merged
