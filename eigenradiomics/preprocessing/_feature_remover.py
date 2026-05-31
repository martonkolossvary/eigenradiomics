"""Feature-removal utilities for Pictologics-style radiomics tables."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from eigenradiomics._utils import _assert_dense_matrix, _check_feature_names

SelectorInput: TypeAlias = str | list[str] | tuple[str, ...] | set[str] | None
CatalogInput: TypeAlias = pd.DataFrame | str | Path | None
MetadataColumns: TypeAlias = Literal["auto"] | list[str] | tuple[str, ...] | None
TableLike: TypeAlias = pd.DataFrame | NDArray

ENCODING = "utf-8"


@dataclass(frozen=True)
class RadiomicsFeatureSplit:
    """In-memory result from splitting selected radiomics feature columns."""

    kept_table: TableLike
    removed_table: TableLike
    kept_columns: tuple[str, ...]
    removed_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]
    unmatched_selectors: tuple[str, ...]


@dataclass(frozen=True)
class RadiomicsFeatureSplitFiles:
    """File paths and metadata produced by :func:`split_radiomics_file`."""

    input_table: Path
    stripped_table: Path
    removed_table: Path
    manifest: Path
    input_catalog: Path | None
    stripped_catalog: Path | None
    removed_catalog: Path | None
    kept_columns: tuple[str, ...]
    removed_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]
    unmatched_selectors: tuple[str, ...]


@dataclass(frozen=True)
class _ParsedColumn:
    name: str
    index: int
    config: str | None
    feature_key: str | None

    @property
    def is_feature(self) -> bool:
        return self.config is not None and self.feature_key is not None


@dataclass(frozen=True)
class _SelectionResult:
    removed_indices: tuple[int, ...]
    kept_indices: tuple[int, ...]
    metadata_indices: tuple[int, ...]
    unmatched_selectors: tuple[str, ...]


def _normalize_selectors(values: SelectorInput) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(str(value) for value in values)


def _parse_columns(names: list[str]) -> list[_ParsedColumn]:
    parsed: list[_ParsedColumn] = []
    for idx, name in enumerate(names):
        if "__" in name:
            config, feature_key = name.split("__", 1)
            if config and feature_key:
                parsed.append(_ParsedColumn(name, idx, config, feature_key))
                continue
        parsed.append(_ParsedColumn(name, idx, None, None))
    return parsed


def _matches(value: str, pattern: str) -> bool:
    return fnmatchcase(value, pattern)


def _feature_selector_matches(column: _ParsedColumn, selector: str) -> bool:
    if _matches(column.name, selector):
        return True
    return column.feature_key is not None and _matches(column.feature_key, selector)


def _config_matches(column: _ParsedColumn, config_selectors: tuple[str, ...]) -> bool:
    if not config_selectors:
        return True
    if column.config is None:
        return False
    return any(_matches(column.config, selector) for selector in config_selectors)


def _load_catalog(catalog: CatalogInput) -> pd.DataFrame | None:
    if catalog is None:
        return None
    if isinstance(catalog, pd.DataFrame):
        return catalog.copy()
    return pd.read_csv(Path(catalog), encoding=ENCODING)


def _normalize_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    if {"config", "feature_key"}.issubset(catalog.columns):
        normalized = catalog.copy()
        normalized["config"] = normalized["config"].astype(str)
        normalized["feature_key"] = normalized["feature_key"].astype(str)
        normalized["_column_name"] = normalized["config"] + "__" + normalized["feature_key"]
        return normalized

    if "feature" in catalog.columns:
        normalized = catalog.copy()
        feature = normalized["feature"].astype(str)
        config: list[str | None] = []
        feature_key: list[str] = []
        for value in feature:
            if "__" in value:
                cfg, key = value.split("__", 1)
                config.append(cfg or None)
                feature_key.append(key)
            else:
                config.append(None)
                feature_key.append(value)
        normalized["config"] = config
        normalized["feature_key"] = feature_key
        normalized["_column_name"] = feature
        return normalized

    raise ValueError(
        "Feature catalog must contain either `config` + `feature_key` columns "
        "or a legacy `feature` column."
    )


def _catalog_selector_matches(
    catalog: pd.DataFrame,
    column: _ParsedColumn,
    *,
    families: tuple[str, ...],
    family_groups: tuple[str, ...],
) -> bool:
    if not families and not family_groups:
        return False

    if column.feature_key is None:
        return False

    mask = pd.Series(True, index=catalog.index)
    config_col = catalog["config"]
    has_config = config_col.notna()
    mask &= (catalog["feature_key"] == column.feature_key) & (
        (~has_config) | (config_col == column.config)
    )

    family_match = False
    if families:
        if "family" not in catalog.columns:  # pragma: no cover - validated before selection
            raise ValueError("Catalog selectors by family require a `family` column.")
        family_values = catalog["family"].astype(str).str.lower()
        wanted_families = {value.lower() for value in families}
        family_match = bool((mask & family_values.isin(wanted_families)).any())

    group_match = False
    if family_groups:
        if "family_group" not in catalog.columns:  # pragma: no cover - validated before selection
            raise ValueError("Catalog selectors by family group require a `family_group` column.")
        group_values = catalog["family_group"].astype(str).str.lower()
        wanted_groups = {value.lower() for value in family_groups}
        group_match = bool((mask & group_values.isin(wanted_groups)).any())

    return family_match or group_match


def _catalog_selector_has_match(
    catalog: pd.DataFrame,
    parsed: list[_ParsedColumn],
    *,
    selector_type: str,
    selector: str,
    config_selectors: tuple[str, ...],
) -> bool:
    if selector_type == "family":
        if "family" not in catalog.columns:
            raise ValueError("Catalog selectors by family require a `family` column.")
        rows = catalog[catalog["family"].astype(str).str.lower() == selector.lower()]
    elif selector_type == "family_group":
        if "family_group" not in catalog.columns:
            raise ValueError("Catalog selectors by family group require a `family_group` column.")
        rows = catalog[catalog["family_group"].astype(str).str.lower() == selector.lower()]
    else:  # pragma: no cover - internal guard
        raise ValueError(f"Unknown catalog selector type: {selector_type}")

    if rows.empty:
        return False

    candidate_columns: set[str] = set()
    candidate_features: set[str] = set()
    for _, row in rows.iterrows():
        feature_key = str(row["feature_key"])
        candidate_features.add(feature_key)
        config = row["config"]
        if pd.notna(config):
            candidate_columns.add(f"{config}__{feature_key}")

    for column in parsed:
        if not _config_matches(column, config_selectors):
            continue
        if column.name in candidate_columns:
            return True
        if column.feature_key in candidate_features and not candidate_columns:
            return True
    return False


def _resolve_selection(
    names: list[str],
    *,
    features: tuple[str, ...],
    configs: tuple[str, ...],
    families: tuple[str, ...],
    family_groups: tuple[str, ...],
    catalog: pd.DataFrame | None,
    metadata_columns: MetadataColumns,
    allow_missing: bool,
) -> _SelectionResult:
    parsed = _parse_columns(names)
    has_name_based_selectors = bool(features or configs or families or family_groups)
    if has_name_based_selectors and all(not column.is_feature for column in parsed):
        raise ValueError(
            "Name-based radiomics feature removal requires named columns. "
            "Pass a pandas DataFrame with Pictologics-style column names."
        )

    normalized_catalog = _normalize_catalog(catalog) if catalog is not None else None
    if (families or family_groups) and normalized_catalog is None:
        raise ValueError("`families` and `family_groups` selectors require a feature catalog.")

    unmatched: list[str] = []
    if features:
        for selector in features:
            if not any(
                _feature_selector_matches(column, selector) and _config_matches(column, configs)
                for column in parsed
            ):
                unmatched.append(f"feature:{selector}")

    if configs:
        for selector in configs:
            if not any(
                column.is_feature and _matches(cast(str, column.config), selector)
                for column in parsed
            ):
                unmatched.append(f"config:{selector}")

    if normalized_catalog is not None:
        for selector in families:
            if not _catalog_selector_has_match(
                normalized_catalog,
                parsed,
                selector_type="family",
                selector=selector,
                config_selectors=configs,
            ):
                unmatched.append(f"family:{selector}")
        for selector in family_groups:
            if not _catalog_selector_has_match(
                normalized_catalog,
                parsed,
                selector_type="family_group",
                selector=selector,
                config_selectors=configs,
            ):
                unmatched.append(f"family_group:{selector}")

    if unmatched and not allow_missing:
        raise ValueError(f"Selector(s) matched no columns: {', '.join(unmatched)}")

    removed_indices: list[int] = []
    for column in parsed:
        if features:
            feature_selected = any(
                _feature_selector_matches(column, selector) for selector in features
            )
        else:
            feature_selected = False

        catalog_selected = False
        if normalized_catalog is not None:
            catalog_selected = _catalog_selector_matches(
                normalized_catalog,
                column,
                families=families,
                family_groups=family_groups,
            )

        if not features and not families and not family_groups and configs:
            selected = column.is_feature
        else:
            selected = feature_selected or catalog_selected

        if selected and _config_matches(column, configs):
            removed_indices.append(column.index)

    removed_set = set(removed_indices)
    kept_indices = [idx for idx in range(len(names)) if idx not in removed_set]
    metadata_indices = _resolve_metadata_indices(parsed, metadata_columns, removed_set)

    return _SelectionResult(
        removed_indices=tuple(removed_indices),
        kept_indices=tuple(kept_indices),
        metadata_indices=tuple(metadata_indices),
        unmatched_selectors=tuple(unmatched),
    )


def _resolve_metadata_indices(
    parsed: list[_ParsedColumn],
    metadata_columns: MetadataColumns,
    removed_set: set[int],
) -> list[int]:
    if metadata_columns is None:
        return []

    if metadata_columns == "auto":
        return [
            column.index
            for column in parsed
            if not column.is_feature and column.index not in removed_set
        ]

    requested = _normalize_selectors(metadata_columns)
    name_to_index = {column.name: column.index for column in parsed}
    missing = [name for name in requested if name not in name_to_index]
    if missing:
        raise ValueError(f"Requested metadata column(s) not found: {missing}")
    seen: set[int] = set()
    indices: list[int] = []
    for name in requested:
        idx = name_to_index[name]
        if idx not in seen and idx not in removed_set:
            indices.append(idx)
            seen.add(idx)
    return indices


def _validate_2d_table(X: TableLike, estimator: Any, *, reset: bool) -> tuple[NDArray, list[str]]:
    _assert_dense_matrix(X)

    if isinstance(X, pd.DataFrame):
        names = [str(column) for column in X.columns]
        X_arr = validate_data(
            estimator,
            X=X,
            y="no_validation",
            reset=reset,
            dtype=None,
            ensure_2d=True,
            ensure_all_finite="allow-nan",
        )
        if reset:
            estimator.feature_names_in_ = np.asarray(names, dtype=str)
        return cast(NDArray, X_arr), names

    X_arr = validate_data(
        estimator,
        X=np.asarray(X),
        y="no_validation",
        reset=reset,
        dtype=None,
        ensure_2d=True,
        ensure_all_finite="allow-nan",
    )
    names = [f"feature_{idx}" for idx in range(X_arr.shape[1])]
    if reset:
        estimator.feature_names_in_ = np.asarray(names, dtype=str)
    return cast(NDArray, X_arr), names


class RadiomicsFeatureRemover(TransformerMixin, BaseEstimator):
    """Remove selected columns from Pictologics-style wide radiomics tables.

    Parameters
    ----------
    features : str or iterable of str, optional
        Exact feature keys, full column names, or ``*`` wildcards.
    configs : str or iterable of str, optional
        Optional global filter over Pictologics configuration names.
    families, family_groups : str or iterable of str, optional
        Catalog-backed selectors using Pictologics ``describe_features()`` metadata.
    catalog : DataFrame, path-like, or None
        Optional feature catalog.
    metadata_columns : "auto", iterable of str, or None
        Metadata columns to keep beside removed features in ``split`` output.
    allow_missing : bool
        If False, unmatched selectors raise during ``fit``.
    """

    def __init__(
        self,
        features: SelectorInput = None,
        configs: SelectorInput = None,
        families: SelectorInput = None,
        family_groups: SelectorInput = None,
        catalog: CatalogInput = None,
        metadata_columns: MetadataColumns = "auto",
        allow_missing: bool = False,
    ) -> None:
        self.features = features
        self.configs = configs
        self.families = families
        self.family_groups = family_groups
        self.catalog = catalog
        self.metadata_columns = metadata_columns
        self.allow_missing = allow_missing

    def _get_tags(self) -> dict[str, Any]:
        """Return scikit-learn tags for estimator checks (sklearn < 1.6)."""
        tags = super()._get_tags() if hasattr(super(), "_get_tags") else {}  # type: ignore[misc]
        tags.update(
            {
                "allow_nan": True,
                "X_types": ["2darray", "string"],
            }
        )
        return tags

    def __sklearn_tags__(self) -> Any:
        """Return scikit-learn tags for estimator checks (sklearn >= 1.6)."""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        return tags

    def fit(self, X: TableLike, y: None = None) -> RadiomicsFeatureRemover:
        """Resolve selected radiomics columns."""
        _, names = _validate_2d_table(X, self, reset=True)
        features = _normalize_selectors(self.features)
        configs = _normalize_selectors(self.configs)
        families = _normalize_selectors(self.families)
        family_groups = _normalize_selectors(self.family_groups)
        catalog = _load_catalog(self.catalog)

        if not isinstance(X, pd.DataFrame) and (features or configs or families or family_groups):
            raise ValueError(
                "Name-based radiomics feature removal requires a pandas DataFrame input "
                "with feature names during fit."
            )

        selection = _resolve_selection(
            names,
            features=features,
            configs=configs,
            families=families,
            family_groups=family_groups,
            catalog=catalog,
            metadata_columns=self.metadata_columns,
            allow_missing=bool(self.allow_missing),
        )

        self.catalog_ = catalog
        self.removed_indices_ = np.asarray(selection.removed_indices, dtype=int)
        self.kept_indices_ = np.asarray(selection.kept_indices, dtype=int)
        self.metadata_indices_ = np.asarray(selection.metadata_indices, dtype=int)
        self.removed_feature_names_ = np.asarray(
            [names[idx] for idx in selection.removed_indices],
            dtype=str,
        )
        self.kept_feature_names_ = np.asarray(
            [names[idx] for idx in selection.kept_indices],
            dtype=str,
        )
        self.metadata_columns_ = np.asarray(
            [names[idx] for idx in selection.metadata_indices],
            dtype=str,
        )
        self.unmatched_selectors_ = selection.unmatched_selectors
        return self

    def transform(self, X: TableLike) -> TableLike:
        """Return *X* with selected feature columns removed."""
        check_is_fitted(self, "kept_indices_")
        return self._take_columns(X, self.kept_indices_)

    def split(self, X: TableLike) -> RadiomicsFeatureSplit:
        """Return kept and removed tables for *X* using fitted column selections."""
        check_is_fitted(self, "kept_indices_")
        kept = self._take_columns(X, self.kept_indices_)
        removed_indices = _unique_indices(
            tuple(int(idx) for idx in self.metadata_indices_)
            + tuple(int(idx) for idx in self.removed_indices_)
        )
        removed = self._take_columns(X, np.asarray(removed_indices, dtype=int))
        return RadiomicsFeatureSplit(
            kept_table=kept,
            removed_table=removed,
            kept_columns=tuple(self.kept_feature_names_.tolist()),
            removed_columns=tuple(self.removed_feature_names_.tolist()),
            metadata_columns=tuple(self.metadata_columns_.tolist()),
            unmatched_selectors=tuple(self.unmatched_selectors_),
        )

    def get_feature_names_out(self, input_features: NDArray | None = None) -> NDArray:
        """Return feature names after removal."""
        check_is_fitted(self, "kept_feature_names_")
        return cast(NDArray, self.kept_feature_names_.copy())

    def _take_columns(self, X: TableLike, indices: NDArray) -> TableLike:
        names = _validate_transform_names(X, self)
        # Strict same-order check only when transforming a named DataFrame; a bare
        # array of the same width (positional selection) is supported and must not
        # raise, matching scikit-learn's cross-type behaviour.
        if isinstance(X, pd.DataFrame):
            _check_feature_names(
                self.feature_names_in_,
                np.asarray(names, dtype=str),
                type(self).__name__,
            )
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, indices].copy()
        X_arr = np.asarray(X)
        return cast(NDArray, X_arr[:, indices])


def _unique_indices(indices: tuple[int, ...]) -> tuple[int, ...]:
    seen: set[int] = set()
    result: list[int] = []
    for idx in indices:
        if idx not in seen:
            result.append(idx)
            seen.add(idx)
    return tuple(result)


def _validate_transform_names(X: TableLike, estimator: Any) -> list[str]:
    _assert_dense_matrix(X)
    if isinstance(X, pd.DataFrame):
        validate_data(
            estimator,
            X=X,
            y="no_validation",
            reset=False,
            dtype=None,
            ensure_2d=True,
            ensure_all_finite="allow-nan",
        )
        return [str(column) for column in X.columns]

    X_arr = np.asarray(X)
    had_feature_names = hasattr(estimator, "feature_names_in_")
    stored_feature_names = getattr(estimator, "feature_names_in_", None)
    if had_feature_names:
        delattr(estimator, "feature_names_in_")
    try:
        validate_data(
            estimator,
            X=X_arr,
            y="no_validation",
            reset=False,
            dtype=None,
            ensure_2d=True,
            ensure_all_finite="allow-nan",
        )
    finally:
        if had_feature_names:
            estimator.feature_names_in_ = stored_feature_names
    return [f"feature_{idx}" for idx in range(X_arr.shape[1])]


def split_radiomics_table(
    table: pd.DataFrame,
    *,
    features: SelectorInput = None,
    configs: SelectorInput = None,
    families: SelectorInput = None,
    family_groups: SelectorInput = None,
    catalog: CatalogInput = None,
    metadata_columns: MetadataColumns = "auto",
    allow_missing: bool = False,
) -> RadiomicsFeatureSplit:
    """Split selected feature columns out of an in-memory radiomics table."""
    remover = RadiomicsFeatureRemover(
        features=features,
        configs=configs,
        families=families,
        family_groups=family_groups,
        catalog=catalog,
        metadata_columns=metadata_columns,
        allow_missing=allow_missing,
    )
    return remover.fit(table).split(table)


def split_radiomics_file(
    input_table: str | Path,
    output_dir: str | Path,
    *,
    features: SelectorInput = None,
    configs: SelectorInput = None,
    families: SelectorInput = None,
    family_groups: SelectorInput = None,
    catalog: CatalogInput = None,
    metadata_columns: MetadataColumns = "auto",
    allow_missing: bool = False,
    stripped_table_name: str | None = None,
    removed_table_name: str = "removed_features.csv",
    stripped_catalog_name: str = "features_catalog.csv",
    removed_catalog_name: str = "removed_features_catalog.csv",
    manifest_name: str = "radiomics_feature_split_manifest.csv",
    encoding: str = ENCODING,
) -> RadiomicsFeatureSplitFiles:
    """Read a radiomics CSV, split selected features, and write output CSV files."""
    input_path = Path(input_table)
    output_path = Path(output_dir)
    table = pd.read_csv(input_path, encoding=encoding)
    catalog_df = _load_catalog(catalog)

    split = split_radiomics_table(
        table,
        features=features,
        configs=configs,
        families=families,
        family_groups=family_groups,
        catalog=catalog_df,
        metadata_columns=metadata_columns,
        allow_missing=allow_missing,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    stripped_name = (
        stripped_table_name or f"{input_path.stem}_stripped{input_path.suffix or '.csv'}"
    )
    stripped_path = output_path / stripped_name
    removed_path = output_path / removed_table_name
    cast(pd.DataFrame, split.kept_table).to_csv(stripped_path, index=False, encoding=encoding)
    cast(pd.DataFrame, split.removed_table).to_csv(removed_path, index=False, encoding=encoding)

    input_catalog_path = Path(catalog) if isinstance(catalog, (str, Path)) else None
    stripped_catalog_path: Path | None = None
    removed_catalog_path: Path | None = None
    if catalog_df is not None:
        stripped_catalog, removed_catalog = _split_catalog_by_columns(
            catalog_df,
            removed_columns=set(split.removed_columns),
        )
        stripped_catalog_path = output_path / stripped_catalog_name
        removed_catalog_path = output_path / removed_catalog_name
        stripped_catalog.to_csv(stripped_catalog_path, index=False, encoding=encoding)
        removed_catalog.to_csv(removed_catalog_path, index=False, encoding=encoding)

    manifest_path = output_path / manifest_name
    _write_manifest(
        manifest_path,
        input_table=input_path,
        stripped_table=stripped_path,
        removed_table=removed_path,
        input_catalog=input_catalog_path,
        stripped_catalog=stripped_catalog_path,
        removed_catalog=removed_catalog_path,
        split=split,
        features=_normalize_selectors(features),
        configs=_normalize_selectors(configs),
        families=_normalize_selectors(families),
        family_groups=_normalize_selectors(family_groups),
        encoding=encoding,
    )

    return RadiomicsFeatureSplitFiles(
        input_table=input_path,
        stripped_table=stripped_path,
        removed_table=removed_path,
        manifest=manifest_path,
        input_catalog=input_catalog_path,
        stripped_catalog=stripped_catalog_path,
        removed_catalog=removed_catalog_path,
        kept_columns=split.kept_columns,
        removed_columns=split.removed_columns,
        metadata_columns=split.metadata_columns,
        unmatched_selectors=split.unmatched_selectors,
    )


def _split_catalog_by_columns(
    catalog: pd.DataFrame,
    *,
    removed_columns: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized = _normalize_catalog(catalog)
    removed_mask = normalized["_column_name"].isin(removed_columns)
    if not removed_mask.any():
        feature_removed = normalized["feature_key"].isin(
            {column.split("__", 1)[1] for column in removed_columns if "__" in column}
        )
        config_removed = normalized["config"].isna()
        removed_mask = feature_removed & config_removed
    helper_columns = [column for column in normalized.columns if column.startswith("_")]
    stripped = normalized.loc[~removed_mask].drop(columns=helper_columns)
    removed = normalized.loc[removed_mask].drop(columns=helper_columns)
    return stripped.copy(), removed.copy()


def _write_manifest(
    path: Path,
    *,
    input_table: Path,
    stripped_table: Path,
    removed_table: Path,
    input_catalog: Path | None,
    stripped_catalog: Path | None,
    removed_catalog: Path | None,
    split: RadiomicsFeatureSplit,
    features: tuple[str, ...],
    configs: tuple[str, ...],
    families: tuple[str, ...],
    family_groups: tuple[str, ...],
    encoding: str,
) -> None:
    manifest = pd.DataFrame(
        [
            {
                "input_table": input_table,
                "stripped_table": stripped_table,
                "removed_table": removed_table,
                "input_catalog": input_catalog,
                "stripped_catalog": stripped_catalog,
                "removed_catalog": removed_catalog,
                "features": ";".join(features),
                "configs": ";".join(configs),
                "families": ";".join(families),
                "family_groups": ";".join(family_groups),
                "metadata_columns": ";".join(split.metadata_columns),
                "removed_columns": ";".join(split.removed_columns),
                "unmatched_selectors": ";".join(split.unmatched_selectors),
                "n_kept_columns": len(split.kept_columns),
                "n_removed_columns": len(split.removed_columns),
            }
        ]
    )
    manifest.to_csv(path, index=False, encoding=encoding)
