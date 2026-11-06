"""Helpers for ingesting Pictologics radiomics exports.

Pictologics names feature columns ``{config}__{feature_key}`` and ships a sidecar
catalog (``config``/``feature_key``/``family``/``family_group``). Reproducibility
studies pivot the wide table into observer-paired columns
``{observer}_{config}__{feature_key}`` (e.g. ``O1_total_orig__mean_intensity_Q4LE``).
These helpers parse those names and reshape observer-paired tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def _split_pictologics_name(
    name: str,
    observer_prefixes: Sequence[str] = (),
) -> tuple[str | None, str | None, str | None]:
    """Split a column name into ``(observer, config, feature_key)``.

    The longest matching entry of *observer_prefixes* (e.g. ``"O1_"``) is
    stripped first, so ``"R10_..."`` goes to ``"R10"`` and not to ``"R1"``. The
    remainder is split once on ``"__"``. Returns ``(observer, None, None)`` when
    the remainder is not a ``config__feature_key`` feature name.
    """
    observer: str | None = None
    base = name
    matches = [prefix for prefix in observer_prefixes if prefix and name.startswith(prefix)]
    if matches:
        observer = max(matches, key=len)
        base = name[len(observer) :]
    if "__" in base:
        config, feature_key = base.split("__", 1)
        if config and feature_key:
            return observer, config, feature_key
    return observer, None, None


def split_observer_tables(
    table: pd.DataFrame | str | Path,
    observer_prefixes: Sequence[str],
    *,
    id_columns: str | Sequence[str] | None = None,
) -> list[pd.DataFrame]:
    """Split an observer-paired table into one feature matrix per observer.

    Reproducibility tables carry each feature once per observer as
    ``{observer}_{config}__{feature_key}``. This returns a list (in
    *observer_prefixes* order) of DataFrames whose columns are the base
    ``config__feature_key`` names (the observer prefix stripped), ready to pass to
    :func:`~eigenradiomics.compute_reproducibility`.

    Parameters
    ----------
    table : DataFrame or CSV path
        The observer-paired wide table. A CSV path is read with the
        *id_columns* as text, so zero-padded IDs such as ``"001"`` stay as they are.
    observer_prefixes : sequence of str
        At least two distinct, non-empty prefixes, e.g. ``("O1_", "O2_")``. No
        prefix may start another one (``"R1"`` and ``"R10"`` overlap; use
        ``"R1_"`` and ``"R10_"``).
    id_columns : str or sequence of str, optional
        Identifier column(s) to use as a shared index so the per-observer tables
        align by sample. The keys must be unique and non-missing.
    """
    prefixes = list(observer_prefixes)
    if len(prefixes) < 2:
        raise ValueError("observer_prefixes must list at least two observer prefixes.")
    if not all(isinstance(prefix, str) and prefix for prefix in prefixes):
        raise ValueError("observer_prefixes must be non-empty strings.")
    if len(set(prefixes)) != len(prefixes):
        raise ValueError(f"observer_prefixes must be distinct, got {prefixes}.")
    for first in prefixes:
        for second in prefixes:
            if first != second and second.startswith(first):
                raise ValueError(
                    f"observer prefix {first!r} also matches the columns of {second!r}; "
                    "add a separator (for example 'R1_' and 'R10_')."
                )

    id_cols: list[str] = []
    if id_columns is not None:
        id_cols = [id_columns] if isinstance(id_columns, str) else list(id_columns)
    if isinstance(table, (str, Path)):
        table = pd.read_csv(table, dtype=dict.fromkeys(id_cols, str) or None)

    index: pd.Index | None = None
    if id_cols:
        keys = table[id_cols]
        if keys.isna().any().any():
            raise ValueError(f"id_columns {id_cols} have missing values.")
        index = pd.MultiIndex.from_frame(keys) if len(id_cols) > 1 else pd.Index(keys[id_cols[0]])
        if not index.is_unique:
            duplicated = index[index.duplicated()].unique()[:5].tolist()
            raise ValueError(f"id_columns {id_cols} have duplicate keys: {duplicated}.")

    tables: list[pd.DataFrame] = []
    for prefix in prefixes:
        cols = [c for c in table.columns if isinstance(c, str) and c.startswith(prefix)]
        if not cols:
            raise ValueError(f"no columns found for observer prefix {prefix!r}.")
        sub = table[cols].copy()
        sub.columns = [c[len(prefix) :] for c in cols]
        if index is not None:
            sub.index = index
        tables.append(sub)
    return tables
