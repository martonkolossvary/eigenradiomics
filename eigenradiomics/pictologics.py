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

    A leading entry of *observer_prefixes* (e.g. ``"O1_"``) is stripped first;
    the remainder is split once on ``"__"``. Returns ``(observer, None, None)``
    when the remainder is not a ``config__feature_key`` feature name.
    """
    observer: str | None = None
    base = name
    for prefix in observer_prefixes:
        if name.startswith(prefix):
            observer = prefix
            base = name[len(prefix):]
            break
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
        The observer-paired wide table.
    observer_prefixes : sequence of str
        At least two prefixes, e.g. ``("O1_", "O2_")``.
    id_columns : str or sequence of str, optional
        Identifier column(s) to use as a shared index so the per-observer tables
        align by sample.
    """
    if isinstance(table, (str, Path)):
        table = pd.read_csv(table)
    if len(observer_prefixes) < 2:
        raise ValueError("observer_prefixes must list at least two observer prefixes.")

    index: pd.Index | None = None
    if id_columns is not None:
        id_cols = [id_columns] if isinstance(id_columns, str) else list(id_columns)
        index = (
            pd.MultiIndex.from_frame(table[id_cols])
            if len(id_cols) > 1
            else pd.Index(table[id_cols[0]])
        )

    tables: list[pd.DataFrame] = []
    for prefix in observer_prefixes:
        cols = [c for c in table.columns if isinstance(c, str) and c.startswith(prefix)]
        if not cols:
            raise ValueError(f"no columns found for observer prefix {prefix!r}.")
        sub = table[cols].copy()
        sub.columns = [c[len(prefix):] for c in cols]
        if index is not None:
            sub.index = index
        tables.append(sub)
    return tables
