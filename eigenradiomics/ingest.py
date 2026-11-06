"""Ingestion primitives: normalize identifiers, resolve duplicates, merge tables.

These study-agnostic helpers cover the "get a clean n x m matrix" entry point:
aligning radiomics tables with clinical/metadata tables by sample identifier,
with explicit diagnostics so silent misalignment cannot occur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Strings treated as "blank" identifiers and masked to missing during normalization.
_BLANK_TOKENS = {"", "nan", "none", "<na>", "null", "na"}
_SENTINEL = "\x00"


def _integer_float_as_int(value: object) -> object:
    """Return an integer-valued float as an int; return other values unchanged."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    return value


def normalize_id_column(
    df: pd.DataFrame,
    column: str,
    *,
    lower: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize an identifier column and report what changed.

    Strips leading/trailing whitespace, collapses internal whitespace runs to a
    single space, and masks blank-like tokens (``""``, ``"nan"``, ``"none"``,
    ...) to missing values. An integer-valued float ID becomes integer text
    (``1.0`` -> ``"1"``); a real decimal such as ``1.5`` stays ``"1.5"``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table (not modified in place).
    column : str
        Identifier column to normalize.
    lower : bool
        If True, lowercase the identifier as well.

    Returns
    -------
    normalized : pandas.DataFrame
        Copy of *df* with *column* normalized.
    changes : pandas.DataFrame
        Rows whose value changed, and rows whose normalized ID is shared with a
        row that had a different raw value. Columns: ``before`` (the raw value
        as text), ``after``, ``reason`` (``integer-valued float``,
        ``whitespace``, ``lowercase``, ``blank token``, joined by ``"; "``) and
        ``collision`` (True when different raw values give this normalized ID).
    """
    if column not in df.columns:
        raise ValueError(f"Column {column!r} not found in the table.")

    raw_text = pd.Series(
        [pd.NA if pd.isna(value) else str(value) for value in df[column]],
        index=df.index,
        dtype="string",
    )
    # Write an integer-valued float ID as an integer ("1", not "1.0"), so it
    # matches the same ID that is stored as an integer.
    values = [_integer_float_as_int(value) for value in df[column]]
    original = pd.Series(values, index=df.index, dtype=object).astype("string")
    spaced = original.str.strip().str.replace(r"\s+", " ", regex=True)
    cased = spaced.str.lower() if lower else spaced
    cleaned = cased.mask(cased.str.lower().isin(_BLANK_TOKENS))

    def differs(a: pd.Series, b: pd.Series) -> pd.Series:
        return a.fillna(_SENTINEL) != b.fillna(_SENTINEL)

    reasons = pd.DataFrame(
        {
            "integer-valued float": differs(raw_text, original),
            "whitespace": differs(original, spaced),
            "lowercase": differs(spaced, cased),
            "blank token": differs(cased, cleaned),
        }
    )
    reason = reasons.apply(lambda row: "; ".join(row.index[row]), axis=1)
    known = cleaned.notna()
    n_raw = raw_text[known].groupby(cleaned[known]).transform("nunique")
    collision = pd.Series(False, index=df.index)
    collision[known] = n_raw.to_numpy() > 1
    report = differs(raw_text, cleaned) | collision
    changes = pd.DataFrame(
        {
            column: raw_text[report].to_numpy(),
            "before": raw_text[report].to_numpy(),
            "after": cleaned[report].to_numpy(),
            "reason": reason[report].to_numpy(),
            "collision": collision[report].to_numpy(),
        }
    )

    out = df.copy()
    out[column] = cleaned
    return out, changes


def resolve_duplicates(
    df: pd.DataFrame,
    key: str | list[str],
    *,
    policy: str = "error",
    order_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve rows that share the same *key*.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table (not modified in place).
    key : str or list of str
        Column(s) that should uniquely identify a row.
    policy : {"error", "first", "last", "most_complete", "min", "max"}
        How to resolve duplicates:

        - ``"error"`` (default): raise if any duplicates exist.
        - ``"first"`` / ``"last"``: keep the first / last occurrence.
        - ``"most_complete"``: keep the row with the most non-missing values.
        - ``"min"`` / ``"max"``: keep the row with the smallest / largest value
          of ``order_column`` (e.g. earliest procedure date).
    order_column : str, optional
        Required for ``policy="min"`` or ``"max"``.

    Returns
    -------
    deduplicated : pandas.DataFrame
        Table with one row per key (original row order preserved).
    dropped : pandas.DataFrame
        The rows that were removed.
    """
    key_cols = [key] if isinstance(key, str) else list(key)
    missing = [col for col in key_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Key column(s) not found: {missing}")

    work = df.reset_index(drop=True)
    dup_mask = work.duplicated(subset=key_cols, keep=False)
    if not dup_mask.any():
        return df.copy(), work.iloc[0:0].copy()

    if policy == "error":
        examples = work.loc[dup_mask, key_cols].drop_duplicates().head(5)
        raise ValueError(
            f"Found duplicate rows for key {key_cols}. "
            f"First duplicated key(s):\n{examples.to_string(index=False)}"
        )

    if policy in {"first", "last"}:
        kept_index = work.drop_duplicates(subset=key_cols, keep=policy).index
    elif policy == "most_complete":
        completeness = work.notna().sum(axis=1)
        ordered = work.assign(_completeness=completeness).sort_values(
            "_completeness", ascending=False, kind="stable"
        )
        kept_index = ordered.drop_duplicates(subset=key_cols, keep="first").index
    elif policy in {"min", "max"}:
        if order_column is None:
            raise ValueError(f"policy={policy!r} requires order_column.")
        if order_column not in work.columns:
            raise ValueError(f"order_column {order_column!r} not found in the table.")
        ordered = work.sort_values(order_column, ascending=(policy == "min"), kind="stable")
        kept_index = ordered.drop_duplicates(subset=key_cols, keep="first").index
    else:
        raise ValueError(f"Unknown duplicate policy: {policy!r}")

    kept_positions = sorted(kept_index)
    kept = work.loc[kept_positions].reset_index(drop=True)
    dropped = work.drop(index=kept_index).reset_index(drop=True)
    return kept, dropped


@dataclass(frozen=True)
class MergeResult:
    """Result of :func:`merge_tables` with alignment diagnostics."""

    merged: pd.DataFrame
    left_only: pd.DataFrame
    right_only: pd.DataFrame
    n_matched: int
    #: Rows of each table whose key matched a row of the other table.
    n_left_matched_rows: int = 0
    n_right_matched_rows: int = 0
    #: Rows of each table with a missing key value (they match no row).
    n_left_missing_keys: int = 0
    n_right_missing_keys: int = 0
    #: Rows in ``merged``.
    n_merged_rows: int = 0


def merge_tables(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_on: str | list[str],
    right_on: str | list[str],
    how: str = "left",
    validate: str | None = None,
    suffixes: tuple[str, str] = ("", "_clinical"),
) -> MergeResult:
    """Merge two tables by key, returning the join plus mismatch diagnostics.

    Parameters
    ----------
    left, right : pandas.DataFrame
        Tables to merge (e.g. radiomics features and clinical metadata).
    left_on, right_on : str or list of str
        Join key column(s) in each table.
    how : str
        Pandas merge strategy (``"left"``, ``"inner"``, ...).
    validate : str, optional
        Pandas merge cardinality check, e.g. ``"one_to_one"`` or
        ``"one_to_many"``. Raises if violated.
    suffixes : tuple of str
        Suffixes for overlapping non-key columns.

    Returns
    -------
    result : MergeResult
        ``merged`` plus ``left_only`` / ``right_only`` (rows whose key matched
        nothing in the other table), ``n_matched`` (number of distinct matched
        keys), the matched row counts of each side, the missing-key row counts
        of each side, and ``n_merged_rows``.
        A row with a missing key value matches no row, so it goes to
        ``left_only`` / ``right_only`` and does not count in ``n_matched``.
    """
    left_cols = [left_on] if isinstance(left_on, str) else list(left_on)
    right_cols = [right_on] if isinstance(right_on, str) else list(right_on)
    if len(left_cols) != len(right_cols):
        raise ValueError("left_on and right_on must have the same number of columns.")

    # Key-level match diagnostics (independent of `how`). A row with a
    # missing key part matches no row.
    left_ok = left[left_cols].notna().all(axis=1)
    right_ok = right[right_cols].notna().all(axis=1)
    left_keys = left[left_cols].apply(tuple, axis=1)
    right_keys = right[right_cols].apply(tuple, axis=1)
    matched = set(left_keys[left_ok]) & set(right_keys[right_ok])
    left_only = left[~(left_ok & left_keys.isin(matched))].reset_index(drop=True)
    right_only = right[~(right_ok & right_keys.isin(matched))].reset_index(drop=True)

    # pandas joins a missing key to a missing key. Give each row with a
    # missing key its own guard value, so that row matches no row.
    guard = "__eigenradiomics_merge_guard__"
    left_guard = np.where(left_ok, 0, -np.arange(1, len(left) + 1))
    right_guard = np.where(right_ok, 0, np.arange(1, len(right) + 1))
    merged = (
        left.assign(**{guard: left_guard})
        .merge(
            right.assign(**{guard: right_guard}),
            left_on=[*left_cols, guard],
            right_on=[*right_cols, guard],
            how=how,
            validate=validate,
            suffixes=suffixes,
        )
        .drop(columns=guard)
    )
    return MergeResult(
        merged=merged,
        left_only=left_only,
        right_only=right_only,
        n_matched=len(matched),
        n_left_matched_rows=len(left) - len(left_only),
        n_right_matched_rows=len(right) - len(right_only),
        n_left_missing_keys=int((~left_ok).sum()),
        n_right_missing_keys=int((~right_ok).sum()),
        n_merged_rows=len(merged),
    )
