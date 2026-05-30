"""Tests for ingestion primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eigenradiomics import merge_tables, normalize_id_column, resolve_duplicates


class TestNormalizeIdColumn:
    def test_strip_and_collapse_whitespace(self) -> None:
        df = pd.DataFrame({"id": [" 001 ", "a  b", "002"]})
        out, changes = normalize_id_column(df, "id")
        assert out["id"].tolist() == ["001", "a b", "002"]
        assert len(changes) == 2

    def test_blank_tokens_masked(self) -> None:
        df = pd.DataFrame({"id": ["", "nan", "NONE", "x"]})
        out, _ = normalize_id_column(df, "id")
        assert out["id"].isna().sum() == 3
        assert out["id"].iloc[3] == "x"

    def test_lower(self) -> None:
        df = pd.DataFrame({"id": ["AbC"]})
        out, _ = normalize_id_column(df, "id", lower=True)
        assert out["id"].iloc[0] == "abc"

    def test_does_not_mutate_input(self) -> None:
        df = pd.DataFrame({"id": [" 1 "]})
        normalize_id_column(df, "id")
        assert df["id"].iloc[0] == " 1 "

    def test_missing_column_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            normalize_id_column(pd.DataFrame({"a": [1]}), "id")


class TestResolveDuplicates:
    def test_no_duplicates_returns_copy(self) -> None:
        df = pd.DataFrame({"k": [1, 2, 3], "v": [1, 2, 3]})
        kept, dropped = resolve_duplicates(df, "k")
        assert len(kept) == 3
        assert len(dropped) == 0

    def test_error_policy_raises(self) -> None:
        df = pd.DataFrame({"k": [1, 1], "v": [1, 2]})
        with pytest.raises(ValueError, match="duplicate"):
            resolve_duplicates(df, "k", policy="error")

    def test_first_last(self) -> None:
        df = pd.DataFrame({"k": [1, 1, 2], "v": [10, 11, 20]})
        kept_first, _ = resolve_duplicates(df, "k", policy="first")
        kept_last, _ = resolve_duplicates(df, "k", policy="last")
        assert kept_first.loc[kept_first["k"] == 1, "v"].iloc[0] == 10
        assert kept_last.loc[kept_last["k"] == 1, "v"].iloc[0] == 11

    def test_most_complete(self) -> None:
        df = pd.DataFrame({"k": [1, 1], "a": [np.nan, 1.0], "b": [np.nan, 2.0]})
        kept, dropped = resolve_duplicates(df, "k", policy="most_complete")
        assert len(kept) == 1
        assert kept["a"].iloc[0] == 1.0
        assert len(dropped) == 1

    def test_min_max_by_order_column(self) -> None:
        df = pd.DataFrame({"k": [1, 1], "date": [2020, 2018], "v": ["new", "old"]})
        kept_min, _ = resolve_duplicates(df, "k", policy="min", order_column="date")
        kept_max, _ = resolve_duplicates(df, "k", policy="max", order_column="date")
        assert kept_min["v"].iloc[0] == "old"  # earliest
        assert kept_max["v"].iloc[0] == "new"  # latest

    def test_min_without_order_column_raises(self) -> None:
        df = pd.DataFrame({"k": [1, 1]})
        with pytest.raises(ValueError, match="order_column"):
            resolve_duplicates(df, "k", policy="min")

    def test_min_order_column_not_found_raises(self) -> None:
        df = pd.DataFrame({"k": [1, 1], "v": [1, 2]})
        with pytest.raises(ValueError, match="order_column 'ghost' not found"):
            resolve_duplicates(df, "k", policy="min", order_column="ghost")

    def test_unknown_policy_raises(self) -> None:
        df = pd.DataFrame({"k": [1, 1]})
        with pytest.raises(ValueError, match="Unknown duplicate policy"):
            resolve_duplicates(df, "k", policy="bogus")

    def test_missing_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Key column"):
            resolve_duplicates(pd.DataFrame({"a": [1]}), "k")

    def test_multicolumn_key(self) -> None:
        df = pd.DataFrame({"p": [1, 1, 1], "m": ["a", "a", "b"], "v": [1, 2, 3]})
        kept, dropped = resolve_duplicates(df, ["p", "m"], policy="first")
        assert len(kept) == 2
        assert len(dropped) == 1


class TestMergeTables:
    def test_matched_and_unmatched(self) -> None:
        left = pd.DataFrame({"PatientID": ["1", "2", "999"], "f": [1.0, 2.0, 3.0]})
        right = pd.DataFrame({"TAJ": ["1", "2", "888"], "age": [60, 70, 80]})
        result = merge_tables(left, right, left_on="PatientID", right_on="TAJ", how="left")
        assert result.n_matched == 2
        assert result.left_only["PatientID"].tolist() == ["999"]
        assert result.right_only["TAJ"].tolist() == ["888"]
        assert result.merged.shape[0] == 3

    def test_validate_one_to_one_violation_raises(self) -> None:
        left = pd.DataFrame({"id": ["1", "1"], "f": [1.0, 2.0]})
        right = pd.DataFrame({"id": ["1"], "age": [60]})
        with pytest.raises(Exception, match="one_to_one|unique|merge"):
            merge_tables(left, right, left_on="id", right_on="id", validate="one_to_one")

    def test_inner_join(self) -> None:
        left = pd.DataFrame({"id": ["1", "2", "3"], "f": [1.0, 2.0, 3.0]})
        right = pd.DataFrame({"id": ["2", "3"], "age": [70, 80]})
        result = merge_tables(left, right, left_on="id", right_on="id", how="inner")
        assert result.merged.shape[0] == 2
        assert result.n_matched == 2

    def test_mismatched_key_lengths_raises(self) -> None:
        left = pd.DataFrame({"a": [1], "b": [2]})
        right = pd.DataFrame({"c": [1]})
        with pytest.raises(ValueError, match="same number"):
            merge_tables(left, right, left_on=["a", "b"], right_on="c")
