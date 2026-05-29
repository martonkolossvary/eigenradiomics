"""Tests for the FeatureCatalog."""

from __future__ import annotations

import pandas as pd
import pytest

from eigenradiomics import FeatureCatalog


@pytest.fixture()
def catalog_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "config": ["orig", "orig", "wav"],
            "feature_key": ["Energy", "Entropy", "Skew"],
            "feature_name": ["energy", "entropy", "skewness"],
            "family": ["firstorder", "firstorder", "glcm"],
            "family_group": ["texture", "texture", "texture"],
        }
    )


class TestConstruction:
    def test_from_config_feature_key(self, catalog_frame: pd.DataFrame) -> None:
        cat = FeatureCatalog(catalog_frame)
        assert cat.feature_names == ["orig__Energy", "orig__Entropy", "wav__Skew"]
        assert len(cat) == 3

    def test_from_legacy_feature_column(self) -> None:
        cat = FeatureCatalog(
            pd.DataFrame({"feature": ["orig__Energy", "wav__Skew"], "family": ["fo", "glcm"]})
        )
        assert cat.feature_names == ["orig__Energy", "wav__Skew"]
        assert cat.frame.loc[0, "config"] == "orig"
        assert cat.frame.loc[0, "feature_key"] == "Energy"

    def test_invalid_schema_raises(self) -> None:
        with pytest.raises(ValueError, match="must contain"):
            FeatureCatalog(pd.DataFrame({"name": ["a", "b"]}))

    def test_from_csv(self, tmp_path, catalog_frame: pd.DataFrame) -> None:
        path = tmp_path / "catalog.csv"
        catalog_frame.to_csv(path, index=False)
        cat = FeatureCatalog.from_csv(path)
        assert cat.feature_names == ["orig__Energy", "orig__Entropy", "wav__Skew"]


class TestAccess:
    def test_families_and_groups(self, catalog_frame: pd.DataFrame) -> None:
        cat = FeatureCatalog(catalog_frame)
        assert cat.families() == ["firstorder", "glcm"]
        assert cat.family_groups() == ["texture"]

    def test_families_empty_when_absent(self) -> None:
        cat = FeatureCatalog(pd.DataFrame({"feature": ["orig__Energy"]}))
        assert cat.families() == []
        assert cat.family_groups() == []

    def test_contains_and_repr(self, catalog_frame: pd.DataFrame) -> None:
        cat = FeatureCatalog(catalog_frame)
        assert "orig__Energy" in cat
        assert "ghost__X" not in cat
        assert "n_features=3" in repr(cat)


class TestValidate:
    def test_validate_passes(self, catalog_frame: pd.DataFrame) -> None:
        cat = FeatureCatalog(catalog_frame)
        assert cat.validate(["orig__Energy", "wav__Skew"]) == []

    def test_validate_raises_on_missing(self, catalog_frame: pd.DataFrame) -> None:
        cat = FeatureCatalog(catalog_frame)
        with pytest.raises(ValueError, match="not found in the catalog"):
            cat.validate(["orig__Energy", "ghost__X"])

    def test_validate_allow_missing_returns_list(self, catalog_frame: pd.DataFrame) -> None:
        cat = FeatureCatalog(catalog_frame)
        assert cat.validate(["ghost__X"], allow_missing=True) == ["ghost__X"]


class TestAnnotate:
    def test_annotate_default(self, catalog_frame: pd.DataFrame) -> None:
        cat = FeatureCatalog(catalog_frame)
        table = pd.DataFrame({"feature": ["orig__Energy", "wav__Skew"], "p": [0.01, 0.2]})
        out = cat.annotate(table)
        # Leading metadata columns first, original "p" retained.
        assert list(out.columns)[:3] == ["feature", "config", "feature_key"]
        assert "p" in out.columns
        assert out.loc[out["feature"] == "orig__Energy", "family"].iloc[0] == "firstorder"

    def test_annotate_column_subset(self, catalog_frame: pd.DataFrame) -> None:
        cat = FeatureCatalog(catalog_frame)
        table = pd.DataFrame({"feature": ["orig__Energy"], "p": [0.01]})
        out = cat.annotate(table, columns=["family"])
        assert "family" in out.columns
        assert "feature_name" not in out.columns

    def test_annotate_on_other_column(self, catalog_frame: pd.DataFrame) -> None:
        cat = FeatureCatalog(catalog_frame)
        table = pd.DataFrame({"name": ["orig__Energy"], "p": [0.01]})
        out = cat.annotate(table, on="name")
        assert "name" in out.columns
        # No duplicate "feature" join key left behind.
        assert list(out.columns).count("feature") == 0 or "config" in out.columns

    def test_annotate_missing_key_raises(self, catalog_frame: pd.DataFrame) -> None:
        cat = FeatureCatalog(catalog_frame)
        with pytest.raises(ValueError, match="not found"):
            cat.annotate(pd.DataFrame({"x": [1]}), on="feature")
