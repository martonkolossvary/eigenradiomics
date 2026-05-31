"""Tests for RadiomicsDataset and StudyDesign."""

from __future__ import annotations

import pandas as pd
import pytest

from eigenradiomics import FeatureCatalog, RadiomicsDataset, StudyDesign


@pytest.fixture()
def wide() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PatientID": ["p1", "p2", "p3", "p4"],
            "Center": ["A", "B", "A", "B"],
            "fu_years": [1.2, 3.4, 2.1, 0.9],
            "death": [1, 0, 1, 0],
            "orig__Energy": [1.0, 2.0, 3.0, 4.0],
            "orig__Entropy": [4.0, 3.0, 2.0, 1.0],
        }
    )


class TestStudyDesign:
    def test_role_properties(self) -> None:
        design = StudyDesign(roles={"group": "PatientID", "batch": "Center"})
        assert design.group == "PatientID"
        assert design.batch == "Center"
        assert design.time is None
        assert design.columns() == ["PatientID", "Center"]

    def test_extra_roles_available(self) -> None:
        design = StudyDesign(roles={"observer": "Reader", "phase": "Phase"})
        assert design.roles["observer"] == "Reader"
        assert design.group is None


class TestConstruction:
    def test_auto_detect_features_by_separator(self, wide: pd.DataFrame) -> None:
        ds = RadiomicsDataset(wide)
        assert ds.feature_columns == ["orig__Energy", "orig__Entropy"]
        assert "PatientID" in ds.metadata_columns

    def test_auto_detect_features_by_catalog(self) -> None:
        df = pd.DataFrame({"id": ["p1"], "Energy": [1.0], "Entropy": [2.0]})
        cat = FeatureCatalog(
            pd.DataFrame({"feature": ["Energy"], "family": ["fo"]})
        )
        ds = RadiomicsDataset(df, catalog=cat)
        assert ds.feature_columns == ["Energy"]

    def test_from_wide_with_roles(self, wide: pd.DataFrame) -> None:
        ds = RadiomicsDataset.from_wide(
            wide, group="PatientID", batch="Center", time="fu_years", event="death"
        )
        assert ds.design.group == "PatientID"
        assert ds.shape == (4, 2)

    def test_from_wide_catalog_from_dataframe(self, wide: pd.DataFrame) -> None:
        cat_df = pd.DataFrame(
            {"config": ["orig", "orig"], "feature_key": ["Energy", "Entropy"]}
        )
        ds = RadiomicsDataset.from_wide(wide, catalog=cat_df)
        assert isinstance(ds.catalog, FeatureCatalog)
        assert ds.feature_columns == ["orig__Energy", "orig__Entropy"]

    def test_from_wide_catalog_from_csv(self, tmp_path, wide: pd.DataFrame) -> None:
        cat_path = tmp_path / "catalog.csv"
        pd.DataFrame(
            {"config": ["orig", "orig"], "feature_key": ["Energy", "Entropy"]}
        ).to_csv(cat_path, index=False)
        ds = RadiomicsDataset.from_wide(wide, catalog=str(cat_path))
        assert isinstance(ds.catalog, FeatureCatalog)
        assert ds.feature_columns == ["orig__Energy", "orig__Entropy"]

    def test_non_dataframe_raises(self) -> None:
        with pytest.raises(TypeError, match="DataFrame"):
            RadiomicsDataset([[1, 2], [3, 4]])

    def test_missing_feature_column_raises(self, wide: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not present"):
            RadiomicsDataset(wide, feature_columns=["ghost"])

    def test_missing_design_column_raises(self, wide: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="missing column"):
            RadiomicsDataset(wide, design=StudyDesign(roles={"group": "ghost"}))


class TestAccess:
    def test_features_and_metadata(self, wide: pd.DataFrame) -> None:
        ds = RadiomicsDataset.from_wide(wide, group="PatientID")
        assert list(ds.features.columns) == ["orig__Energy", "orig__Entropy"]
        assert "PatientID" in ds.metadata.columns
        assert "orig__Energy" not in ds.metadata.columns

    def test_groups(self, wide: pd.DataFrame) -> None:
        ds = RadiomicsDataset.from_wide(wide, group="PatientID")
        assert ds.groups.tolist() == ["p1", "p2", "p3", "p4"]

    def test_groups_none_without_role(self, wide: pd.DataFrame) -> None:
        ds = RadiomicsDataset(wide)
        assert ds.groups is None

    def test_survival_y(self, wide: pd.DataFrame) -> None:
        ds = RadiomicsDataset.from_wide(wide, time="fu_years", event="death")
        X, y = ds.to_pipeline_input()
        assert list(X.columns) == ["orig__Energy", "orig__Entropy"]
        assert list(y.columns) == ["fu_years", "death"]

    def test_target_y(self, wide: pd.DataFrame) -> None:
        ds = RadiomicsDataset.from_wide(wide, target="death")
        _, y = ds.to_pipeline_input()
        assert isinstance(y, pd.Series)
        assert y.name == "death"

    def test_no_outcome_y_is_none(self, wide: pd.DataFrame) -> None:
        ds = RadiomicsDataset(wide)
        assert ds.y() is None
        _, y = ds.to_pipeline_input()
        assert y is None

    def test_len_and_repr(self, wide: pd.DataFrame) -> None:
        ds = RadiomicsDataset.from_wide(wide, group="PatientID")
        assert len(ds) == 4
        assert "n_features=2" in repr(ds)

    def test_groups_with_missing_raises(self) -> None:
        df = pd.DataFrame(
            {"PatientID": ["p1", None, "p3"], "orig__E": [1.0, 2.0, 3.0]}
        )
        ds = RadiomicsDataset(df, design=StudyDesign(roles={"group": "PatientID"}))
        with pytest.raises(ValueError, match="missing values"):
            _ = ds.groups

    def test_half_specified_survival_raises(self, wide: pd.DataFrame) -> None:
        # 'time' without 'event' (or vice versa) is a config error, not a silent skip.
        ds = RadiomicsDataset(wide, design=StudyDesign(roles={"time": "fu_years"}))
        with pytest.raises(ValueError, match="both 'time' and 'event'"):
            ds.y()


class TestCatalogIntegration:
    def test_catalog_zero_match_warns_and_falls_back(self) -> None:
        df = pd.DataFrame({"id": ["p1"], "alpha": [1.0], "beta": [2.0]})
        cat = FeatureCatalog(pd.DataFrame({"feature": ["orig__Energy"], "family": ["fo"]}))
        with pytest.warns(UserWarning, match="none of its features match"):
            ds = RadiomicsDataset(df, catalog=cat)
        assert ds.feature_columns == []  # no '__' columns either -> empty
