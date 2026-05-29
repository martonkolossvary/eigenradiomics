"""Tests for radiomics table preprocessing utilities."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eigenradiomics.preprocessing import (
    RadiomicsFeatureRemover,
    RadiomicsPrepTransformer,
    split_radiomics_file,
    split_radiomics_table,
)


@pytest.fixture()
def pictologics_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PatientID": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "Timepoint": ["t0", "t0", "t1", "t1", "t2", "t2"],
            "standard_fbn_8__mean_intensity_Q4LE": [1, 2, 3, 4, 5, 6],
            "standard_fbn_8__volume_RNU0": [10, 11, 12, 13, 14, 15],
            "standard_fbn_8__joint_entropy_TU9B": [3, 4, 5, 6, 7, 8],
            "standard_fbs_8__volume_RNU0": [20, 21, 22, 23, 24, 25],
            "standard_fbs_8__joint_entropy_TU9B": [6, 7, 8, 9, 10, 11],
            "custom__surface_area_C0JK": [30, 31, 32, 33, 34, 35],
        }
    )


@pytest.fixture()
def pictologics_catalog() -> pd.DataFrame:
    rows = []
    specs = {
        "standard_fbn_8": [
            ("mean_intensity_Q4LE", "intensity", "Intensity"),
            ("volume_RNU0", "morphology", "Morphology"),
            ("joint_entropy_TU9B", "glcm", "Texture"),
        ],
        "standard_fbs_8": [
            ("volume_RNU0", "morphology", "Morphology"),
            ("joint_entropy_TU9B", "glcm", "Texture"),
        ],
        "custom": [
            ("surface_area_C0JK", "morphology", "Morphology"),
        ],
    }
    for config, features in specs.items():
        for feature_key, family, family_group in features:
            rows.append(
                {
                    "config": config,
                    "feature_key": feature_key,
                    "feature_name": feature_key.rsplit("_", 1)[0],
                    "family": family,
                    "family_group": family_group,
                }
            )
    return pd.DataFrame(rows)


class TestRadiomicsFeatureSelection:
    def test_exact_feature_key_removes_all_config_occurrences(
        self,
        pictologics_table: pd.DataFrame,
    ) -> None:
        remover = RadiomicsFeatureRemover(features="volume_RNU0").fit(pictologics_table)
        assert list(remover.removed_feature_names_) == [
            "standard_fbn_8__volume_RNU0",
            "standard_fbs_8__volume_RNU0",
        ]

    def test_full_column_name_removes_one_occurrence(
        self,
        pictologics_table: pd.DataFrame,
    ) -> None:
        remover = RadiomicsFeatureRemover(
            features="standard_fbn_8__volume_RNU0",
        ).fit(pictologics_table)
        assert list(remover.removed_feature_names_) == ["standard_fbn_8__volume_RNU0"]

    def test_feature_and_config_wildcards(self, pictologics_table: pd.DataFrame) -> None:
        remover = RadiomicsFeatureRemover(
            features="*__volume_RNU0",
            configs="standard_fbn_*",
        ).fit(pictologics_table)
        assert list(remover.removed_feature_names_) == ["standard_fbn_8__volume_RNU0"]

    def test_config_only_removes_all_features_in_config(
        self,
        pictologics_table: pd.DataFrame,
    ) -> None:
        remover = RadiomicsFeatureRemover(configs="standard_fbs_*").fit(pictologics_table)
        assert list(remover.removed_feature_names_) == [
            "standard_fbs_8__volume_RNU0",
            "standard_fbs_8__joint_entropy_TU9B",
        ]

    def test_duplicate_selectors_do_not_duplicate_removed_columns(
        self,
        pictologics_table: pd.DataFrame,
    ) -> None:
        remover = RadiomicsFeatureRemover(
            features=["volume_RNU0", "*__volume_RNU0"],
        ).fit(pictologics_table)
        assert list(remover.removed_feature_names_).count("standard_fbn_8__volume_RNU0") == 1

    def test_strict_missing_selector_raises(self, pictologics_table: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="matched no columns"):
            RadiomicsFeatureRemover(features="missing_feature").fit(pictologics_table)

    def test_allow_missing_records_unmatched(self, pictologics_table: pd.DataFrame) -> None:
        remover = RadiomicsFeatureRemover(
            features=["volume_RNU0", "missing_feature"],
            allow_missing=True,
        ).fit(pictologics_table)
        assert remover.unmatched_selectors_ == ("feature:missing_feature",)
        assert "standard_fbn_8__volume_RNU0" in remover.removed_feature_names_

    def test_catalog_family_selector(
        self,
        pictologics_table: pd.DataFrame,
        pictologics_catalog: pd.DataFrame,
    ) -> None:
        remover = RadiomicsFeatureRemover(
            families="morphology",
            catalog=pictologics_catalog,
        ).fit(pictologics_table)
        assert list(remover.removed_feature_names_) == [
            "standard_fbn_8__volume_RNU0",
            "standard_fbs_8__volume_RNU0",
            "custom__surface_area_C0JK",
        ]

    def test_catalog_family_selector_with_config_filter(
        self,
        pictologics_table: pd.DataFrame,
        pictologics_catalog: pd.DataFrame,
    ) -> None:
        remover = RadiomicsFeatureRemover(
            configs="standard_fbs_*",
            families="morphology",
            catalog=pictologics_catalog,
        ).fit(pictologics_table)
        assert list(remover.removed_feature_names_) == ["standard_fbs_8__volume_RNU0"]

    def test_catalog_family_row_for_missing_config_records_unmatched(
        self,
        pictologics_table: pd.DataFrame,
    ) -> None:
        catalog = pd.DataFrame(
            {
                "config": ["not_present"],
                "feature_key": ["volume_RNU0"],
                "family": ["morphology"],
                "family_group": ["Morphology"],
            }
        )
        remover = RadiomicsFeatureRemover(
            families="morphology",
            catalog=catalog,
            allow_missing=True,
        ).fit(pictologics_table)
        assert remover.unmatched_selectors_ == ("family:morphology",)

    def test_catalog_texture_group_selector(
        self,
        pictologics_table: pd.DataFrame,
        pictologics_catalog: pd.DataFrame,
    ) -> None:
        remover = RadiomicsFeatureRemover(
            family_groups="texture",
            catalog=pictologics_catalog,
        ).fit(pictologics_table)
        assert list(remover.removed_feature_names_) == [
            "standard_fbn_8__joint_entropy_TU9B",
            "standard_fbs_8__joint_entropy_TU9B",
        ]

    def test_catalog_family_requires_catalog(self, pictologics_table: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="require a feature catalog"):
            RadiomicsFeatureRemover(families="morphology").fit(pictologics_table)

    def test_selector_on_non_feature_dataframe_raises(self) -> None:
        table = pd.DataFrame({"PatientID": ["p1", "p2"], "Timepoint": ["t0", "t1"]})
        with pytest.raises(ValueError, match="requires named columns"):
            RadiomicsFeatureRemover(features="PatientID").fit(table)

    def test_unmatched_config_and_family_group_are_recorded(
        self,
        pictologics_table: pd.DataFrame,
        pictologics_catalog: pd.DataFrame,
    ) -> None:
        remover = RadiomicsFeatureRemover(
            configs="missing_config",
            family_groups="missing_group",
            catalog=pictologics_catalog,
            allow_missing=True,
        ).fit(pictologics_table)
        assert remover.unmatched_selectors_ == (
            "config:missing_config",
            "family_group:missing_group",
        )

    def test_config_filter_prevents_metadata_selector_match(
        self,
        pictologics_table: pd.DataFrame,
    ) -> None:
        remover = RadiomicsFeatureRemover(
            features="PatientID",
            configs="standard_fbn_*",
            allow_missing=True,
        ).fit(pictologics_table)
        assert remover.unmatched_selectors_ == ("feature:PatientID",)

    def test_legacy_catalog_full_feature_column(
        self,
        pictologics_table: pd.DataFrame,
    ) -> None:
        catalog = pd.DataFrame(
            {
                "feature": ["standard_fbn_8__volume_RNU0"],
                "family": ["morphology"],
                "family_group": ["Morphology"],
            }
        )
        remover = RadiomicsFeatureRemover(families="morphology", catalog=catalog).fit(
            pictologics_table
        )
        assert list(remover.removed_feature_names_) == ["standard_fbn_8__volume_RNU0"]

    def test_legacy_catalog_base_feature_column(
        self,
        pictologics_table: pd.DataFrame,
    ) -> None:
        catalog = pd.DataFrame(
            {
                "feature": ["volume_RNU0"],
                "family": ["morphology"],
                "family_group": ["Morphology"],
            }
        )
        remover = RadiomicsFeatureRemover(families="morphology", catalog=catalog).fit(
            pictologics_table
        )
        assert list(remover.removed_feature_names_) == [
            "standard_fbn_8__volume_RNU0",
            "standard_fbs_8__volume_RNU0",
        ]

    def test_invalid_catalog_schema_raises(self, pictologics_table: pd.DataFrame) -> None:
        catalog = pd.DataFrame({"name": ["volume_RNU0"]})
        with pytest.raises(ValueError, match="Feature catalog must contain"):
            RadiomicsFeatureRemover(families="morphology", catalog=catalog).fit(
                pictologics_table
            )

    def test_catalog_missing_family_column_raises(
        self,
        pictologics_table: pd.DataFrame,
    ) -> None:
        catalog = pd.DataFrame(
            {
                "config": ["standard_fbn_8"],
                "feature_key": ["volume_RNU0"],
                "family_group": ["Morphology"],
            }
        )
        with pytest.raises(ValueError, match="require a `family` column"):
            RadiomicsFeatureRemover(families="morphology", catalog=catalog).fit(
                pictologics_table
            )

    def test_catalog_missing_family_group_column_raises(
        self,
        pictologics_table: pd.DataFrame,
    ) -> None:
        catalog = pd.DataFrame(
            {
                "config": ["standard_fbn_8"],
                "feature_key": ["volume_RNU0"],
                "family": ["morphology"],
            }
        )
        with pytest.raises(ValueError, match="require a `family_group` column"):
            RadiomicsFeatureRemover(family_groups="Morphology", catalog=catalog).fit(
                pictologics_table
            )


class TestRadiomicsFeatureRemover:
    def test_transform_preserves_dataframe_rows_and_kept_order(
        self,
        pictologics_table: pd.DataFrame,
    ) -> None:
        transformed = RadiomicsFeatureRemover(features="volume_RNU0").fit_transform(
            pictologics_table
        )
        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape[0] == pictologics_table.shape[0]
        assert list(transformed.columns) == [
            "PatientID",
            "Timepoint",
            "standard_fbn_8__mean_intensity_Q4LE",
            "standard_fbn_8__joint_entropy_TU9B",
            "standard_fbs_8__joint_entropy_TU9B",
            "custom__surface_area_C0JK",
        ]

    def test_split_returns_metadata_prefixed_removed_table(
        self,
        pictologics_table: pd.DataFrame,
    ) -> None:
        split = RadiomicsFeatureRemover(features="volume_RNU0").fit(pictologics_table).split(
            pictologics_table
        )
        assert isinstance(split.removed_table, pd.DataFrame)
        assert list(split.removed_table.columns) == [
            "PatientID",
            "Timepoint",
            "standard_fbn_8__volume_RNU0",
            "standard_fbs_8__volume_RNU0",
        ]
        assert split.metadata_columns == ("PatientID", "Timepoint")

    def test_explicit_metadata_columns(self, pictologics_table: pd.DataFrame) -> None:
        split = RadiomicsFeatureRemover(
            features="volume_RNU0",
            metadata_columns=["PatientID"],
        ).fit(pictologics_table).split(pictologics_table)
        assert list(split.removed_table.columns) == [
            "PatientID",
            "standard_fbn_8__volume_RNU0",
            "standard_fbs_8__volume_RNU0",
        ]

    def test_get_feature_names_out(self, pictologics_table: pd.DataFrame) -> None:
        remover = RadiomicsFeatureRemover(features="volume_RNU0").fit(pictologics_table)
        assert list(remover.get_feature_names_out()) == list(remover.kept_feature_names_)

    def test_pickle_roundtrip(self, pictologics_table: pd.DataFrame) -> None:
        remover = RadiomicsFeatureRemover(features="volume_RNU0").fit(pictologics_table)
        loaded = pickle.loads(pickle.dumps(remover))  # noqa: S301
        pd.testing.assert_frame_equal(
            remover.transform(pictologics_table),
            loaded.transform(pictologics_table),
        )

    def test_ndarray_noop_transform(self) -> None:
        X = np.arange(12).reshape(4, 3)
        transformed = RadiomicsFeatureRemover().fit_transform(X)
        np.testing.assert_array_equal(transformed, X)

    def test_ndarray_named_selector_fit_raises(self) -> None:
        X = np.arange(12).reshape(4, 3)
        with pytest.raises(ValueError, match="requires a pandas DataFrame"):
            RadiomicsFeatureRemover(features="volume_RNU0").fit(X)

    def test_sparse_fit_raises(self) -> None:
        with pytest.raises(TypeError, match="Sparse matrices are not supported"):
            RadiomicsFeatureRemover().fit(scipy.sparse.csr_matrix(np.eye(3)))

    def test_sparse_transform_raises(self) -> None:
        remover = RadiomicsFeatureRemover().fit(np.eye(3))
        with pytest.raises(TypeError, match="Sparse matrices are not supported"):
            remover.transform(scipy.sparse.csr_matrix(np.eye(3)))

    def test_sklearn_tags(self) -> None:
        remover = RadiomicsFeatureRemover()
        tags = remover._get_tags()
        sklearn_tags = remover.__sklearn_tags__()
        assert tags["allow_nan"]
        assert sklearn_tags.input_tags.string

    def test_dataframe_column_order_must_match(self, pictologics_table: pd.DataFrame) -> None:
        remover = RadiomicsFeatureRemover(features="volume_RNU0").fit(pictologics_table)
        with pytest.raises(ValueError, match="same order"):
            remover.transform(pictologics_table.iloc[:, ::-1])

    def test_explicit_metadata_missing_raises(self, pictologics_table: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="metadata column"):
            RadiomicsFeatureRemover(
                features="volume_RNU0",
                metadata_columns=["missing"],
            ).fit(pictologics_table)

    def test_metadata_columns_none(self, pictologics_table: pd.DataFrame) -> None:
        split = RadiomicsFeatureRemover(
            features="volume_RNU0",
            metadata_columns=None,
        ).fit(pictologics_table).split(pictologics_table)
        assert split.metadata_columns == ()
        assert list(split.removed_table.columns) == [
            "standard_fbn_8__volume_RNU0",
            "standard_fbs_8__volume_RNU0",
        ]

    def test_pipeline_before_numeric_sklearn_steps(self) -> None:
        X = pd.DataFrame(
            {
                "standard_fbn_8__mean_intensity_Q4LE": [1.0, 2.0, 3.0, 4.0],
                "standard_fbn_8__volume_RNU0": [10.0, 11.0, 12.0, 13.0],
                "standard_fbn_8__joint_entropy_TU9B": [3.0, 4.0, 5.0, 6.0],
            }
        )
        pipe = Pipeline(
            [
                ("remove", RadiomicsFeatureRemover(features="volume_RNU0")),
                ("var", VarianceThreshold(threshold=0.0)),
                ("scale", StandardScaler()),
            ]
        )
        transformed = pipe.fit_transform(X)
        assert transformed.shape == (4, 2)


class TestRadiomicsFeatureSplitHelpers:
    def test_split_radiomics_table(
        self,
        pictologics_table: pd.DataFrame,
        pictologics_catalog: pd.DataFrame,
    ) -> None:
        split = split_radiomics_table(
            pictologics_table,
            families="glcm",
            catalog=pictologics_catalog,
        )
        assert split.removed_columns == (
            "standard_fbn_8__joint_entropy_TU9B",
            "standard_fbs_8__joint_entropy_TU9B",
        )

    def test_split_radiomics_file_writes_outputs(
        self,
        tmp_path: Path,
        pictologics_table: pd.DataFrame,
        pictologics_catalog: pd.DataFrame,
    ) -> None:
        input_path = tmp_path / "radiomics.csv"
        catalog_path = tmp_path / "feature_catalog.csv"
        output_dir = tmp_path / "out"
        pictologics_table.to_csv(input_path, index=False)
        pictologics_catalog.to_csv(catalog_path, index=False)

        result = split_radiomics_file(
            input_path,
            output_dir,
            features="volume_RNU0",
            catalog=catalog_path,
            metadata_columns=["PatientID"],
            stripped_table_name="stripped.csv",
        )

        assert result.stripped_table.exists()
        assert result.removed_table.exists()
        assert result.stripped_catalog is not None and result.stripped_catalog.exists()
        assert result.removed_catalog is not None and result.removed_catalog.exists()
        assert result.manifest.exists()

        stripped = pd.read_csv(result.stripped_table)
        removed = pd.read_csv(result.removed_table)
        removed_catalog = pd.read_csv(result.removed_catalog)
        manifest = pd.read_csv(result.manifest)

        assert "standard_fbn_8__volume_RNU0" not in stripped.columns
        assert list(removed.columns) == [
            "PatientID",
            "standard_fbn_8__volume_RNU0",
            "standard_fbs_8__volume_RNU0",
        ]
        assert set(removed_catalog["feature_key"]) == {"volume_RNU0"}
        assert manifest.iloc[0]["n_removed_columns"] == 2

    def test_split_radiomics_file_splits_legacy_base_catalog(
        self,
        tmp_path: Path,
        pictologics_table: pd.DataFrame,
    ) -> None:
        input_path = tmp_path / "radiomics.csv"
        catalog_path = tmp_path / "legacy_catalog.csv"
        output_dir = tmp_path / "out"
        pictologics_table.to_csv(input_path, index=False)
        pd.DataFrame(
            {
                "feature": ["volume_RNU0", "joint_entropy_TU9B"],
                "family": ["morphology", "glcm"],
                "family_group": ["Morphology", "Texture"],
            }
        ).to_csv(catalog_path, index=False)

        result = split_radiomics_file(
            input_path,
            output_dir,
            features="volume_RNU0",
            catalog=catalog_path,
        )

        assert result.removed_catalog is not None
        removed_catalog = pd.read_csv(result.removed_catalog)
        assert removed_catalog["feature"].tolist() == ["volume_RNU0"]


class TestRadiomicsPrepTransformer:
    def test_sklearn_check_estimator(self) -> None:
        from sklearn.utils.estimator_checks import check_estimator

        check_estimator(RadiomicsPrepTransformer())

    def test_get_feature_names_out(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        pt = RadiomicsPrepTransformer().fit(df)
        assert list(pt.get_feature_names_out()) == ["a", "b"]

    def test_reordered_columns_rejected(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 5.0, 6.0, 7.0]})
        pt = RadiomicsPrepTransformer().fit(df)
        with pytest.raises(ValueError, match="feature names"):
            pt.transform(df[["b", "a"]])

    def test_wrong_n_features_rejected(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        pt = RadiomicsPrepTransformer().fit(df)
        with pytest.raises(ValueError):
            pt.transform(df[["a"]])

    def test_nan_preserved_and_dataframe_returned(self) -> None:
        df = pd.DataFrame(
            {"a": [1.0, np.nan, 3.0, 4.0, 5.0], "b": [5.0, 4.0, 3.0, 2.0, 1.0]}
        )
        out = RadiomicsPrepTransformer().fit_transform(df)
        assert isinstance(out, pd.DataFrame)
        assert np.isnan(out.iloc[1, 0])

    def test_numpy_input_returns_ndarray(self) -> None:
        X = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0], [4.0, 7.0]])
        out = RadiomicsPrepTransformer().fit_transform(X)
        assert isinstance(out, np.ndarray)
        assert out.shape == X.shape

    def test_pickle_roundtrip(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0], "b": [4.0, 5.0, 6.0, 7.0]})
        pt = RadiomicsPrepTransformer().fit(df)
        loaded = pickle.loads(pickle.dumps(pt))  # noqa: S301
        pd.testing.assert_frame_equal(pt.transform(df), loaded.transform(df))

    def test_all_nan_column_carried_through(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [np.nan, np.nan, np.nan, np.nan]})
        out = RadiomicsPrepTransformer().fit_transform(df)
        assert out["b"].isna().all()
        assert not out["a"].isna().any()
