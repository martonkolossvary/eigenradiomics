"""Tests for Pictologics ingestion helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eigenradiomics import (
    FeatureCatalog,
    RadiomicsDataset,
    RadiomicsFeatureRemover,
    compute_reproducibility,
    split_observer_tables,
)


def _raw_export(n: int = 8) -> pd.DataFrame:
    """A single-observer Pictologics wide export with a subject_id leak + metadata."""
    rng = np.random.default_rng(0)
    cols: dict[str, object] = {
        "PatientID": [f"P{i}" for i in range(n)],
        "Vessel": ["LAD", "RCA"] * (n // 2),
    }
    for cfg in ("total_orig", "total_fbn_16"):
        for fk in ("mean_intensity_Q4LE", "joint_entropy_TU9B"):
            cols[f"{cfg}__{fk}"] = rng.standard_normal(n)
    cols["total_orig__subject_id"] = [f"P{i}" for i in range(n)]
    return pd.DataFrame(cols)


def _catalog() -> FeatureCatalog:
    return FeatureCatalog(
        pd.DataFrame(
            {
                "config": ["total_orig", "total_orig", "total_fbn_16", "total_fbn_16"],
                "feature_key": [
                    "mean_intensity_Q4LE", "joint_entropy_TU9B",
                    "mean_intensity_Q4LE", "joint_entropy_TU9B",
                ],
                "family": ["intensity", "glcm", "intensity", "glcm"],
                "family_group": ["Intensity", "Texture", "Intensity", "Texture"],
            }
        )
    )


# ---- RadiomicsDataset.from_pictologics ------------------------------------


def test_from_pictologics_dataframe_with_catalog():
    ds = RadiomicsDataset.from_pictologics(_raw_export(), _catalog(), group="PatientID")
    assert set(ds.feature_columns) == {
        "total_orig__mean_intensity_Q4LE", "total_orig__joint_entropy_TU9B",
        "total_fbn_16__mean_intensity_Q4LE", "total_fbn_16__joint_entropy_TU9B",
    }
    assert "total_orig__subject_id" not in ds.data.columns  # leak dropped
    assert set(ds.metadata_columns) == {"PatientID", "Vessel"}
    assert ds.design.group == "PatientID"


def test_from_pictologics_no_catalog_uses_heuristic():
    ds = RadiomicsDataset.from_pictologics(_raw_export())
    assert len(ds.feature_columns) == 4  # config__feature_key columns, subject_id excluded


def test_from_pictologics_keep_subject_id():
    ds = RadiomicsDataset.from_pictologics(_raw_export(), drop_subject_id=False)
    assert "total_orig__subject_id" in ds.data.columns


def test_from_pictologics_warns_on_catalog_mismatch():
    raw = _raw_export()
    raw["mystery__extra_AB12"] = 1.0  # a feature-like column absent from the catalog
    with pytest.warns(UserWarning, match="not in the catalog"):
        ds = RadiomicsDataset.from_pictologics(raw, _catalog())
    assert "mystery__extra_AB12" not in ds.feature_columns  # treated as metadata


def test_from_pictologics_path_autodiscovers_sidecar(tmp_path):
    table_path = tmp_path / "features.csv"
    _raw_export().to_csv(table_path, index=False)
    pd.DataFrame(
        {
            "config": ["total_orig", "total_orig", "total_fbn_16", "total_fbn_16"],
            "feature_key": [
                "mean_intensity_Q4LE", "joint_entropy_TU9B",
                "mean_intensity_Q4LE", "joint_entropy_TU9B",
            ],
            "family": ["intensity", "glcm", "intensity", "glcm"],
        }
    ).to_csv(tmp_path / "features_catalog.csv", index=False)

    ds = RadiomicsDataset.from_pictologics(str(table_path))  # catalog auto-discovered
    assert isinstance(ds.catalog, FeatureCatalog)
    assert len(ds.feature_columns) == 4


# ---- split_observer_tables ------------------------------------------------


def _paired(n: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    cols: dict[str, object] = {"PatientID": [f"P{i}" for i in range(n)]}
    for obs in ("O1_", "O2_"):
        for fk in ("mean_intensity_Q4LE", "joint_entropy_TU9B"):
            cols[f"{obs}total_orig__{fk}"] = rng.standard_normal(n)
    return pd.DataFrame(cols)


def test_split_observer_tables_feeds_reproducibility():
    tables = split_observer_tables(_paired(), ("O1_", "O2_"), id_columns="PatientID")
    assert len(tables) == 2
    assert list(tables[0].columns) == [
        "total_orig__mean_intensity_Q4LE", "total_orig__joint_entropy_TU9B",
    ]
    assert list(tables[0].index) == [f"P{i}" for i in range(8)]
    results = compute_reproducibility(tables, bootstrap_iterations=20)
    assert len(results["ICC"]) == 2


def test_split_observer_tables_multi_id_columns():
    paired = _paired()
    paired["Visit"] = ["v1", "v2"] * 4
    tables = split_observer_tables(paired, ("O1_", "O2_"), id_columns=["PatientID", "Visit"])
    assert isinstance(tables[0].index, pd.MultiIndex)


def test_split_observer_tables_no_id_columns():
    tables = split_observer_tables(_paired(), ("O1_", "O2_"))
    assert len(tables) == 2


def test_split_observer_tables_requires_two_prefixes():
    with pytest.raises(ValueError, match="at least two"):
        split_observer_tables(_paired(), ("O1_",))


def test_split_observer_tables_missing_prefix_raises():
    with pytest.raises(ValueError, match="no columns found"):
        split_observer_tables(_paired(), ("O1_", "O9_"))


def test_split_observer_tables_reads_csv_path(tmp_path):
    path = tmp_path / "paired.csv"
    _paired().to_csv(path, index=False)
    tables = split_observer_tables(str(path), ("O1_", "O2_"), id_columns="PatientID")
    assert len(tables) == 2


# ---- observer-aware selectors + FeatureCatalog in RadiomicsFeatureRemover --


def test_remover_observer_prefixes_family_selector():
    rem = RadiomicsFeatureRemover(
        families="intensity", catalog=_catalog(), observer_prefixes=("O1_", "O2_")
    ).fit(_paired())
    assert set(rem.removed_feature_names_) == {
        "O1_total_orig__mean_intensity_Q4LE", "O2_total_orig__mean_intensity_Q4LE",
    }


def test_remover_accepts_feature_catalog_object():
    raw = _raw_export()
    rem = RadiomicsFeatureRemover(families="intensity", catalog=_catalog()).fit(raw)
    assert "total_orig__mean_intensity_Q4LE" in rem.removed_feature_names_
