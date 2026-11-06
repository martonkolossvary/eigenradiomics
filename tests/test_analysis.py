"""Tests for WGCNA-inspired analysis utilities and accessible plotting functions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from eigenradiomics.analysis import (
    compute_group_enrichment,
    compute_module_membership,
    identify_hub_features,
)
from eigenradiomics.plotting import (
    plot_batch_distributions,
    plot_eigengene_profiles,
    plot_hub_significance,
)
from eigenradiomics.reducers import WGCNAReducer


@pytest.fixture()
def sample_data():
    """Deterministic feature data, components, and assignments for testing."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(size=(20, 6)),
        columns=["feat_1", "feat_2", "feat_3", "feat_4", "feat_5", "feat_6"],
    )
    # 2 components
    eigengenes = pd.DataFrame(rng.normal(size=(20, 2)), columns=["blue", "brown"])
    # Custom assignments
    cluster_labels = pd.Series(
        ["blue", "blue", "blue", "brown", "brown", "brown"], index=X.columns, name="module"
    )
    group_assignments = pd.Series(
        ["firstorder", "firstorder", "glcm", "glcm", "shape", "shape"],
        index=X.columns,
        name="family",
    )
    return X, eigengenes, cluster_labels, group_assignments


class TestAnalysisStats:
    """Validate WGCNA analysis helpers k_ME, hub feature search, and hypergeometric ORA."""

    def test_module_membership_calculation(self, sample_data):
        X, eigengenes, _, _ = sample_data
        kme = compute_module_membership(X, eigengenes, method="spearman")
        assert kme.shape == (6, 2)
        assert list(kme.columns) == ["blue", "brown"]
        assert list(kme.index) == list(X.columns)
        assert kme.abs().max().max() <= 1.0

        # Pearson and Kendall modes
        kme_pearson = compute_module_membership(X, eigengenes, method="pearson")
        assert kme_pearson.shape == (6, 2)
        kme_kendall = compute_module_membership(X, eigengenes, method="kendall")
        assert kme_kendall.shape == (6, 2)

        # Numpy input
        kme_arr = compute_module_membership(X.to_numpy(), eigengenes.to_numpy())
        assert kme_arr.shape == (6, 2)

        # List/Numpy headers fallback
        kme_list_fallback = compute_module_membership(X.to_numpy(), eigengenes)
        assert kme_list_fallback.shape == (6, 2)

    def test_membership_from_reducer(self, sample_data):
        X, _, _, _ = sample_data

        # Sequentially fitted reducer with 1 component mapping
        reducer_single = WGCNAReducer(
            soft_power=6, min_module_size=2, include_grey=True, verbose=0
        )
        reducer_single.fit(X)
        kme_single = compute_module_membership(X, reducer=reducer_single)
        assert kme_single.shape[0] == X.shape[1]

        # Multi-component dummy WGCNAReducer coverage (line 82)
        reducer_multi = WGCNAReducer(
            soft_power=6, min_module_size=2, include_grey=True, n_module_components=2, verbose=0
        )
        reducer_multi.fit(X)
        kme_multi = compute_module_membership(X, reducer=reducer_multi)
        assert kme_multi.shape[0] == X.shape[1]

        # Reducer without module_names_ fallback (line 82)
        from eigenradiomics.reducers import PCAReducer

        pca_red = PCAReducer(n_components=2)
        pca_red.fit(X)
        kme_pca = compute_module_membership(X, reducer=pca_red)
        assert kme_pca.shape[1] == 2

    def test_membership_mismatch_raises(self, sample_data):
        X, eigengenes, _, _ = sample_data
        with pytest.raises(ValueError, match="Either 'eigengenes' or a fitted 'reducer'"):
            compute_module_membership(X)
        with pytest.raises(ValueError, match="Row count mismatch"):
            compute_module_membership(X.iloc[:5, :], eigengenes)
        with pytest.raises(TypeError, match="not a valid scikit-learn estimator"):
            compute_module_membership(X, reducer="invalid_reducer")

    def test_identify_hub_features(self, sample_data):
        X, eigengenes, cluster_labels, _ = sample_data
        hubs = identify_hub_features(X, cluster_labels, eigengenes, top_n=1)
        assert isinstance(hubs, pd.DataFrame)
        assert len(hubs) == 2
        assert list(hubs.columns) == ["cluster", "feature", "k_ME", "rank"]
        assert set(hubs["cluster"]) == {"blue", "brown"}
        assert hubs.loc[hubs["cluster"] == "blue", "rank"].iloc[0] == 1

        # Dict and Sequence labels (trigerring Series coercion lines 185)
        h_bare_series = pd.Series(list(cluster_labels), index=X.columns)
        hubs_dict = identify_hub_features(X, h_bare_series, eigengenes)
        assert len(hubs_dict) == 2

        # Dict input triggers dict coercion (line 185)
        hubs_raw_dict = identify_hub_features(X, cluster_labels.to_dict(), eigengenes)
        assert len(hubs_raw_dict) == 2

        # Sequence with positional list structure (line 185)
        h_flat = list(cluster_labels)
        h_flat_annotated = [h_flat[i] for i in range(len(h_flat))]
        hubs_seq = identify_hub_features(X, h_flat_annotated, eigengenes)
        assert len(hubs_seq) == 2

        # Trigger line 169 (X being a non-DataFrame list/NumPy array seen during transform)
        hubs_raw_numpy = identify_hub_features(
            X.to_numpy(), list(cluster_labels), eigengenes, method="spearman"
        )
        assert len(hubs_raw_numpy) > 0

        # Labels without a matching eigengene column are skipped (no positional map).
        with pytest.warns(UserWarning, match="no matching eigengene column"):
            hubs_fall = identify_hub_features(X, ["ghost1", "ghost2"] * 3, eigengenes)
        assert len(hubs_fall) == 0

        # Trigger line 185 (neither Series nor dict, sequence input)
        hubs_sequence_direct = identify_hub_features(
            X, [None, None, None, None, None, None], eigengenes
        )
        assert isinstance(hubs_sequence_direct, pd.DataFrame)

        # Explicit test triggering lines 185, 199-200, 207 directly
        # Passing a bare dictionary representing label sequence
        h_bare_dict = {f"feat_{i + 1}": "blue" if i < 3 else "brown" for i in range(6)}
        hubs_dict_seq = identify_hub_features(X, h_bare_dict, eigengenes)
        assert len(hubs_dict_seq) == 2

        # Trigger line 185 - non-DataFrame NumPy array input for reducer-fitted assignments
        reducer_dummy = WGCNAReducer(soft_power=6, min_module_size=2, include_grey=True, verbose=0)
        reducer_dummy.fit(X)
        hubs_arr_fall = identify_hub_features(
            X.to_numpy(), list(reducer_dummy.module_colors_), reducer=reducer_dummy
        )
        assert set(hubs_arr_fall["cluster"]) == set(reducer_dummy.module_names_)

        # Unnamed eigengene columns cannot be matched to labels, so nothing is reported.
        with pytest.warns(UserWarning, match="no matching eigengene column"):
            hubs_unlabelled = identify_hub_features(
                X.to_numpy(), list(cluster_labels), eigengenes.to_numpy()
            )
        assert len(hubs_unlabelled) == 0

    def test_compute_group_enrichment(self, sample_data):
        _, _, cluster_labels, group_assignments = sample_data
        enrichment = compute_group_enrichment(cluster_labels, group_assignments)
        assert isinstance(enrichment, pd.DataFrame)
        assert len(enrichment) > 0
        expected_cols = {
            "cluster",
            "group",
            "n_overlap",
            "cluster_size",
            "group_size",
            "total_features",
            "p_value",
            "odds_ratio",
            "fdr_q_value",
        }
        assert expected_cols.issubset(enrichment.columns)

        # Empty assignments triggered coverage of line 330
        empty_enrich = compute_group_enrichment(
            pd.Series([], dtype=str), pd.Series([], dtype=str), feature_names=[]
        )
        assert isinstance(empty_enrich, pd.DataFrame)
        assert len(empty_enrich) == 0

        # Dict, Sequence, and feature_names parameters coverage
        feature_names = list(cluster_labels.index)
        enrich_dict = compute_group_enrichment(
            cluster_labels.to_dict(), group_assignments.to_dict()
        )
        assert len(enrich_dict) > 0

        enrich_seq = compute_group_enrichment(
            list(cluster_labels), list(group_assignments), feature_names=feature_names
        )
        assert len(enrich_seq) > 0

        # Errors
        with pytest.raises(ValueError, match="feature_names is required"):
            compute_group_enrichment(list(cluster_labels), group_assignments)
        with pytest.raises(ValueError, match="feature_names is required"):
            compute_group_enrichment(cluster_labels, list(group_assignments))
        with pytest.raises(ValueError, match="No overlapping feature names"):
            compute_group_enrichment(
                pd.Series(list(cluster_labels), index=["a", "b", "c", "d", "e", "f"]),
                group_assignments,
            )

    def test_compute_group_enrichment_methods_and_alternatives(self, sample_data):
        _, _, cluster_labels, group_assignments = sample_data

        for m in ["fisher", "chi2", "hypergeometric"]:
            for alt in ["greater", "less", "two-sided"]:
                res = compute_group_enrichment(
                    cluster_labels, group_assignments, method=m, alternative=alt
                )
                assert isinstance(res, pd.DataFrame)
                assert len(res) > 0
                assert "p_value" in res.columns
                assert "odds_ratio" in res.columns
                assert "fdr_q_value" in res.columns

        with pytest.raises(ValueError, match="method"):
            compute_group_enrichment(cluster_labels, group_assignments, method="invalid_test")
        with pytest.raises(ValueError, match="alternative"):
            compute_group_enrichment(cluster_labels, group_assignments, alternative="invalid_alt")

    def test_compute_group_enrichment_multi_column_fdr_scope(self, sample_data):
        _, _, cluster_labels, _ = sample_data
        cat_df = pd.DataFrame(
            {
                "feature": list(cluster_labels.index),
                "family": ["firstorder", "firstorder", "glcm", "glcm", "shape", "shape"],
                "category": ["A", "B", "A", "B", "A", "B"],
            }
        )

        res = compute_group_enrichment(cluster_labels, cat_df)
        assert isinstance(res, pd.DataFrame)
        assert "catalog_column" in res.columns
        assert set(res["catalog_column"].unique()) == {"family", "category"}

        # Check that columns argument filters columns
        res_fam = compute_group_enrichment(cluster_labels, cat_df, columns=["family"])
        assert set(res_fam["catalog_column"].unique()) == {"family"}

        with pytest.raises(ValueError, match="Columns not found"):
            compute_group_enrichment(cluster_labels, cat_df, columns=["nonexistent_column"])


class TestAccessiblePlotting:
    """Validate accessible visualization outputs."""

    def test_plot_hub_significance(self, sample_data):
        X, eigengenes, cluster_labels, _ = sample_data
        kme = compute_module_membership(X, eigengenes)
        # Dummy significance
        sig = pd.Series([0.9, 0.8, 0.1, 0.2, 0.3, 0.4], index=X.columns)

        fig = plot_hub_significance(
            kme, sig, cluster_labels, target_cluster="blue", top_n_labels=2, title="Hub plot"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Trigger fallback automatic title branch of Line 616
        fig_auto_title = plot_hub_significance(
            kme, sig, cluster_labels, target_cluster="blue", top_n_labels=2, title=None
        )
        assert isinstance(fig_auto_title, plt.Figure)
        plt.close(fig_auto_title)

        # An unknown module name raises (no silent fallback to the first column).
        with pytest.raises(ValueError, match="not a column"):
            plot_hub_significance(
                kme, sig, cluster_labels.to_dict(), target_cluster="unknown_column", top_n_labels=1
            )

        # File writing path trigger
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            out_file = os.path.join(tmpdir, "hub.png")
            fig_path = plot_hub_significance(
                kme, sig, cluster_labels, target_cluster="blue", path=out_file
            )
            assert os.path.exists(out_file)
            plt.close(fig_path)

    def test_plot_eigengene_profiles_categorical(self, sample_data):
        _, eigengenes, _, _ = sample_data
        # Categorical trait
        trait = pd.Series(["Grade I", "Grade I", "Grade II", "Grade II"] * 5)
        fig = plot_eigengene_profiles(eigengenes, trait, trait_name="Grade")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Trigger fallback automatic title branch of Line 600-620
        fig_no_title = plot_eigengene_profiles(eigengenes, trait, trait_name="Grade", title=None)
        assert isinstance(fig_no_title, plt.Figure)
        plt.close(fig_no_title)

        # Categorical array input without index mapping, and custom title with Path writing
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            out_file = os.path.join(tmpdir, "profiles.png")
            fig_arr = plot_eigengene_profiles(
                eigengenes.to_numpy(),
                trait.to_numpy(),
                trait_name="Grade",
                title="Group Profiles",
                path=out_file,
            )
            assert os.path.exists(out_file)
            plt.close(fig_arr)

    def test_plot_eigengene_profiles_continuous(self, sample_data):
        _, eigengenes, _, _ = sample_data
        # Continuous trait
        rng = np.random.default_rng(100)
        trait = pd.Series(rng.normal(size=20))
        fig = plot_eigengene_profiles(eigengenes, trait, trait_name="ContinuousTrait")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_batch_distributions(self, sample_data):
        X, _, _, _ = sample_data
        # Duplicate dataframe to represent 'after'
        X_after = X * 0.9
        # Batch annotations
        batch_ids = pd.Series(["Center_A", "Center_B"] * 10)

        fig = plot_batch_distributions(X, X_after, batch_ids, feature_name="feat_1")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Triggers hist fallback branches by having zero variance
        X_const = X.copy()
        X_const["feat_1"] = 1.2
        X_const_after = X_const * 1.0
        fig_const = plot_batch_distributions(
            X_const, X_const_after, batch_ids, feature_name="feat_1"
        )
        assert isinstance(fig_const, plt.Figure)
        plt.close(fig_const)

        # Non-DataFrame arrays, with path trigger
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            out_file = os.path.join(tmpdir, "batch.png")
            fig_arr = plot_batch_distributions(
                X.to_numpy(),
                X_after.to_numpy(),
                batch_ids.to_numpy(),
                feature_name=0,
                title="Batch comparison",
                path=out_file,
            )
            assert os.path.exists(out_file)
            plt.close(fig_arr)
            fig_arr = plot_batch_distributions(
                X.to_numpy(),
                X_after.to_numpy(),
                batch_ids.to_numpy(),
                feature_name=0,
                title="Batch comparison",
                path=out_file,
            )
            assert os.path.exists(out_file)
            plt.close(fig_arr)


class TestAnalysisCoverageGaps:
    """Targeted tests for remaining uncovered branches in analysis.py."""

    def test_identify_hub_features_edge_cases(self):
        from eigenradiomics.analysis import identify_hub_features

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(20, 3)), columns=["f1", "f2", "f3"])
        eigengenes = pd.DataFrame(rng.normal(size=(20, 1)), columns=["col_1"])

        # Cluster label c2 has features, but c3 has none in k_me (hits line 184)
        labels_sparse = pd.Series(["c1", "c2", "c3"], index=["f1", "f2", "f_other"])
        eigengenes = eigengenes.rename(columns={"col_1": "c1"})
        hubs_sparse = identify_hub_features(X, cluster_labels=labels_sparse, eigengenes=eigengenes)
        assert list(hubs_sparse["cluster"]) == ["c1"]

        # Clusters without their own eigengene column are skipped, not mapped by position.
        labels_many = pd.Series(["c1", "c2", "c3"], index=["f1", "f2", "f3"])
        with pytest.warns(UserWarning, match="2 cluster label"):
            hubs_many = identify_hub_features(X, cluster_labels=labels_many, eigengenes=eigengenes)
        assert list(hubs_many["cluster"]) == ["c1"]

        # Empty eigengenes (hits line 202)
        eigengenes_empty = pd.DataFrame(index=X.index)
        hubs_empty = identify_hub_features(
            X, cluster_labels=labels_many, eigengenes=eigengenes_empty
        )
        assert len(hubs_empty) == 0

    def test_single_column_enrichment_chi2_and_invalid(self):
        from eigenradiomics.analysis import _single_column_enrichment

        # Chi2 with zero row causing chi2_contingency to raise ValueError (hits lines 281-282)
        zero_cat = pd.Series(["g1", "g1", "g1"], index=["f1", "f2", "f3"])
        cluster_labels = pd.Series(["c1", "c1", "c1"], index=["f1", "f2", "f3"])
        res = _single_column_enrichment(
            cluster_labels, zero_cat, method="chi2", alternative="greater"
        )
        assert np.isnan(res["p_value"].iloc[0])

        # Hypergeometric alternative='less' (hits line 295)
        res_less = _single_column_enrichment(
            cluster_labels, zero_cat, method="hypergeometric", alternative="less"
        )
        assert not np.isnan(res_less["p_value"].iloc[0])

        # Invalid method in _single_column_enrichment (hits line 294)
        with pytest.raises(ValueError, match="method must be"):
            _single_column_enrichment(
                cluster_labels, zero_cat, method="unsupported_method", alternative="greater"
            )

    def test_compute_group_enrichment_catalog_and_methods(self):
        from eigenradiomics.analysis import compute_group_enrichment
        from eigenradiomics.catalog import FeatureCatalog

        cluster_labels = pd.Series(["m1", "m1", "m2", "m2"], index=["f1", "f2", "f3", "f4"])

        # 1. FeatureCatalog input
        catalog_df = pd.DataFrame(
            {
                "feature": ["f1", "f2", "f3", "f4"],
                "family": ["A", "A", "B", "B"],
            }
        )
        catalog = FeatureCatalog(catalog_df)
        res_cat = compute_group_enrichment(cluster_labels, catalog)
        assert isinstance(res_cat, pd.DataFrame)
        assert len(res_cat) > 0

        # 2. DataFrame without 'feature' column
        df_no_feat = pd.DataFrame(
            {
                "family": ["A", "A", "B", "B"],
            },
            index=["f1", "f2", "f3", "f4"],
        )
        res_no_feat = compute_group_enrichment(cluster_labels, df_no_feat)
        assert isinstance(res_no_feat, pd.DataFrame)
        assert len(res_no_feat) > 0

        # 3. Numeric-only catalog columns (fallback cols_to_run)
        df_numeric = pd.DataFrame(
            {
                "numeric_group": [1, 1, 2, 2],
            },
            index=["f1", "f2", "f3", "f4"],
        )
        res_numeric = compute_group_enrichment(cluster_labels, df_numeric)
        assert isinstance(res_numeric, pd.DataFrame)

        # 4. Hypergeometric two-sided alternative
        res_hyper_two = compute_group_enrichment(
            cluster_labels, df_no_feat, method="hypergeometric", alternative="two-sided"
        )
        assert isinstance(res_hyper_two, pd.DataFrame)

        # 5. Invalid method error
        with pytest.raises(ValueError, match="method must be"):
            compute_group_enrichment(cluster_labels, df_no_feat, method="invalid_method")

        # 6. All empty results fallback
        df_empty_vals = pd.DataFrame(
            {
                "family": [np.nan, np.nan, np.nan, np.nan],
            },
            index=["f1", "f2", "f3", "f4"],
        )
        res_empty = compute_group_enrichment(cluster_labels, df_empty_vals)
        assert isinstance(res_empty, pd.DataFrame)
        assert len(res_empty) == 0


def test_hubs_use_own_module_with_multiple_components():
    """Each module's hub uses its own first component, also for 2 components per module."""
    rng = np.random.default_rng(0)
    n = 80
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    X = pd.DataFrame(
        np.hstack(
            [
                f1[:, None] + 0.4 * rng.normal(size=(n, 20)),
                f2[:, None] + 0.4 * rng.normal(size=(n, 20)),
            ]
        ),
        columns=[f"a{i}" for i in range(20)] + [f"b{i}" for i in range(20)],
    )
    r = WGCNAReducer(soft_power=6, min_module_size=10, n_module_components=2).fit(X)
    labels = pd.Series(r.module_colors_, index=X.columns)
    hubs = identify_hub_features(X, labels, reducer=r, top_n=1, method="pearson")
    assert len(hubs) == len(r.module_names_) >= 2
    assert (hubs["k_ME"].abs() > 0.8).all()


def test_membership_rejects_unknown_method_and_aligns_eigengenes(sample_data):
    X, eigengenes, _, _ = sample_data
    with pytest.raises(ValueError, match="method must be"):
        compute_module_membership(X, eigengenes, method="pearsonn")
    ref = compute_module_membership(X, eigengenes, method="pearson")
    shuffled = eigengenes.sample(frac=1, random_state=2)
    pd.testing.assert_frame_equal(ref, compute_module_membership(X, shuffled, method="pearson"))
    with pytest.raises(ValueError, match="same unique index labels"):
        compute_module_membership(X, eigengenes.set_axis(range(1, 21)), method="pearson")


def test_chi2_enrichment_respects_alternative():
    labels = pd.Series(["m1"] * 20 + ["m2"] * 20, index=[f"f{i}" for i in range(40)])
    groups = pd.Series(["g1"] * 18 + ["g2"] * 2 + ["g1"] * 2 + ["g2"] * 18, index=labels.index)
    res = compute_group_enrichment(labels, groups, method="chi2", alternative="greater")
    less = compute_group_enrichment(labels, groups, method="chi2", alternative="less")
    two = compute_group_enrichment(labels, groups, method="chi2", alternative="two-sided")
    key = (res["cluster"] == "m1") & (res["group"] == "g1")
    assert res.loc[key, "p_value"].iloc[0] < 1e-4
    assert less.loc[key, "p_value"].iloc[0] > 0.99
    assert np.isclose(res.loc[key, "p_value"].iloc[0] * 2, two.loc[key, "p_value"].iloc[0])


def test_missing_group_label_is_not_background():
    idx = [f"f{i}" for i in range(12)]
    labels = pd.Series(["m1"] * 6 + ["m2"] * 6, index=idx)
    groups = pd.Series(["a"] * 4 + [None] * 2 + ["b"] * 6, index=idx)
    series_res = compute_group_enrichment(labels, groups)
    catalog_res = compute_group_enrichment(
        labels, pd.DataFrame({"feature": idx, "family": groups.to_numpy()}), columns="family"
    )
    assert (series_res["total_features"] == 10).all()
    assert np.allclose(
        series_res.sort_values(["cluster", "group"])["p_value"].to_numpy(),
        catalog_res.sort_values(["cluster", "group"])["p_value"].to_numpy(),
    )
