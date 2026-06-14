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
        columns=["feat_1", "feat_2", "feat_3", "feat_4", "feat_5", "feat_6"]
    )
    # 2 components
    eigengenes = pd.DataFrame(
        rng.normal(size=(20, 2)),
        columns=["blue", "brown"]
    )
    # Custom assignments
    cluster_labels = pd.Series(
        ["blue", "blue", "blue", "brown", "brown", "brown"],
        index=X.columns,
        name="module"
    )
    group_assignments = pd.Series(
        ["firstorder", "firstorder", "glcm", "glcm", "shape", "shape"],
        index=X.columns,
        name="family"
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
            soft_power=6, min_module_size=2, include_grey=True,
            n_module_components=2, verbose=0
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

        # Non-overlapping cluster labels (triggers specific falls - Line 199-200)
        hubs_fall = identify_hub_features(X, ["ghost1", "ghost2"] * 3, eigengenes)
        assert len(hubs_fall) > 0

        # Trigger line 185 (neither Series nor dict, sequence input)
        hubs_sequence_direct = identify_hub_features(
            X, [None, None, None, None, None, None], eigengenes
        )
        assert isinstance(hubs_sequence_direct, pd.DataFrame)

        # Explicit test triggering lines 185, 199-200, 207 directly
        # Passing a bare dictionary representing label sequence
        h_bare_dict = {f"feat_{i+1}": "blue" if i < 3 else "brown" for i in range(6)}
        hubs_dict_seq = identify_hub_features(X, h_bare_dict, eigengenes)
        assert len(hubs_dict_seq) == 2

        # Trigger line 185 - non-DataFrame NumPy array input for reducer-fitted assignments
        reducer_dummy = WGCNAReducer(
            soft_power=6, min_module_size=2, include_grey=True, verbose=0
        )
        reducer_dummy.fit(X)
        hubs_arr_fall = identify_hub_features(
            X.to_numpy(), list(cluster_labels), reducer=reducer_dummy
        )
        assert len(hubs_arr_fall) > 0

        # No column names whatsoever in a fallback scenario where model does not resolve names
        hubs_unlabelled = identify_hub_features(
            X.to_numpy(), list(cluster_labels), eigengenes.to_numpy()
        )
        assert len(hubs_unlabelled) > 0

        # Trigger fallback col_name matching for ghost labels (Line 199-200)
        hubs_ghost = identify_hub_features(X, ["ghost1", "ghost2"] * 3, eigengenes)
        assert len(hubs_ghost) > 0

        # Reducer and custom inputs. Set min_module_size lower to avoid empty module returns.
        reducer = WGCNAReducer(
            soft_power=6, min_module_size=2, include_grey=True, verbose=0
        )
        reducer.fit(X)
        hubs_red = identify_hub_features(X.to_numpy(), list(cluster_labels), reducer=reducer)
        assert len(hubs_red) > 0

    def test_compute_group_enrichment(self, sample_data):
        _, _, cluster_labels, group_assignments = sample_data
        enrichment = compute_group_enrichment(cluster_labels, group_assignments)
        assert isinstance(enrichment, pd.DataFrame)
        assert len(enrichment) > 0
        expected_cols = {
            "cluster", "group", "n_overlap", "cluster_size",
            "group_size", "total_features", "p_value", "odds_ratio", "fdr_q_value"
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
                group_assignments
            )


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

        # Mismatched name index coverage trigger
        fig_mismatch = plot_hub_significance(
            kme, sig, cluster_labels.to_dict(),
            target_cluster="unknown_column", top_n_labels=1
        )
        assert isinstance(fig_mismatch, plt.Figure)
        plt.close(fig_mismatch)

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
                eigengenes.to_numpy(), trait.to_numpy(),
                trait_name="Grade", title="Group Profiles", path=out_file
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
                X.to_numpy(), X_after.to_numpy(), batch_ids.to_numpy(),
                feature_name=0, title="Batch comparison", path=out_file
            )
            assert os.path.exists(out_file)
            plt.close(fig_arr)
            fig_arr = plot_batch_distributions(
                X.to_numpy(), X_after.to_numpy(), batch_ids.to_numpy(),
                feature_name=0, title="Batch comparison", path=out_file
            )
            assert os.path.exists(out_file)
            plt.close(fig_arr)
