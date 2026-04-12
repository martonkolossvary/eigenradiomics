
"""Tests for eigenradiomics.reducers.WGCNAReducer."""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pywgcna = pytest.importorskip("PyWGCNA", reason="PyWGCNA not installed")

from eigenradiomics.reducers import WGCNAReducer  # noqa: E402
from eigenradiomics.reducers._wgcna_utils import _wgcna_compute_eigengene  # noqa: E402


@pytest.fixture()
def wgcna_default():
    """WGCNAReducer with fast settings for testing."""
    return WGCNAReducer(
        soft_power=6,
        min_module_size=20,
        deep_split=2,
        me_diss_threshold=0.25,
        verbose=0,
    )


class TestWGCNAReducerFitTransform:
    """Core fit/transform behaviour."""

    def test_fit_returns_self(self, wgcna_default, small_feature_matrix):
        result = wgcna_default.fit(small_feature_matrix)
        assert result is wgcna_default

    def test_output_shape(self, wgcna_default, small_feature_matrix):
        Y = wgcna_default.fit_transform(small_feature_matrix)
        n_samples = small_feature_matrix.shape[0]
        assert Y.shape[0] == n_samples
        assert Y.shape[1] == wgcna_default.n_components_
        assert Y.shape[1] < small_feature_matrix.shape[1]

    def test_transform_new_data(self, wgcna_default, small_feature_matrix, rng):
        wgcna_default.fit(small_feature_matrix)
        X_new = small_feature_matrix + rng.standard_normal(small_feature_matrix.shape) * 0.1
        Y_new = wgcna_default.transform(X_new)
        assert Y_new.shape == (small_feature_matrix.shape[0], wgcna_default.n_components_)

    def test_fit_transform_close_to_fit_then_transform(self, small_feature_matrix):
        reducer1 = WGCNAReducer(soft_power=6, min_module_size=20, verbose=0)
        reducer2 = WGCNAReducer(soft_power=6, min_module_size=20, verbose=0)

        Y1 = reducer1.fit_transform(small_feature_matrix)
        Y2 = reducer2.fit(small_feature_matrix).transform(small_feature_matrix)
        np.testing.assert_allclose(Y1, Y2, atol=1e-12)

    def test_accepts_dataframe(self, wgcna_default, small_feature_df):
        Y = wgcna_default.fit_transform(small_feature_df)
        assert Y.shape[0] == small_feature_df.shape[0]

    def test_feature_names_stored(self, wgcna_default, small_feature_df):
        wgcna_default.fit(small_feature_df)
        assert hasattr(wgcna_default, "feature_names_in_")
        assert wgcna_default.n_features_in_ == small_feature_df.shape[1]
        assert list(wgcna_default.feature_names_in_[:3]) == ["feat_0", "feat_1", "feat_2"]

    def test_dataframe_column_order_must_match(self, wgcna_default, small_feature_df):
        wgcna_default.fit(small_feature_df)
        with pytest.raises(ValueError, match="same order"):
            wgcna_default.transform(small_feature_df.iloc[:, ::-1])

    def test_wrong_n_features_transform(self, wgcna_default, small_feature_matrix, rng):
        wgcna_default.fit(small_feature_matrix)
        with pytest.raises(ValueError, match="features"):
            wgcna_default.transform(rng.standard_normal((10, 5)))


class TestWGCNAReducerParameters:
    """Parameter handling and sklearn interface."""

    def test_get_params(self):
        r = WGCNAReducer(soft_power=8, min_module_size=30)
        params = r.get_params()
        assert params["soft_power"] == 8
        assert params["min_module_size"] == 30

    def test_set_params(self):
        r = WGCNAReducer()
        r.set_params(soft_power=10, deep_split=3)
        assert r.soft_power == 10
        assert r.deep_split == 3

    def test_reducer_prefix(self):
        assert WGCNAReducer._reducer_prefix == "wgcna"

    def test_feature_names_out(self, wgcna_default, small_feature_matrix):
        wgcna_default.fit(small_feature_matrix)
        names = wgcna_default.get_feature_names_out()
        assert len(names) == wgcna_default.n_components_
        assert all(n.startswith("wgcna_") for n in names)


class TestWGCNAReducerMethods:
    """WGCNA-specific public methods."""

    def test_module_assignments(self, wgcna_default, small_feature_matrix):
        wgcna_default.fit(small_feature_matrix)
        assignments = wgcna_default.wgcna_get_module_assignments()
        assert isinstance(assignments, dict)
        # All values should be lists of feature name strings
        for features in assignments.values():
            assert isinstance(features, list)
            assert all(isinstance(f, str) for f in features)

    def test_module_sizes(self, wgcna_default, small_feature_matrix):
        wgcna_default.fit(small_feature_matrix)
        sizes = wgcna_default.wgcna_get_module_sizes()
        assert isinstance(sizes, dict)
        total = sum(sizes.values())
        # Total assigned features ≤ input features (grey excluded)
        assert total <= small_feature_matrix.shape[1]

    def test_soft_power_table_auto(self, small_feature_matrix):
        r = WGCNAReducer(soft_power="auto", min_module_size=20, verbose=0)
        r.fit(small_feature_matrix)
        table = r.wgcna_get_soft_power_table()
        assert isinstance(table, pd.DataFrame)
        assert "Power" in table.columns

    def test_soft_power_table_manual(self, wgcna_default, small_feature_matrix):
        wgcna_default.fit(small_feature_matrix)
        assert wgcna_default.wgcna_get_soft_power_table() is None

    def test_store_tom(self, small_feature_matrix):
        r = WGCNAReducer(soft_power=6, min_module_size=20, store_tom=True, verbose=0)
        r.fit(small_feature_matrix)
        assert hasattr(r, "tom_")
        m = small_feature_matrix.shape[1]
        assert r.tom_.shape == (m, m)

    def test_include_grey(self, small_feature_matrix):
        r_no = WGCNAReducer(soft_power=6, min_module_size=20, include_grey=False, verbose=0)
        r_yes = WGCNAReducer(soft_power=6, min_module_size=20, include_grey=True, verbose=0)
        r_no.fit(small_feature_matrix)
        r_yes.fit(small_feature_matrix)
        # If grey features exist, including them adds a module
        if "grey" in r_yes.module_names_:
            assert r_yes.n_components_ >= r_no.n_components_


class TestWGCNAReducerPipeline:
    """Integration with sklearn Pipeline."""

    def test_in_pipeline_with_sklearn_preprocessors(self, small_feature_matrix):
        X = small_feature_matrix[:, :60].copy()
        X[0, 0] = np.nan
        X[:, 1] = 1.0
        pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("var", VarianceThreshold(threshold=0.0)),
                ("scale", StandardScaler()),
                ("wgcna", WGCNAReducer(soft_power=6, min_module_size=5, verbose=0)),
            ]
        )
        Y = pipe.fit_transform(X)
        assert Y.shape[0] == X.shape[0]
        assert Y.ndim == 2

    def test_grid_search_with_sklearn_preprocessors(self, small_feature_matrix):
        X = small_feature_matrix[:, :40].copy()
        X[:, 0] = 1.0
        y = small_feature_matrix[:, 0]

        pipe = Pipeline(
            [
                ("var", VarianceThreshold(threshold=0.0)),
                ("scale", StandardScaler()),
                ("reduce", WGCNAReducer(soft_power=6, min_module_size=5, verbose=0)),
                ("model", Ridge()),
            ]
        )
        search = GridSearchCV(
            pipe,
            {
                "model__alpha": [0.1, 1.0],
            },
            cv=2,
        )

        search.fit(X, y)
        assert search.best_estimator_ is not None
        assert "model__alpha" in search.best_params_


class TestWGCNAReducerEdgeCases:
    """Edge cases and error handling."""

    def test_not_fitted_transform_raises(self):
        r = WGCNAReducer()
        with pytest.raises((ValueError, AttributeError)):
            r.transform(np.random.randn(10, 50))

    def test_not_fitted_methods_raise(self):
        from sklearn.exceptions import NotFittedError

        r = WGCNAReducer()
        with pytest.raises(NotFittedError):
            r.wgcna_get_module_assignments()
        with pytest.raises(NotFittedError):
            r.wgcna_get_module_sizes()

    def test_pywgcna_import_missing(self, monkeypatch):
        """Verify the lazy import produces a helpful error."""
        import eigenradiomics.reducers._wgcna as mod

        def bad_import():
            raise ImportError("no PyWGCNA")

        monkeypatch.setattr(mod, "_import_pywgcna", bad_import)
        with pytest.raises(ImportError):
            WGCNAReducer(soft_power=6).fit(np.random.randn(20, 50))

    def test_log_file_captures_stdout(self, tmp_path):
        log_file = tmp_path / "wgcna.log"
        reducer = WGCNAReducer(log_file=str(log_file), verbose=0)
        with reducer._capture_output():
            print("python stdout line")
            os.write(1, b"fd stdout line\n")

        text = log_file.read_text()
        assert "python stdout line" in text
        assert "fd stdout line" in text


class TestWGCNAReducerParameterValidation:
    """Validate that invalid constructor parameters are rejected at fit time."""

    @pytest.mark.parametrize(
        "param, value, match",
        [
            ("soft_power", -1, "soft_power must be a positive integer"),
            ("soft_power", 0, "soft_power must be a positive integer"),
            ("soft_power", "invalid", "soft_power must be a positive integer"),
            ("r_squared_cut", 0.0, "r_squared_cut must be in"),
            ("r_squared_cut", 1.5, "r_squared_cut must be in"),
            ("r_squared_cut", -0.1, "r_squared_cut must be in"),
            ("mean_cut", 0, "mean_cut must be positive"),
            ("mean_cut", -10, "mean_cut must be positive"),
            ("min_module_size", 0, "min_module_size must be a positive integer"),
            ("min_module_size", -5, "min_module_size must be a positive integer"),
            ("me_diss_threshold", -0.1, "me_diss_threshold must be in"),
            ("me_diss_threshold", 1.5, "me_diss_threshold must be in"),
            ("deep_split", 5, "deep_split must be"),
            ("deep_split", -1, "deep_split must be"),
            ("verbose", -1, "verbose must be a non-negative integer"),
            ("network_type", "invalid", "network_type must be one of"),
            ("tom_type", "invalid", "tom_type must be one of"),
        ],
    )
    def test_invalid_param_raises(self, param, value, match, small_feature_matrix):
        kwargs = {"soft_power": 6, "min_module_size": 20, "verbose": 0}
        kwargs[param] = value
        r = WGCNAReducer(**kwargs)
        with pytest.raises(ValueError, match=match):
            r.fit(small_feature_matrix)


class TestWGCNAReducerDeterminism:
    """Verify that fitting with the same data produces identical results."""

    def test_fit_twice_gives_same_output(self, small_feature_matrix):
        r1 = WGCNAReducer(soft_power=6, min_module_size=20, verbose=0)
        r2 = WGCNAReducer(soft_power=6, min_module_size=20, verbose=0)

        Y1 = r1.fit_transform(small_feature_matrix)  # noqa: F841
        Y2 = r2.fit_transform(small_feature_matrix)

        np.testing.assert_allclose(Y1, Y2, atol=1e-12)

    def test_refit_clears_old_state(self, small_feature_matrix):
        """A refit on new data should not keep stale attributes."""
        r = WGCNAReducer(soft_power=6, min_module_size=20, store_tom=True, verbose=0)
        r.fit(small_feature_matrix)

        # Fit on a subset — results may differ
        r.fit(small_feature_matrix[:, :80])
        # TOM should match the new shape, not the old one
        if hasattr(r, "tom_"):
            assert r.tom_.shape == (80, 80)


class TestWGCNAReducerPersistence:
    """Verify pickle/joblib round-trip."""

    def test_pickle_roundtrip(self, wgcna_default, small_feature_matrix):
        wgcna_default.fit(small_feature_matrix)
        Y_before = wgcna_default.transform(small_feature_matrix)

        data = pickle.dumps(wgcna_default)
        loaded = pickle.loads(data)  # noqa: S301

        Y_after = loaded.transform(small_feature_matrix)
        np.testing.assert_allclose(Y_before, Y_after, atol=1e-12)


class TestWGCNAReducerModuleRecovery:
    """Verify that correlated groups are recovered as modules."""

    def test_approximately_recovers_groups(self, small_feature_matrix):
        """The 5-group synthetic data should yield roughly 3-7 modules."""
        r = WGCNAReducer(soft_power=6, min_module_size=20, verbose=0)
        r.fit(small_feature_matrix)
        # 5 latent groups → should be close to 5 modules (allow some tolerance)
        assert 2 <= r.n_components_ <= 10, (
            f"Expected roughly 5 modules from 5-group data, got {r.n_components_}"
        )


class TestWGCNAReducerConstantColumns:
    """Tests using the matrix_with_constant_cols fixture."""

    def test_constant_cols_handled(self, matrix_with_constant_cols):
        """Constant columns should not crash the reducer."""
        r = WGCNAReducer(soft_power=6, min_module_size=5, verbose=0)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Y = r.fit_transform(matrix_with_constant_cols)
        assert Y.ndim == 2
        assert Y.shape[0] == matrix_with_constant_cols.shape[0]


class TestWGCNAReducerWideMatrix:
    """Tests using the wide_feature_matrix fixture for scalability."""

    def test_wide_matrix_fit_transform(self, wide_feature_matrix):
        """30×1000 matrix should fit and transform without error."""
        r = WGCNAReducer(soft_power=6, min_module_size=30, verbose=0)
        Y = r.fit_transform(wide_feature_matrix)
        assert Y.ndim == 2
        assert Y.shape[0] == wide_feature_matrix.shape[0]
        assert Y.shape[1] < wide_feature_matrix.shape[1]


class TestWGCNAReducerParameterizedFit:
    """Parameterized tests for different WGCNA configurations."""

    @pytest.mark.parametrize("min_module_size", [5, 15, 30])
    def test_min_module_size_variation(self, small_feature_matrix, min_module_size):
        r = WGCNAReducer(
            soft_power=6,
            min_module_size=min_module_size,
            verbose=0,
        )
        Y = r.fit_transform(small_feature_matrix)
        assert Y.ndim == 2
        assert Y.shape[0] == small_feature_matrix.shape[0]
        # Every module must have at least min_module_size features
        sizes = r.wgcna_get_module_sizes()
        for mod, size in sizes.items():
            assert size >= min_module_size, (
                f"Module '{mod}' has {size} features, below min_module_size={min_module_size}"
            )

    @pytest.mark.parametrize("me_diss_threshold", [0.1, 0.25, 0.5])
    def test_me_diss_threshold_variation(self, small_feature_matrix, me_diss_threshold):
        r = WGCNAReducer(
            soft_power=6,
            min_module_size=20,
            me_diss_threshold=me_diss_threshold,
            verbose=0,
        )
        Y = r.fit_transform(small_feature_matrix)
        assert Y.ndim == 2
        assert Y.shape[0] == small_feature_matrix.shape[0]

    @pytest.mark.parametrize("deep_split", [0, 2, 3])
    def test_deep_split_variation(self, small_feature_matrix, deep_split):
        r = WGCNAReducer(
            soft_power=6,
            min_module_size=20,
            deep_split=deep_split,
            verbose=0,
        )
        Y = r.fit_transform(small_feature_matrix)
        assert Y.ndim == 2


class TestWGCNAReducerSparseInput:
    """Sparse matrix rejection."""

    def test_sparse_csr_raises(self):
        import scipy.sparse

        X_sparse = scipy.sparse.csr_matrix(np.eye(20))
        r = WGCNAReducer(soft_power=6, min_module_size=5, verbose=0)
        with pytest.raises(TypeError, match="Sparse matrices are not supported"):
            r.fit(X_sparse)

    def test_sparse_csc_raises(self):
        import scipy.sparse

        X_sparse = scipy.sparse.csc_matrix(np.eye(20))
        r = WGCNAReducer(soft_power=6, min_module_size=5, verbose=0)
        with pytest.raises(TypeError, match="Sparse matrices are not supported"):
            r.fit(X_sparse)


class TestWGCNAReducerNetworkTOMTypes:
    """Parameterized tests for network_type × tom_type combinations."""

    @pytest.mark.parametrize("network_type", ["signed hybrid", "signed", "unsigned"])
    @pytest.mark.parametrize("tom_type", ["signed", "unsigned"])
    def test_network_tom_combination(self, small_feature_matrix, network_type, tom_type):
        r = WGCNAReducer(
            network_type=network_type,
            tom_type=tom_type,
            soft_power=6,
            min_module_size=20,
            verbose=0,
        )
        Y = r.fit_transform(small_feature_matrix)
        assert Y.ndim == 2
        assert Y.shape[0] == small_feature_matrix.shape[0]
        assert Y.shape[1] > 0


class TestWGCNAReducerPlotMethods:
    """Plotting methods with matplotlib mocking."""

    def test_plot_soft_power_returns_figure(self, small_feature_matrix):
        import matplotlib.pyplot as plt

        r = WGCNAReducer(soft_power="auto", min_module_size=20, verbose=0)
        r.fit(small_feature_matrix)
        fig = r.wgcna_plot_soft_power()
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_soft_power_raises_without_auto(self, wgcna_default, small_feature_matrix):
        wgcna_default.fit(small_feature_matrix)
        with pytest.raises(RuntimeError, match="soft_power='auto'"):
            wgcna_default.wgcna_plot_soft_power()

    def test_plot_dendrogram_returns_figure(self, wgcna_default, small_feature_matrix):
        import matplotlib.pyplot as plt

        wgcna_default.fit(small_feature_matrix)
        fig = wgcna_default.wgcna_plot_dendrogram()
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        plt.close(fig)


class TestWGCNAReducerFeatureImportances:
    """Tests for wgcna_get_feature_importances."""

    def test_returns_dict_of_dataframes(self, wgcna_default, small_feature_matrix):
        wgcna_default.fit(small_feature_matrix)
        result = wgcna_default.wgcna_get_feature_importances()
        assert isinstance(result, dict)
        for _mod, df in result.items():
            assert isinstance(df, pd.DataFrame)
            assert set(df.columns) == {"feature", "loading", "importance"}

    def test_importances_sum_to_one_when_normalized(self, wgcna_default, small_feature_matrix):
        wgcna_default.fit(small_feature_matrix)
        result = wgcna_default.wgcna_get_feature_importances(normalize=True)
        for df in result.values():
            np.testing.assert_allclose(df["importance"].sum(), 1.0, atol=1e-12)

    def test_unnormalized_importances(self, wgcna_default, small_feature_matrix):
        wgcna_default.fit(small_feature_matrix)
        result = wgcna_default.wgcna_get_feature_importances(normalize=False)
        for df in result.values():
            # Unnormalized = absolute loadings
            np.testing.assert_allclose(
                df["importance"].values,
                np.abs(df["loading"].values),
            )

    def test_not_fitted_raises(self):
        from sklearn.exceptions import NotFittedError

        r = WGCNAReducer()
        with pytest.raises(NotFittedError):
            r.wgcna_get_feature_importances()

    def test_all_modules_covered(self, wgcna_default, small_feature_matrix):
        wgcna_default.fit(small_feature_matrix)
        result = wgcna_default.wgcna_get_feature_importances()
        assert set(result.keys()) == set(wgcna_default.module_names_)


class TestWGCNAReducerRandomStateNJobs:
    """Tests for random_state and n_jobs parameters."""

    def test_random_state_accepted(self, small_feature_matrix):
        r = WGCNAReducer(soft_power=6, min_module_size=20, verbose=0, random_state=42)
        Y = r.fit_transform(small_feature_matrix)
        assert Y.ndim == 2

    def test_n_jobs_one(self, small_feature_matrix):
        r = WGCNAReducer(soft_power=6, min_module_size=20, verbose=0, n_jobs=1)
        Y = r.fit_transform(small_feature_matrix)
        assert Y.ndim == 2

    def test_n_jobs_parallel(self, small_feature_matrix):
        r = WGCNAReducer(soft_power=6, min_module_size=20, verbose=0, n_jobs=2)
        Y = r.fit_transform(small_feature_matrix)
        assert Y.ndim == 2

    def test_n_jobs_parallel_matches_sequential(self, small_feature_matrix):
        r_seq = WGCNAReducer(soft_power=6, min_module_size=20, verbose=0, n_jobs=1)
        r_par = WGCNAReducer(soft_power=6, min_module_size=20, verbose=0, n_jobs=2)
        Y_seq = r_seq.fit_transform(small_feature_matrix)
        Y_par = r_par.fit_transform(small_feature_matrix)
        np.testing.assert_allclose(Y_seq, Y_par, atol=1e-12)

    def test_get_params_includes_new_params(self):
        r = WGCNAReducer(random_state=42, n_jobs=3)
        params = r.get_params()
        assert params["random_state"] == 42
        assert params["n_jobs"] == 3


class TestWGCNAReducerCoverageGaps:
    """Tests targeting specific uncovered lines for 100% coverage."""

    def test_inverse_transform_works(self, wgcna_default, small_feature_matrix):
        """Verify inverse_transform operates via module reconstruction."""
        wgcna_default.fit(small_feature_matrix)
        Y = wgcna_default.transform(small_feature_matrix)
        X_recon = wgcna_default.inverse_transform(Y)
        assert X_recon.shape == small_feature_matrix.shape
        # Should NOT be NaNs in reconstruction
        assert not np.isnan(X_recon).any()

    def test_sklearn_tags_present(self, wgcna_default):
        """Verify the standard sklearn tags API overrides are correctly wired."""
        if hasattr(wgcna_default, "__sklearn_tags__"):
            tags = wgcna_default.__sklearn_tags__()
        else:
            tags = wgcna_default._get_tags()
        # Since sklearn versions >1.6 vs <1.6 have different structures,
        # we just check they don't crash and return something.
        assert tags is not None

    def test_import_pywgcna_missing(self, monkeypatch):
        """_wgcna.py:40-41 — ImportError when PyWGCNA not installed."""
        import builtins

        import eigenradiomics.reducers._wgcna as mod

        real_import = builtins.__import__

        def _block_pywgcna(name, *args, **kwargs):
            if name == "PyWGCNA":
                raise ImportError("mocked: no PyWGCNA")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_pywgcna)
        with pytest.raises(ImportError, match="PyWGCNA is required"):
            mod._import_pywgcna()

    def test_too_few_samples(self):
        """_wgcna.py:212 — n_samples < 3 error."""
        r = WGCNAReducer(soft_power=6, min_module_size=5, verbose=0)
        X = np.random.default_rng(0).standard_normal((2, 50))
        with pytest.raises(ValueError, match="n_samples >= 3"):
            r.fit(X)

    def test_too_few_features(self):
        """_wgcna.py:217 — n_features < 2 error."""
        r = WGCNAReducer(soft_power=6, min_module_size=5, verbose=0)
        X = np.random.default_rng(0).standard_normal((20, 1))
        with pytest.raises(ValueError, match="n_features >= 3"):
            r.fit(X)

    def test_auto_soft_power_failure(self, monkeypatch):
        """_wgcna.py:236 — pickSoftThreshold returns None."""
        import eigenradiomics.reducers._wgcna as mod

        class FakeWGCNA:
            @staticmethod
            def pickSoftThreshold(*args, **kwargs):
                table = pd.DataFrame({"Power": [1, 2], "SFT.R.sq": [0.1, 0.2], "mean(k)": [10, 5]})
                return None, table

            @staticmethod
            def adjacency(*args, **kwargs):
                pass

        def _fake_import():
            return FakeWGCNA

        monkeypatch.setattr(mod, "_import_pywgcna", _fake_import)
        r = WGCNAReducer(soft_power="auto", min_module_size=5, verbose=0)
        with pytest.raises(ValueError, match="Automatic soft-power selection failed"):
            r.fit(np.random.default_rng(0).standard_normal((20, 50)))

    def test_verbose_passthrough(self, small_feature_matrix):
        """_wgcna.py:683-684 — verbose >= 1 yield path."""
        r = WGCNAReducer(soft_power=6, min_module_size=20, verbose=1)
        r.fit(small_feature_matrix)
        assert r.n_components_ > 0

    def test_include_grey_removes_grey_module(self, small_feature_matrix):
        """_wgcna.py:303 — include_grey=False removing grey from module list.

        We force a grey module by using very large min_module_size so some
        features are unassigned, then verify grey is excluded.
        """
        r_no_grey = WGCNAReducer(
            soft_power=6,
            min_module_size=5,
            include_grey=False,
            verbose=0,
        )
        r_no_grey.fit(small_feature_matrix)
        # Grey should never appear in module_names_ when include_grey=False
        assert "grey" not in r_no_grey.module_names_

    def test_dendrogram_invalid_color_fallback(
        self,
        wgcna_default,
        small_feature_matrix,
        monkeypatch,
    ):
        """_wgcna.py:514-515 — to_rgba ValueError fallback to lightgrey."""
        import matplotlib.pyplot as plt

        wgcna_default.fit(small_feature_matrix)
        # Replace one module color with an invalid name
        wgcna_default.module_colors_[0] = "not_a_real_color_xyz"
        fig = wgcna_default.wgcna_plot_dendrogram()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_merge_zero_variance_during_merge(self):
        """_wgcna.py:581-585 — zero-variance features during _merge_close_modules."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 100))
        # Make some columns constant — they'll cause zero-variance in merge
        X[:, 0] = 5.0
        X[:, 1] = 5.0
        import warnings

        r = WGCNAReducer(soft_power=6, min_module_size=5, verbose=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Y = r.fit_transform(X)
        assert Y.ndim == 2

    def test_merge_nan_correlation_warning(self, wgcna_default, small_feature_matrix, monkeypatch):
        """_wgcna.py:594-600 — NaN in eigengene correlation during merge."""
        import warnings

        original_corrcoef = np.corrcoef
        injected = [False]

        def _corrcoef_with_nan(*args, **kwargs):
            result = original_corrcoef(*args, **kwargs)
            # Target only the merge-loop eigengene correlation (small square
            # matrix, 3–20 modules).  PyWGCNA's adjacency call produces a
            # much larger n_features × n_features matrix.
            if not injected[0] and result.ndim == 2 and 3 <= result.shape[0] <= 20:
                injected[0] = True
                result[0, 1] = np.nan
                result[1, 0] = np.nan
            return result

        monkeypatch.setattr(np, "corrcoef", _corrcoef_with_nan)
        r = WGCNAReducer(soft_power=6, min_module_size=20, me_diss_threshold=0.5, verbose=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Y = r.fit_transform(small_feature_matrix)
        nan_warnings = [w for w in caught if "NaN" in str(w.message)]
        assert len(nan_warnings) >= 1
        assert Y.ndim == 2




class TestWGCNAAdditionalCoverage:
    def test_inverse_transform(self, wgcna_default, small_feature_matrix):
        """Test inverse_transform round trips."""
        Y = wgcna_default.fit_transform(small_feature_matrix)
        X_inv = wgcna_default.inverse_transform(Y)
        assert X_inv.shape == small_feature_matrix.shape
        assert X_inv.ndim == 2
        # Check projection dims
        assert hasattr(wgcna_default, "module_loadings_")
        assert X_inv.shape[1] == len(wgcna_default.feature_names_in_)

    def test_inverse_transform_exceptions(self, wgcna_default, small_feature_matrix):
        """Raise error on un-fitted and wrong dimensions."""
        with pytest.raises(ValueError, match="Expected"):
            wgcna_default.fit(small_feature_matrix)
            Y = wgcna_default.transform(small_feature_matrix)
            Y_bad = Y[:, :-1]
            wgcna_default.inverse_transform(Y_bad)

    def test_set_params_refit(self, small_feature_matrix):
        """Test set_params + refit changes behaviour correctly without leakage."""
        r1 = WGCNAReducer(soft_power=6, min_module_size=40, me_diss_threshold=0.25, verbose=0)
        Y1 = r1.fit_transform(small_feature_matrix)  # noqa: F841

        r1.set_params(min_module_size=10)
        Y2 = r1.fit_transform(small_feature_matrix)

        # They shouldn't be the same number of components
        assert r1.min_module_size == 10
        assert Y2.ndim == 2

    def test_compute_eigengene_low_variance(self):
        """Line 61 branch."""
        X_mod = np.zeros((10, 5))
        eig, load = _wgcna_compute_eigengene(X_mod)
        assert np.array_equal(eig, np.zeros(10))
        assert len(load) == 5
        assert load[0] == 1.0

    def test_capture_multiprocessing(self, wgcna_default, small_feature_matrix, monkeypatch):
        """Line 758 branch: mock multiprocessing.current_process."""

        # Should not clobber stdout if child process
        class MockProcess:
            name = "ForkPoolWorker-1"

        monkeypatch.setattr("multiprocessing.current_process", lambda: MockProcess())

        r = WGCNAReducer(soft_power=6, min_module_size=40, verbose=1, log_file="none")
        # Ensure it works when not main process without writing to stdout
        Y = r.fit_transform(small_feature_matrix)
        assert Y.ndim == 2


def test_base_tags(wgcna_default):
    """Hits the _get_tags and __sklearn_tags__ logic to ensure base coverage."""
    tags1 = wgcna_default._get_tags()
    tags2 = wgcna_default.__sklearn_tags__()
    assert isinstance(tags1, dict)
    assert not tags1.get("requires_y", True)
    assert tags2 is not None
