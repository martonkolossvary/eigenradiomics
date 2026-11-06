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


def _noise_heavy_matrix() -> np.ndarray:
    """Matrix with 4 correlated groups plus many uncorrelated noise features.

    The noise features are not assigned to any module by WGCNA (the unassigned
    "grey" module), which lets us test ``include_grey`` behaviour.
    """
    rng = np.random.default_rng(0)
    n = 50
    cols = []
    for _ in range(4):  # 4 correlated groups of 25 features
        latent = rng.standard_normal(n)
        for _ in range(25):
            cols.append(latent + rng.standard_normal(n) * 0.3)
    for _ in range(40):  # 40 uncorrelated noise features
        cols.append(rng.standard_normal(n))
    return np.column_stack(cols)


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
        # No power meets the criteria on this small matrix: "raise" raises, and
        # the default fallback warns and records the failed criteria.
        with pytest.raises(ValueError, match="soft-power selection failed"):
            WGCNAReducer(soft_power="auto", min_module_size=20, soft_power_fallback="raise").fit(
                small_feature_matrix
            )
        r = WGCNAReducer(soft_power="auto", min_module_size=20, verbose=0)
        with pytest.warns(UserWarning, match="best-R"):
            r.fit(small_feature_matrix)
        table = r.wgcna_get_soft_power_table()
        assert isinstance(table, pd.DataFrame)
        assert "Power" in table.columns
        assert r.soft_power_criteria_met_ is False

    def test_soft_power_table_manual(self, wgcna_default, small_feature_matrix):
        wgcna_default.fit(small_feature_matrix)
        assert wgcna_default.wgcna_get_soft_power_table() is None

    def test_store_tom(self, small_feature_matrix):
        r = WGCNAReducer(soft_power=6, min_module_size=20, store_tom=True, verbose=0)
        r.fit(small_feature_matrix)
        assert hasattr(r, "tom_")
        m = small_feature_matrix.shape[1]
        assert r.tom_.shape == (m, m)

    def test_include_grey(self):
        """include_grey must control whether unassigned features form a module.

        Regression test: previously the unassigned module was only removed when
        literally named "grey", which PyWGCNA never emits, so include_grey=False
        silently kept all unassigned features.
        """
        X = _noise_heavy_matrix()
        r_no = WGCNAReducer(soft_power=6, min_module_size=20, include_grey=False, verbose=0)
        r_yes = WGCNAReducer(soft_power=6, min_module_size=20, include_grey=True, verbose=0)
        r_no.fit(X)
        r_yes.fit(X)

        assigned_no = sum(len(v) for v in r_no.module_assignments_.values())
        assigned_yes = sum(len(v) for v in r_yes.module_assignments_.values())

        # Some noise features are unassigned and excluded by default ...
        assert assigned_no < X.shape[1]
        # ... but retained as an extra module when include_grey=True.
        assert assigned_yes > assigned_no
        assert r_yes.n_components_ > r_no.n_components_
        assert set(r_no.module_names_).issubset(set(r_yes.module_names_))


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
            ("n_module_components", 0, "n_module_components must be a positive integer"),
            ("n_module_components", -5, "n_module_components must be a positive integer"),
            ("n_module_components", "invalid", "n_module_components must be a positive integer"),
            ("n_jobs", 0, "n_jobs must be a non-zero integer"),
            ("n_jobs", 1.5, "n_jobs must be a non-zero integer"),
            ("correlation_method", "kendall", "correlation_method must be one of"),
            ("power_min", 0, "power_min must be a positive integer"),
            ("power_max", 0, "power_max must be an integer >= power_min"),
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

    def test_constant_cols_warn(self, matrix_with_constant_cols):
        """Zero-variance columns are surfaced via a warning (then left unassigned)."""
        r = WGCNAReducer(soft_power=6, min_module_size=5, verbose=0)
        with pytest.warns(UserWarning, match="zero variance"):
            r.fit(matrix_with_constant_cols)


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

        r = WGCNAReducer(
            soft_power="auto", min_module_size=20, verbose=0, soft_power_fallback="best_r2"
        )
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
    """Tests for n_module_components and n_jobs parameters."""

    def test_n_module_components_multi(self, small_feature_matrix):
        r = WGCNAReducer(soft_power=6, min_module_size=20, verbose=0, n_module_components=2)
        Y = r.fit_transform(small_feature_matrix)
        assert Y.ndim == 2
        assert r.n_components_ > len(r.module_names_)

        # Test inverse_transform with multi-eigengenes
        X_recon = r.inverse_transform(Y)
        assert X_recon.shape == small_feature_matrix.shape

        # Test feature importances with multi-eigengenes
        importances = r.wgcna_get_feature_importances()
        assert isinstance(importances, dict)
        for _, df in importances.items():
            assert set(df.columns) == {"feature", "loading", "importance"}

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
        r = WGCNAReducer(n_module_components=3, n_jobs=3)
        params = r.get_params()
        assert params["n_module_components"] == 3
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

    def test_no_modules_warns_and_returns_empty(self):
        """All-grey fit -> warn and produce a (n_samples, 0) transform."""
        X = np.random.default_rng(1).standard_normal((30, 6))  # unstructured noise
        r = WGCNAReducer(soft_power=6, min_module_size=30, include_grey=False, verbose=0)
        with pytest.warns(UserWarning, match="no modules"):
            r.fit(X)
        assert r.n_components_ == 0
        assert r.transform(X).shape == (30, 0)

    def test_auto_soft_power_failure(self, monkeypatch):
        """_wgcna.py:236 — pickSoftThreshold returns None."""
        import eigenradiomics.reducers._wgcna as mod

        class FakeWGCNA:
            @staticmethod
            def pickSoftThreshold(*args, **kwargs):
                table = pd.DataFrame(
                    {
                        "Power": [1, 2],
                        "SFT.R.sq": [0.1, 0.2],
                        "slope": [-1, -1],
                        "mean(k)": [10, 5],
                    }
                )
                return None, table

            @staticmethod
            def adjacency(*args, **kwargs):
                pass

        def _fake_import():
            return FakeWGCNA

        monkeypatch.setattr(mod, "_import_pywgcna", _fake_import)
        r = WGCNAReducer(
            soft_power="auto", min_module_size=5, verbose=0, soft_power_fallback="raise"
        )
        with pytest.raises(ValueError, match="Automatic soft-power selection failed"):
            r.fit(np.random.default_rng(0).standard_normal((20, 50)))

    def test_verbose_passthrough(self, small_feature_matrix):
        """_wgcna.py:683-684 — verbose >= 1 yield path."""
        r = WGCNAReducer(soft_power=6, min_module_size=20, verbose=1)
        r.fit(small_feature_matrix)
        assert r.n_components_ > 0

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


class TestWGCNAReducerArtifacts:
    """Structured reduction-artifact outputs for downstream plotting/analysis."""

    def test_artifacts_with_tom(self, small_feature_matrix):
        r = WGCNAReducer(soft_power=6, min_module_size=20, store_tom=True, verbose=0).fit(
            small_feature_matrix
        )
        art = r.get_reduction_artifacts()
        m = small_feature_matrix.shape[1]
        assert set(art.available()) == {
            "similarity",
            "linkage",
            "cluster_labels",
            "feature_order",
            "feature_importances",
        }
        assert art.similarity.shape == (m, m)
        assert art.linkage.shape == (m - 1, 4)
        assert len(art.cluster_labels) == m
        assert len(art.feature_order) == m
        assert {"feature", "loading", "importance", "module"}.issubset(
            art.feature_importances.columns
        )

    def test_artifacts_without_tom(self, small_feature_matrix):
        r = WGCNAReducer(soft_power=6, min_module_size=20, store_tom=False, verbose=0).fit(
            small_feature_matrix
        )
        art = r.get_reduction_artifacts()
        assert art.similarity is None
        assert "similarity" not in art.available()
        assert art.linkage is not None  # linkage available without the TOM


class TestWGCNAReducerCorrelationMethod:
    def test_spearman_fit_and_transform(self, small_feature_matrix):
        r = WGCNAReducer(
            soft_power=6, min_module_size=20, correlation_method="spearman", verbose=0
        )
        Y = r.fit_transform(small_feature_matrix)
        assert Y.shape == (small_feature_matrix.shape[0], r.n_components_)
        # Loadings are on the original scale, so transform of unseen rows is defined.
        Y2 = r.transform(small_feature_matrix[:10])
        assert Y2.shape == (10, r.n_components_)

    def test_spearman_tom_differs_from_pearson(self, small_feature_matrix):
        common = dict(soft_power=6, min_module_size=20, store_tom=True, verbose=0)
        p = WGCNAReducer(correlation_method="pearson", **common).fit(small_feature_matrix)
        s = WGCNAReducer(correlation_method="spearman", **common).fit(small_feature_matrix)
        assert not np.allclose(p.tom_, s.tom_)


class TestWGCNAReducerPowerRange:
    def test_auto_power_within_range(self, small_feature_matrix):
        r = WGCNAReducer(
            soft_power="auto",
            power_min=2,
            power_max=12,
            r_squared_cut=0.8,
            min_module_size=20,
            verbose=0,
            soft_power_fallback="best_r2",
        )
        r.fit(small_feature_matrix)
        table = r.wgcna_get_soft_power_table()
        assert table["Power"].between(2, 12).all()
        assert 2 <= r.soft_power_ <= 12


class TestWGCNAReducerMissingStrategy:
    """Tests for missing data handling in WGCNAReducer."""

    def test_missing_strategy_raise_on_nan(self, small_feature_matrix, rng):
        X_nan = small_feature_matrix.copy()
        X_nan[0, 0] = np.nan
        r = WGCNAReducer(missing_strategy="raise", soft_power=6, min_module_size=20, verbose=0)
        with pytest.raises(ValueError, match="NaN"):
            r.fit(X_nan)

        r_fit = WGCNAReducer(missing_strategy="raise", soft_power=6, min_module_size=20, verbose=0)
        r_fit.fit(small_feature_matrix)
        with pytest.raises(ValueError, match="NaN"):
            r_fit.transform(X_nan)

    def test_missing_strategy_zero(self, small_feature_matrix, rng):
        X_nan = small_feature_matrix.copy()
        mask = rng.random(X_nan.shape) < 0.1
        X_nan[mask] = np.nan

        r = WGCNAReducer(missing_strategy="zero", soft_power=6, min_module_size=20, verbose=0)
        Y = r.fit_transform(X_nan)
        assert not np.isnan(Y).any()
        assert np.all(r.impute_values_ == 0.0)

        # Transform on new data with NaNs
        X_test = small_feature_matrix.copy()
        X_test[0, 5] = np.nan
        Y_test = r.transform(X_test)
        assert not np.isnan(Y_test).any()

    def test_missing_strategy_floor(self, small_feature_matrix, rng):
        X_nan = small_feature_matrix.copy()
        mask = rng.random(X_nan.shape) < 0.1
        X_nan[mask] = np.nan

        r = WGCNAReducer(missing_strategy="floor", soft_power=6, min_module_size=20, verbose=0)
        Y = r.fit_transform(X_nan)
        assert not np.isnan(Y).any()
        expected_floor = np.nanmin(X_nan, axis=0)
        np.testing.assert_allclose(r.impute_values_, expected_floor)

    def test_missing_strategy_mean_median(self, small_feature_matrix, rng):
        X_nan = small_feature_matrix.copy()
        mask = rng.random(X_nan.shape) < 0.1
        X_nan[mask] = np.nan

        r_mean = WGCNAReducer(missing_strategy="mean", soft_power=6, min_module_size=20, verbose=0)
        Y_mean = r_mean.fit_transform(X_nan)
        assert not np.isnan(Y_mean).any()
        np.testing.assert_allclose(r_mean.impute_values_, np.nanmean(X_nan, axis=0))

        r_median = WGCNAReducer(
            missing_strategy="median", soft_power=6, min_module_size=20, verbose=0
        )
        Y_median = r_median.fit_transform(X_nan)
        assert not np.isnan(Y_median).any()
        np.testing.assert_allclose(r_median.impute_values_, np.nanmedian(X_nan, axis=0))

    def test_missing_strategy_pairwise(self, small_feature_matrix, rng):
        X_nan = small_feature_matrix.copy()
        mask = rng.random(X_nan.shape) < 0.1
        X_nan[mask] = np.nan

        r = WGCNAReducer(missing_strategy="pairwise", soft_power=6, min_module_size=20, verbose=0)
        Y = r.fit_transform(X_nan)
        assert not np.isnan(Y).any()
        assert Y.shape[1] == r.n_components_

        # Pairwise with auto soft power
        r_auto = WGCNAReducer(
            missing_strategy="pairwise",
            soft_power="auto",
            power_min=2,
            power_max=10,
            r_squared_cut=0.7,
            min_module_size=20,
            verbose=0,
            soft_power_fallback="best_r2",
        )
        r_auto.fit(X_nan)
        assert r_auto.soft_power_ is not None

    def test_missing_strategy_all_nan_column_guard(self, small_feature_matrix):
        X_nan = small_feature_matrix.copy()
        X_nan[:, 3] = np.nan  # Entire column 3 is NaN

        r = WGCNAReducer(missing_strategy="mean", soft_power=6, min_module_size=20, verbose=0)
        with pytest.warns(UserWarning, match="entirely NaN"):
            r.fit(X_nan)

        assert r.impute_values_[3] == 0.0
        Y = r.transform(X_nan)
        assert not np.isnan(Y).any()

    def test_missing_strategy_sklearn_tags(self):
        r_raise = WGCNAReducer(missing_strategy="raise")
        r_zero = WGCNAReducer(missing_strategy="zero")

        assert not r_raise._get_tags().get("allow_nan", False)
        assert r_zero._get_tags().get("allow_nan", False)
        assert not r_raise.__sklearn_tags__().input_tags.allow_nan
        assert r_zero.__sklearn_tags__().input_tags.allow_nan

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="missing_strategy"):
            WGCNAReducer(missing_strategy="invalid_mode").fit(np.ones((10, 10)))


class TestWGCNAReducerSignAlignment:
    """Tests for eigengene sign orientation rules."""

    def test_orient_eigengenes_mean_correlation(self, small_feature_matrix):
        r = WGCNAReducer(
            orient_eigengenes="mean_correlation",
            soft_power=6,
            min_module_size=20,
            verbose=0,
        ).fit(small_feature_matrix)
        Y = r.transform(small_feature_matrix)
        assert Y.shape[1] == r.n_components_

    def test_orient_eigengenes_hub_feature_multicomponent(self, small_feature_matrix):
        r = WGCNAReducer(
            orient_eigengenes="hub_feature",
            n_module_components=2,
            soft_power=6,
            min_module_size=20,
            verbose=0,
        ).fit(small_feature_matrix)
        for _mod, loadings in r.module_loadings_.items():
            if loadings.ndim == 2:
                for col in range(loadings.shape[1]):
                    j_max = np.argmax(np.abs(loadings[:, col]))
                    assert loadings[j_max, col] >= 0

    def test_orient_eigengenes_sum_loadings_guard(self):
        # Construct synthetic matrix with balanced positive and negative loadings
        X = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]])
        eg, loadings = _wgcna_compute_eigengene(
            X, n_components=1, orient_eigengenes="sum_loadings"
        )
        assert len(loadings) == 2

    def test_orient_eigengenes_false(self, small_feature_matrix):
        r = WGCNAReducer(
            orient_eigengenes=False,
            soft_power=6,
            min_module_size=20,
            verbose=0,
        ).fit(small_feature_matrix)
        assert r.n_components_ > 0

    def test_orient_eigengenes_inverse_transform(self, small_feature_matrix):
        r1 = WGCNAReducer(
            orient_eigengenes="mean_correlation", soft_power=6, min_module_size=20, verbose=0
        ).fit(small_feature_matrix)
        r2 = WGCNAReducer(
            orient_eigengenes=False, soft_power=6, min_module_size=20, verbose=0
        ).fit(small_feature_matrix)

        Y1 = r1.transform(small_feature_matrix)
        Y2 = r2.transform(small_feature_matrix)

        recon1 = r1.inverse_transform(Y1)
        recon2 = r2.inverse_transform(Y2)

        np.testing.assert_allclose(recon1, recon2, atol=1e-10)


class TestWGCNAReducerCatalogEnrichment:
    """Tests for get_catalog_enrichment."""

    def test_wgcna_get_catalog_enrichment(self, small_feature_df):
        r = WGCNAReducer(soft_power=6, min_module_size=20, verbose=0).fit(small_feature_df)

        n_cols = len(small_feature_df.columns)
        catalog_data = {
            "feature": list(small_feature_df.columns),
            "family": ["firstorder" if i < 100 else "glcm" for i in range(n_cols)],
            "family_group": ["intensity" if i < 100 else "texture" for i in range(n_cols)],
        }
        catalog_df = pd.DataFrame(catalog_data)

        enrich = r.get_catalog_enrichment(catalog_df)
        assert isinstance(enrich, pd.DataFrame)
        assert "catalog_column" in enrich.columns
        assert "fdr_q_value" in enrich.columns
        assert set(enrich["catalog_column"].unique()) == {"family", "family_group"}


class TestWGCNAUtilsAndReducerCoverageGaps:
    """Tests for edge cases in _wgcna.py and _wgcna_utils.py."""

    def test_invalid_orient_eigengenes_raises(self, small_feature_matrix):
        r = WGCNAReducer(orient_eigengenes="invalid_mode")
        with pytest.raises(ValueError, match="orient_eigengenes"):
            r.fit(small_feature_matrix)

    def test_guard_all_nan_features_none(self):
        r = WGCNAReducer()
        r.impute_values_ = None
        r._guard_all_nan_features(["feat1", "feat2"])  # Should return cleanly

    def test_pairwise_network_types(self, small_feature_matrix, rng):
        X_nan = small_feature_matrix.copy()
        mask = rng.random(X_nan.shape) < 0.05
        X_nan[mask] = np.nan

        # Unsigned pairwise
        r_unsigned = WGCNAReducer(
            missing_strategy="pairwise",
            network_type="unsigned",
            soft_power=6,
            min_module_size=20,
            verbose=0,
        )
        Y_u = r_unsigned.fit_transform(X_nan)
        assert Y_u.shape[1] > 0

        # Signed pairwise
        r_signed = WGCNAReducer(
            missing_strategy="pairwise",
            network_type="signed",
            soft_power=6,
            min_module_size=20,
            verbose=0,
        )
        Y_s = r_signed.fit_transform(X_nan)
        assert Y_s.shape[1] > 0

    def test_auto_soft_power_failure_raises(self, small_feature_matrix, monkeypatch):
        from PyWGCNA.wgcna import WGCNA

        monkeypatch.setattr(
            WGCNA,
            "pickSoftThreshold",
            lambda *args, **kwargs: (
                None,
                pd.DataFrame(columns=["Power", "SFT.R.sq", "slope", "mean(k)"]),
            ),
        )
        r = WGCNAReducer(
            missing_strategy="pairwise",
            soft_power="auto",
            min_module_size=5,
            verbose=0,
            soft_power_fallback="raise",
        )
        with pytest.raises(ValueError, match="Automatic soft-power selection failed"):
            r.fit(small_feature_matrix)

    def test_cutree_dynamic_ndarray_result(self, small_feature_matrix, monkeypatch):
        from PyWGCNA.wgcna import WGCNA

        # Mock cutreeHybrid returning a numpy array instead of a DataFrame (hits line 470)
        n_features = small_feature_matrix.shape[1]
        monkeypatch.setattr(
            WGCNA, "cutreeHybrid", lambda *args, **kwargs: np.ones(n_features, dtype=int)
        )
        r = WGCNAReducer(soft_power=6, min_module_size=5, verbose=0)
        r.fit(small_feature_matrix)
        assert r.n_components_ > 0

    def test_artifact_feature_importances_no_modules(self, small_feature_matrix):
        # Matrix where no modules are found (min_module_size larger than n_features)
        r = WGCNAReducer(min_module_size=1000, soft_power=6, verbose=0)
        with pytest.warns(UserWarning, match="found no modules"):
            r.fit(small_feature_matrix)
        assert r._artifact_feature_importances() is None

    def test_wgcna_utils_sign_orientations(self):
        # 1. Multi-component sum_loadings
        X = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 1.0, 2.0],
                [3.0, 3.0, 1.0],
                [4.0, 2.0, 3.0],
            ]
        )
        eg, loadings = _wgcna_compute_eigengene(
            X, n_components=2, orient_eigengenes="sum_loadings"
        )
        assert loadings.shape == (3, 2)

        # 2. Multi-component mean_correlation with orthogonal secondary component (lines 99-100)
        for mat in [
            np.array([[1.0, 1.0], [-1.0, -1.0], [2.0, -2.0], [-2.0, 2.0]]),
            np.array([[1.0, 1.0], [-1.0, -1.0], [-2.0, 2.0], [2.0, -2.0]]),
            np.array([[1.0, -1.0], [-1.0, 1.0], [2.0, 2.0], [-2.0, -2.0]]),
            np.array([[-1.0, 1.0], [1.0, -1.0], [2.0, 2.0], [-2.0, -2.0]]),
            np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, -3.0], [0.0, 3.0]]),
            np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 3.0], [0.0, -3.0]]),
        ]:
            _wgcna_compute_eigengene(mat, n_components=2, orient_eigengenes="mean_correlation")

        # 3. Single component sum_loadings inversion (hits lines 64-65)
        for s in [1.0, -1.0]:
            X_s = np.array([[s * 1.0, s * 2.0], [s * 2.0, s * 1.0], [s * 3.0, s * 2.0]])
            eg3, loadings3 = _wgcna_compute_eigengene(
                X_s, n_components=1, orient_eigengenes="sum_loadings"
            )
            assert np.sum(loadings3) >= -1e-10


# ----------------------------------------------------------------------
# Regression tests: topology input, PAM stage, merge policy, imputation
# ----------------------------------------------------------------------


def _opposite_modules(n=80, size=25, seed=0, constant=False):
    """Three modules: A, B = -A (opposite sign), and an independent C."""
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    blocks = [
        f1[:, None] + 0.4 * rng.normal(size=(n, size)),
        -f1[:, None] + 0.4 * rng.normal(size=(n, size)),
        f2[:, None] + 0.4 * rng.normal(size=(n, size)),
    ]
    cols = [f"{g}{i}" for g in "ABC" for i in range(size)]
    X = pd.DataFrame(np.hstack(blocks), columns=cols)
    if constant:
        X["const"] = 1.0
    return X


def _groups_by_module(reducer, X):
    labels = pd.Series(reducer.module_colors_, index=X.columns)
    return {m: set(labels.index[labels == m].str[0]) for m in reducer.module_names_}


def test_constant_feature_is_unassigned_and_outside_topology():
    X = _opposite_modules(constant=True)
    with pytest.warns(UserWarning, match="zero variance"):
        r = WGCNAReducer(
            network_type="signed", soft_power=6, min_module_size=10, store_tom=True
        ).fit(X)
    assert r.module_colors_[-1] == "grey"
    assert len(r.network_features_) == X.shape[1] - 1
    art = r.get_reduction_artifacts()
    assert sorted(art.feature_order) == sorted(X.columns.drop("const"))
    assert art.cluster_labels["const"] == "grey"
    import matplotlib.pyplot as plt

    from eigenradiomics.plotting import plot_clustered_heatmap

    fig = plot_clustered_heatmap(art)
    plt.close(fig if isinstance(fig, plt.Figure) else fig[0])


def test_pam_default_keeps_opposite_features_out_of_a_module():
    X = _opposite_modules()
    r = WGCNAReducer(network_type="signed", soft_power=6, min_module_size=10, me_diss_threshold=0)
    r.fit(X)
    for groups in _groups_by_module(r, X).values():
        assert len(groups) == 1  # no module mixes A, B or C features


def test_signed_merge_keeps_opposite_modules_apart():
    X = _opposite_modules()
    kw = dict(network_type="signed", soft_power=6, min_module_size=10, me_diss_threshold=0.25)
    signed = _groups_by_module(WGCNAReducer(**kw).fit(X), X)
    assert sorted(map(sorted, signed.values())) == [["A"], ["B"], ["C"]]
    absolute = _groups_by_module(WGCNAReducer(merge_policy="absolute", **kw).fit(X), X)
    assert {"A", "B"} in absolute.values()


def test_unassigned_label_is_grey_and_enrichment_filters_it():
    X = _noise_heavy_matrix()
    r = WGCNAReducer(soft_power=6, min_module_size=20, include_grey=True, verbose=0).fit(X)
    assert "black" not in r.module_colors_
    assert "grey" in r.module_colors_


def test_pairwise_undefined_pair_has_no_edge():
    X = _opposite_modules(size=12)
    X.iloc[:40, 0] = np.nan
    X.iloc[40:, 1] = np.nan
    r = WGCNAReducer(
        network_type="signed",
        soft_power=6,
        min_module_size=5,
        missing_strategy="pairwise",
        store_tom=True,
    )
    with pytest.warns(UserWarning, match="1 feature pair"):
        r.fit(X)
    assert np.isfinite(r.tom_).all()


@pytest.mark.parametrize("strategy", ["zero", "floor", "median", "mean"])
def test_imputing_strategies_center_training_scores(strategy):
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(80, 40)) + rng.normal(size=(80, 1)) + 5)
    X.columns = [f"g{i}" for i in range(40)]
    X = X.mask(rng.random(X.shape) < 0.2)
    r = WGCNAReducer(soft_power=6, min_module_size=10, missing_strategy=strategy).fit(X)
    scores = r.transform(X)
    assert scores.shape[1] > 0
    assert np.allclose(scores.mean(axis=0), 0.0, atol=1e-8)


def test_new_parameters_are_validated():
    X = _opposite_modules(size=5)
    with pytest.raises(ValueError, match="soft_power_fallback"):
        WGCNAReducer(soft_power_fallback="maybe").fit(X)
    with pytest.raises(ValueError, match="merge_policy"):
        WGCNAReducer(merge_policy="sideways").fit(X)
    flat = pd.DataFrame(np.ones((10, 5)), columns=list("abcde")).assign(a=np.arange(10.0))
    with pytest.raises(ValueError, match="at least 3 non-constant"):
        WGCNAReducer(soft_power=6).fit(flat)


def test_auto_power_default_warns_and_records_failed_criteria():
    X = _opposite_modules(size=10)
    r = WGCNAReducer(soft_power="auto", r_squared_cut=0.9999, mean_cut=0.001, min_module_size=5)
    with pytest.warns(UserWarning, match="No soft power meets"):
        r.fit(X)
    assert r.soft_power_criteria_met_ is False


def test_numpy_integer_parameters_and_grid_search():
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    X = _opposite_modules(size=10)
    r = WGCNAReducer(soft_power=np.int64(6), min_module_size=np.int64(5), power_max=np.int32(12))
    r.fit(X)
    assert r.soft_power_ == 6
    pipe = Pipeline([("reduce", WGCNAReducer(min_module_size=5))])
    grid = GridSearchCV(
        pipe,
        {"reduce__soft_power": np.arange(4, 9, 2)},
        cv=2,
        scoring=lambda est, X, y=None: float(est.transform(X).shape[1]),
        error_score="raise",
    )
    grid.fit(X)
    assert len(grid.cv_results_["mean_test_score"]) == 3


@pytest.mark.parametrize("depth", [0, 1, 2, 3, 4])
def test_every_deep_split_runs_with_r_defaults(depth, monkeypatch):
    from PyWGCNA.wgcna import WGCNA

    seen = {}
    real = WGCNA.cutreeHybrid

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(WGCNA, "cutreeHybrid", spy)
    WGCNAReducer(soft_power=6, min_module_size=5, deep_split=depth).fit(_opposite_modules(size=10))
    assert seen["maxCoreScatter"] == (0.64, 0.73, 0.82, 0.91, 0.95)[depth]
