"""PCA and Sparse PCA dimensionality reduction as sklearn-compatible estimators.

These reducers wrap standard sklearn implementations but inherit from BaseReducer
and adapt them to the eigenradiomics pipeline architecture:
- Names of output columns are prefixed with 'pca_` or 'sparse_pca_`
- Full column-order and name contract validation on fit/transform
- Support for detailed reduction artifacts
- Full, clean support for inverse_transform mapping back to original space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.decomposition import PCA, SparsePCA
from sklearn.utils.validation import check_is_fitted

from eigenradiomics.reducers._base import BaseReducer


class PCAReducer(BaseReducer):
    """Dimensionality reduction via Principal Component Analysis (PCA).

    Wraps sklearn's PCA but integrates seamlessly into the eigenradiomics pipeline
    system, including strict verification of feature column sequence and name contracts,
    support for structured ReductionArtifacts, and inverse_transform mapping back to
    the original feature space.

    Parameters
    ----------
    n_components : int, float or None (default=None)
        Number of components to keep. If None, all components are kept.
        If 0 < n_components < 1, select the number of components such that
        the amount of variance that needs to be explained is greater than
        the percentage specified.
    whiten : bool (default=False)
        When True, the components_ vectors are multiplied by the square root
        of n_samples and then divided by the singular values to ensure uncorrelated
        outputs with unit component-wise variances.
    svd_solver : {"auto", "full", "arpack", "randomized"} (default="auto")
        SVD solver to use.
    tol : float (default=0.0)
        Tolerance for singular values estimated by svd_solver == "arpack".
    iterated_power : int or "auto" (default="auto")
        Number of iterations for the power method in randomized SVD solver.
    random_state : int, RandomState instance or None (default=None)
        Used when randomized or arpack solver is used.
    """

    _reducer_prefix: str = "pca"

    def __init__(
        self,
        n_components: int | float | None = None,
        *,
        whiten: bool = False,
        svd_solver: str = "auto",
        tol: float = 0.0,
        iterated_power: int | str = "auto",
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

    def fit(self, X: NDArray | pd.DataFrame, y: None = None) -> PCAReducer:
        """Fit the PCA model with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self : PCAReducer
            Fitted estimator.
        """
        X_arr = self._validate_input(X, reset=True, allow_nan=False)

        self.pca_ = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            tol=self.tol,
            iterated_power=self.iterated_power,
            random_state=self.random_state,
        )

        self.pca_.fit(X_arr)
        self.n_components_ = self.pca_.n_components_

        # Expose components_ and explained_variance_ etc.
        self.components_ = self.pca_.components_
        self.explained_variance_ = self.pca_.explained_variance_
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        self.singular_values_ = self.pca_.singular_values_
        self.mean_ = self.pca_.mean_
        self.noise_variance_ = self.pca_.noise_variance_

        return self

    def transform(self, X: NDArray | pd.DataFrame) -> NDArray:
        """Apply dimensionality reduction to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components_)
            Projection of X onto components.
        """
        check_is_fitted(self, "pca_")
        X_arr = self._validate_input(X, reset=False, allow_nan=False)
        return np.asarray(self.pca_.transform(X_arr))

    def inverse_transform(self, Y: NDArray) -> NDArray:
        """Transform data back to its original space.

        Parameters
        ----------
        Y : array-like of shape (n_samples, n_components_)
            Projected data to invert.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Reconstructed data.
        """
        check_is_fitted(self, "pca_")
        Y_arr = np.asarray(Y)
        return np.asarray(self.pca_.inverse_transform(Y_arr))

    def _artifact_feature_importances(self) -> pd.DataFrame | None:
        """Return loadings as feature importances."""
        check_is_fitted(self, "components_")
        columns = [f"component_{i}" for i in range(self.n_components_)]
        df = pd.DataFrame(
            self.components_.T,
            index=self.feature_names_in_,
            columns=columns,
        )
        # Add a column for max absolute loading across components as an importance summary
        df["importance"] = np.abs(self.components_).max(axis=0)
        return df


class SparsePCAReducer(BaseReducer):
    """Dimensionality reduction via Sparse Principal Component Analysis (SparsePCA).

    Wraps sklearn's SparsePCA to integrate into the modern eigenradiomics pipeline
    system, verifying feature column name sequence, and including clean support
    for inverse_transform.

    Parameters
    ----------
    n_components : int or None (default=None)
        Number of sparse components to extract. If None, all components are kept.
    alpha : float (default=1.0)
        Sparsity controlling parameter. Higher values lead to sparser components.
    ridge_alpha : float (default=0.01)
        Amount of ridge shrinkage to apply in order to improve conditioning
        when calling transform.
    max_iter : int (default=1000)
        Maximum number of iterations to run.
    tol : float (default=1e-8)
        Tolerance for stopping criterion.
    method : {"lars", "cd"} (default="lars")
        lars: uses the least angle regression method.
        cd: uses coordinate descent.
        lars is faster but cd is more stable for large matrices.
    n_jobs : int or None (default=None)
        Number of parallel jobs to run.
    random_state : int, RandomState instance or None (default=None)
        Used for initialization.
    """

    _reducer_prefix: str = "sparse_pca"

    def __init__(
        self,
        n_components: int | None = None,
        *,
        alpha: float = 1.0,
        ridge_alpha: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-8,
        method: str = "lars",
        n_jobs: int | None = None,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.n_components = n_components
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: NDArray | pd.DataFrame, y: None = None) -> SparsePCAReducer:
        """Fit the SparsePCA model with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self : SparsePCAReducer
            Fitted estimator.
        """
        X_arr = self._validate_input(X, reset=True, allow_nan=False)

        self.sparse_pca_ = SparsePCA(
            n_components=self.n_components,
            alpha=self.alpha,
            ridge_alpha=self.ridge_alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            method=self.method,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        self.sparse_pca_.fit(X_arr)
        # sklearn <= 1.5 doesn't set n_components_ explicitly or might depend on fitted state
        self.n_components_ = self.sparse_pca_.components_.shape[0]

        # Exposing components_, error_ etc.
        self.components_ = self.sparse_pca_.components_
        self.error_ = self.sparse_pca_.error_
        self.n_iter_ = self.sparse_pca_.n_iter_
        self.mean_ = self.sparse_pca_.mean_

        return self

    def transform(self, X: NDArray | pd.DataFrame) -> NDArray:
        """Apply dimensionality reduction to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components_)
            Projection of X.
        """
        check_is_fitted(self, "sparse_pca_")
        X_arr = self._validate_input(X, reset=False, allow_nan=False)
        return np.asarray(self.sparse_pca_.transform(X_arr))

    def inverse_transform(self, Y: NDArray) -> NDArray:
        """Transform data back to its original space.

        For SparsePCA, because the components are not orthonormal, this uses
        reconstruction via pseudoinverse or direct linear matrix multiplication
        conforming to standard projection mappings:
        `X_reconstructed = Y @ components_ + mean_`.

        Parameters
        ----------
        Y : array-like of shape (n_samples, n_components_)
            Projected data to invert.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Reconstructed data.
        """
        check_is_fitted(self, "sparse_pca_")
        Y_arr = np.asarray(Y)
        # Map back to original dimensions: code = Y, dictionary = components_
        # X_reconstructed = code @ dictionary + mean
        return np.asarray(Y_arr @ self.components_ + self.mean_)

    def _artifact_feature_importances(self) -> pd.DataFrame | None:
        """Return sparse loadings as feature importances."""
        check_is_fitted(self, "components_")
        columns = [f"component_{i}" for i in range(self.n_components_)]
        df = pd.DataFrame(
            self.components_.T,
            index=self.feature_names_in_,
            columns=columns,
        )
        df["importance"] = np.abs(self.components_).max(axis=0)
        return df
