"""Manifold learning and embedding reducers as sklearn-compatible estimators.

This module provides standard scikit-learn manifold estimators (t-SNE, MDS, Isomap,
Spectral Embedding, Locally Linear Embedding) and optional advanced reducers
(UMAP, PaCMAP, TriMAP) wrapped to inherit from BaseReducer.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.utils.validation import check_is_fitted

from eigenradiomics.reducers._base import BaseReducer


class TSNEReducer(BaseReducer):
    """t-Distributed Stochastic Neighbor Embedding (t-SNE) reducer.

    TSNE is a transductive-only method and does not support transforming
    unseen data out-of-sample.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.
    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms.
    early_exaggeration : float, default=12.0
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them.
    learning_rate : float or str, default="auto"
        The learning rate for t-SNE is usually in the range [10.0, 1000.0].
    n_iter : int, default=1000
        Maximum number of iterations for the optimization.
    random_state : int, RandomState instance or None, default=None
        Determines the random number generator.
    **kwargs : dict
        Additional keyword arguments passed to sklearn.manifold.TSNE.
    """

    _reducer_prefix: str = "tsne"

    def __init__(
        self,
        n_components: int = 2,
        *,
        perplexity: float = 30.0,
        early_exaggeration: float = 12.0,
        learning_rate: float | str = "auto",
        max_iter: int = 1000,
        random_state: int | np.random.RandomState | None = None,
        **kwargs: Any,
    ) -> None:
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(self, X: NDArray | pd.DataFrame, y: None = None) -> TSNEReducer:
        X_arr = self._validate_input(X, reset=True, allow_nan=False)
        self.tsne_ = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            early_exaggeration=self.early_exaggeration,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            random_state=self.random_state,
            **self.kwargs,
        )
        self.embedding_ = self.tsne_.fit_transform(X_arr)
        self.n_components_ = self.n_components
        return self

    def transform(self, X: NDArray | pd.DataFrame) -> NDArray:
        raise NotImplementedError(
            "TSNEReducer is transductive-only and does not support transform on new data. "
            "Use fit_transform on training data."
        )

    def fit_transform(self, X: NDArray | pd.DataFrame, y: None = None) -> NDArray:
        self.fit(X, y)
        return cast(NDArray, self.embedding_)


class MDSReducer(BaseReducer):
    """Multidimensional Scaling (MDS) reducer.

    MDS is a transductive-only method and does not support transforming
    unseen data out-of-sample.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities.
    metric : bool, default=True
        If True, perform metric MDS; otherwise, perform nonmetric MDS.
    n_init : int, default=4
        Number of times the SMACOF algorithm will be run with different
        initializations.
    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.
    eps : float, default=1e-3
        Relative tolerance with respect to stress at which to declare convergence.
    random_state : int, RandomState instance or None, default=None
        Determines the random number generator.
    **kwargs : dict
        Additional keyword arguments passed to sklearn.manifold.MDS.
    """

    _reducer_prefix: str = "mds"

    def __init__(
        self,
        n_components: int = 2,
        *,
        metric: bool = True,
        n_init: int = 4,
        max_iter: int = 300,
        eps: float = 1e-3,
        random_state: int | np.random.RandomState | None = None,
        **kwargs: Any,
    ) -> None:
        self.n_components = n_components
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.eps = eps
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(self, X: NDArray | pd.DataFrame, y: None = None) -> MDSReducer:
        X_arr = self._validate_input(X, reset=True, allow_nan=False)
        self.mds_ = MDS(
            n_components=self.n_components,
            metric=self.metric,
            n_init=self.n_init,
            max_iter=self.max_iter,
            eps=self.eps,
            random_state=self.random_state,
            **self.kwargs,
        )
        self.embedding_ = self.mds_.fit_transform(X_arr)
        self.n_components_ = self.n_components
        return self

    def transform(self, X: NDArray | pd.DataFrame) -> NDArray:
        raise NotImplementedError(
            "MDSReducer is transductive-only and does not support transform on new data. "
            "Use fit_transform on training data."
        )

    def fit_transform(self, X: NDArray | pd.DataFrame, y: None = None) -> NDArray:
        self.fit(X, y)
        return cast(NDArray, self.embedding_)


class SpectralReducer(BaseReducer):
    """Spectral Embedding reducer.

    Spectral Embedding is a transductive-only method and does not support
    transforming unseen data out-of-sample.

    Parameters
    ----------
    n_components : int, default=2
        The dimension of the projected subspace.
    affinity : str, default="nearest_neighbors"
        How to construct the affinity matrix.
    gamma : float, default=None
        Kernel coefficient for rbf, poly, sigmoid, laplacian kernels.
    random_state : int, RandomState instance or None, default=None
        Determines the random number generator.
    **kwargs : dict
        Additional keyword arguments passed to sklearn.manifold.SpectralEmbedding.
    """

    _reducer_prefix: str = "spectral"

    def __init__(
        self,
        n_components: int = 2,
        *,
        affinity: str = "nearest_neighbors",
        gamma: float | None = None,
        random_state: int | np.random.RandomState | None = None,
        **kwargs: Any,
    ) -> None:
        self.n_components = n_components
        self.affinity = affinity
        self.gamma = gamma
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(self, X: NDArray | pd.DataFrame, y: None = None) -> SpectralReducer:
        X_arr = self._validate_input(X, reset=True, allow_nan=False)
        self.spectral_ = SpectralEmbedding(
            n_components=self.n_components,
            affinity=self.affinity,
            gamma=self.gamma,
            random_state=self.random_state,
            **self.kwargs,
        )
        self.embedding_ = self.spectral_.fit_transform(X_arr)
        self.n_components_ = self.n_components
        return self

    def transform(self, X: NDArray | pd.DataFrame) -> NDArray:
        raise NotImplementedError(
            "SpectralReducer is transductive-only and does not support transform on new data. "
            "Use fit_transform on training data."
        )

    def fit_transform(self, X: NDArray | pd.DataFrame, y: None = None) -> NDArray:
        self.fit(X, y)
        return cast(NDArray, self.embedding_)


class IsomapReducer(BaseReducer):
    """Isomap (Isometric Feature Mapping) reducer.

    Supports true inductive out-of-sample projection via transform.

    Parameters
    ----------
    n_components : int, default=2
        Number of coordinates for the manifold.
    n_neighbors : int, default=5
        Number of neighbors to consider for each point.
    **kwargs : dict
        Additional keyword arguments passed to sklearn.manifold.Isomap.
    """

    _reducer_prefix: str = "isomap"

    def __init__(
        self,
        n_components: int = 2,
        *,
        n_neighbors: int = 5,
        **kwargs: Any,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.kwargs = kwargs

    def fit(self, X: NDArray | pd.DataFrame, y: None = None) -> IsomapReducer:
        X_arr = self._validate_input(X, reset=True, allow_nan=False)
        self.isomap_ = Isomap(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            **self.kwargs,
        )
        self.isomap_.fit(X_arr)
        self.n_components_ = self.n_components
        return self

    def transform(self, X: NDArray | pd.DataFrame) -> NDArray:
        check_is_fitted(self, "n_components_")
        X_arr = self._validate_input(X, reset=False, allow_nan=False)
        return cast(NDArray, self.isomap_.transform(X_arr))


class LLEReducer(BaseReducer):
    """Locally Linear Embedding (LLE) reducer.

    Supports true inductive out-of-sample projection via transform.

    Parameters
    ----------
    n_components : int, default=2
        Number of coordinates for the manifold.
    n_neighbors : int, default=5
        Number of neighbors to consider for each point.
    method : {"standard", "ltsa", "hessian", "modified"}, default="standard"
        Algorithm method to use.
    random_state : int, RandomState instance or None, default=None
        Determines the random number generator.
    **kwargs : dict
        Additional keyword arguments passed to sklearn.manifold.LocallyLinearEmbedding.
    """

    _reducer_prefix: str = "lle"

    def __init__(
        self,
        n_components: int = 2,
        *,
        n_neighbors: int = 5,
        method: str = "standard",
        random_state: int | np.random.RandomState | None = None,
        **kwargs: Any,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.method = method
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(self, X: NDArray | pd.DataFrame, y: None = None) -> LLEReducer:
        X_arr = self._validate_input(X, reset=True, allow_nan=False)
        self.lle_ = LocallyLinearEmbedding(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            method=self.method,
            random_state=self.random_state,
            **self.kwargs,
        )
        self.lle_.fit(X_arr)
        self.n_components_ = self.n_components
        return self

    def transform(self, X: NDArray | pd.DataFrame) -> NDArray:
        check_is_fitted(self, "n_components_")
        X_arr = self._validate_input(X, reset=False, allow_nan=False)
        return cast(NDArray, self.lle_.transform(X_arr))


class UMAPReducer(BaseReducer):
    """Uniform Manifold Approximation and Projection (UMAP) reducer.

    Supports true inductive out-of-sample projection via transform.
    Requires optional 'umap-learn' package.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.
    densmap : bool, default=False
        Whether to use densMAP to preserve local density representation.
    random_state : int, RandomState instance or None, default=None
        Determines the random number generator.
    **kwargs : dict
        Additional keyword arguments passed to umap.UMAP.
    """

    _reducer_prefix: str = "umap"

    def __init__(
        self,
        n_components: int = 2,
        *,
        densmap: bool = False,
        random_state: int | np.random.RandomState | None = None,
        **kwargs: Any,
    ) -> None:
        self.n_components = n_components
        self.densmap = densmap
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(self, X: NDArray | pd.DataFrame, y: None = None) -> UMAPReducer:
        try:
            import umap
        except ImportError as e:
            raise ImportError(
                "UMAPReducer requires the 'umap-learn' package. "
                "Install it with: pip install umap-learn"
            ) from e

        X_arr = self._validate_input(X, reset=True, allow_nan=False)
        self.umap_ = umap.UMAP(
            n_components=self.n_components,
            densmap=self.densmap,
            random_state=self.random_state,
            **self.kwargs,
        )
        self.umap_.fit(X_arr)
        self.n_components_ = self.n_components
        return self

    def transform(self, X: NDArray | pd.DataFrame) -> NDArray:
        check_is_fitted(self, "n_components_")
        X_arr = self._validate_input(X, reset=False, allow_nan=False)
        return cast(NDArray, self.umap_.transform(X_arr))


class PaCMAPReducer(BaseReducer):
    """Pairwise Controlled Manifold Approximation (PaCMAP) reducer.

    Supports true inductive out-of-sample projection via transform.
    Requires optional 'pacmap' package.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.
    random_state : int, RandomState instance or None, default=None
        Determines the random number generator.
    **kwargs : dict
        Additional keyword arguments passed to pacmap.PaCMAP.
    """

    _reducer_prefix: str = "pacmap"

    def __init__(
        self,
        n_components: int = 2,
        *,
        random_state: int | np.random.RandomState | None = None,
        **kwargs: Any,
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(self, X: NDArray | pd.DataFrame, y: None = None) -> PaCMAPReducer:
        try:
            import pacmap
        except ImportError as e:
            raise ImportError(
                "PaCMAPReducer requires the 'pacmap' package. "
                "Install it with: pip install pacmap"
            ) from e

        X_arr = self._validate_input(X, reset=True, allow_nan=False)
        self.pacmap_ = pacmap.PaCMAP(
            n_dims=self.n_components,
            random_state=self.random_state,
            **self.kwargs,
        )
        self.pacmap_.fit(X_arr)
        self.n_components_ = self.n_components
        return self

    def transform(self, X: NDArray | pd.DataFrame) -> NDArray:
        check_is_fitted(self, "n_components_")
        X_arr = self._validate_input(X, reset=False, allow_nan=False)
        return cast(NDArray, self.pacmap_.transform(X_arr))


class TriMAPReducer(BaseReducer):
    """TriMap dimensionality reduction reducer.

    Supports true inductive out-of-sample projection via transform.
    Requires optional 'trimap' package.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.
    **kwargs : dict
        Additional keyword arguments passed to trimap.TRIMAP.
    """

    _reducer_prefix: str = "trimap"

    def __init__(
        self,
        n_components: int = 2,
        **kwargs: Any,
    ) -> None:
        self.n_components = n_components
        self.kwargs = kwargs

    def fit(self, X: NDArray | pd.DataFrame, y: None = None) -> TriMAPReducer:
        try:
            import trimap
        except ImportError as e:
            raise ImportError(
                "TriMAPReducer requires the 'trimap' package. "
                "Install it with: pip install trimap"
            ) from e

        X_arr = self._validate_input(X, reset=True, allow_nan=False)
        self.trimap_ = trimap.TRIMAP(
            n_dims=self.n_components,
            **self.kwargs,
        )
        if hasattr(self.trimap_, "fit"):
            self.trimap_.fit(X_arr)
        else:
            self.embedding_ = self.trimap_.fit_transform(X_arr)
        self.n_components_ = self.n_components
        return self

    def transform(self, X: NDArray | pd.DataFrame) -> NDArray:
        check_is_fitted(self, "n_components_")
        X_arr = self._validate_input(X, reset=False, allow_nan=False)
        if hasattr(self.trimap_, "transform"):
            return cast(NDArray, self.trimap_.transform(X_arr))
        raise NotImplementedError(
            "The installed 'trimap' library does not support transform on new data."
        )
