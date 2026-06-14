# Manifold Learning and Embeddings

This page guide explains how to use the default and optional manifold learning and embedding reducers in `eigenradiomics` to visualize and inspect high-dimensional radiomic feature spaces.

```mermaid
flowchart TD
    subgraph Default (Scikit-Learn Native)
        TSNE[TSNEReducer<br/>Transductive-only]
        MDS[MDSReducer<br/>Transductive-only]
        Spectral[SpectralReducer<br/>Transductive-only]
        Isomap[IsomapReducer<br/>Inductive]
        LLE[LLEReducer<br/>Inductive]
    end
    subgraph Optional (Lazy Imports)
        UMAP[UMAPReducer<br/>Inductive / densMAP]
        PaCMAP[PaCMAPReducer<br/>Inductive]
        TriMAP[TriMAPReducer<br/>Inductive]
    end
```

---

## Default vs Optional Reducers

### 1. Default (scikit-learn Native)
These are available out of the box without any extra dependencies:
- **`TSNEReducer`**: t-Distributed Stochastic Neighbor Embedding, excellent for capturing local cluster structure.
- **`MDSReducer`**: Multidimensional Scaling, preserves global pairwise distances.
- **`SpectralReducer`**: Spectral Embedding, uses graph Laplacian eigenvalues to find non-linear structure.
- **`IsomapReducer`**: Isometric Feature Mapping, preserves geodesic distances on a neighborhood graph.
- **`LLEReducer`**: Locally Linear Embedding, preserves local linear relationships.

### 2. Optional (Advanced Embeddings)
These require installing additional optional packages:
- **`UMAPReducer`**: Uniform Manifold Approximation and Projection (requires `umap-learn`). Supports preserving local densities via `densmap=True`.
- **`PaCMAPReducer`**: Pairwise Controlled Manifold Approximation (requires `pacmap`), optimizes global and local structure preservation.
- **`TriMAPReducer`**: Trilet-based dimensionality reduction (requires `trimap`), preserves global structure using triplet constraints.

To use optional reducers, install the corresponding packages:
```bash
pip install umap-learn pacmap trimap
```

---

## Inductive vs Transductive Limitations

Dimensionality reduction methods fall into two categories regarding how they handle new, unseen test data:

1. **Inductive (Out-of-sample mapping supported)**:
   Methods like `IsomapReducer`, `LLEReducer`, `UMAPReducer`, `PaCMAPReducer`, and `TriMAPReducer` support the standard `.transform()` method. You can fit them on training data and project new test data into the exact same embedding space:
   ```python
   reducer = IsomapReducer(n_components=2)
   reducer.fit(X_train)
   Y_train = reducer.transform(X_train)
   Y_test = reducer.transform(X_test)  # Safe out-of-sample projection
   ```

2. **Transductive (No out-of-sample mapping)**:
   Methods like `TSNEReducer`, `MDSReducer`, and `SpectralReducer` do not support projecting new points into an existing embedding space. Attempting to call `.transform()` on them will raise a `NotImplementedError`. Instead, you must use `.fit_transform()` to embed all data points together:
   ```python
   reducer = TSNEReducer(n_components=2, perplexity=15.0)
   Y_all = reducer.fit_transform(X_all)  # Transformed all at once
   ```

---

## Code Example

Here is how you can initialize, tune, and run a manifold learning reducer in `eigenradiomics`:

```python
import pandas as pd
from eigenradiomics.reducers import TSNEReducer, UMAPReducer

# 1. Using a default reducer (t-SNE)
tsne = TSNEReducer(
    n_components=2,
    perplexity=20.0,
    max_iter=1000,
    random_state=42
)
# t-SNE is transductive, so we fit_transform all at once:
Y_tsne = tsne.fit_transform(X)

# 2. Using an optional reducer (UMAP with densMAP enabled)
# Ensure `umap-learn` is installed!
umap_reducer = UMAPReducer(
    n_components=2,
    densmap=True,       # Preserves local density information
    random_state=42
)
umap_reducer.fit(X_train)
Y_train = umap_reducer.transform(X_train)
Y_test = umap_reducer.transform(X_test)
```
