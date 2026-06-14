# PCA & Sparse PCA Reducers

The package implements two standard linear projection techniques wrapped to conform to the `eigenradiomics` pipeline and validation specifications: **Principal Component Analysis (PCA)** and **Sparse PCA**.

Unlike standard scikit-learn transformers, these estimators integrate feature-name checking, automatic input consistency checks, structured `ReductionArtifacts`, and robust reconstruction back to the original space via `inverse_transform`.

---

## Principal Component Analysis (`PCAReducer`)

`PCAReducer` wraps standard `sklearn.decomposition.PCA`. It derives orthogonal axes that maximize the explained variance of the data.

### Use Case
PCA is highly effective for reducing high-dimensional collinear features (like first-order or texture radiomics) when the primary goal is statistical efficiency and decorrelation.

```python
from eigenradiomics.reducers import PCAReducer

# Retain 95% of total variance
pca = PCAReducer(n_components=0.95)
Y = pca.fit_transform(X)

print(Y.shape)  # output columns are named 'pca_0', 'pca_1', ...
```

---

## Sparse PCA (`SparsePCAReducer`)

`SparsePCAReducer` wraps `sklearn.decomposition.SparsePCA`. It introduces an $L_1$ penalty to find sparse components, where each component is a linear combination of only a small subset of the original features.

### Use Case
While standard PCA components are linear combinations of *all* input features, Sparse PCA forces many coefficient loadings to be exactly zero. This makes the resulting components far more interpretable for clinical applications, as each factor can be traced directly to a handful of physical radiomic descriptors.

```python
from eigenradiomics.reducers import SparsePCAReducer

# Extract 3 sparse components with sparsity penalty alpha=1.0
spca = SparsePCAReducer(n_components=3, alpha=1.0, random_state=42)
Y = spca.fit_transform(X)
```

---

## Reconstruction and Inverse Transformation

Both reducers implement `.inverse_transform(Y)` to project the reduced scores back into the original multi-dimensional space, supporting exact validation of model fit and reconstruction loss.

- **`PCAReducer`**: Maps back using the transpose of the orthonormal loading matrix.
- **`SparsePCAReducer`**: Maps back via linear projection conforming to:
  $$\hat{X} = Y \cdot C + \mu$$
  where $C \in \mathbb{R}^{k \times m}$ is the sparse component matrix, $Y \in \mathbb{R}^{n \times k}$ are the component scores, and $\mu \in \mathbb{R}^m$ is the feature feature-wise training mean.

```python
# Project back to original space
X_reconstructed = spca.inverse_transform(Y)
reconstruction_error = ((X - X_reconstructed) ** 2).mean()
```

---

## Accessing Loadings & Artifacts

Both classes report feature loadings through `get_reduction_artifacts().feature_importances` as a unified Pandas DataFrame, allowing direct plotting or export.

```python
artifacts = pca.get_reduction_artifacts()
loadings_df = artifacts.feature_importances
# `loadings_df` contains columns: 'component_0', 'component_1', ... and 'importance' (max absolute loading)
```
