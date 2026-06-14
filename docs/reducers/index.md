# Reducers (Algorithms)

**eigenradiomics** is designed around a modular scaffolding that aggregates diverse dimensionality reduction techniques under a unified scikit-learn compatible interface. The core objective is to compress a wide feature matrix \(X \in \mathbb{R}^{n \times m}\) into a dense subspace \(Y \in \mathbb{R}^{n \times k}\) where \(k \ll m\), while preserving meaningful latent structure and enabling transfer of the learned reduction to new, unseen data.

This is particularly critical when working with high-dimensional radiomic feature tables, such as those produced by [Pictologics](https://github.com/martonkolossvary/pictologics), where hundreds or thousands of correlated features must be distilled into a tractable set of components for downstream predictive modelling.

## Unified Reducer Contract

While different algorithms operate under distinct statistical assumptions, all reducers in this package adhere to a shared set of guarantees:

1. **Dense Matrices**: All methods operate on dense NumPy backends. Sparse data arrays are rejected by default.
2. **Prior Scaling**: Dimensionality reduction models are vulnerable to unit-variance skew. Continuous features should be \(z\)-scored (standard scaled) *prior* to passing to a reducer.
3. **Reproducible Loadings**: Every reducer fits and caches its internal parameters (e.g., eigenvector bases, singular value loadings, module membership) during `.fit()`. This ensures that `.transform()` on new data produces deterministic, mathematically equivalent mappings without re-computing the reduction, strictly preventing data leakage.
4. **Inverse Transform**: Where mathematically supported, reducers implement `.inverse_transform()` to project from the reduced subspace back to the original feature space. This enables reconstruction error analysis and intrinsic goodness-of-fit evaluation during hyperparameter tuning.
5. **Full Parameter Exposure**: Every hyperparameter of a given reducer is exposed through the standard `get_params()` / `set_params()` interface. This allows seamless integration with scikit-learn hyperparameter tuning frameworks such as `GridSearchCV` and `RandomizedSearchCV`.

## Hyperparameter Optimization

Because reducers are unsupervised, their parameters are tuned either by downstream
supervised evaluation in a `Pipeline`, by an intrinsic unsupervised score
(reconstruction error, silhouette), or by tracking both with a multi-metric
`GridSearchCV`. The [Pipelines & Grid Search](../user_guide/pipelines_and_grid_search.md)
guide is the single home for those strategies with worked examples.

## Output Nomenclature

For straightforward traceability in downstream coefficient tables or hyperparameter search results, reduced features receive synthetic names scoped by the algorithm:

- WGCNA modules → `wgcna_0`, `wgcna_1`, `wgcna_2`, ...
- Future reducers → `pca_0`, `nmf_0`, etc.

These names are returned by `get_feature_names_out()` and are propagated through scikit-learn pipelines automatically.

## Reduction Artifacts

Beyond the transformed matrix, a reducer can expose **structured intermediate
outputs** for downstream plotting and analysis through one uniform interface:

```python
artifacts = reducer.get_reduction_artifacts()
artifacts.available()    # e.g. ['similarity', 'linkage', 'cluster_labels',
                         #       'feature_order', 'feature_importances']
artifacts.similarity     # feature × feature DataFrame, or None
```

Each reducer populates whichever elements it can produce and leaves the rest as
`None`, so a method that only estimates a similarity matrix and one that also
assigns clusters both feed the same `ReductionArtifacts` container — and the
same downstream code (clustered heatmaps, exports, analysis).

| Element | Meaning |
|---------|---------|
| `feature_names` | input feature names (always present) |
| `similarity` | symmetric feature × feature similarity (e.g. WGCNA TOM) |
| `linkage` | SciPy hierarchical-clustering linkage matrix |
| `cluster_labels` | per-feature cluster / module label |
| `feature_order` | features in a meaningful display order (e.g. dendrogram leaves) |
| `feature_importances` | per-feature contribution to the reduction |

`WGCNAReducer` populates all of these (`similarity`/TOM requires
`store_tom=True`). [`plot_clustered_heatmap`](../user_guide/clustered_heatmap.md)
consumes a `ReductionArtifacts` directly.

## Currently Implemented Reducers

| Reducer | Technique | Reference |
|---------|-----------|-----------|
| [`WGCNAReducer`](wgcna.md) | Weighted Gene Co-expression Network Analysis | [Langfelder & Horvath, 2008](https://pubmed.ncbi.nlm.nih.gov/19114008/); [PyWGCNA](https://pubmed.ncbi.nlm.nih.gov/37399090/) |
| [`PCAReducer`](pca.md) | Principal Component Analysis (PCA) | [Pearson, 1901](https://doi.org/10.1080/14786440109462720); scikit-learn |
| [`SparsePCAReducer`](pca.md) | Sparse Principal Component Analysis | [Zou et al., 2006](https://doi.org/10.1198/106186006X113430); scikit-learn |
