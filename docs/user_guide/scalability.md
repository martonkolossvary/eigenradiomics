# Scalability Guide

When working with radiomics features, datasets can quickly grow to thousands of features across hundreds of samples. This guide describes the performance characteristics, memory limits, and optimal configurations for running `eigenradiomics` pipelines at scale.

## Memory Complexity

Dimensionality reduction algorithms often require computing dense pairwise distance or correlation matrices. 
For `WGCNAReducer`, the primary memory overhead is the Topological Overlap Matrix (TOM), which scales quadratically with the number of input features ($O(n^2)$).

- **1,000 features:** ~8 MB for TOM ($1000 \times 1000$ 64-bit strict floats)
- **10,000 features:** ~800 MB for TOM
- **50,000 features:** ~20 GB for TOM

**Recommendation:** For workflows exceeding 20,000 features, ensure you have an environment with at least 32-64GB of RAM. Always leave `store_tom=False` (the default) unless you specifically need to inspect the topological matrix downstream.

## Computational Complexity & `n_samples` vs `n_features`

The calculation of the adjacency matrix scales with $O(n^2 \times m)$ where $n$ is `n_features` and $m$ is `n_samples`.
While radiomics datasets typically have a wide aspect ratio ($m \ll n$), `PyWGCNA` performs optimizations under the hood.

1. **Large `n_features`:** The hierarchical clustering routine (averaging linkages) can become a bottleneck ($O(n^3)$ worst-case, typically $O(n^2)$). If execution takes too long on dense datasets > 20,000 features, consider an upstream variance threshold filter or univariate feature selection before the WGCNA step.
2. **Large `n_samples`:** The correlation calculation will dominate. The default implementation relies on optimized NumPy and pandas operations. 

**Pre-Filtering Example:**
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from eigenradiomics.reducers import WGCNAReducer

pipe = Pipeline([
    ("filter", VarianceThreshold(threshold=0.01)),
    ("wgcna", WGCNAReducer(soft_power=6, min_module_size=20))
])
```

## Parallelization Trade-offs

`WGCNAReducer` leverages multiprocessing locally during `fit` and `transform` via `joblib`.
Control the number of workers via the `n_jobs` parameter.

- **`n_jobs=1`:** Stable, minimal RAM overhead. Recommended for highly iterative GridSearches where `GridSearchCV` is already parallelizing cross-validation.
- **`n_jobs=-1`:** Utilizes all CPU cores. Ideal for processing a single massive matrix when fitting or transforming. Note that `joblib` needs to copy the data into worker processes, temporarily increasing memory usage. 

!!! warning "Nested parallelization"
    Running `GridSearchCV(n_jobs=-1, estimator=WGCNAReducer(n_jobs=-1))`
    multiplies the number of spawned processes, which can thrash the CPU or
    exhaust memory. Set `n_jobs=-1` on the `GridSearchCV` and `n_jobs=1` on the
    reducer itself.

