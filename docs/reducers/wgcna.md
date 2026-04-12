# WGCNA Reducer (Weighted Gene Co-expression Network Analysis)

`WGCNAReducer` operates as an unsupervised manifold clustering process originally formulated for gene expression clusters, highly optimized functionally via the `PyWGCNA` backbone. It aggregates tightly correlated variables spanning very wide feature spaces into isolated blocks grouped by correlation strength, then maps representations deterministically back against a constrained latent topology.

## Mathematical Formulation

Unlike typical global factorization procedures (i.e., Principal Component Analysis) which evaluate variance globally, WGCNA fundamentally isolates structural dependencies bottom-up leveraging deterministic network distance bounds.

### 1. Adjacency Matrix

Initial variable clusters are extracted by estimating continuous topological distances between inputs across the feature plane. A power function soft-thresholds the Pearson correlation metric \(\text{corr}(x_i, x_j)\), driving weak connections toward zero and enhancing significant overlaps asymptotically.

\[
A_{ij} = \lvert \text{corr}(x_{i}, x_{j}) \rvert ^ {\beta}
\]

Where \(\beta\) (`soft_power`) represents the scaling amplification factor maximizing scale-free topological dependencies. 

### 2. Topological Overlap Matrix (TOM)

To theoretically protect clustering boundaries against localized noise spikes, the Adjacency mapping transfers to evaluating shared neighbourhood dependencies via the Topological Overlap Matrix.

\[
\text{TOM}_{ij} = \frac{\sum_u A_{iu} \cdot A_{uj} + A_{ij}}{\min(K_i, K_j) + 1 - A_{ij}}
\]

Dissimilarity matrices are constructed as \(1 - \text{TOM}_{ij}\). Hierarchical clustering is then applied to this dissimilarity, iteratively grouping the most tightly connected features into modules (bounded by `min_module_size`).

### 3. Latent Vector Mapping (Module Eigengenes)

Extracted features are consolidated mathematically for stable predictive transformation mapping against unseen validation vectors. 

Every identified module defines a Module Eigengene (\(Y_{\text{mod}}\)), equivalent to the first principal component (PC1) derived via Singular Value Decomposition across the centered and scaled features belonging **only** to that module.

\[
X^{(\text{scaled})} = U \Sigma V^T
\]

\[
Y_{\text{mod}} = X^{(\text{scaled})} \cdot V_1
\]

The loading vector \(V_1\) is stored during `.fit()` and reused during `.transform()` to project unseen data deterministically without re-running the clustering algorithm. This is also the basis for `.inverse_transform()`, which reconstructs an approximation of the original feature space from the reduced representation.

---

## Python Interface

The transformation output condenses wide features strictly into dimensional matrix boundaries equivalent to the number of discovered clusters functionally, resolving unscaled dependencies reliably over arbitrary vectors.

### Training Execution

```python
from eigenradiomics import WGCNAReducer

reducer = WGCNAReducer(
    soft_power="auto",
    min_module_size=30,    # Target parameter volume limits
    me_diss_threshold=0.2, # Eigengene merging dissimilarity targets
    deep_split=2,          # Spatial map density bounds
)

Y_train = reducer.fit_transform(X_train)
```

### Unseen Data Projections (No-Leakage Constraints)

Fitted topologies organically project representations targeting multi-center distributions dynamically. Eigenradiomics fundamentally blocks arbitrary training data leakages natively.

```python
# Deterministic static mapping projections referencing only historical singular models statically:
Y_test = reducer.transform(X_test)

# Mathematical matrix expansions calculating localized feature reconstructions statistically scaling backward explicitly:
X_reconstructed = reducer.inverse_transform(Y_test)
```

---

## Diagnostics Engine

The model natively inherits visualization utilities exposing deep state attributes:

- `wgcna_get_module_assignments()`
- `wgcna_get_module_sizes()`
- `wgcna_get_soft_power_table()`
- `wgcna_plot_soft_power()`
- `wgcna_plot_dendrogram()`

### Plotting Demonstrations

The diagnostic scale-free topological index verifies dependency fits dynamically (R²) charting against the algorithmic connectivity bounds. The explicit power targets are highlighted universally via vertical boundaries.

```python
fig = reducer.wgcna_plot_soft_power(figsize=(10, 5))
fig.savefig("soft_power.png", dpi=150)
```

Concurrently, classical dendrogram dependencies plot localized hierarchical clusters with matching explicit sub-module tracking colors dynamically across the array topology.

```python
fig = reducer.wgcna_plot_dendrogram(figsize=(12, 4))
fig.savefig("dendrogram.png", dpi=150)
```

## Parameter Guide

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `soft_power` | `"auto"` | Power variable driving the Adjacency Matrix bounding calculations. Native string `"auto"` estimates bounds organically globally; integer limits deterministic pipelines unconditionally. |
| `r_squared_cut` | `0.9` | Mathematical R² limit bounds resolving topological scale-free estimates automatically natively. |
| `min_module_size` | `50` | Volume floor boundaries blocking highly diminutive micro-clusters natively globally. Highly dependent on initial matrix width spans natively. |
| `me_diss_threshold` | `0.2` | Core structural grouping variables binding independent eigengenes statistically. |
| `store_tom` | `False` | Resolves intense deep variable tree mappings explicitly against RAM limits natively (O(n²)). |

## References

- Langfelder, P., & Horvath, S. (2008). **WGCNA: an R package for weighted correlation network analysis**. *BMC Bioinformatics*, 9, 559. [PubMed](https://pubmed.ncbi.nlm.nih.gov/19114008/)
- Rezaie, N., et al. (2023). **PyWGCNA: A Python package for weighted gene co-expression network analysis**. *Bioinformatics*, 39(7). [PubMed](https://pubmed.ncbi.nlm.nih.gov/37399090/)

