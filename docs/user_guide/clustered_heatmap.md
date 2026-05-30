# Clustered Heatmap

`plot_clustered_heatmap` renders a feature-by-feature **similarity heatmap**
ordered by hierarchical clustering, with an aligned left dendrogram and a
per-feature cluster colour strip. It is deliberately generic — bring your own
similarity matrix and, optionally, a cluster assignment, linkage, and ordering —
so it works for a WGCNA TOM or any other symmetric similarity.

## From a fitted reducer

Any reducer that produces [reduction artifacts](../reducers/index.md#reduction-artifacts)
feeds the heatmap directly:

```python
from eigenradiomics import WGCNAReducer, plot_clustered_heatmap

reducer = WGCNAReducer(soft_power="auto", min_module_size=20, store_tom=True).fit(X)
fig = plot_clustered_heatmap(
    reducer.get_reduction_artifacts(),   # similarity (TOM), linkage, cluster labels, order
    cmap="magma",
    vmin=0.0,
    vmax=1.0,
    below_cutoff_color="#050505",
    colorbar_label="Topological overlap",
)
fig.savefig("wgcna_tom.png", dpi=150)
```

![WGCNA TOM heatmap with dendrogram and module colour strip](../assets/figures/clustered_heatmap.png)

`store_tom=True` is required so the TOM similarity is available in the artifacts.

## Bring your own inputs

You don't need a reducer — pass any symmetric matrix and, optionally, a
per-feature cluster assignment, a SciPy linkage, and/or an explicit order:

```python
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from eigenradiomics import plot_clustered_heatmap

# `sim` is an (n, n) DataFrame of |correlation|, TOM, kernel similarity, ...
z = linkage(squareform(1 - sim.to_numpy(), checks=False), method="average")

fig = plot_clustered_heatmap(
    sim,
    linkage=z,                # left dendrogram + leaf ordering
    cluster_labels=labels,    # per-feature module/cluster -> left colour strip
)
```

If you omit `linkage`, rows are ordered by `cluster_labels` (or left as given);
omit `cluster_labels` and the colour strip is dropped.

!!! note "What's next"
    This is the core of the heatmap. Planned additions layer optional annotation
    tracks — a feature-family strip, a clinical-correlation panel, and a
    significance bar — onto the same figure, sourced from a `FeatureCatalog` and
    `RadiomicsDataset`.

## Key options

| Argument | Purpose |
|----------|---------|
| `similarity` | symmetric matrix / DataFrame, or a `ReductionArtifacts` |
| `cluster_labels` | per-feature label → colour strip and default ordering |
| `linkage` | SciPy linkage → dendrogram and leaf ordering |
| `order` | explicit feature ordering (overrides the linkage leaves) |
| `cluster_colors` | `{label: colour}` (otherwise an automatic palette) |
| `cmap`, `vmin`, `vmax`, `below_cutoff_color` | similarity colour scaling |
| `show_dendrogram`, `show_cluster_strip` | toggle the side panels |
| `labels` | `"auto"`, an explicit list, or `None` |

See the [API reference](../api/plotting.md) for the full signature.
