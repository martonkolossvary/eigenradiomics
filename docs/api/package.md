# Package API

`import eigenradiomics` re-exports the main public objects. Full signatures and
docstrings live on the per-area API pages linked below.

## Public exports

| Object | Area | Reference |
|--------|------|-----------|
| `FeatureCatalog` | Ingestion | [Ingestion & Datasets](ingestion.md) |
| `RadiomicsDataset`, `StudyDesign` | Ingestion | [Ingestion & Datasets](ingestion.md) |
| `normalize_id_column`, `resolve_duplicates`, `merge_tables`, `MergeResult` | Ingestion | [Ingestion & Datasets](ingestion.md) |
| `RadiomicsFeatureRemover`, `RadiomicsPrepTransformer` | Preprocessing | [Preprocessing](preprocessing.md) |
| `compute_reproducibility`, `plot_reproducibility_histograms`, `write_reproducibility_excel` | Reproducibility | [Reproducibility](reproducibility.md) |
| `compute_batch_effects`, `plot_batch_effects`, `write_batch_effects_excel` | Batch effects | [Batch Effects](batch_effects.md) |
| `BaseReducer`, `WGCNAReducer` | Reducers | [Reducers](reducers.md) |
| `__version__` | — | package version string |

```python
from eigenradiomics import (
    # ingestion
    FeatureCatalog, RadiomicsDataset, StudyDesign,
    normalize_id_column, resolve_duplicates, merge_tables,
    # preprocessing
    RadiomicsFeatureRemover, RadiomicsPrepTransformer,
    # pre-analysis statistics
    compute_reproducibility, compute_batch_effects,
    # dimensionality reduction
    WGCNAReducer,
)
```
