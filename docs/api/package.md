# Package API

`import eigenradiomics` re-exports the public objects below. Full signatures and
docstrings live on the per-area API pages linked in each row.

## Public exports

| Area | Objects | Reference |
|------|---------|-----------|
| Ingestion & datasets | `FeatureCatalog`, `RadiomicsDataset`, `StudyDesign`, `MergeResult`, `normalize_id_column`, `resolve_duplicates`, `merge_tables`, `split_observer_tables` | [Ingestion & Datasets](ingestion.md) |
| Preprocessing & selection | `RadiomicsPrepTransformer`, `RadiomicsFeatureRemover`, `FeatureScoreSelector`, `ComBatHarmonizer` | [Preprocessing](preprocessing.md) |
| Reproducibility QC | `compute_reproducibility`, `plot_reproducibility_histograms`, `write_reproducibility_excel` | [Reproducibility](reproducibility.md) |
| Batch-effect QC | `compute_batch_effects`, `plot_batch_effects`, `write_batch_effects_excel` | [Batch Effects](batch_effects.md) |
| Dimensionality reduction | `BaseReducer`, `WGCNAReducer`, `ReductionArtifacts` | [Reducers](reducers.md) |
| Visualization | `plot_clustered_heatmap`, `Strip`, `Bar`, `CorrPanel` | [Plotting](plotting.md) |
| Downstream stats | `compute_clinical_correlations`, `compute_module_trait_associations`, `encode_clinical_series` | [Clinical & Module-Trait](clinical.md) |
| Version | `__version__` | package version string |

```python
from eigenradiomics import (
    # ingestion & datasets
    FeatureCatalog, RadiomicsDataset, StudyDesign,
    normalize_id_column, resolve_duplicates, merge_tables, split_observer_tables,
    # preprocessing & selection
    RadiomicsPrepTransformer, RadiomicsFeatureRemover, FeatureScoreSelector, ComBatHarmonizer,
    # pre-analysis QC
    compute_reproducibility, compute_batch_effects,
    # dimensionality reduction
    WGCNAReducer, ReductionArtifacts,
    # visualization
    plot_clustered_heatmap, Strip, Bar, CorrPanel,
    # downstream statistical analysis
    compute_clinical_correlations, compute_module_trait_associations,
)
```

`RadiomicsDataset.from_pictologics(...)` loads a Pictologics export directly — see
[Load & align data](../user_guide/data_ingestion.md#loading-a-pictologics-export).
