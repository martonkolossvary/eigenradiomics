from eigenradiomics._version import __version__
from eigenradiomics.artifacts import ReductionArtifacts
from eigenradiomics.batch_effects import (
    compute_batch_effects,
    plot_batch_effects,
    write_batch_effects_excel,
)
from eigenradiomics.catalog import FeatureCatalog
from eigenradiomics.dataset import RadiomicsDataset, StudyDesign
from eigenradiomics.ingest import (
    MergeResult,
    merge_tables,
    normalize_id_column,
    resolve_duplicates,
)
from eigenradiomics.plotting import Strip, plot_clustered_heatmap
from eigenradiomics.preprocessing import RadiomicsFeatureRemover, RadiomicsPrepTransformer
from eigenradiomics.reducers import BaseReducer, WGCNAReducer
from eigenradiomics.reproducibility import (
    compute_reproducibility,
    plot_reproducibility_histograms,
    write_reproducibility_excel,
)

__all__ = [
    "__version__",
    "BaseReducer",
    "ReductionArtifacts",
    "RadiomicsFeatureRemover",
    "WGCNAReducer",
    "RadiomicsPrepTransformer",
    "FeatureCatalog",
    "RadiomicsDataset",
    "StudyDesign",
    "MergeResult",
    "merge_tables",
    "normalize_id_column",
    "resolve_duplicates",
    "compute_reproducibility",
    "plot_reproducibility_histograms",
    "write_reproducibility_excel",
    "compute_batch_effects",
    "plot_batch_effects",
    "write_batch_effects_excel",
    "plot_clustered_heatmap",
    "Strip",
]
