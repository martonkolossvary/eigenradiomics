from eigenradiomics._version import __version__
from eigenradiomics.artifacts import ReductionArtifacts
from eigenradiomics.batch_effects import (
    compute_batch_effects,
    plot_batch_effects,
    write_batch_effects_excel,
)
from eigenradiomics.catalog import FeatureCatalog
from eigenradiomics.clinical import (
    compute_clinical_correlations,
    compute_module_trait_associations,
    encode_clinical_series,
)
from eigenradiomics.dataset import RadiomicsDataset, StudyDesign
from eigenradiomics.ingest import (
    MergeResult,
    merge_tables,
    normalize_id_column,
    resolve_duplicates,
)
from eigenradiomics.pictologics import split_observer_tables
from eigenradiomics.plotting import Bar, CorrPanel, Strip, plot_clustered_heatmap
from eigenradiomics.preprocessing import (
    FeatureScoreSelector,
    RadiomicsFeatureRemover,
    RadiomicsPrepTransformer,
)
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
    "FeatureScoreSelector",
    "WGCNAReducer",
    "RadiomicsPrepTransformer",
    "FeatureCatalog",
    "RadiomicsDataset",
    "StudyDesign",
    "MergeResult",
    "merge_tables",
    "normalize_id_column",
    "resolve_duplicates",
    "split_observer_tables",
    "compute_reproducibility",
    "plot_reproducibility_histograms",
    "write_reproducibility_excel",
    "compute_batch_effects",
    "plot_batch_effects",
    "write_batch_effects_excel",
    "plot_clustered_heatmap",
    "Strip",
    "Bar",
    "CorrPanel",
    "compute_clinical_correlations",
    "compute_module_trait_associations",
    "encode_clinical_series",
]
