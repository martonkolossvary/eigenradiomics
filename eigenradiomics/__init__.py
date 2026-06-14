from eigenradiomics._version import __version__
from eigenradiomics.analysis import (
    compute_group_enrichment,
    compute_module_membership,
    identify_hub_features,
)
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
from eigenradiomics.feature_models import (
    FeatureAssociationResult,
    compute_feature_associations,
    plot_volcano,
)
from eigenradiomics.ingest import (
    MergeResult,
    merge_tables,
    normalize_id_column,
    resolve_duplicates,
)
from eigenradiomics.pictologics import split_observer_tables
from eigenradiomics.plotting import (
    Bar,
    CorrPanel,
    Strip,
    plot_batch_distributions,
    plot_clustered_heatmap,
    plot_eigengene_profiles,
    plot_hub_significance,
)
from eigenradiomics.preprocessing import (
    ComBatHarmonizer,
    FeatureScoreSelector,
    RadiomicsFeatureRemover,
    RadiomicsPrepTransformer,
)
from eigenradiomics.reducers import BaseReducer, PCAReducer, SparsePCAReducer, WGCNAReducer
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
    "ComBatHarmonizer",
    "WGCNAReducer",
    "PCAReducer",
    "SparsePCAReducer",
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
    "plot_hub_significance",
    "plot_eigengene_profiles",
    "plot_batch_distributions",
    "compute_clinical_correlations",
    "compute_module_trait_associations",
    "encode_clinical_series",
    "compute_feature_associations",
    "FeatureAssociationResult",
    "plot_volcano",
    "compute_module_membership",
    "identify_hub_features",
    "compute_group_enrichment",
]
