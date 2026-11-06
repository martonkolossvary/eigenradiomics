# Best Practices & Pitfalls

A consolidated checklist for using eigenradiomics correctly end-to-end. Each item
links to the guide with the details.

## Workflow order

The steps are designed to run in this order; doing them out of order is the most
common source of trouble.

```mermaid
flowchart LR
    I[Ingest &amp; align] --> Q[QC: reproducibility + batch]
    Q --> P[Preprocess] --> S[Select reliable features] --> R[Reduce] --> D[Downstream stats]
```

1. **Ingest and align first** with [`from_pictologics`](data_ingestion.md#loading-a-pictologics-export) / `merge_tables`, so features and metadata stay index-aligned for everything downstream.
2. **Screen before you model.** Run [reproducibility](reproducibility.md) and [batch-effect](batch_effects.md) QC on the *raw* features — both need information (multiple readers, batch labels) that does not exist inside a single-`X` fit.
3. **Preprocess, then reduce.** Put `RadiomicsPrepTransformer` *before* the reducer, in one `Pipeline`. WGCNA uses correlations, so a change of feature units does not change its result. Winsorizing and Yeo-Johnson do change the correlations: they decrease the effect of outliers and skew. PCA and most other reducers also need the z-score step.

## Leakage safety

- **Fit on train only.** Every transformer learns its mapping in `fit` and replays it in `transform`. Inside cross-validation, put the *whole* pipeline (including the reducer) in the splitter so the reduction is fit per fold — never `fit_transform` the full cohort first.
- **Group by patient.** Use `dataset.groups` with `GroupKFold` so repeated scans of one patient don't straddle train/test. See [Downstream Analysis](downstream_analysis.md#feeding-eigengenes-into-a-model-leakage-safe-cv).
- **Keep labels through the pipeline.** Call `.set_output(transform="pandas")` so eigengenes keep their sample index and `wgcna_*` names for joining to traits/outcomes.

## Feature identity

- **Prefer DataFrames.** Name-based steps (`RadiomicsFeatureRemover`, `FeatureScoreSelector`, catalog selectors) need column names. Fitting on a DataFrame and transforming an equivalent array is supported (it warns), but a DataFrame end-to-end is safest.
- **The `config__feature_key` convention** is assumed throughout. For observer-paired reproducibility tables (`O1_…`, `O2_…`), pass `observer_prefixes` to `RadiomicsFeatureRemover` and use [`split_observer_tables`](data_ingestion.md#loading-a-pictologics-export).

## QC choices

- **ICC threshold.** ≥ 0.80 is a common "good reliability" cut; judge from the ICC estimate **and its CI vs the threshold**, not the p-value (the ICC p-value only tests whether subjects differ). Drop weak features with [`FeatureScoreSelector`](downstream_analysis.md#qc-driven-feature-selection-in-a-pipeline).
- **Bootstrap CIs need samples.** ICC CIs from only a handful of subjects are unreliable regardless of iterations.
- **ComBat is a sensitivity check, not a default.** `compute_batch_effects` reports how much a result *would* change under ComBat; only harmonize when a real, non-biological batch effect is shown — and do it leakage-safely with [`ComBatHarmonizer`](harmonization.md) (correction fit on train, applied to test).

## Reduction

- **`missing_strategy` choice:**
    - `"pairwise"` is ideal when missingness is low-to-moderate and features are missing at random (MAR): it computes exact pairwise correlations without distorting feature variances.
    - `"floor"` is recommended when missingness represents true biological absence (e.g., non-contrast scans lacking enhancement features): filling with minimum observed values preserves the physical ground truth.
    - `"mean"` / `"median"` or upstream `KNNImputer` / `IterativeImputer` are recommended for general imputation.
- **`orient_eigengenes` in predictive vs longitudinal studies:**
    - `"mean_correlation"` (default) aligns the eigengene positive sign with the overall average feature trajectory of that module, making positive eigengene changes intuitive in longitudinal and phenotype analyses.
    - `"hub_feature"` aligns the eigengene strictly with the single most central hub feature in the module.
- **`soft_power="auto"`** picks the lowest power that meets the scale-free criteria (R², mean connectivity, negative slope); if none does, it uses the best-R² power, warns, and sets `soft_power_criteria_met_` to False (`soft_power_fallback="raise"` stops instead); inspect `wgcna_plot_soft_power()` if modules look off.
- **Module merging** uses `merge_policy="auto"`: signed (`1 - r`) for signed and signed hybrid networks, so modules with opposite eigengenes stay apart.
- **`min_module_size`** controls granularity; too large yields all-grey (no modules → a `(n, 0)` output, with a warning). Lower it or set `include_grey=True`.
- **Constant features** can't enter the network (they're left unassigned with a warning) — drop them upstream.

## Figures

- The continuous maps are **colourblind-safe and greyscale-legible** (`magma` for similarity, `RdBu_r` for correlation). Categorical strips, module strips, and volcano points use the package's eight-colour `EIGEN_VIBRANT` palette. This palette is not a published colourblind-safe set. For a colourblind-safe set, pass explicit colours (for example, the Okabe-Ito colours). Beyond **eight** categorical levels the palette changes to `tab20`, which is **not** safe. See the [Clustered Heatmap](clustered_heatmap.md#accessibility) guide.

## Performance

- **Nested parallelism.** If a reducer's `n_jobs > 1` runs inside a `GridSearchCV`/`cross_validate` that is *also* parallel, you can oversubscribe cores. Parallelize at one level. See [Scalability](scalability.md).
- **Pre-filter wide tables.** WGCNA builds an \(O(m^2)\) similarity matrix; remove clearly-irrelevant features (by family, or by QC score) before reducing thousands of columns.
