# Downstream Statistical Analysis

The reducer turns a wide feature matrix into a handful of **module eigengenes**.
This page shows how to take those eigengenes (and the reduction artifacts) into
statistical analysis: module–trait associations, QC-driven feature selection,
and leakage-safe modelling.

```mermaid
flowchart LR
    P[Pipeline<br/>prep · select · reduce] --> Y[Module eigengenes<br/>n_samples × n_modules]
    Y --> MT[compute_module_trait_associations<br/>r · p · FDR]
    Y --> M[Your model<br/>GroupKFold via dataset.groups]
    QC[(ICC / batch QC)] -.-> S[FeatureScoreSelector] -.-> P
```

## Get labeled eigengenes

`WGCNAReducer.transform` returns a NumPy array. To carry the **sample index** and
**eigengene names** (`wgcna_0`, `wgcna_1`, …) into downstream stats, opt into
scikit-learn's pandas output:

```python
reducer = WGCNAReducer(soft_power="auto", min_module_size=20, store_tom=True)
reducer.set_output(transform="pandas")          # eigengenes come back as a DataFrame
eigengenes = reducer.fit_transform(X)            # index = samples, columns = wgcna_*
```

`set_output(transform="pandas")` works on the whole `Pipeline` too, so the final
step's output keeps its labels.

## Module–trait relationships

`compute_module_trait_associations` correlates each module eigengene with each
clinical trait and reports the coefficient, its p-value, and a
Benjamini-Hochberg FDR across the table — the standard WGCNA *module–trait
relationship*. Mixed-type traits are encoded automatically.

```python
from eigenradiomics import compute_module_trait_associations

mtr = compute_module_trait_associations(
    eigengenes,                      # samples × modules
    dataset,                         # a RadiomicsDataset (or a traits DataFrame)
    ["Age", "Stage", "Sex", "Event"],
    method="spearman",
)
mtr["r"]       # modules × traits correlation matrix
mtr["p"]       # matching p-values
mtr["p_fdr"]   # Benjamini-Hochberg FDR
```

`mtr["r"]` (with `mtr["p_fdr"]` for significance) is a compact summary linking the
reduced space to outcomes.

## QC-driven feature selection in a Pipeline

Reproducibility and batch QC run *outside* a single-`X` fit (they need multiple
readers / a batch label). Compute the scores once, then drop weak features inside
the pipeline with `FeatureScoreSelector` — so selection is part of the fitted,
leakage-safe transform:

```python
from eigenradiomics import compute_reproducibility, FeatureScoreSelector, WGCNAReducer
from sklearn.pipeline import Pipeline

repro = compute_reproducibility([reader1, reader2])          # per-feature ICC
icc = repro["ICC"]                                            # has 'feature' + 'icc_2_1'

pipe = Pipeline([
    ("prep", RadiomicsPrepTransformer().set_output(transform="pandas")),
    ("reliable", FeatureScoreSelector(icc, threshold=0.80, score_column="icc_2_1")),
    ("reduce", WGCNAReducer(soft_power="auto", min_module_size=20)),
]).set_output(transform="pandas")
```

The same selector drops **batch-confounded** features — pass a batch-effect size
column with `keep="below"` (keep features whose effect is small).

## Feeding eigengenes into a model (leakage-safe CV)

A `RadiomicsDataset` carries the target and the grouping needed for leakage-safe
cross-validation. Use `dataset.groups` (e.g. `PatientID`) with `GroupKFold` so no
patient appears in both train and test:

```python
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.linear_model import LogisticRegression

X, y = dataset.to_pipeline_input()      # X = features, y from the study design
cv = GroupKFold(n_splits=5)
scores = cross_val_score(
    Pipeline([("reduce", WGCNAReducer(soft_power="auto", min_module_size=20)),
              ("clf", LogisticRegression())]),
    X, y, groups=dataset.groups, cv=cv,
)
```

Because the reducer fits its mapping on the training fold only and applies it to
the test fold, the reduction is part of the cross-validated estimate — not fit on
the whole cohort.

!!! note "Survival outcomes"
    A survival `StudyDesign` (`time` + `event`) makes `dataset.y()` return a
    `[time, event]` frame. eigenradiomics does not ship a survival model; hand the
    eigengenes + that frame to a Cox model (e.g. `lifelines` or
    `scikit-survival`), grouping by `dataset.groups` for validation.

See the [End-to-End Workflow](end_to_end.md) for the full pipeline in one script.
