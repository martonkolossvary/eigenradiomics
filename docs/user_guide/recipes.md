# Cookbook & Recipes

Focused snippets that go beyond the basics. For the happy path see the
[Quick Start](quick_start.md); for leakage-safe cross-validation and the
recommended workflow order see [Best Practices](best_practices.md).

---

## Longitudinal Radiomics & Response Assessment

When analyzing pre- and post-treatment imaging, `transform_paired` calculates trajectory metrics in the reduced latent space while strictly locking the reduction parameters learned at baseline:

```python
import pandas as pd
from eigenradiomics import WGCNAReducer

# Load baseline (pre) and follow-up (post) radiomics tables
X_pre = pd.read_csv("baseline_features.csv", index_col="PatientID")
X_post = pd.read_csv("followup_features.csv", index_col="PatientID")

# 1. Fit WGCNA reducer on baseline scans
reducer = WGCNAReducer(
    soft_power="auto",
    min_module_size=20,
    orient_eigengenes="mean_correlation",
)
reducer.fit(X_pre)

# 2. Compute paired trajectory metrics
paired = reducer.transform_paired(
    X_pre=X_pre,
    X_post=X_post,
    relative=True,           # Compute Δ / (|baseline| + ε)
    compute_distance=True,   # Compute ||Δ||_2 Euclidean trajectory length
    align_indices=True,      # Match patients present in both scans
)

# Access individual result attributes (pandas DataFrames/Series)
print(paired.delta.head())              # Latent shifts per module
print(paired.trajectory_distance.head()) # Overall magnitude of radiomic change
print(paired.effect_size)                # Standardized Cohen's d per module

# Merge into a single wide DataFrame for downstream modeling:
longitudinal_df = paired.to_dataframe(prefix_mode="suffix")
print(longitudinal_df.columns)
# -> ['wgcna_0_baseline', 'wgcna_0_followup', 'wgcna_0_delta', 'wgcna_0_relative_change', 'trajectory_distance']
```

---

## Handling Incomplete Feature Tables & Missingness

When anatomical structures or sub-threshold regions result in missing values (NaNs), choose between native reducer missingness or upstream multivariate imputation:

### Option A: Native Pairwise WGCNA

Avoids distorting correlation topologies by computing pairwise-complete associations directly:

```python
from eigenradiomics import RadiomicsPrepTransformer, WGCNAReducer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("prep", RadiomicsPrepTransformer()),  # Cleans distribution, preserves NaNs
    ("wgcna", WGCNAReducer(
        missing_strategy="pairwise",       # Pairwise complete correlation
        orient_eigengenes="mean_correlation",
    )),
])

Y = pipeline.fit_transform(X_with_nans)
```

### Option B: MICE / KNN in a Leakage-Safe Pipeline

For reducers requiring full dense matrices (e.g. `PCAReducer`, `UMAPReducer`):

```python
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from eigenradiomics import PCAReducer, RadiomicsPrepTransformer

pipeline = Pipeline([
    ("prep", RadiomicsPrepTransformer()),
    ("impute", KNNImputer(n_neighbors=5)),
    ("pca", PCAReducer(n_components=10)),
])

Y = pipeline.fit_transform(X_with_nans)
```

---

## Catalog-Driven Module Functional Profiling

Interpret discovered modules against standard IBSI feature families and filter configurations using Over-Representation Analysis (ORA):

```python
from eigenradiomics import FeatureCatalog, WGCNAReducer

# 1. Fit WGCNA reducer
reducer = WGCNAReducer(soft_power=6, min_module_size=20)
reducer.fit(X_df)

# 2. Query enrichment across IBSI catalog categories
catalog = FeatureCatalog.from_preset("ibsi")
enrichment = reducer.get_catalog_enrichment(
    catalog,
    columns=["family", "family_group"],
    method="fisher",         # Fisher's Exact test
    alternative="greater",   # Over-representation test
)

# View top enriched module-family pairs
significant_enrichment = enrichment[enrichment["fdr_q_value"] < 0.05]
print(significant_enrichment[["catalog_column", "cluster", "group", "n_overlap", "odds_ratio", "fdr_q_value"]])
```

---

## Compose with Stock scikit-learn Preprocessors

`WGCNAReducer` is a normal transformer, so it slots in after any scikit-learn preprocessing:

```python
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from eigenradiomics import WGCNAReducer

X = pd.read_csv("pictologics_features.csv")

pipe = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
        ("scale", StandardScaler()),
        ("reduce", WGCNAReducer(soft_power="auto", min_module_size=30)),
    ]
)

X_reduced = pipe.fit_transform(X)
```

---

## Reconstruct Features (Inverse Transform)

```python
Y_test = pipe.transform(X_test)

# Approximately reconstruct the original features:
X_test_reconstructed = pipe.named_steps["reduce"].inverse_transform(Y_test)
```

---

## Extracting Multiple Eigengenes per Module

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from eigenradiomics import WGCNAReducer

pipe = Pipeline(
    [
        ("scale", StandardScaler()),
        ("reduce", WGCNAReducer(soft_power=6, min_module_size=20, n_module_components=2)),
    ]
)

Y_train = pipe.fit_transform(X_train)
Y_test = pipe.transform(X_test)

reducer = pipe.named_steps["reduce"]
importances = reducer.wgcna_get_feature_importances()
print(importances["blue"].head())
```