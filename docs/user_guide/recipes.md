# Cookbook & Recipes

Focused snippets that go beyond the basics. For the happy path see the
[Quick Start](quick_start.md); for leakage-safe cross-validation and the
recommended workflow order see [Best Practices](best_practices.md).

## Compose with stock scikit-learn preprocessors

`WGCNAReducer` is a normal transformer, so it slots in after any scikit-learn
preprocessing — useful when you want generic imputation/variance filtering
instead of (or before) the radiomics-specific `RadiomicsPrepTransformer`:

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

## Reconstruct features (inverse transform)

Reducers that store loadings (e.g. `WGCNAReducer`) can approximately reconstruct
the original feature space from the reduced representation — handy for
reconstruction-error diagnostics.

```python
Y_test = pipe.transform(X_test)

# Approximately reconstruct the original features:
X_test_reconstructed = pipe.named_steps["reduce"].inverse_transform(Y_test)
```

## Keep the diagnostics

The reducer retains its fitted internals, so you can inspect or visualize the
model after `.fit()`.

```python
reducer = pipe.named_steps["reduce"]

# Tabular diagnostics:
module_sizes = reducer.wgcna_get_module_sizes()
module_assignments = reducer.wgcna_get_module_assignments()

# Soft-power diagnostic plot:
fig_sp = reducer.wgcna_plot_soft_power(figsize=(10, 5))
fig_sp.savefig("soft_power.png")

# Feature dendrogram with module colours:
fig_dendro = reducer.wgcna_plot_dendrogram(figsize=(12, 4))
fig_dendro.savefig("dendrogram.png")
```

## Work with numpy Arrays

```python
import numpy as np

X = np.random.default_rng(42).normal(size=(40, 500))
Y = WGCNAReducer(soft_power=6, min_module_size=25).fit_transform(X)
```

When using `numpy.ndarray`, synthetic feature names are generated internally, so later transforms must preserve the same column order even though no explicit names are attached.

## Use a Custom Transformer Upstream

```python
from sklearn.pipeline import Pipeline

from my_project.transformers import CorrelationPruner

from eigenradiomics import WGCNAReducer

pipe = Pipeline(
    [
        ("corr", CorrelationPruner(threshold=0.9, method="kendall")),
        ("reduce", WGCNAReducer(soft_power=6, min_module_size=20)),
    ]
)
```

This is useful for domain-specific or project-local preprocessing.

## Extracting Multiple Eigengenes per Module

If a single principal component (PC1) explains too little variance of a module's features, you can capture richer variations by extracting multiple principal components (eigengenes) using `n_module_components`:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from eigenradiomics import WGCNAReducer

pipe = Pipeline(
    [
        ("scale", StandardScaler()),
        # Extract the top 2 principal components per module
        ("reduce", WGCNAReducer(soft_power=6, min_module_size=20, n_module_components=2)),
    ]
)

# Project training and testing data: the output dimensions will be sum(k_i) across modules
Y_train = pipe.fit_transform(X_train)
Y_test = pipe.transform(X_test)

# The total number of columns Y will be n_modules * n_module_components (capped by module sizes)
print(f"Total reduced components: {Y_train.shape[1]}")
```

When calling `wgcna_get_feature_importances()`, the framework automatically uses the **L2 Euclidean Norm** of the SVD loadings across the selected components to reflect each feature's overall contribution:

```python
reducer = pipe.named_steps["reduce"]
importances = reducer.wgcna_get_feature_importances()

# Dataframe schema: ["feature", "loading", "importance"]
# "loading" corresponds to the primary (first) component,
# while "importance" represents the combined weight across both components.
print(importances["blue"].head())
```