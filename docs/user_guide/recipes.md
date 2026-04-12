# Recipes

This page collects common usage patterns for eigenradiomics.

## Reduce a [Pictologics](https://github.com/martonkolossvary/pictologics) Feature Table

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

## Strict Data Leakage Prevention

Always invoke `.fit` entirely on explicit model boundaries preventing structural cross-contamination natively. 

```python
pipe.fit(X_train)
Y_train = pipe.transform(X_train)
Y_test = pipe.transform(X_test[X_train.columns])
```

## Mathematical Reconstruction (Inverse Mapping)

Once compressed organically, methods supporting dimensional loading topologies (e.g. `WGCNAReducer` natively via PC1) permit symmetric structural reconstructions natively.

```python
Y_test = pipe.transform(X_test)

# Expand representations returning approximate original topological layouts identically:
X_test_reconstructed = pipe.named_steps["reduce"].inverse_transform(Y_test)
```

## Keep the Diagnostics

You natively retain deep internal parameter attributes generated dynamically during `.fit()`.

```python
reducer = pipe.named_steps["reduce"]

# Standard text topology metrics natively tracking parameters:
module_sizes = reducer.wgcna_get_module_sizes()
module_assignments = reducer.wgcna_get_module_assignments()

# Generate visual diagnostic renderings internally testing scale-free topological index:
fig_sp = reducer.wgcna_plot_soft_power(figsize=(10, 5))
fig_sp.savefig("soft_power.png")

# Native module color allocations against structural hierarchical distances symmetrically:
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