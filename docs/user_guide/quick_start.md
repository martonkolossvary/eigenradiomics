# Quick Start

This guide gets you from a wide feature matrix to a reduced representation with minimal setup.

## Minimal Example

```python
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eigenradiomics import WGCNAReducer

# X is a samples x features table.
X = pd.read_csv("radiomics_features.csv")

pipe = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
        ("scale", StandardScaler()),
        ("reduce", WGCNAReducer(soft_power="auto", min_module_size=30)),
    ]
)

Y = pipe.fit_transform(X)
print(Y.shape)
```

The output `Y` is an \\(n \\times k\\) matrix with `k << m` and reducer-specific output names such as `wgcna_0`, `wgcna_1`, and so on.

Upstream preprocessing stays in the same pipeline.

## Train / Test Pattern

The core use case is to fit on training data and later apply the same fitted reducer to unseen samples.

```python
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

pipe.fit(X_train)
Y_train = pipe.transform(X_train)
Y_test = pipe.transform(X_test)
```

If you fit the reducer directly on a pandas DataFrame, later transforms must use the same feature names in the same order. After upstream sklearn steps convert the data to an ndarray, validation falls back to feature order.

## Inspecting the WGCNA Fit

```python
reducer = pipe.named_steps["reduce"]

module_sizes = reducer.wgcna_get_module_sizes()
assignments = reducer.wgcna_get_module_assignments()
soft_power_table = reducer.wgcna_get_soft_power_table()
```

These diagnostics make it easier to decide whether to keep the auto-selected soft power or refit with an explicit value.

## Next Steps

- Read [Input Data Model](input_data_model.md) to understand accepted matrix formats.
- Read [Pipelines and Grid Search](pipelines_and_grid_search.md) before tuning unsupervised reducers inside sklearn model selection workflows.