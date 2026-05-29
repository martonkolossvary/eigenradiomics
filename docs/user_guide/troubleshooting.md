# Troubleshooting

Common issues and fixes when using **eigenradiomics**, grouped into
framework-level problems and WGCNA/PyWGCNA-specific ones.

---

## 1. Framework issues

### Installation & dependencies

| Problem | Solution |
|---|---|
| `pip install eigenradiomics` fails | Ensure Python ≥ 3.10 and a recent pip (`pip install --upgrade pip`), then retry. |
| `ModuleNotFoundError: No module named 'PyWGCNA'` | WGCNA is optional. Install the extra: `pip install 'eigenradiomics[wgcna]'`. |
| ComBat step is skipped | Install the ComBat extra: `pip install 'eigenradiomics[combat]'` (requires Python ≥ 3.11). |

### Sparse matrix input

```
TypeError: Sparse matrices are not supported.
```

Convert to a dense array first:

```python
X_dense = X_sparse.toarray()
reducer.fit(X_dense)
```

### Feature-name mismatch

```
ValueError: ... requires input features to be in the same order as during fit.
```

If you fit on a DataFrame, `transform` must receive the **same column names in
the same order**. Reindex the new data back to the fitted columns:

```python
Y = reducer.transform(X_new[reducer.feature_names_in_])
```

### Feature-count mismatch

```
ValueError: X has N features, but ... is expecting M features
```

`transform` must receive the same number of features as `fit`. This usually
means an upstream step (e.g. `VarianceThreshold`) selected different columns —
fit the whole pipeline together rather than steps in isolation.

### NotFittedError

Fit the pipeline before calling `predict`/`transform`:

```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eigenradiomics import WGCNAReducer

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("reduce", WGCNAReducer(soft_power=6, min_module_size=20)),
    ("model", Ridge()),
])
pipe.fit(X_train, y_train)
```

### Grid search is very slow

`GridSearchCV` refits the whole pipeline for every parameter combination × fold.
Parallelize the **search** (`GridSearchCV(n_jobs=-1)`) and keep the reducer
single-threaded (`WGCNAReducer(n_jobs=1)`) to avoid nested processes. See
[Scalability](scalability.md).

---

## 2. WGCNA-specific issues

### Automatic soft-power selection fails

```
ValueError: Automatic soft-power selection failed — no power reached the R² threshold.
```

**Cause:** no candidate power reached `r_squared_cut` for the scale-free
topology fit.

**Fixes:**

1. Lower the threshold: `WGCNAReducer(soft_power="auto", r_squared_cut=0.8)`.
2. Set the power explicitly: `WGCNAReducer(soft_power=6)`.
3. Inspect the table to choose a power yourself:

```python
reducer = WGCNAReducer(soft_power="auto", r_squared_cut=0.7).fit(X)
print(reducer.wgcna_get_soft_power_table())
```

### Too few samples or features

```
ValueError: WGCNAReducer requires n_samples >= 3
```

Correlation-based networks need enough samples to be meaningful. The hard
minimum is 3 samples / 3 features; **15+ samples** is recommended for stable
module detection.

### Only one module detected

**Likely causes:** `min_module_size` is too large for the data, or
`me_diss_threshold` merged everything together.

**Fixes:**

- Lower `min_module_size` (e.g. `20`).
- Lower `me_diss_threshold` (e.g. `0.15`) so fewer modules are merged.
- Increase `deep_split` (e.g. `3`) to cut the dendrogram more aggressively.

### Most features land in the grey module

The grey module holds features that don't fit any cluster. If almost everything
is grey, the network found little structure. Lower `min_module_size`, or set
`include_grey=True` to keep those features as a pseudo-module.

### High memory use

The Topological Overlap Matrix is `O(n_features²)`. Keep `store_tom=False` (the
default) so it isn't retained after `fit`, and pre-filter very wide tables (see
[Scalability](scalability.md)).

### Suppressing PyWGCNA console output

PyWGCNA prints progress to stdout. Silence it or route it to a file:

```python
reducer = WGCNAReducer(verbose=0)                       # suppress
reducer = WGCNAReducer(verbose=0, log_file="wgcna.log") # send to a file
```

---

## 3. PyWGCNA backend notes

### Hanging during `fit`

If `reducer.fit(X)` hangs, the input most likely contains `NaN`, `Inf`, or
all-zero columns, which can deadlock PyWGCNA's internal checks. Impute and
variance-filter first:

```python
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
# SimpleImputer(strategy="median") then VarianceThreshold(threshold=0.0) upstream
```

### Zero-variance feature warning

```
UserWarning: Module 'grey' contains 5 zero-variance feature(s)
```

eigenradiomics safely sets the scale of constant features to `1.0`, but they
carry no signal. Add a `VarianceThreshold` before the reducer to drop them.
