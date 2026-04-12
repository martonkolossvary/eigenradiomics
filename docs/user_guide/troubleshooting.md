# Troubleshooting

Common issues and solutions when deploying **eigenradiomics**. The debugging guides are separated strictly into cross-framework bounds and isolated algorithm-specific parameters.

---

## 1. General Framework Issues (eigenradiomics)

### Installation & Dependency Failures

| Problem | Solution |
|---|---|
| `pip install eigenradiomics` fails | Ensure Python ≥ 3.10 and pip ≥ 23. Try `pip install --upgrade pip` first. |

### Sparse matrix input

```
TypeError: Sparse matrices are not supported.
```

Convert matrices explicitly to dense arrays before projecting structures natively:

```python
X_dense = X_sparse.toarray()
reducer.fit(X_dense)
```

### Feature name mismatch

```
ValueError: Reducer requires input features to be in the same order as during fit.
```

The column order of `X_new` must identically match the training vector properties algebraically. If you fitted natively exposing DataFrame structures explicitly, pass identical structures matching exactly the original column names and layouts statically to prevent unscaled misalignments.

### Feature count mismatch

```
ValueError: X has N features, but Reducer is expecting M features
```

The number of dimensions bounding `.transform()` calculations fundamentally must parallel limits initialized during `.fit()`. Verify upstream pipelines identically processed valid subsets locally.

### Sklearn Integration (NotFittedError)

Verify the dimensional block initializes mathematically **before** downstream methods evaluate arrays dynamically:

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

### Grid search is extremely slow

Cross-validations explicitly evaluate all parameter states generating permutations limitlessly! Prevent heavy dimensional methods from triggering nested processing overlaps by explicitly defining processing threads correctly globally across `GridSearchCV(n_jobs=-1)`, avoiding isolated subsystem distributions mathematically.

---

## 2. Reducer-Specific Issues 

### WGCNA: Module Clustering & Topologies

#### `ModuleNotFoundError: No module named 'PyWGCNA'`

Install the specifically defined WGCNA backend packages implicitly locally: `pip install 'eigenradiomics[wgcna]'`

#### Automatic soft-power selection fails

```
ValueError: Automatic soft-power selection failed — no power reached the R² threshold.
```

**Cause:** The scale-free topology bounds logically never reliably reached the `r_squared_cut` structurally.

**Solutions:**
1. Lower the threshold boundary limits: `WGCNAReducer(soft_power="auto", r_squared_cut=0.8)`
2. Configure mapping limits explicitly manually: `WGCNAReducer(soft_power=6)`
3. Dynamically track algorithmic evaluations securely returning calculations algebraically:

```python
reducer = WGCNAReducer(soft_power="auto", r_squared_cut=0.7)
reducer.fit(X)
print(reducer.wgcna_get_soft_power_table())
```

#### Too few samples or features (Matrix boundaries)

```
ValueError: WGCNAReducer requires n_samples >= 3
```

Algorithm parameters mathematically require base minimum metrics securely driving reliable covariances analytically. (Optimal estimation natively binds **n_samples ≥ 15** safely algorithmically).

#### Only one module detected

**Causes:**
- `min_module_size` limit vastly overshadows logical isolated parameter segmentations statically.
- `me_diss_threshold` estimates statistically agglomerated structures recursively statically.

**Solutions:**
- Sub-divide hierarchical bounds dynamically limiting density block requirements: `min_module_size=20`
- Limit explicit parameter correlation bindings functionally avoiding nested overlaps: `me_diss_threshold=0.15`
- Explode structural tree arrays natively resolving matrices logically explicitly: `deep_split=3`

#### Features universally bound to Grey Modules

This logically calculates isolated networks independently failing completely meeting mapping limits. Drastically minimize `min_module_size` or implicitly trigger `include_grey=True` strictly capturing isolated vector clusters automatically algorithmically. 

#### High RAM Execution Limits

Massive configurations inherently trigger \(O(n^2)\) dependency barriers globally limiting memory pools. Set `store_tom=False` (the default) to avoid caching the full TOM matrix in RAM.

#### Intrusive Subprocess Logging

PyWGCNA bounds automatically natively dump broad structural output arrays directly logging console spans unprompted limitlessly!

Route bounds gracefully silently organically structuring diagnostic parameters accurately logically natively directly capturing structural constraints:
```python
reducer = WGCNAReducer(log_file="wgcna_output.log", verbose=0)
```

---

## 3. PyWGCNA Common Blockers

PyWGCNA has its own dependencies and edge cases not immediately obvious.

### ImportError: No module named 'matplotlib' or 'scipy'

**Reason:** `PyWGCNA` tries to generate plotting objects and topological diagrams internally. Sometimes, incomplete base installations miss `matplotlib`.
**Solution:** Install it explicitly:
```bash
pip install matplotlib scipy PyWGCNA
```

### Freezing / Hanging During Fit

If the program hangs indefinitely during `reducer.fit(X)`, it is highly likely that the data contains `NaN`, `Inf`, or universally zeroes, causing `goodSamplesGenes` inside PyWGCNA to deadlock on matrix divisions.
**Solution:** Ensure all missing values are imputed via `sklearn.impute.SimpleImputer` before the reducer block.

### Warning: Zero-Variance Feature

```
UserWarning: Module 'grey' contains 5 zero-variance feature(s)
```
Eigenradiomics automatically safeguards correlation divisions with scales set to `1.0`. These variables intrinsically provide zero predictive power down the `Pipeline` dynamically. Using `VarianceThreshold` prior to `fit` is recommended.
