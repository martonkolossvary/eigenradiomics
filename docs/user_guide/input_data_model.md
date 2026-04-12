# Input Data Model

eigenradiomics accepts generic feature matrices with shape `n_samples x n_features`.

## Accepted Input Types

| Input type <div style="min-width: 180px;"></div> | Supported | Notes |
|:--|:--:|:--|
| `numpy.ndarray` | Yes | Synthetic feature names are generated internally (`feature_0`, `feature_1`, ...) |
| `pandas.DataFrame` | Yes | Column names are preserved and validated across `fit` / `transform` |

## Row and Column Semantics

- **Rows** correspond to samples, studies, patients, lesions, or any observation-level unit.
- **Columns** correspond to features.

This is the same orientation expected by scikit-learn and by the low-level PyWGCNA static functions used inside `WGCNAReducer`.

## Feature Identity Rules

When you fit on a DataFrame, the estimator stores the feature names it saw during `fit`.

During `transform`:

- missing columns are rejected
- unexpected extra columns are rejected
- the same columns in a different order are rejected

This is especially important for wide radiomics tables, where a column-order mismatch can produce plausible-looking but wrong transformed outputs.

!!! warning "Do not reorder columns between fit and transform"
    If your upstream feature engineering pipeline changes column order, reindex the new DataFrame back to the fitted feature order before calling `transform`.

## Interaction with sklearn Preprocessors

Many scikit-learn preprocessors emit `numpy.ndarray` outputs rather than `pandas.DataFrame` objects. That is fully compatible with eigenradiomics reducers.

When a reducer receives an ndarray after upstream preprocessing:

- synthetic feature names are generated internally
- transform-time validation falls back to feature order rather than original column labels

Use the same fitted sklearn pipeline for train and inference to preserve consistency. If you want explicit column-name validation at the reducer boundary, keep the data as a DataFrame until reduction or reconstruct a DataFrame with the intended feature names before fitting the reducer.

## [Pictologics](https://github.com/martonkolossvary/pictologics)-Derived Tables

[Pictologics](https://github.com/martonkolossvary/pictologics) produces wide feature tables where each column is a named radiomics feature. Those tables are a natural input for eigenradiomics because:

- they are already sample-by-feature matrices
- they often contain strong correlation structure
- preserving named columns matters for traceability and interpretability

Recommended pattern:

```python
X = pictologics_results_df
pipe.fit(X_train)
Y_new = pipe.transform(X_new[X_train.columns])
```

## Output Naming

All reducers produce reducer-scoped output names through `get_feature_names_out()`.

Examples:

- `wgcna_0`, `wgcna_1`, `wgcna_2`
- future reducers would follow the same pattern, e.g. `pca_0`, `nmf_3`