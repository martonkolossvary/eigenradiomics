# Pipelines and Grid Search

eigenradiomics estimators are designed to work inside sklearn pipelines.

Use standard sklearn or project-local transformers ahead of the reducer.

## Basic Pipeline

```python
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eigenradiomics import WGCNAReducer

pipe = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
        ("scale", StandardScaler()),
        ("reduce", WGCNAReducer(soft_power=6, min_module_size=30)),
    ]
)
```

## Important Note on GridSearchCV

`GridSearchCV` requires either:

- a scoring function, or
- a downstream estimator with a `score()` method

Because eigenradiomics reducers are unsupervised transformers, a pipeline that ends at a reducer does **not** provide a default score.

!!! tip "Avoid nested parallelism"
    When tuning `WGCNAReducer` inside `GridSearchCV`, set `n_jobs=-1` on the
    search and `n_jobs=1` on the reducer. Nesting both multiplies the number of
    processes and can thrash the CPU or exhaust memory. See
    [Scalability](scalability.md).

## Strategies for unsupervised optimization

Because there is no target to score against, you must give `GridSearchCV`
something to optimize. There are two practical strategies for judging the quality
of an unsupervised reduction.

### Strategy 1: downstream supervised evaluation

The most common approach is to put a supervised model (e.g. logistic or ridge
regression) after the reducer in the same `Pipeline` and let prediction quality
judge the reduction. The grid then asks: did this parameter setting (like
`me_diss_threshold`) produce a subspace that improves the downstream model?

```python
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eigenradiomics import WGCNAReducer

pipe = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
        ("scale", StandardScaler()),
        ("reduce", WGCNAReducer(soft_power=6, min_module_size=20, verbose=0)),
        ("model", RidgeClassifier()),
    ]
)

# The grid automatically assesses the reducer based on final accuracy
search = GridSearchCV(
    pipe,
    {
        "reduce__deep_split": [1, 2, 3],
        "reduce__me_diss_threshold": [0.15, 0.2, 0.25],
        "model__alpha": [0.1, 1.0, 10.0],
    },
    scoring="accuracy",
    cv=3,
)

# Execute grid operations
search.fit(X_train, y_train)

# best_estimator_ is the full pipeline refit on all training data:
y_pred = search.predict(X_test)
```

### Strategy 2: intrinsic unsupervised scoring

To tune the reducer without labels, score the reduction itself with a custom
scorer. Useful intrinsic metrics include:

1. **Reconstruction error (MSE):** how faithfully `inverse_transform` recovers the original features.
2. **Explained variance:** how much of the original variance the reduced space retains (as in PCA/SVD).
3. **Silhouette score:** how cleanly the discovered modules separate.

Pairing such a scorer with the search lets it evaluate reductions without any
target `y`.

```python
from sklearn.metrics import mean_squared_error, make_scorer

# Define a function returning negative MSE (since GridSearchCV inherently maximizes scores)
def reconstruction_scorer(estimator, X, y=None):
    # Grab the fitted reducer step
    reducer = estimator.named_steps["reduce"]

    # Project, then reconstruct
    Y_subspace = reducer.transform(X)
    X_reconstructed = reducer.inverse_transform(Y_subspace)

    return -mean_squared_error(X, X_reconstructed)

unsupervised_search = GridSearchCV(
    pipe, 
    param_grid={"reduce__me_diss_threshold": [0.15, 0.2, 0.3]}, 
    scoring=reconstruction_scorer, 
    cv=3
)

# No y needed: the scorer measures reconstruction error
unsupervised_search.fit(X_train)

# Transform new data with the best-found parameters:
Y_optimal = unsupervised_search.transform(X_test)
```

### Advanced: multi-metric tracking

You can track several scores at once — for example a supervised metric *and*
reconstruction error. scikit-learn evaluates each metric per fold and averages
them separately; it does not combine them into one number. Use `refit="<key>"`
to choose which metric selects the final `best_estimator_`; the others are
logged in `cv_results_` for inspection:

```python
multi_metrics = {
    "supervised_accuracy": "accuracy",
    "reconstruction": reconstruction_scorer
}

robust_search = GridSearchCV(
    pipe,
    param_grid={"reduce__soft_power": [6, 8, 10]},
    scoring=multi_metrics,
    refit="supervised_accuracy", # Target defining final best_estimator_ state
    cv=3,
)

# Execute operations
robust_search.fit(X_train, y_train)

# Report both metrics at the selected parameter setting
best_index = robust_search.best_index_
print(f"Optimal Test Accuracy: {robust_search.cv_results_['mean_test_supervised_accuracy'][best_index]}")
print(f"Optimal Reconstruction Loss: {robust_search.cv_results_['mean_test_reconstruction'][best_index]}")

# predict() uses the model refit on supervised_accuracy:
y_test_pred = robust_search.predict(X_test)
```

## Parameter Access

Upstream components and the reducer expose constructor parameters through `get_params()` and `set_params()`, so standard sklearn parameter naming works:

- `impute__strategy`
- `var__threshold`
- `scale__with_mean`
- `reduce__soft_power`
- `reduce__deep_split`
- `model__alpha`

## Custom Preprocessing

Any sklearn-compatible transformer can be inserted before the reducer.

```python
from sklearn.pipeline import Pipeline

from my_project.transformers import CorrelationPruner

from eigenradiomics import WGCNAReducer

pipe = Pipeline(
    [
        ("corr", CorrelationPruner(threshold=0.9, method="spearman")),
        ("reduce", WGCNAReducer(soft_power=6, min_module_size=20)),
    ]
)
```

## Reproducibility Advice

- pin upstream hyperparameters explicitly
- persist the fitted pipeline if you need identical deployment behavior
- keep DataFrame columns consistent between train and inference