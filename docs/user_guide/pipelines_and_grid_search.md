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

## Strategies for Unsupervised Optimization

Optimizing unsupervised transformers isolated inside generic searches requires explicit structuring since there are no ground-truth targets to validate against natively. There are two primary strategies for evaluating the goodness-of-fit of reductions cleanly.

### Strategy 1: Downstream Supervised Evaluation

The most pragmatic way to optimize an unsupervised reduction method is to couple it inside a `Pipeline` directly connected to a supervised predictive model (e.g., Logistic Regression or Ridge Regression). 

Rather than judging the reducer purely on matrix reconstructions mathematically, `GridSearchCV` evaluates whether a specific parameter state (like `me_diss_threshold`) generated a feature subspace that statistically improved clinical prediction metrics.

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

# The search automatically binds 'best_estimator_' dynamically,
# directly predicting validations seamlessly natively:
y_pred = search.predict(X_test)
```

### Strategy 2: Intrinsic Unsupervised Scoring (Goodness-of-Fit)

To optimize the reducer entirely independent of supervised labels, construct internal custom `make_scorer` configurations predicting mathematical limits natively. Commonly adapted unsupervised metric families include:

1. **Reconstruction Error (MSE):** Measures algorithmic loss leveraging `inverse_transform`. Excellent for methods calculating how faithfully the constrained topologies replicate original noise signatures organically.
2. **Variance Retained (Explained Variance):** Evaluates matrix distributions preserving the mathematical footprints dominating variance arrays cleanly after reduction (common across SVD / PCA estimations natively).
3. **Internal Node Cohesion (Silhouette Score):** Numerically determines if internal grouping parameters (like WGCNA clustering bounds) accurately generated tight dimensional populations strongly decoupled optimally from adjacent structures mathematically.

By pairing `make_scorer` against one of these topologies (e.g. measuring reconstruction error dynamically), the internal search natively evaluates reductions cleanly without relying natively on explicit target distributions `y`.

```python
from sklearn.metrics import mean_squared_error, make_scorer

# Define a function returning negative MSE (since GridSearchCV inherently maximizes scores)
def reconstruction_scorer(estimator, X, y=None):
    # Fetch the standalone reduction steps natively
    reducer = estimator.named_steps["reduce"]
    
    # Calculate dimensional projection representations
    Y_subspace = reducer.transform(X)
    X_reconstructed = reducer.inverse_transform(Y_subspace)
    
    return -mean_squared_error(X, X_reconstructed)

unsupervised_search = GridSearchCV(
    pipe, 
    param_grid={"reduce__me_diss_threshold": [0.15, 0.2, 0.3]}, 
    scoring=reconstruction_scorer, 
    cv=3
)

# Execute grid mapping natively against raw vectors alone identically
unsupervised_search.fit(X_train)

# Transform dynamically exploiting optimally grouped cluster dependencies reliably:
Y_optimal = unsupervised_search.transform(X_test)
```

### Advanced: Multi-Metric Tracking

For highly rigorous diagnostic pipelines, you may track internal mathematical degradation matrices *concurrently* alongside standard downstream clinical predictions dynamically!

Provide a mapping dictionary defining arbitrary scoring targets implicitly. Note that scikit-learn evaluates each specific validation fold entirely independently calculating simultaneous boundaries, eventually averaging every metric distinctly (`mean_test_score`). It does **not** dynamically aggregate multiple criteria into a single global combination variable blindly!

You must structurally instruct the algorithmic loop specifically through the `refit="key"` parameter dictating explicitly *which* distinct isolated evaluation mathematically defines algorithmic convergence globally locating optimal estimator instances securely. The orthogonal configurations remaining are cleanly logged statistically for your external analytic review:

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

# Print concurrent tracking performance outputs locally structurally mapping iterations
best_index = robust_search.best_index_
print(f"Optimal Test Accuracy: {robust_search.cv_results_['mean_test_supervised_accuracy'][best_index]}")
print(f"Optimal Reconstruction Loss: {robust_search.cv_results_['mean_test_reconstruction'][best_index]}")

# Operations immediately deploy tracking logic maximizing 'supervised_accuracy' entirely logically:
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