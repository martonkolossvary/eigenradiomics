# Changelog

<!-- managed by towncrier — do not edit manually -->

## 0.1.0 (2026-04-12)

Project initiation.

- sklearn-compatible reducer framework with abstract `BaseReducer` base class
- `WGCNAReducer`: module-based dimensionality reduction using PyWGCNA network
  construction and stored SVD loadings for transform-time eigengene projection
- Thread-safe PyWGCNA output capture with fd-level redirect
- Sparse matrix rejection, NaN and zero-variance guards
- Joblib-parallel module loading computation and transform (`n_jobs`, `random_state`)
- Inverse transform via SVD loadings
- Interpretability utilities (`wgcna_get_feature_importances`, diagnostic plots)
- Full parameter validation at fit time
- GitHub Actions CI/CD (Python 3.10–3.12 matrix, ruff, mypy, pytest + Codecov, PyPI publish)
- MkDocs documentation with user guide, API reference, and generated quality report
- Towncrier changelog management
- 100% test coverage (105 tests)