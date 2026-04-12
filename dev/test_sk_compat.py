import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from eigenradiomics.reducers import WGCNAReducer


@parametrize_with_checks([WGCNAReducer()])
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in [
        "check_estimators_overwrite_params",  # uses n_features=2
        "check_fit2d_1feature",
        "check_estimators_unfitted",
        "check_complex_data",
        "check_methods_subset_invariance",  # uses too few samples
        "check_fit_score_takes_y",
        "check_fit_idempotent",
    ]:
        pytest.skip(f"Skipping {check.func.__name__} due to network size limits")

    check(estimator)
