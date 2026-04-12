from sklearn.utils.estimator_checks import check_estimator

from eigenradiomics.reducers import WGCNAReducer

try:
    check_estimator(WGCNAReducer())
except SystemExit as e:
    print("Caught SystemExit:", e)
