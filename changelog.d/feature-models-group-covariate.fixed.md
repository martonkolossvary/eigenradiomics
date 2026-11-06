`compute_feature_associations` raises a `ValueError` when one column is both the `groups` column and a covariate. Before, the working table got two columns with the same name.
