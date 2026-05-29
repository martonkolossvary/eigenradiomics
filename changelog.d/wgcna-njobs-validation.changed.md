`WGCNAReducer` now validates `n_jobs` (must be `None` or a non-zero integer) and frees large intermediate adjacency/distance matrices during `fit` to reduce peak memory on wide feature tables.
