`RadiomicsPrepTransformer` now accepts `winsor_lower=None` and `winsor_upper=None` to turn clipping off. Quantiles 0 and 1 still clip new data to the training range. Invalid quantile pairs now raise.
