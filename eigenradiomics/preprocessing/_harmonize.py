"""Leakage-safe ComBat harmonization as a scikit-learn transformer.

`inmoose.pycombat_norm` is transductive — it corrects a whole matrix at once and
returns only the corrected data. To use ComBat in a `Pipeline` / cross-validation
we instead recover its fitted parameters on the training data (grand mean, pooled
variance, and per-batch empirical-Bayes additive/multiplicative effects) via
inmoose's own estimation helpers, store them, and replay the per-batch adjustment
on new data. Samples whose batch was seen during ``fit`` are corrected with those
stored parameters; no test data ever touches the estimation.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin

from eigenradiomics._utils import check_constant_features, validate_estimator_input


def _import_combat() -> tuple[Any, Any, Any, Any, Any]:
    """Import inmoose's ComBat estimation helpers (optional dependency)."""
    try:
        from inmoose.pycombat.covariates import make_design_matrix
        from inmoose.pycombat.pycombat_norm import (
            calculate_mean_var,
            calculate_stand_mean,
            fit_model,
            standardise_data,
        )
    except ImportError as exc:
        raise ImportError(
            "ComBatHarmonizer requires the optional 'inmoose' dependency. "
            "Install it with `pip install inmoose` (or the package's 'combat' extra)."
        ) from exc
    return (
        make_design_matrix,
        calculate_mean_var,
        calculate_stand_mean,
        standardise_data,
        fit_model,
    )


class ComBatHarmonizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Leakage-safe ComBat batch harmonization.

    Removes additive and multiplicative batch (e.g. scanner/center) effects while
    optionally preserving biological covariates. Unlike running ComBat on a whole
    cohort, this estimates the correction on the ``fit`` data only and replays it
    on ``transform`` data, so it composes safely inside a scikit-learn
    ``Pipeline`` / cross-validation.

    Parameters
    ----------
    reference_batch : optional
        Batch label to use as the reference; its samples are left unchanged.
    parametric : bool
        Parametric empirical-Bayes priors (default) vs non-parametric.
    on_unseen_batch : {"passthrough", "error"}
        Behaviour for a transform-time batch not seen during ``fit``: pass the
        samples through unchanged with a warning, or raise.

    Notes
    -----
    ``batch`` (required) and ``covariates`` (optional) are passed as keyword
    arguments to :meth:`fit` / :meth:`transform`; scikit-learn metadata routing
    (``set_fit_request`` / ``set_transform_request``) lets them flow through a
    ``Pipeline``. When ``X`` is a DataFrame, a ``batch`` Series and a
    ``covariates`` DataFrame align to ``X`` by index; arrays are positional.
    Missing batch labels raise; a batch that was not seen during ``fit`` follows
    ``on_unseen_batch``. Covariates must be a **numeric** model matrix (encode
    categoricals first) with no missing values, and ``transform`` needs every
    fitted covariate column. Features that are constant in the training data are
    passed through unchanged. A feature with (near) zero pooled within-batch
    variance, or a fit with non-finite batch parameters, raises a ValueError.
    This is a cross-sectional ComBat: it does not model repeated measurements of
    one subject. Requires the optional ``inmoose`` dependency.
    """

    def __init__(
        self,
        *,
        reference_batch: Any = None,
        parametric: bool = True,
        on_unseen_batch: str = "passthrough",
    ) -> None:
        self.reference_batch = reference_batch
        self.parametric = parametric
        self.on_unseen_batch = on_unseen_batch

    # ------------------------------------------------------------------
    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        batch: Sequence[Any] | NDArray | None = None,
        covariates: pd.DataFrame | None = None,
    ) -> ComBatHarmonizer:
        """Estimate and store the ComBat correction from the training data."""
        # Cheap validation first (no optional dependency needed), so configuration
        # errors raise clearly even where inmoose is not installed.
        if self.on_unseen_batch not in ("passthrough", "error"):
            raise ValueError(
                f"on_unseen_batch must be 'passthrough' or 'error', got {self.on_unseen_batch!r}."
            )
        X_arr = validate_estimator_input(self, X, reset=True, allow_nan=False)
        n_samples = X_arr.shape[0]
        batch_arr = self._check_batch(batch, X, n_samples)
        categories = list(pd.unique(batch_arr))
        if len(categories) < 2:
            raise ValueError(
                f"ComBatHarmonizer needs at least 2 batches to harmonize; got {len(categories)}."
            )
        sizes = pd.Series(batch_arr).astype(str).value_counts()
        if (sizes < 2).any():
            raise ValueError(
                "ComBatHarmonizer needs at least 2 samples in each batch; these batches have 1: "
                f"{', '.join(sizes.index[sizes < 2])}."
            )
        if self.reference_batch is not None and str(self.reference_batch) not in {
            str(c) for c in categories
        }:
            raise ValueError(f"reference_batch {self.reference_batch!r} is not among the batches.")

        # Constant (train) features cannot be standardized; pass them through.
        self.constant_mask_ = check_constant_features(X_arr, threshold=1e-12)
        constant_indices = np.flatnonzero(self.constant_mask_)
        if constant_indices.size > 0:
            names = ", ".join(self.feature_names_in_[i] for i in constant_indices[:5])
            suffix = " ..." if constant_indices.size > 5 else ""
            warnings.warn(
                f"{constant_indices.size} feature(s) have near-zero variance and will not "
                f"be harmonized (passed through unchanged): {names}{suffix}.",
                stacklevel=2,
            )
        active = ~self.constant_mask_
        if not active.any():
            raise ValueError("all features are constant; nothing to harmonize.")

        covariate_frame = self._prepare_fit_covariates(covariates, X, n_samples)

        (
            make_design_matrix,
            calculate_mean_var,
            calculate_stand_mean,
            standardise_data,
            fit_model,
        ) = _import_combat()
        dat = X_arr[:, active].T  # genes (features) x samples, as inmoose expects

        vci = make_design_matrix(
            dat,
            list(batch_arr),
            covariate_frame,
            self.reference_batch,
            na_cov_action="raise",
        )
        design = np.transpose(vci.design)
        n_batch = vci.n_batch
        batches_ind = [vci.batch_composition[b] for b in vci.batch.categories]
        batch_sizes = [len(idx) for idx in batches_ind]
        ref = vci.ref_batch_idx
        n = dat.shape[1]

        b_hat, grand_mean, var_pooled = calculate_mean_var(
            design, batches_ind, ref, dat, batch_sizes, n_batch, n
        )
        # A feature without within-batch (residual) variance cannot be
        # standardized, and it makes the shared priors of all features
        # non-finite. Stop with a clear error.
        pooled = np.asarray(var_pooled, dtype=float).ravel()
        total = X_arr[:, active].var(axis=0, ddof=1)
        unstable = ~(np.isfinite(pooled) & (pooled > 1e-10 * total))
        if unstable.any():
            names = ", ".join(map(str, self.feature_names_in_[active][unstable][:5]))
            raise ValueError(
                f"{int(unstable.sum())} feature(s) have (near) zero pooled within-batch "
                f"variance and cannot be harmonized: {names}. Remove them before ComBat."
            )
        stand_mean = calculate_stand_mean(grand_mean, n, design, n_batch, b_hat)
        s_data = standardise_data(dat, stand_mean, var_pooled, n)
        gamma_star, delta_star, _ = fit_model(
            design, n_batch, s_data, batches_ind, False, self.parametric, None, ref
        )

        fitted_categories = [str(c) for c in vci.batch.categories]
        gamma_star = np.asarray(gamma_star, dtype=float)
        delta_star = np.asarray(delta_star, dtype=float)
        if (
            not (np.isfinite(gamma_star).all() and np.isfinite(delta_star).all())
            or (delta_star <= 0).any()
        ):
            raise ValueError("ComBat estimation gave non-finite or non-positive batch parameters.")
        self.batches_ = fitted_categories
        self.grand_mean_ = np.asarray(grand_mean).ravel()
        self.var_pooled_ = np.asarray(var_pooled).ravel()
        self.gamma_star_ = {cat: gamma_star[i] for i, cat in enumerate(fitted_categories)}
        self.delta_star_ = {cat: delta_star[i] for i, cat in enumerate(fitted_categories)}
        # Numeric covariates map 1:1 to design columns, so B_hat's covariate rows
        # align with covariate_columns_ (validated numeric in _prepare_fit_covariates).
        self.b_covar_: NDArray | None = (
            np.asarray(b_hat)[n_batch:] if self.covariate_columns_ is not None else None
        )
        return self

    # ------------------------------------------------------------------
    def transform(
        self,
        X: Any,
        *,
        batch: Sequence[Any] | NDArray | None = None,
        covariates: pd.DataFrame | None = None,
    ) -> pd.DataFrame | NDArray:
        """Apply the stored per-batch correction to *X*."""
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "gamma_star_")
        X_arr = validate_estimator_input(self, X, reset=False, allow_nan=False)
        n_samples = X_arr.shape[0]
        batch_arr = self._check_batch(batch, X, n_samples)
        covar_aligned = self._align_transform_covariates(covariates, X, n_samples)

        active = ~self.constant_mask_
        out = X_arr.astype(float).copy()
        out_active = X_arr[:, active].astype(float)
        sd = np.sqrt(self.var_pooled_)

        if covar_aligned is not None and self.b_covar_ is not None:
            stand_mean_all = self.grand_mean_[None, :] + covar_aligned @ self.b_covar_
        else:
            stand_mean_all = np.broadcast_to(self.grand_mean_, (n_samples, active.sum()))

        reference = None if self.reference_batch is None else str(self.reference_batch)
        unseen: set[str] = set()
        for label in {str(b) for b in batch_arr}:
            rows = np.flatnonzero(np.asarray([str(b) for b in batch_arr]) == label)
            if label == reference:
                continue  # reference batch is left unchanged
            if label not in self.gamma_star_:
                if self.on_unseen_batch == "error":
                    raise ValueError(f"batch {label!r} was not seen during fit.")
                unseen.add(label)
                continue
            sm = stand_mean_all[rows]
            standardized = (out_active[rows] - sm) / sd
            adjusted = (standardized - self.gamma_star_[label]) / np.sqrt(self.delta_star_[label])
            out_active[rows] = adjusted * sd + sm
        if unseen:
            warnings.warn(
                f"{len(unseen)} batch(es) not seen during fit were left uncorrected: "
                f"{', '.join(sorted(unseen))}.",
                stacklevel=2,
            )
        out[:, active] = out_active

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(out, index=X.index, columns=X.columns)
        return out

    def fit_transform(
        self,
        X: Any,
        y: Any = None,
        *,
        batch: Sequence[Any] | NDArray | None = None,
        covariates: pd.DataFrame | None = None,
    ) -> pd.DataFrame | NDArray:
        """Fit then transform *X*, threading ``batch`` / ``covariates`` to both."""
        return self.fit(X, y, batch=batch, covariates=covariates).transform(
            X, batch=batch, covariates=covariates
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _check_batch(batch: Any, X: Any, n_samples: int) -> NDArray:
        if batch is None:
            raise ValueError("ComBatHarmonizer requires `batch` (a per-sample batch label).")
        if isinstance(batch, pd.Series) and isinstance(X, pd.DataFrame):
            if not batch.index.is_unique:
                raise ValueError("batch has duplicate index labels.")
            missing = X.index.difference(batch.index)
            if len(missing) > 0:
                raise ValueError(
                    f"batch has no label for {len(missing)} sample(s) of X, "
                    f"e.g. {list(missing[:3])}."
                )
            batch_arr = batch.loc[X.index].to_numpy()
        else:
            batch_arr = np.asarray(batch)
            if batch_arr.ndim != 1 or batch_arr.shape[0] != n_samples:
                raise ValueError(
                    f"batch has {batch_arr.shape[0]} labels but X has {n_samples} samples."
                )
        if pd.isna(batch_arr).any():
            raise ValueError(f"batch has {int(pd.isna(batch_arr).sum())} missing label(s).")
        labels = [str(b) for b in pd.unique(batch_arr)]
        if len(set(labels)) != len(labels):
            raise ValueError("Two batch labels have the same text form (e.g. 1 and '1').")
        checked: NDArray = batch_arr
        return checked

    @staticmethod
    def _align_covariates(covariates: Any, X: Any, n_samples: int) -> pd.DataFrame:
        """Align a covariate DataFrame to X by index; use other inputs by position."""
        frame = covariates if isinstance(covariates, pd.DataFrame) else pd.DataFrame(covariates)
        if isinstance(covariates, pd.DataFrame) and isinstance(X, pd.DataFrame):
            if not frame.index.is_unique:
                raise ValueError("covariates has duplicate index labels.")
            missing = X.index.difference(frame.index)
            if len(missing) > 0:
                raise ValueError(
                    f"covariates has {frame.shape[0]} rows but X has {n_samples} samples; "
                    f"{len(missing)} sample key(s) of X have no covariate row. Pass a "
                    "NumPy array for positional covariates."
                )
            return frame.loc[X.index]
        if frame.shape[0] != n_samples:
            raise ValueError(
                f"covariates has {frame.shape[0]} rows but X has {n_samples} samples."
            )
        return frame

    def _prepare_fit_covariates(
        self, covariates: pd.DataFrame | None, X: Any, n_samples: int
    ) -> pd.DataFrame | None:
        if covariates is None:
            self.covariate_columns_ = None
            return None
        frame = self._align_covariates(covariates, X, n_samples)
        if not all(pd.api.types.is_numeric_dtype(frame[col]) for col in frame.columns):
            raise ValueError(
                "covariates must be a numeric model matrix; encode categorical covariates "
                "first (e.g. one-hot / encode_clinical_series)."
            )
        self.covariate_columns_ = [str(c) for c in frame.columns]
        frame = frame.copy()
        # inmoose builds a patsy formula from the column names. Use plain names,
        # because "0", "1" (NumPy input) or a name with a space breaks it.
        frame.columns = [f"covariate_{i}" for i in range(frame.shape[1])]
        return frame.reset_index(drop=True)

    def _align_transform_covariates(
        self, covariates: pd.DataFrame | None, X: Any, n_samples: int
    ) -> NDArray | None:
        if self.covariate_columns_ is None:
            if covariates is not None:
                raise ValueError("fitted without covariates; do not pass covariates at transform.")
            return None
        if covariates is None:
            raise ValueError("fitted with covariates; transform requires the same covariates.")
        frame = self._align_covariates(covariates, X, n_samples).copy()
        frame.columns = [str(c) for c in frame.columns]
        missing_cols = [c for c in self.covariate_columns_ if c not in frame.columns]
        if missing_cols:
            raise ValueError(f"transform covariates lack fitted column(s): {missing_cols}.")
        aligned = frame[self.covariate_columns_]
        if not all(pd.api.types.is_numeric_dtype(aligned[col]) for col in aligned.columns):
            raise ValueError("covariates must be a numeric model matrix.")
        result: NDArray = aligned.to_numpy(dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("transform covariates have missing or non-finite values.")
        return result

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        return tags
