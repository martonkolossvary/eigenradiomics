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

from eigenradiomics._utils import validate_estimator_input


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
    ``Pipeline``. Covariates must be a **numeric** model matrix (encode
    categoricals first); features that are constant in the training data are
    passed through unchanged. Requires the optional ``inmoose`` dependency.
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
        batch_arr = self._check_batch(batch, n_samples)
        categories = list(pd.unique(batch_arr))
        if len(categories) < 2:
            raise ValueError(
                f"ComBatHarmonizer needs at least 2 batches to harmonize; got {len(categories)}."
            )
        if self.reference_batch is not None and str(self.reference_batch) not in {
            str(c) for c in categories
        }:
            raise ValueError(f"reference_batch {self.reference_batch!r} is not among the batches.")

        # Constant (train) features cannot be standardized; pass them through.
        self.constant_mask_ = X_arr.std(axis=0) <= 1e-12
        active = ~self.constant_mask_
        if not active.any():
            raise ValueError("all features are constant; nothing to harmonize.")

        covariate_frame = self._prepare_fit_covariates(covariates, n_samples)

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
        stand_mean = calculate_stand_mean(grand_mean, n, design, n_batch, b_hat)
        s_data = standardise_data(dat, stand_mean, var_pooled, n)
        gamma_star, delta_star, _ = fit_model(
            design, n_batch, s_data, batches_ind, False, self.parametric, None, ref
        )

        fitted_categories = [str(c) for c in vci.batch.categories]
        gamma_star = np.asarray(gamma_star)
        delta_star = np.asarray(delta_star)
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
        batch_arr = self._check_batch(batch, n_samples)
        covar_aligned = self._align_transform_covariates(covariates, n_samples)

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
    def _check_batch(batch: Any, n_samples: int) -> NDArray:
        if batch is None:
            raise ValueError("ComBatHarmonizer requires `batch` (a per-sample batch label).")
        batch_arr = np.asarray(batch)
        if batch_arr.shape[0] != n_samples:
            raise ValueError(
                f"batch has {batch_arr.shape[0]} labels but X has {n_samples} samples."
            )
        return batch_arr

    def _prepare_fit_covariates(
        self, covariates: pd.DataFrame | None, n_samples: int
    ) -> pd.DataFrame | None:
        if covariates is None:
            self.covariate_columns_ = None
            return None
        frame = covariates if isinstance(covariates, pd.DataFrame) else pd.DataFrame(covariates)
        if frame.shape[0] != n_samples:
            raise ValueError(
                f"covariates has {frame.shape[0]} rows but X has {n_samples} samples."
            )
        if not all(pd.api.types.is_numeric_dtype(frame[col]) for col in frame.columns):
            raise ValueError(
                "covariates must be a numeric model matrix; encode categorical covariates "
                "first (e.g. one-hot / encode_clinical_series)."
            )
        self.covariate_columns_ = [str(c) for c in frame.columns]
        frame = frame.copy()
        frame.columns = self.covariate_columns_
        return frame.reset_index(drop=True)

    def _align_transform_covariates(
        self, covariates: pd.DataFrame | None, n_samples: int
    ) -> NDArray | None:
        if self.covariate_columns_ is None:
            if covariates is not None:
                raise ValueError("fitted without covariates; do not pass covariates at transform.")
            return None
        if covariates is None:
            raise ValueError("fitted with covariates; transform requires the same covariates.")
        frame = covariates if isinstance(covariates, pd.DataFrame) else pd.DataFrame(covariates)
        if frame.shape[0] != n_samples:
            raise ValueError(
                f"covariates has {frame.shape[0]} rows but X has {n_samples} samples."
            )
        frame = frame.copy()
        frame.columns = [str(c) for c in frame.columns]
        aligned = frame.reindex(columns=self.covariate_columns_, fill_value=0.0)
        result: NDArray = aligned.to_numpy(dtype=float)
        return result

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        return tags
