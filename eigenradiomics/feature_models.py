"""Per-feature outcome association models (volcano analysis).

For each radiomics feature, fits an outcome model (optionally adjusted for
clinical covariates) and collects the effect, confidence interval, p-value, and
Benjamini-Hochberg FDR into a tidy table — the input to a volcano plot and to the
clustered-heatmap annotation tracks.

This module is built in phases. The continuous outcome engine (OLS with HC3
robust standard errors) is dependency-free; survival (Cox) and binary (logistic),
their mixed/clustered variants, the volcano plot, and the heatmap/Excel bridges
are layered on in later phases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from eigenradiomics._features import resolve_analysis_features
from eigenradiomics._stats import _fdr_correct
from eigenradiomics.catalog import FeatureCatalog
from eigenradiomics.dataset import RadiomicsDataset

#: Columns every fitted row carries (before FDR and catalog annotation).
_RESULT_FIELDS = (
    "model",
    "feature",
    "model_family",
    "outcome_type",
    "covariates",
    "coef",
    "effect",
    "effect_name",
    "se",
    "statistic",
    "p_value",
    "ci_low",
    "ci_high",
    "n",
    "n_events",
    "n_missing",
    "status",
    "error",
)


@dataclass
class FeatureAssociationResult:
    """Result of :func:`compute_feature_associations`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per (model tier, feature): effect / CI / p-value / FDR, fit
        diagnostics (``n``, ``status``, ...), and catalog annotations.
    outcome_type : str
        ``"continuous"``, ``"binary"``, or ``"survival"``.
    tiers : list of str
        Model-tier names in order (e.g. ``["Univariable", "Adjusted"]``).
    """

    table: pd.DataFrame
    outcome_type: str
    tiers: list[str]

    def top_hits(
        self,
        *,
        mode: str = "fdr",
        alpha: float = 0.05,
        per_panel: int | None = None,
    ) -> pd.DataFrame:
        """Return the most notable rows per model tier.

        Parameters
        ----------
        mode : {"fdr", "nominal", "ranked"}
            ``"fdr"`` keeps FDR-significant rows, ``"nominal"`` keeps p < ``alpha``,
            ``"ranked"`` keeps the lowest-p rows regardless of significance.
        alpha : float
            Significance threshold for ``"fdr"`` / ``"nominal"``.
        per_panel : int, optional
            Cap on rows kept per tier (applied after sorting by p-value).
        """
        if mode not in ("fdr", "nominal", "ranked"):
            raise ValueError(f"mode must be 'fdr', 'nominal', or 'ranked', got {mode!r}.")
        fitted = self.table[self.table["status"] == "ok"]
        if mode == "fdr":
            fitted = fitted[fitted["p_fdr"] < alpha]
        elif mode == "nominal":
            fitted = fitted[fitted["p_value"] < alpha]
        parts = []
        for _, group in fitted.groupby("model", sort=False):
            ordered = group.sort_values("p_value")
            parts.append(ordered.head(per_panel) if per_panel is not None else ordered)
        if not parts:
            return fitted
        return pd.concat(parts, ignore_index=True)


def _fit_ols_hc3(y: NDArray, design: NDArray, feature_idx: int) -> dict[str, Any]:
    """OLS with HC3 (heteroskedasticity-robust) inference for one coefficient."""
    n, k = design.shape
    if n < k + 2:
        return {"status": "not_enough_degrees_of_freedom"}
    xtx = design.T @ design
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:  # pragma: no cover - guarded by the constant/df checks
        return {"status": "rank_deficient_design"}
    beta = xtx_inv @ design.T @ y
    resid = y - design @ beta
    leverage = np.clip(np.diag(design @ xtx_inv @ design.T), None, 1 - 1e-10)
    omega = resid**2 / (1 - leverage) ** 2  # HC3 weights
    cov = xtx_inv @ (design.T * omega) @ design @ xtx_inv
    se = np.sqrt(np.diag(cov))
    df = n - k
    coef = float(beta[feature_idx])
    se_i = float(se[feature_idx])
    statistic = coef / se_i if se_i > 0 else np.nan
    p_value = float(2 * stats.t.sf(abs(statistic), df)) if np.isfinite(statistic) else np.nan
    crit = float(stats.t.ppf(0.975, df))
    return {
        "status": "ok",
        "model_family": "ols_hc3",
        "coef": coef,
        "effect": coef,
        "effect_name": "beta",
        "se": se_i,
        "statistic": statistic,
        "p_value": p_value,
        "ci_low": coef - crit * se_i,
        "ci_high": coef + crit * se_i,
        "n": int(n),
    }


def _fit_feature_continuous(
    frame: pd.DataFrame,
    feature: str,
    covariates: Sequence[str],
    outcome_col: str,
    min_unique: int,
) -> dict[str, Any]:
    """Complete-case continuous (OLS+HC3) fit for one feature in one tier."""
    columns = [outcome_col, feature, *covariates]
    subset = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    base = {
        "n": int(len(subset)),
        "n_events": np.nan,
        "n_missing": int(len(frame) - len(subset)),
    }
    if subset.empty:
        return {**base, "status": "no_complete_cases"}
    if subset[feature].nunique() < min_unique:
        return {**base, "status": "constant_feature"}
    y = subset[outcome_col].to_numpy(dtype=float)
    design = np.column_stack(
        [np.ones(len(subset)), subset[feature].to_numpy(dtype=float)]
        + [subset[col].to_numpy(dtype=float) for col in covariates]
    )
    fit = _fit_ols_hc3(y, design, feature_idx=1)
    fit["n_missing"] = base["n_missing"]
    fit.setdefault("n", base["n"])
    fit.setdefault("n_events", np.nan)
    return fit


def _resolve_outcome(
    X: pd.DataFrame | RadiomicsDataset,
    outcome: Any,
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Resolve the outcome to a frame of outcome columns + an inferred type."""
    if outcome is None:
        if not isinstance(X, RadiomicsDataset):
            raise ValueError("outcome is required unless X is a RadiomicsDataset with a design.")
        outcome = X.y()
        if outcome is None:
            raise ValueError(
                "the RadiomicsDataset has no outcome; set a 'target' or 'time'/'event' "
                "study-design role, or pass outcome= explicitly."
            )
    if isinstance(outcome, str):
        outcome = data[outcome]
    if isinstance(outcome, pd.Series):
        return outcome.to_frame(name="__outcome__"), "binary_or_continuous"
    if isinstance(outcome, pd.DataFrame):
        return outcome, "survival"
    raise TypeError(
        f"outcome must be a Series, DataFrame, or column name; got {type(outcome).__name__}."
    )


def _infer_outcome_type(outcome_frame: pd.DataFrame) -> str:
    if outcome_frame.shape[1] >= 2:
        return "survival"
    values = pd.to_numeric(outcome_frame.iloc[:, 0], errors="coerce").dropna()
    return "binary" if values.nunique() <= 2 else "continuous"


def compute_feature_associations(
    X: pd.DataFrame | RadiomicsDataset,
    outcome: Any = None,
    *,
    outcome_type: str = "auto",
    model_tiers: Mapping[str, Sequence[str]] | None = None,
    adjust_for: Sequence[str] | None = None,
    covariate_data: pd.DataFrame | None = None,
    groups: Any = None,
    catalog: FeatureCatalog | pd.DataFrame | None = None,
    features: Any = None,
    min_unique: int = 2,
) -> FeatureAssociationResult:
    """Model each feature against an outcome, with optional clinical adjustment.

    Parameters
    ----------
    X : DataFrame or RadiomicsDataset
        Feature matrix (samples x features). A :class:`RadiomicsDataset` also
        supplies the outcome (study-design roles), catalog, and covariate source.
    outcome : Series, DataFrame, column name, or None
        Continuous/binary outcome (Series or column), survival ``[time, event]``
        (2-column DataFrame), or ``None`` to take it from the dataset's design.
    outcome_type : {"auto", "continuous", "binary", "survival"}
        Outcome family; ``"auto"`` infers it (2 columns -> survival, <=2 unique
        values -> binary, else continuous).
    model_tiers : mapping, optional
        ``{tier_name: [covariate columns]}`` fitted in parallel. Defaults to a
        ``"Univariable"`` tier plus an ``"Adjusted"`` tier when *adjust_for* is set.
    adjust_for : sequence of str, optional
        Convenience covariate list for a single adjusted tier.
    covariate_data : DataFrame, optional
        Where covariate columns live when *X* is a bare feature DataFrame
        (defaults to the dataset metadata, or *X* itself).
    groups : optional
        Cluster / repeated-measures identifier (used by the mixed engines).
    catalog : FeatureCatalog or DataFrame, optional
        Adds ``family`` / ``family_group`` (and any other catalog columns).
    features : optional
        Pictologics-style selectors to restrict which columns are modelled.
    min_unique : int
        Minimum distinct feature values required to fit (else ``constant_feature``).

    Returns
    -------
    FeatureAssociationResult
    """
    if isinstance(X, RadiomicsDataset):
        feature_matrix = X.features
        data = X.data
        if catalog is None:
            catalog = X.catalog
    else:
        feature_matrix = X
        data = covariate_data if covariate_data is not None else X

    outcome_frame, _ = _resolve_outcome(X, outcome, data)
    outcome_frame = outcome_frame.reindex(feature_matrix.index)
    resolved_type = _infer_outcome_type(outcome_frame) if outcome_type == "auto" else outcome_type
    if resolved_type not in ("continuous", "binary", "survival"):
        raise ValueError(
            f"outcome_type must be continuous/binary/survival, got {resolved_type!r}."
        )
    if resolved_type != "continuous":
        raise NotImplementedError(
            f"{resolved_type} models are added in a later phase; pass outcome_type='continuous' "
            "for now."
        )

    feature_names = list(resolve_analysis_features(feature_matrix, features=features))
    if not feature_names:
        raise ValueError("no feature columns to model.")

    tiers = _build_tiers(model_tiers, adjust_for)
    covariate_cols = sorted({col for cols in tiers.values() for col in cols})
    missing = [col for col in covariate_cols if col not in data.columns]
    if missing:
        raise KeyError(f"covariate column(s) not found: {', '.join(missing[:5])}.")

    outcome_col = outcome_frame.columns[0]
    work = pd.concat(
        [feature_matrix[feature_names], outcome_frame[[outcome_col]], data[covariate_cols]],
        axis=1,
    )

    rows = []
    for tier_name, tier_covariates in tiers.items():
        for feature in feature_names:
            fit = _fit_feature_continuous(work, feature, tier_covariates, outcome_col, min_unique)
            rows.append(
                {
                    "model": tier_name,
                    "feature": feature,
                    "outcome_type": resolved_type,
                    "covariates": ", ".join(tier_covariates),
                    "error": "",
                    **fit,
                }
            )

    table = pd.DataFrame(rows).reindex(columns=[*_RESULT_FIELDS, "p_fdr"])
    table = _apply_fdr(table)
    table = _annotate_catalog(table, catalog)
    return FeatureAssociationResult(table=table, outcome_type=resolved_type, tiers=list(tiers))


def _build_tiers(
    model_tiers: Mapping[str, Sequence[str]] | None,
    adjust_for: Sequence[str] | None,
) -> dict[str, list[str]]:
    if model_tiers is not None:
        return {name: list(cols) for name, cols in model_tiers.items()}
    tiers: dict[str, list[str]] = {"Univariable": []}
    if adjust_for:
        tiers["Adjusted"] = list(adjust_for)
    return tiers


def _apply_fdr(table: pd.DataFrame) -> pd.DataFrame:
    for _, idx in table.groupby("model", sort=False).groups.items():
        mask = table.index.isin(idx) & table["p_value"].notna()
        if mask.any():
            table.loc[mask, "p_fdr"] = _fdr_correct(table.loc[mask, "p_value"].to_numpy())
    return table


def _annotate_catalog(
    table: pd.DataFrame,
    catalog: FeatureCatalog | pd.DataFrame | None,
) -> pd.DataFrame:
    if catalog is None:
        return table
    frame = catalog.frame if isinstance(catalog, FeatureCatalog) else FeatureCatalog(catalog).frame
    keep = [c for c in ("feature", "family", "family_group", "config") if c in frame.columns]
    return table.merge(frame[keep], on="feature", how="left", validate="many_to_one")
