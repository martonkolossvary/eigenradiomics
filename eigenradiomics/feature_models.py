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
    "c_index",
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


@dataclass(frozen=True)
class _OutcomeSpec:
    outcome_type: str
    outcome_cols: list[str]
    groups_col: str | None
    mixed_method: str
    penalizer: float
    min_events: int
    min_unique: int


def _import_lifelines() -> Any:
    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:
        raise ImportError(
            "survival models require the optional 'lifelines' dependency "
            "(`pip install eigenradiomics[survival]`)."
        ) from exc
    return CoxPHFitter


def _import_statsmodels() -> Any:
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError(
            "binary / mixed models require the optional 'statsmodels' dependency "
            "(`pip install eigenradiomics[modeling]`)."
        ) from exc
    return sm


def _clean_error(exc: Exception) -> str:
    import re

    return re.sub(r"\s+", " ", str(exc)).strip()[:300]


def _design(subset: pd.DataFrame, feature: str, covariates: Sequence[str]) -> NDArray:
    """[intercept, feature, *covariates] design matrix; the feature is column 1."""
    return np.column_stack(
        [np.ones(len(subset)), subset[feature].to_numpy(dtype=float)]
        + [subset[col].to_numpy(dtype=float) for col in covariates]
    )


def _ratio_result(family: str, coef: float, se: float, p: float, lo: float, hi: float) -> dict:
    """Result row for a ratio effect (HR/OR) on the exp scale."""
    return {
        "status": "ok",
        "model_family": family,
        "coef": coef,
        "effect": float(np.exp(coef)),
        "effect_name": "HR" if family.startswith("cox") else "OR",
        "se": se,
        "statistic": coef / se if se > 0 else np.nan,
        "p_value": p,
        "ci_low": float(np.exp(lo)),
        "ci_high": float(np.exp(hi)),
    }


def _fit_cox(
    subset: pd.DataFrame, feature: str, covariates: Sequence[str], spec: _OutcomeSpec
) -> dict:
    cox_cls = _import_lifelines()
    time_col, event_col = spec.outcome_cols
    keep = [time_col, event_col, feature, *covariates]
    fit_kwargs = {"duration_col": time_col, "event_col": event_col, "show_progress": False}
    family = "cox"
    if spec.groups_col:
        keep = [*keep, spec.groups_col]
        fit_kwargs["cluster_col"] = spec.groups_col
        fit_kwargs["robust"] = True
        family = "cox_clustered"
    try:
        cph = cox_cls(penalizer=spec.penalizer)
        cph.fit(subset[keep], **fit_kwargs)
        row = cph.summary.loc[feature]
    except Exception as exc:
        return {"status": "fit_failed", "error": _clean_error(exc)}
    result = _ratio_result(
        family,
        float(row["coef"]),
        float(row["se(coef)"]),
        float(row["p"]),
        np.log(float(row["exp(coef) lower 95%"])),
        np.log(float(row["exp(coef) upper 95%"])),
    )
    result["statistic"] = float(row["z"])
    result["c_index"] = float(cph.concordance_index_)
    return result


def _fit_binary(
    subset: pd.DataFrame, feature: str, covariates: Sequence[str], spec: _OutcomeSpec
) -> dict:
    if spec.groups_col and spec.mixed_method == "glmm":
        return _fit_glmm(subset, feature, covariates, spec)
    sm = _import_statsmodels()
    y = subset[spec.outcome_cols[0]].to_numpy(dtype=float)
    design = _design(subset, feature, covariates)
    try:
        if spec.groups_col:
            model = sm.GEE(
                y,
                design,
                groups=subset[spec.groups_col].to_numpy(),
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable(),
            )
            family = "gee_logit"
        else:
            model = sm.Logit(y, design)
            family = "logit"
        res = model.fit(disp=0) if family == "logit" else model.fit()
        params, bse, pvals = np.asarray(res.params), np.asarray(res.bse), np.asarray(res.pvalues)
        conf = np.asarray(res.conf_int())
    except Exception as exc:
        return {"status": "fit_failed", "error": _clean_error(exc)}
    return _ratio_result(
        family,
        float(params[1]),
        float(bse[1]),
        float(pvals[1]),
        float(conf[1, 0]),
        float(conf[1, 1]),
    )


def _fit_glmm(
    subset: pd.DataFrame, feature: str, covariates: Sequence[str], spec: _OutcomeSpec
) -> dict:
    _import_statsmodels()
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    renames = {spec.outcome_cols[0]: "y", feature: "x"}
    renames.update({col: f"c{i}" for i, col in enumerate(covariates)})
    safe = subset.rename(columns=renames).copy()
    safe["g"] = subset[spec.groups_col].to_numpy()
    rhs = " + ".join(["x", *(f"c{i}" for i in range(len(covariates)))])
    try:
        model = BinomialBayesMixedGLM.from_formula(f"y ~ {rhs}", {"g": "0 + C(g)"}, safe)
        res = model.fit_vb()
        idx = list(res.model.fep_names).index("x")
        coef, sd = float(res.fe_mean[idx]), float(res.fe_sd[idx])
    except Exception as exc:
        return {"status": "fit_failed", "error": _clean_error(exc)}
    z = coef / sd if sd > 0 else np.nan
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return _ratio_result("glmm", coef, sd, p, coef - 1.96 * sd, coef + 1.96 * sd)


def _fit_continuous(
    subset: pd.DataFrame, feature: str, covariates: Sequence[str], spec: _OutcomeSpec
) -> dict:
    if spec.groups_col:
        sm = _import_statsmodels()
        design = _design(subset, feature, covariates)
        y = subset[spec.outcome_cols[0]].to_numpy(dtype=float)
        try:
            res = sm.MixedLM(y, design, groups=subset[spec.groups_col].to_numpy()).fit(
                method=["lbfgs"], reml=True
            )
            params, bse, pvals = (
                np.asarray(res.params),
                np.asarray(res.bse),
                np.asarray(res.pvalues),
            )
            conf = np.asarray(res.conf_int())
        except Exception as exc:
            return {"status": "fit_failed", "error": _clean_error(exc)}
        coef, se = float(params[1]), float(bse[1])
        return {
            "status": "ok",
            "model_family": "mixedlm",
            "coef": coef,
            "effect": coef,
            "effect_name": "beta",
            "se": se,
            "statistic": coef / se if se > 0 else np.nan,
            "p_value": float(pvals[1]),
            "ci_low": float(conf[1, 0]),
            "ci_high": float(conf[1, 1]),
        }
    y = subset[spec.outcome_cols[0]].to_numpy(dtype=float)
    return _fit_ols_hc3(y, _design(subset, feature, covariates), feature_idx=1)


def _fit_feature(
    work: pd.DataFrame, feature: str, covariates: Sequence[str], spec: _OutcomeSpec
) -> dict:
    """Complete-case fit for one feature in one tier, dispatched by outcome type."""
    model_cols = [*spec.outcome_cols, feature, *covariates]
    subset = work[model_cols].apply(pd.to_numeric, errors="coerce")
    if spec.groups_col:
        subset[spec.groups_col] = work[spec.groups_col].to_numpy()
    subset = subset.dropna()
    base = {"n": int(len(subset)), "n_events": np.nan, "n_missing": int(len(work) - len(subset))}
    if subset.empty:
        return {**base, "status": "no_complete_cases"}
    if subset[feature].nunique() < spec.min_unique:
        return {**base, "status": "constant_feature"}

    if spec.outcome_type == "survival":
        base["n_events"] = int(subset[spec.outcome_cols[1]].sum())
        if base["n_events"] < spec.min_events:
            return {**base, "status": "no_events"}
        fit = _fit_cox(subset, feature, covariates, spec)
    elif spec.outcome_type == "binary":
        outcome_values = subset[spec.outcome_cols[0]]
        base["n_events"] = int(outcome_values.sum())
        if outcome_values.nunique() < 2:
            return {**base, "status": "no_events"}
        fit = _fit_binary(subset, feature, covariates, spec)
    else:
        fit = _fit_continuous(subset, feature, covariates, spec)

    fit["n_missing"] = base["n_missing"]
    fit.setdefault("n", base["n"])
    fit.setdefault("n_events", base["n_events"])
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
    mixed_method: str = "glmm",
    penalizer: float = 0.0,
    min_events: int = 1,
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
    groups : array-like or column name, optional
        Cluster / repeated-measures identifier. When given, the engine switches to
        a mixed/clustered variant: MixedLM (continuous), GLMM or GEE (binary), and
        cluster-robust Cox (survival).
    mixed_method : {"glmm", "gee"}
        Mixed binary model when *groups* is given (random-intercept GLMM, default,
        vs cluster-robust GEE).
    penalizer : float
        Optional Cox ridge penalizer for convergence support.
    min_events : int
        Minimum events required to fit a survival/binary model (else ``no_events``).
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

    if mixed_method not in ("glmm", "gee"):
        raise ValueError(f"mixed_method must be 'glmm' or 'gee', got {mixed_method!r}.")

    outcome_frame, _ = _resolve_outcome(X, outcome, data)
    outcome_frame = outcome_frame.reindex(feature_matrix.index)
    resolved_type = _infer_outcome_type(outcome_frame) if outcome_type == "auto" else outcome_type
    if resolved_type not in ("continuous", "binary", "survival"):
        raise ValueError(
            f"outcome_type must be continuous/binary/survival, got {resolved_type!r}."
        )

    feature_names = list(resolve_analysis_features(feature_matrix, features=features))
    if not feature_names:
        raise ValueError("no feature columns to model.")

    tiers = _build_tiers(model_tiers, adjust_for)
    covariate_cols = sorted({col for cols in tiers.values() for col in cols})
    missing = [col for col in covariate_cols if col not in data.columns]
    if missing:
        raise KeyError(f"covariate column(s) not found: {', '.join(missing[:5])}.")

    # Resolve the clustering/repeated-measures identifier (column name or array).
    groups_col: str | None = None
    groups_series: pd.Series | None = None
    if isinstance(X, RadiomicsDataset) and groups is None and X.design.group:
        groups = X.design.group
    if groups is not None:
        if isinstance(groups, str):
            groups_col, groups_series = groups, data[groups]
        else:
            groups_col = "__groups__"
            groups_series = pd.Series(np.asarray(groups), index=feature_matrix.index)

    outcome_cols = list(outcome_frame.columns)
    frames = [feature_matrix[feature_names], outcome_frame, data[covariate_cols]]
    if groups_series is not None:
        frames.append(groups_series.rename(groups_col).to_frame())
    work = pd.concat(frames, axis=1)

    spec = _OutcomeSpec(
        outcome_type=resolved_type,
        outcome_cols=outcome_cols,
        groups_col=groups_col,
        mixed_method=mixed_method,
        penalizer=penalizer,
        min_events=min_events,
        min_unique=min_unique,
    )

    rows = []
    for tier_name, tier_covariates in tiers.items():
        for feature in feature_names:
            fit = _fit_feature(work, feature, list(tier_covariates), spec)
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
