"""Per-feature outcome association models (volcano analysis).

For each radiomics feature, fits an outcome model (optionally adjusted for
clinical covariates) and collects the effect, confidence interval, p-value, and
Benjamini-Hochberg FDR into a tidy table — the input to a volcano plot and to the
clustered-heatmap annotation tracks.

Supported outcomes: continuous (OLS + HC3 robust SE, dependency-free), survival
(Cox PH, ``lifelines``), and binary (logistic, ``statsmodels``). When a cluster /
repeated-measures identifier is given, each switches to a mixed/clustered variant
(MixedLM, GLMM or GEE, cluster-robust Cox). :func:`plot_volcano` renders the
results, and :class:`FeatureAssociationResult` bridges to the clustered heatmap
and to Excel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from eigenradiomics._features import resolve_analysis_features
from eigenradiomics._stats import _fdr_correct
from eigenradiomics.catalog import FeatureCatalog
from eigenradiomics.dataset import RadiomicsDataset

if TYPE_CHECKING:
    from pathlib import Path

    import matplotlib.pyplot as plt

    from eigenradiomics.plotting import Bar, CorrPanel, Strip

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

    def _tier_value(self, tier: str | None, value: str) -> pd.Series:
        tier = tier if tier is not None else self.tiers[0]
        if tier not in self.tiers:
            raise ValueError(f"unknown tier {tier!r}; available: {self.tiers}.")
        sub = self.table[self.table["model"] == tier].set_index("feature")
        return _value_column(sub, value)

    def bar(
        self,
        tier: str | None = None,
        *,
        value: str = "neg_log10_p",
        title: str | None = None,
        reference: float | None = None,
        color: str = "by_module",
    ) -> Bar:
        """Return a heatmap :class:`~eigenradiomics.Bar` of a per-feature statistic.

        ``value`` is one of ``"neg_log10_p"``, ``"neg_log10_fdr"``, ``"log2_effect"``,
        ``"coef"``, ``"effect"``, ``"statistic"``. The bar (indexed by feature) drops
        straight into ``plot_clustered_heatmap(..., bottom=[bar])``.
        """
        from eigenradiomics.plotting import Bar

        series = self._tier_value(tier, value)
        label = title if title is not None else value
        return Bar(series.rename(label), title=label, reference=reference, color=color)

    def matrix(self, *, value: str = "coef", tiers: Sequence[str] | None = None) -> pd.DataFrame:
        """Return a feature x model-tier matrix of *value* (for a heatmap ``CorrPanel``)."""
        tiers = list(tiers) if tiers is not None else self.tiers
        return pd.DataFrame({tier: self._tier_value(tier, value) for tier in tiers})

    def to_excel(
        self,
        path: Any,
        *,
        top_hits_mode: str = "fdr",
        alpha: float = 0.05,
        per_panel: int | None = 50,
    ) -> None:
        """Write the full results and a top-hits sheet to a styled Excel workbook."""
        from eigenradiomics._excel import write_styled_workbook

        sheets = {
            "associations": self.table,
            "top_hits": self.top_hits(mode=top_hits_mode, alpha=alpha, per_panel=per_panel),
        }
        write_styled_workbook(sheets, path, _association_number_format)


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


def _value_column(table: pd.DataFrame, value: str) -> pd.Series:
    """Per-feature Series for a bar/matrix value (feature-indexed)."""
    tiny = np.nextafter(0, 1)
    if value == "neg_log10_p":
        return -np.log10(table["p_value"].clip(lower=tiny))
    if value == "neg_log10_fdr":
        return -np.log10(table["p_fdr"].clip(lower=tiny))
    if value == "log2_effect":
        return np.log2(table["effect"].clip(lower=tiny))
    if value in ("coef", "effect", "statistic", "ci_low", "ci_high"):
        return table[value]
    raise ValueError(
        "value must be one of neg_log10_p, neg_log10_fdr, log2_effect, coef, effect, "
        f"statistic, ci_low, ci_high; got {value!r}."
    )


def _association_number_format(col_name: str, val: float) -> str | None:
    if "p_value" in col_name or "p_fdr" in col_name:
        return "0.00E+00" if val < 1e-4 else "0.0000"
    return "0.000"


def _annotate_catalog(
    table: pd.DataFrame,
    catalog: FeatureCatalog | pd.DataFrame | None,
) -> pd.DataFrame:
    if catalog is None:
        return table
    frame = catalog.frame if isinstance(catalog, FeatureCatalog) else FeatureCatalog(catalog).frame
    keep = [c for c in ("feature", "family", "family_group", "config") if c in frame.columns]
    return table.merge(frame[keep], on="feature", how="left", validate="many_to_one")


# ----------------------------------------------------------------------
# Volcano plot
# ----------------------------------------------------------------------

#: Exact grids for small panel counts; 5-9 panels use a 3x3 grid.
_VOLCANO_LAYOUTS = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2)}
_VOLCANO_MARKERS = ["o", "^", "s", "D", "v", "P", "X", "*", "p"]
_X_LABELS = {
    "survival": r"$\log_2$ hazard ratio",
    "binary": r"$\log_2$ odds ratio",
    "continuous": "coefficient",
}
_NONSIG_COLOR = "#B8B8B8"


def _resolve_layout(layout: str | tuple[int, int], n: int) -> tuple[int, int]:
    if isinstance(layout, tuple):
        rows, cols = layout
        if rows * cols < n:
            raise ValueError(f"layout {layout} has fewer cells than the {n} panels requested.")
        return rows, cols
    return _VOLCANO_LAYOUTS.get(n, (3, 3))


def _symmetric_xlim(x: NDArray, strategy: str, percentile: float) -> tuple[float, float]:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return (-1.0, 1.0)
    extent = (
        np.max(np.abs(finite))
        if strategy == "include"
        else np.percentile(np.abs(finite), percentile)
    )
    extent = max(float(extent), 1e-6)
    return (-extent * 1.08, extent * 1.08)


def plot_volcano(
    result: FeatureAssociationResult,
    *,
    tiers: Sequence[str] | None = None,
    color_by: str | None = "family_group",
    marker_by: str | None = None,
    fdr_alpha: float = 0.05,
    layout: str | tuple[int, int] = "auto",
    axis_mode: str = "panel",
    outlier_strategy: str = "clip",
    core_percentile: float = 99.0,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Any:
    """Volcano plot of the feature-association results, one panel per model tier.

    The panel grid is chosen from the number of tiers shown: 1 -> 1x1, 2 -> 1x2,
    3 -> 1x3, 4 -> 2x2, and 5-9 -> 3x3 (unused cells hidden). Pass an explicit
    ``layout=(rows, cols)`` to override.

    Parameters
    ----------
    result : FeatureAssociationResult
        Output of :func:`compute_feature_associations`.
    tiers : sequence of str, optional
        Which model tiers to show as panels (default: all, capped at 9).
    color_by : str, optional
        Catalog column whose categories colour the FDR-significant points
        (default ``"family_group"``); others are grey.
    marker_by : str, optional
        Catalog column whose categories set the point markers.
    fdr_alpha : float
        FDR threshold for "significant" colouring and the horizontal reference.
    layout : "auto" or (rows, cols)
        Panel grid.
    axis_mode : {"panel", "shared"}
        Per-panel or shared axis limits.
    outlier_strategy : {"clip", "include"}
        ``"clip"`` bounds the x-axis to ``core_percentile`` of \\|effect\\| (extreme
        points pinned to the edge); ``"include"`` shows the full range.
    core_percentile : float
        Percentile of \\|x\\| used for the "clip" x-limit.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from eigenradiomics._plotting import apply_science_style
    from eigenradiomics.plotting import _assign_colors

    if axis_mode not in ("panel", "shared"):
        raise ValueError(f"axis_mode must be 'panel' or 'shared', got {axis_mode!r}.")
    if outlier_strategy not in ("clip", "include"):
        raise ValueError(
            f"outlier_strategy must be 'clip' or 'include', got {outlier_strategy!r}."
        )

    panels = list(tiers) if tiers is not None else list(result.tiers)
    if not 1 <= len(panels) <= 9:
        raise ValueError(f"plot_volcano supports 1-9 panels, got {len(panels)}.")

    is_ratio = result.outcome_type in ("survival", "binary")
    shown = result.table[
        result.table["model"].isin(panels)
        & result.table["status"].eq("ok")
        & result.table["p_value"].notna()
        & result.table["p_value"].gt(0)
    ].copy()
    if is_ratio:
        shown["_x"] = np.log2(shown["effect"].clip(lower=np.finfo(float).tiny))
    else:
        shown["_x"] = shown["coef"]
    shown["_y"] = -np.log10(shown["p_value"].clip(lower=np.nextafter(0, 1)))
    shown["_sig"] = shown["p_fdr"].lt(fdr_alpha)

    has_color = bool(color_by) and color_by in shown.columns
    categories = (
        sorted(shown.loc[shown["_sig"], color_by].dropna().astype(str).unique())
        if has_color
        else []
    )
    color_map = _assign_colors(categories, None) if categories else {}
    has_marker = bool(marker_by) and marker_by in shown.columns
    marker_cats = sorted(shown[marker_by].dropna().astype(str).unique()) if has_marker else []
    marker_map = {
        c: _VOLCANO_MARKERS[i % len(_VOLCANO_MARKERS)] for i, c in enumerate(marker_cats)
    }

    rows, cols = _resolve_layout(layout, len(panels))
    apply_science_style()
    fig, axes = plt.subplots(
        rows, cols, figsize=figsize or (cols * 3.4, rows * 3.3), squeeze=False
    )
    flat = axes.ravel()

    shared_xlim = _symmetric_xlim(shown["_x"].to_numpy(), outlier_strategy, core_percentile)
    shared_ylim = (0.0, float(shown["_y"].max()) * 1.08 if len(shown) else 1.0)
    x_label = _X_LABELS[result.outcome_type]

    for index, tier in enumerate(panels):
        ax = flat[index]
        tier_data = shown[shown["model"].eq(tier)]
        if axis_mode == "shared":
            xlim, ylim = shared_xlim, shared_ylim
        else:
            xlim = _symmetric_xlim(tier_data["_x"].to_numpy(), outlier_strategy, core_percentile)
            ylim = (0.0, float(tier_data["_y"].max()) * 1.08 if len(tier_data) else 1.0)
        _draw_volcano_panel(
            ax,
            tier_data,
            tier,
            color_by,
            marker_by,
            color_map,
            marker_map,
            fdr_alpha,
            xlim,
            ylim,
            x_label,
        )
        if index % cols == 0:
            ax.set_ylabel(r"$-\log_{10}$(p-value)")
    for hidden in flat[len(panels) :]:
        hidden.set_visible(False)

    handles: list[Any] = []
    for category, colour in color_map.items():
        handles.append(Patch(facecolor=colour, edgecolor="0.3", label=str(category)))
    if color_map or has_color:
        handles.append(
            Patch(facecolor=_NONSIG_COLOR, edgecolor="0.3", label=f"FDR ≥ {fdr_alpha:g}")
        )
    for category, marker in marker_map.items():
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="",
                markerfacecolor="white",
                markeredgecolor="black",
                label=str(category),
            )
        )
    handles.append(
        Line2D([0], [0], color="black", linestyle=":", label=f"p = {fdr_alpha:g} (FDR)")
    )
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False)
    if title:
        fig.suptitle(title, weight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def _draw_volcano_panel(
    ax: Any,
    data: pd.DataFrame,
    tier: str,
    color_by: str | None,
    marker_by: str | None,
    color_map: Mapping[str, Any],
    marker_map: Mapping[str, str],
    fdr_alpha: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    x_label: str,
) -> None:
    ax.set_title(tier)
    ax.set_xlabel(x_label)
    if data.empty:
        ax.text(0.5, 0.5, "no fitted features", ha="center", va="center", transform=ax.transAxes)
        return
    x = np.clip(data["_x"].to_numpy(), xlim[0], xlim[1])
    y = data["_y"].to_numpy()
    for i, (_, row) in enumerate(data.iterrows()):
        if row["_sig"] and color_map and color_by:
            colour = color_map.get(str(row[color_by]), _NONSIG_COLOR)
        else:
            colour = "#D55E00" if row["_sig"] else _NONSIG_COLOR
        marker = marker_map.get(str(row[marker_by]), "o") if (marker_by and marker_map) else "o"
        ax.scatter(
            x[i],
            y[i],
            c=colour,
            marker=marker,
            s=16,
            alpha=0.7 if row["_sig"] else 0.3,
            edgecolors="none",
        )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axhline(-np.log10(fdr_alpha), color="black", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_rwas_manhattan(
    result: FeatureAssociationResult | pd.DataFrame,
    catalog: FeatureCatalog | pd.DataFrame | None = None,
    *,
    tier: str | None = None,
    group_by: str = "family",
    order: Sequence[str] | None = None,
    strips: Sequence[Strip] | None = None,
    bars: Sequence[Bar] | None = None,
    corr_panel: CorrPanel | None = None,
    fdr_alpha: float = 0.05,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    path: str | Path | None = None,
) -> plt.Figure:
    """Plot an RWAS Manhattan plot of feature associations.

    Features are plotted on the horizontal axis grouped by their catalog family
    or group, with alternating shading and labels. The vertical axis represents
    the -log10 association p-value. Custom tracks (categorical strips, numeric
    bars, and correlation panels) can be aligned horizontally under the plot.

    Parameters
    ----------
    result : FeatureAssociationResult or pandas.DataFrame
        Output of `compute_feature_associations`.
    catalog : FeatureCatalog or pandas.DataFrame, optional
        Used to resolve feature groups/families.
    tier : str, optional
        Which model tier to show (default: first tier in results).
    group_by : str, default="family"
        Catalog column to group features by on the horizontal axis.
    order : sequence of str, optional
        Custom ordering of features. If None, features are ordered by group_by
        then by feature name.
    strips : sequence of Strip, optional
        Categorical annotation strips to draw below the plot.
    bars : sequence of Bar, optional
        Numeric annotation bars to draw below the plot.
    corr_panel : CorrPanel, optional
        A feature-by-variable correlation panel to draw transposed below the plot.
    fdr_alpha : float, default=0.05
        FDR threshold for "significant" points and the reference line.
    figsize : tuple of float, optional
        Figure size.
    title : str, optional
        Figure title.
    path : str or Path, optional
        If set, save the figure to this file path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from eigenradiomics._plotting import apply_science_style
    from eigenradiomics.plotting import (
        _assign_colors,
        _draw_bottom_bars,
        _draw_top_strips,
    )

    # Resolve DataFrame
    df = result.table.copy() if isinstance(result, FeatureAssociationResult) else result.copy()

    # Annotate catalog
    if catalog is not None:
        df = _annotate_catalog(df, catalog)

    if group_by not in df.columns:
        df[group_by] = "All Features"

    # Handle tier
    if "model" in df.columns:
        if tier is None:
            available_tiers = list(df["model"].dropna().unique())
            if len(available_tiers) > 0:
                tier = available_tiers[0]
        if tier is not None:
            df = df[df["model"] == tier]

    df = df[df["status"].eq("ok") & df["p_value"].notna() & df["p_value"].gt(0)].copy()

    if df.empty:
        raise ValueError("No fitted features available to plot.")

    # Determine unique features in this subset
    all_features = df["feature"].unique()

    # Determine ordering of features
    if order is not None:
        order_names = [feat for feat in order if feat in all_features]
    else:
        # Sort by group_by first, then by feature name
        df_sorted = df.sort_values(by=[group_by, "feature"])
        order_names = list(df_sorted["feature"].unique())

    if not order_names:
        raise ValueError("No features match the specified order or are present in results.")

    n = len(order_names)

    # Reindex df to order_names
    df_ordered = df.set_index("feature").reindex(order_names).reset_index()
    df_ordered["_y"] = -np.log10(df_ordered["p_value"].clip(lower=np.nextafter(0, 1)))
    df_ordered["_sig"] = df_ordered["p_fdr"].lt(fdr_alpha)

    # Assign group colors
    unique_groups = sorted(df_ordered[group_by].dropna().unique())
    group_colors = _assign_colors(unique_groups, None)

    # Color map for the points
    colors = []
    for _, row in df_ordered.iterrows():
        if row["_sig"]:
            colors.append(group_colors.get(row[group_by], "#B8B8B8"))
        else:
            colors.append("#B8B8B8")

    # Layout dimensions
    n_strips = len(strips) if strips else 0
    n_bars = len(bars) if bars else 0
    has_corr = corr_panel is not None

    # Calculate grid ratios
    height_ratios = [10.0]  # Manhattan plot
    for _ in range(n_strips):
        height_ratios.append(0.32)
    for _ in range(n_bars):
        height_ratios.append(1.1)
    if corr_panel is not None:
        n_vars = corr_panel.data.shape[1]
        height_ratios.append(max(1.0, 0.3 * n_vars))
    height_ratios.append(1.0)  # spacing for colorbar/ticks

    # Determine grid columns (main + legend)
    strip_color_maps: list[dict[Any, Any]] = []
    legend_blocks: list[tuple[str, dict[Any, Any]]] = []
    if strips:
        for index, strip in enumerate(strips):
            ordered_cats = strip.data.reindex(order_names)
            unique_cats = list(dict.fromkeys(ordered_cats.dropna()))
            color_map = _assign_colors(unique_cats, strip.colors)
            strip_color_maps.append(color_map)
            legend_blocks.append((strip.title or f"strip {index + 1}", color_map))

    if group_colors:
        legend_blocks.append((group_by.capitalize(), group_colors))

    has_legend = bool(legend_blocks)
    width_ratios = [10.0]
    if has_legend:
        width_ratios.append(2.6)

    # Subplots configuration
    apply_science_style()
    if figsize is None:
        width = 7.0 + (2.6 if has_legend else 0.0)
        height = 4.0 + 0.32 * n_strips + 1.1 * n_bars + (1.5 if has_corr else 0.0)
        figsize = (width, height)

    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(
        len(height_ratios),
        len(width_ratios),
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=0.02,
        hspace=0.08,
    )

    # 1. Main Manhattan Plot
    ax_main = fig.add_subplot(grid[0, 0])

    # Shade alternating groups (contiguous blocks of family)
    boundaries = []
    current_family = None
    start_idx = 0
    for idx, _name in enumerate(order_names):
        fam = df_ordered.loc[idx, group_by]
        if idx == 0:
            current_family = fam
        elif fam != current_family:
            boundaries.append((current_family, start_idx, idx - 1))
            current_family = fam
            start_idx = idx
    boundaries.append((current_family, start_idx, n - 1))

    for i, (_fam, start, end) in enumerate(boundaries):
        if i % 2 == 1:
            ax_main.axvspan(start - 0.5, end + 0.5, color="#F5F5F5", alpha=0.6, zorder=1)

    # Plot scatter
    x_coords = np.arange(n)
    y_coords = df_ordered["_y"].to_numpy()

    ax_main.scatter(
        x_coords,
        y_coords,
        c=colors,
        s=18,
        alpha=0.8,
        edgecolors="none",
        zorder=3,
    )

    # Draw reference lines
    ax_main.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    sig_df = df_ordered[df_ordered["_sig"]]
    if not sig_df.empty:
        max_sig_p = sig_df["p_value"].max()
        ax_main.axhline(
            -np.log10(max_sig_p), color="black", linestyle=":", linewidth=1.0, alpha=0.8
        )

    # Decoration
    ax_main.set_xlim(-0.5, n - 0.5)
    ax_main.set_ylim(0, max(1.5, float(df_ordered["_y"].max()) * 1.1))
    ax_main.set_ylabel(r"$-\log_{10}$(p-value)")

    # Ticks for families (centered on each block)
    tick_positions = [(start + end) / 2 for _, start, end in boundaries]
    tick_labels_main = [str(fam) for fam, _, _ in boundaries]
    ax_main.set_xticks(tick_positions)
    ax_main.set_xticklabels(tick_labels_main, fontsize=8)

    # Hide ticks and bottom spine if there are bottom tracks to avoid overlap
    if n_strips > 0 or n_bars > 0 or has_corr:
        ax_main.tick_params(labelbottom=False)

    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    # Determine tick labels for features (if small enough)
    tick_labels = order_names if n <= 60 else None

    # 2. Draw categorical strips
    if strips is not None and len(strips) > 0:
        _draw_top_strips(
            fig=fig,
            grid=grid,
            strips=strips,
            strip_color_maps=strip_color_maps,
            order_names=order_names,
            col_idx=0,
            row_offset=1,
        )

    # 3. Draw numeric bars
    if bars is not None and len(bars) > 0:
        tick_labels_bars = tick_labels if corr_panel is None else None
        _draw_bottom_bars(
            fig=fig,
            grid=grid,
            row_offset=1 + n_strips,
            bars=bars,
            labels_series=None,
            module_color_map=None,
            order_names=order_names,
            col_idx=0,
            tick_labels=tick_labels_bars,
        )

    # 4. Draw correlation panel (feature-by-variable correlation heatmap)
    if corr_panel is not None:
        corr_row = 1 + n_strips + n_bars
        panel = corr_panel.data.reindex(order_names).T
        ax_corr = fig.add_subplot(grid[corr_row, 0])
        corr_image = ax_corr.imshow(
            panel.to_numpy(dtype=float),
            aspect="auto",
            cmap=corr_panel.cmap,
            vmin=corr_panel.vmin,
            vmax=corr_panel.vmax,
            interpolation="nearest",
            origin="upper",
        )
        ax_corr.set_yticks(range(panel.shape[0]))
        ax_corr.set_yticklabels(list(panel.index), fontsize=7)
        ax_corr.set_xlim(-0.5, n - 0.5)

        if tick_labels is not None:
            ax_corr.set_xticks(range(n))
            ax_corr.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        else:
            ax_corr.set_xticks([])

        # Colorbar for correlation panel
        cbar_row = len(height_ratios) - 1
        corr_cbar_host = grid[cbar_row, 0].subgridspec(1, 3, width_ratios=[0.4, 0.2, 0.4])
        ax_corr_cbar = fig.add_subplot(corr_cbar_host[0, 1])
        fig.colorbar(
            corr_image,
            cax=ax_corr_cbar,
            orientation="horizontal",
            label=corr_panel.label,
        )

    # 5. Right-side stacked legend column
    if has_legend:
        legend_gs = grid[:, 1].subgridspec(
            len(legend_blocks),
            1,
            height_ratios=[len(block_map) + 1.5 for _, block_map in legend_blocks],
        )
        for block_index, (block_title, color_map) in enumerate(legend_blocks):
            ax_legend = fig.add_subplot(legend_gs[block_index])
            handles = [
                Patch(facecolor=color, edgecolor="0.3", label=str(category))
                for category, color in color_map.items()
            ]
            ax_legend.legend(
                handles=handles,
                title=block_title,
                loc="center left",
                frameon=False,
                fontsize=7,
                title_fontsize=8,
                handlelength=1.0,
            )
            ax_legend.axis("off")

    if title:
        fig.suptitle(title, weight="bold")

    fig.tight_layout()
    if path:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    return fig

