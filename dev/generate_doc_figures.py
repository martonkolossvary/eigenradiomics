#!/usr/bin/env python3
"""Generate example output figures for the documentation.

Runs eigenradiomics' own visualization utilities on synthetic-but-realistic
data and saves the resulting figures to ``docs/assets/figures/``. These show the
package's actual output rather than mock-ups. Re-run after changing a plotting
function:

    poetry run python dev/generate_doc_figures.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eigenradiomics import (  # noqa: E402
    Bar,
    RadiomicsPrepTransformer,
    WGCNAReducer,
    compute_batch_effects,
    compute_clinical_correlations,
    compute_reproducibility,
    plot_batch_effects,
    plot_clustered_heatmap,
    plot_reproducibility_histograms,
)

FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(7)


def _save(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(FIG_DIR.parent.parent.parent)}")


def reproducibility_figure() -> None:
    """Two-reader reproducibility with a spread of feature reliabilities."""
    n, n_feat = 90, 45
    cols = [f"original__feat_{i}" for i in range(n_feat)]
    base = RNG.standard_normal((n, n_feat))
    reader1 = pd.DataFrame(base, columns=cols, index=[f"S{i}" for i in range(n)])
    # Reader 2 = reader 1 + per-feature noise whose scale spans high -> low reliability.
    noise_scale = np.linspace(0.1, 2.2, n_feat)
    reader2 = pd.DataFrame(
        base + RNG.standard_normal((n, n_feat)) * noise_scale,
        columns=cols,
        index=reader1.index,
    )
    results = compute_reproducibility([reader1, reader2], bootstrap_iterations=200)
    fig = plot_reproducibility_histograms(results, primary_threshold=0.80)
    _save(fig, "reproducibility_histograms.png")


def batch_effects_figure() -> None:
    """Three-center batch effect, visible in PCA and reduced by ComBat."""
    per_batch = 30
    n_feat = 30
    batches = np.repeat(["Center A", "Center B", "Center C"], per_batch)
    n = len(batches)
    signal = RNG.standard_normal((n, n_feat))
    # Inject a center-specific offset on the first two-thirds of features.
    offset = {"Center A": 0.0, "Center B": 1.6, "Center C": 3.1}
    shift = np.array([offset[b] for b in batches])
    affected = n_feat * 2 // 3
    signal[:, :affected] += shift[:, None] * RNG.uniform(0.6, 1.2, affected)
    cols = [f"original__feat_{i}" for i in range(n_feat)]
    X = pd.DataFrame(signal, columns=cols, index=[f"S{i}" for i in range(n)])
    batch = pd.Series(batches, index=X.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = compute_batch_effects(
            X, batch, permutations=200, min_valid_samples=10,
            min_valid_per_batch=5, min_batches_per_feature=3, no_combat=False,
        )
        fig = plot_batch_effects(results, primary_alpha=0.05)
    _save(fig, "batch_effects.png")


def _wgcna_matrix() -> np.ndarray:
    """50 samples x 200 features in 5 correlated blocks (+ noise)."""
    n, groups, per = 50, 5, 38
    cols = []
    for _ in range(groups):
        latent = RNG.standard_normal(n)
        for _ in range(per):
            cols.append(latent + RNG.standard_normal(n) * 0.35)
    for _ in range(200 - groups * per):
        cols.append(RNG.standard_normal(n))
    return np.column_stack(cols)


def wgcna_figures() -> None:
    """WGCNA soft-power diagnostic and feature dendrogram."""
    X = _wgcna_matrix()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = WGCNAReducer(soft_power="auto", min_module_size=20, verbose=0)
        reducer.fit(X)
    _save(reducer.wgcna_plot_soft_power(figsize=(9, 4)), "wgcna_soft_power.png")
    _save(reducer.wgcna_plot_dendrogram(figsize=(11, 4)), "wgcna_dendrogram.png")


def preprocessing_figure() -> None:
    """Before/after of RadiomicsPrepTransformer on a skewed, outlier-laden feature."""
    n = 400
    raw = np.exp(RNG.normal(0.0, 1.0, n))  # right-skewed (log-normal)
    raw[:8] = raw.max() * RNG.uniform(3, 6, 8)  # extreme outliers
    df = pd.DataFrame({"original__skewed": raw})
    transformed = RadiomicsPrepTransformer(winsor_lower=0.01, winsor_upper=0.99).fit_transform(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.hist(raw, bins=40, color="#CD5C5C", edgecolor="0.25", linewidth=0.6)
    ax1.set_title("Raw feature", weight="bold")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Count")
    ax2.hist(
        transformed["original__skewed"], bins=40, color="#4682B4",
        edgecolor="0.25", linewidth=0.6,
    )
    ax2.set_title("After winsorize → Yeo-Johnson → z-score", weight="bold")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Count")
    fig.tight_layout()
    _save(fig, "preprocessing_before_after.png")


def clustered_heatmap_figure() -> None:
    """Cornerstone heatmap: WGCNA TOM + top family strip, bottom bar, and clinical panel."""
    matrix = _wgcna_matrix()
    cols = [f"original__feat_{i}" for i in range(matrix.shape[1])]
    samples = [f"S{i}" for i in range(matrix.shape[0])]
    X = pd.DataFrame(matrix, columns=cols, index=samples)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = WGCNAReducer(soft_power="auto", min_module_size=20, store_tom=True, verbose=0)
        reducer.fit(X)
    artifacts = reducer.get_reduction_artifacts()
    names = list(artifacts.feature_names)
    families = ["Intensity", "Texture", "Morphology"]
    family = pd.Series([families[i % 3] for i in range(len(names))], index=names, name="Family")
    # A synthetic per-feature -log10 p-value, larger inside the real signal blocks.
    signal = np.r_[np.abs(RNG.normal(2.5, 0.8, 190)), np.abs(RNG.normal(0.4, 0.3, 10))]
    neglogp = pd.Series(signal[: len(names)], index=names, name="-log10 p")
    # Clinical variables tied to specific correlated blocks (+ a categorical, a stage).
    clinical = pd.DataFrame(
        {
            "Biomarker A": matrix[:, :38].mean(axis=1) + RNG.normal(0, 0.4, matrix.shape[0]),
            "Biomarker B": matrix[:, 38:76].mean(axis=1) + RNG.normal(0, 0.4, matrix.shape[0]),
            "Outcome": matrix[:, 76:114].mean(axis=1) + RNG.normal(0, 0.6, matrix.shape[0]),
            "Sex": RNG.choice(["male", "female"], matrix.shape[0]),
            "Stage": RNG.choice(["I", "II", "III", "IV"], matrix.shape[0]),
        },
        index=samples,
    )
    corr = compute_clinical_correlations(X, clinical, method="spearman", min_pairs=20)
    fig = plot_clustered_heatmap(
        artifacts,
        top=[family],
        bottom=[Bar(neglogp, title="-log10 p", reference=float(-np.log10(0.05)))],
        right=corr,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        below_cutoff_color="#050505",
        colorbar_label="Topological overlap",
        title="WGCNA TOM heatmap",
    )
    _save(fig, "clustered_heatmap.png")


def main() -> None:
    reproducibility_figure()
    batch_effects_figure()
    wgcna_figures()
    preprocessing_figure()
    clustered_heatmap_figure()
    print("done")


if __name__ == "__main__":
    main()
