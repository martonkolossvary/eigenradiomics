#!/usr/bin/env python3
"""End-to-end eigenradiomics workflow on synthetic-but-realistic data.

Wires every primitive into one pipeline, from raw tables to the cornerstone
heatmap:

    raw tables ─▶ ingest/merge ─▶ catalog + dataset ─▶ QC (reproducibility,
    batch effects) ─▶ preprocessing ─▶ WGCNA reduction ─▶ clustered heatmap

Run it:

    poetry run python examples/end_to_end.py

It prints what each step produced and writes the figures to ``examples/output/``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eigenradiomics import (  # noqa: E402
    Bar,
    FeatureCatalog,
    RadiomicsDataset,
    RadiomicsPrepTransformer,
    StudyDesign,
    WGCNAReducer,
    compute_batch_effects,
    compute_clinical_correlations,
    compute_reproducibility,
    merge_tables,
    normalize_id_column,
    plot_batch_effects,
    plot_clustered_heatmap,
    plot_reproducibility_histograms,
    resolve_duplicates,
)

RNG = np.random.default_rng(20)
OUT = Path(__file__).resolve().parent / "output"


# --------------------------------------------------------------------------
# 0. Synthetic raw inputs: radiomics (two readers), clinical metadata, catalog
# --------------------------------------------------------------------------
def make_raw_tables() -> dict[str, pd.DataFrame]:
    """Two reader tables, a messy clinical table, and a feature catalog."""
    n_patients, n_per_block, n_blocks = 90, 15, 4
    n_feat = n_per_block * n_blocks  # 60 features in 4 correlated blocks
    centers = RNG.choice(["Center A", "Center B", "Center C"], n_patients)

    # Block-correlated feature signal + a center-specific offset (batch effect).
    blocks, latents = [], {}
    for b in range(n_blocks):
        latent = RNG.standard_normal(n_patients)
        latents[b] = latent
        for _ in range(n_per_block):
            blocks.append(latent + RNG.standard_normal(n_patients) * 0.4)
    signal = np.column_stack(blocks)
    offset = {"Center A": 0.0, "Center B": 1.4, "Center C": 2.6}
    signal[:, : n_feat // 2] += np.array([offset[c] for c in centers])[:, None]

    cols = [f"original__feat_{i}" for i in range(n_feat)]
    ids = [f"P{i:03d}" for i in range(n_patients)]
    reader1 = pd.DataFrame(signal, columns=cols)
    reader1.insert(0, "PatientID", ids)
    # Reader 2: same patients, per-feature noise spanning high -> low reliability.
    reliability_noise = np.linspace(0.1, 1.8, n_feat)
    reader2 = pd.DataFrame(
        signal + RNG.standard_normal(signal.shape) * reliability_noise, columns=cols
    )
    reader2.insert(0, "PatientID", ids)

    # Clinical: messy IDs (whitespace), one duplicate row, mixed-type variables.
    clinical = pd.DataFrame(
        {
            "PatientID": [f"  {i} " for i in ids],  # stray whitespace
            "Center": centers,
            "Age": (60 + 9 * RNG.standard_normal(n_patients)).round(1),
            "Biomarker": latents[0] * 1.6 + RNG.standard_normal(n_patients),  # tied to block 0
            "Sex": RNG.choice(["male", "female"], n_patients),
            "Stage": RNG.choice(["I", "II", "III", "IV"], n_patients),
            "Event": RNG.integers(0, 2, n_patients),
        }
    )
    clinical = pd.concat([clinical, clinical.iloc[[0]]], ignore_index=True)  # a dup row

    # Feature catalog: one row per feature with family / family_group metadata.
    family = ["firstorder", "glcm", "glcm", "shape"]
    family_group = ["Intensity", "Texture", "Texture", "Morphology"]
    catalog = pd.DataFrame(
        {
            "feature": cols,
            "family": [family[i // n_per_block] for i in range(n_feat)],
            "family_group": [family_group[i // n_per_block] for i in range(n_feat)],
        }
    )
    return {"reader1": reader1, "reader2": reader2, "clinical": clinical, "catalog": catalog}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    raw = make_raw_tables()
    feature_cols = [c for c in raw["reader1"].columns if c.startswith("original__")]

    # ----------------------------------------------------------------------
    # 1. Ingest: normalize IDs, resolve duplicate clinical rows, merge tables
    # ----------------------------------------------------------------------
    clinical, _ = normalize_id_column(raw["clinical"], "PatientID")
    clinical, dup_report = resolve_duplicates(clinical, "PatientID", policy="first")
    merge = merge_tables(
        raw["reader1"], clinical, left_on="PatientID", right_on="PatientID",
        how="left", validate="1:1",
    )
    table = merge.merged
    print(f"1. ingest: dropped {len(dup_report)} duplicate clinical row(s); "
          f"merged {merge.n_matched} patients, {len(merge.right_only)} clinical-only.")

    # ----------------------------------------------------------------------
    # 2. Catalog + Dataset: carry features, metadata, and study design together
    # ----------------------------------------------------------------------
    catalog = FeatureCatalog(raw["catalog"])
    dataset = RadiomicsDataset(
        table,
        feature_columns=feature_cols,
        catalog=catalog,
        design=StudyDesign(roles={"group": "PatientID", "batch": "Center", "event": "Event"}),
    )
    print(f"2. dataset: {dataset!r} | families={catalog.family_groups()}")

    # ----------------------------------------------------------------------
    # 3. QC: two-reader reproducibility and multi-center batch effects
    # ----------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        repro = compute_reproducibility(
            [raw["reader1"][feature_cols], raw["reader2"][feature_cols]],
            bootstrap_iterations=200,
        )
        plot_reproducibility_histograms(repro, OUT / "01_reproducibility.png")
        batch = compute_batch_effects(
            dataset.features, dataset.metadata["Center"], permutations=200,
        )
        plot_batch_effects(batch, OUT / "02_batch_effects.png")
    icc = repro["ICC"].set_index("feature")["icc_2_1"]
    print(f"3. QC: {int((icc >= 0.80).sum())}/{len(icc)} features reach ICC >= 0.80; "
          f"reproducibility + batch figures written.")

    # ----------------------------------------------------------------------
    # 4. Preprocess: winsorize -> Yeo-Johnson -> z-score (scikit-learn API)
    # ----------------------------------------------------------------------
    X = RadiomicsPrepTransformer().fit_transform(dataset.features)
    print(f"4. preprocess: transformed {X.shape[1]} features for {X.shape[0]} samples.")

    # ----------------------------------------------------------------------
    # 5. Reduce: WGCNA modules + eigengenes; keep the structured artifacts
    # ----------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = WGCNAReducer(
            soft_power="auto", min_module_size=10, store_tom=True, verbose=0
        )
        eigengenes = reducer.fit_transform(X)
    artifacts = reducer.get_reduction_artifacts()
    names = list(artifacts.feature_names)
    print(f"5. reduce: {X.shape[1]} features -> {eigengenes.shape[1]} module eigengenes; "
          f"artifacts available: {artifacts.available()}")

    # ----------------------------------------------------------------------
    # 6. Clinical correlations: features vs clinical variables (for the panel)
    # ----------------------------------------------------------------------
    corr = compute_clinical_correlations(
        X, dataset.metadata[["Age", "Biomarker", "Sex", "Stage", "Event"]],
        method="spearman", min_pairs=20,
    )

    # ----------------------------------------------------------------------
    # 7. Visualize: the cornerstone heatmap tying every primitive together
    # ----------------------------------------------------------------------
    families = catalog.frame.set_index("feature")["family_group"].reindex(names)
    families.name = "Family"
    fig = plot_clustered_heatmap(
        artifacts,                                   # TOM similarity, linkage, modules, order
        top=[families],                              # catalog family-group strip
        bottom=[Bar(icc.reindex(names), title="ICC", reference=0.80)],  # reproducibility
        right=corr,                                  # feature x clinical correlations
        cmap="magma", vmin=0.0, vmax=1.0, below_cutoff_color="#050505",
        colorbar_label="Topological overlap",
        title="End-to-end: WGCNA modules, reproducibility & clinical links",
    )
    fig.savefig(OUT / "03_cornerstone_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"7. visualize: cornerstone heatmap written to {OUT / '03_cornerstone_heatmap.png'}")


if __name__ == "__main__":
    main()
