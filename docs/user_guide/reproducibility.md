# Radiomics Feature Reproducibility Analysis

Evaluating the robustness and reproducibility of machine learning features across different observers, readers, or software settings is a fundamental step in high-dimensional radiomics pipelines. Non-reproducible features can introduce substantial noise and bias, and should be excluded early in the modeling process.

`eigenradiomics` provides a publication-grade reproducibility framework featuring rigorous Quality Control (QC), multiple-rater statistics, formatted Excel reports, and accessible scientific visuals conforming strictly to OUP journals styling rules.

---

## Why ICC(2,1)?

The framework implements the **two-way random-effects, absolute-agreement, single-measure Intraclass Correlation Coefficient (ICC(2,1))** (McGraw & Wong, 1996; Shrout & Fleiss, 1979). 

* **Generalizability**: By modeling both the subjects (patients) and the observers (readers/settings) as random effects, the reliability coefficients generalize beyond the specific raters in the study to the broader population.
* **Absolute Agreement**: Standardizes absolute measurements rather than just relative ranks (which is critical in clinical imaging).
* **Multi-Observer Support**: The underlying ANOVA Mean Squares formulation natively handles any $K \ge 2$ observers concurrently.
* **Deterministic Bootstrapping**: Computes 95% Confidence Intervals deterministically via feature-name-keyed seed hashing (using BLAKE2b) to guarantee 100% reproducible results across platforms.

---

## Quality Control (QC) & Auto-Alignment

 Mismatched row orders (subjects) or column orders (features) across datasets can lead to catastrophic, silent statistical errors. `eigenradiomics` implements a strict validation layer:

1. **Name-Based Matching (Named DataFrames)**:
   * Asserts that all input DataFrames share the exact same sets of column names and row indices.
   * If any features or subjects are missing or unexpected across datasets, a detailed `ValueError` is raised (detailing the first 5 mismatched items).
   * If the columns or rows match exactly but are out of order, the framework **automatically reorders** them to align perfectly with the first dataset's layout.
2. **Positional Matching (Arrays / RangeIndexes)**:
   * Fallback for numpy arrays or RangeIndex DataFrames. Asserts that all datasets share identical shapes and aligns columns by positional indexes.

---

## 2-Observer Studies ($K = 2$)

For studies comparing exactly two readers, the framework reports detailed correlation metrics for every analyzed feature:
* **Spearman's $\rho$ and Pearson's $r$**: Correlation estimates.
* **95% Confidence Intervals**: Fisher-transformed confidence intervals for both metrics.
* **Multiple Testing Correction**: Raw $p$-values and FDR-corrected $p$-values (Benjamini-Hochberg procedure).

```python
import pandas as pd
from eigenradiomics import compute_reproducibility

# Reader 1 and Reader 2 feature DataFrames
df_reader1 = pd.read_csv("reader1_features.csv", index_col="PatientID")
df_reader2 = pd.read_csv("reader2_features.csv", index_col="PatientID")

# Calculate reproducibility
results = compute_reproducibility(
    datasets=[df_reader1, df_reader2],
    bootstrap_iterations=1000,
    primary_threshold=0.80
)

# results is a dictionary containing Spearman, Pearson, and ICC DataFrames
print(results["ICC"].head())
print(results["Spearman"].head())
```

---

## Multi-Observer Studies ($K > 2$)

When comparing three or more observers/settings, reporting a single correlation between two readers is insufficient. For $K > 2$, the framework computes all $\binom{K}{2}$ pairwise combinations and compiles aggregate statistics for the Spearman and Pearson sheets:
* **mean**: Arithmetic mean of pairwise correlations.
* **median**: Median of pairwise correlations.
* **sd**: Standard deviation of pairwise correlations (capturing observer consensus).
* **q25, q75**: 25th and 75th percentiles.
* **min, max**: Minimum and maximum pairwise correlations.

The ICC sheet computes the unified $n \times K$ table Intraclass Correlation Coefficient.

```python
# 3 different readers
results = compute_reproducibility(
    datasets=[df_reader1, df_reader2, df_reader3],
    bootstrap_iterations=1000
)

# View Spearman aggregate metrics across the rater pairs
print(results["Spearman"][["feature", "mean", "sd", "min", "max"]].head())
```

---

## Feature Selector Filtering

You can calculate reproducibility over a subset of target features using standard Pictologics-style selectors (`features`, `configs`, `families`, `family_groups`, and `catalog`). This behaves exactly like `RadiomicsFeatureRemover`:

```python
# Evaluate only first-order features belonging to the "original" config
results = compute_reproducibility(
    datasets=[df_reader1, df_reader2],
    features=["*Energy", "*Entropy"],
    configs="original",
)
```

---

## Polished Excel Reports

To share statistics with collaborators, export the results dictionary to a pristine Excel workbook. It automatically applies:
* **Frozen headers (`A2`)** and **active auto-filters** on all sheets (`Spearman`, `Pearson`, `ICC`).
* **Auto-fit column widths** with dynamic padding to prevent cropped values.
* **Sleek dark-navy fills** on the header row.
* **Tailored alignments & decimal formatting** (e.g. 3 decimal places for coefficients, 4 decimal places/scientific notation for $p$-values).

```python
from eigenradiomics import write_reproducibility_excel

write_reproducibility_excel(results, "reproducibility_report.xlsx")
```

---

## Accessible Scientific Histograms

Visualize metric distributions using publication-ready scientific plots built on the `scienceplots` package, with accessibility-focused styling (high-contrast colors, dark bar outlines, direct labeling, and sans-serif typography):

```python
from eigenradiomics import plot_reproducibility_histograms

fig = plot_reproducibility_histograms(
    results,
    path="reproducibility_distributions.png",
    primary_threshold=0.80
)
```

The figure shows the Spearman, Pearson, and ICC(2,1) distributions across
features, each with summary statistics and the retention threshold:

![Reproducibility metric histograms for Spearman, Pearson, and ICC](../assets/figures/reproducibility_histograms.png)

!!! tip "Choosing a threshold"
    A common convention treats ICC ≥ 0.75 as *good* and ≥ 0.90 as *excellent*
    reliability. The `primary_threshold` is what populates the `retained_*`
    flags — pick it to match your study's tolerance, then drop the features that
    fall below it before modeling (see the pipeline example below).

**Accessibility and Design features**:
* **Color Independence**: High-contrast, colorblind-friendly colors (Steel Blue, Indian Red, Muted Teal) assigned to each metric.
* **Dark Outlines**: Distinct outlines around histogram bars (`edgecolor='0.25'`) to ensure a contrast ratio $> 3:1$ against the background.
* **Direct Labeling**: The 0.80 cutoff threshold line is directly labeled with a contrasting bounding box, avoiding convoluted legends.
* **Sans-serif Typography**: Uses clear sans-serif typography (Arial/Helvetica) sized between 10 pt and 12 pt for reading accessibility.

---

## End-to-End scikit-learn Pipeline Integration

A standard workflow consists of running a reproducibility analysis, identifying features that fail to meet a reliability threshold (e.g. $ICC < 0.80$), and excluding them before downstream training:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from eigenradiomics import RadiomicsFeatureRemover, WGCNAReducer, compute_reproducibility

# 1. Run reproducibility analysis
results = compute_reproducibility([df_reader1, df_reader2])
icc_df = results["ICC"]

# 2. Identify non-reproducible features. Features whose ICC could not be
#    estimated (NaN, e.g. too few paired samples) are treated as
#    non-reproducible here via fillna(0.0); drop the fillna to keep them.
icc_values = icc_df["icc_2_1"].fillna(0.0)
non_reproducible = icc_df.loc[icc_values < 0.80, "feature"].tolist()

# 3. Feed non-reproducible features to RadiomicsFeatureRemover in a pipeline
pipeline = Pipeline([
    ("exclude_non_reproducible", RadiomicsFeatureRemover(features=non_reproducible)),
    ("scale", StandardScaler()),
    ("reduce", WGCNAReducer(soft_power=6, min_module_size=30))
])

# Fit and transform training data safely
X_train_clean = pipeline.fit_transform(df_reader1)
```
