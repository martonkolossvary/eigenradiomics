# Installation

## Prerequisites

- Python >= 3.10
- `pip` for installation or Poetry for local development

## Install from PyPI

Base package:

```bash
pip install eigenradiomics
```

With WGCNA support:

```bash
pip install 'eigenradiomics[wgcna]'
```

`PyWGCNA` is optional because only `WGCNAReducer` requires it.

With ComBat batch-effect harmonization support:

```bash
pip install 'eigenradiomics[combat]'
```

The `combat` extra installs `inmoose`, which powers the optional ComBat
sensitivity diagnostic in `compute_batch_effects` and the `ComBatHarmonizer`. It
requires Python >= 3.11; on Python 3.10 the ComBat step is skipped with a warning.

For [feature–outcome models](feature_models.md):

```bash
pip install 'eigenradiomics[survival,modeling]'
```

The `survival` extra installs `lifelines` (Cox models) and `modeling` installs
`statsmodels` (logistic, GLMM/GEE, MixedLM). Continuous OLS+HC3 models need
neither. You can combine extras, e.g.
`pip install 'eigenradiomics[wgcna,combat,survival,modeling]'`.

## Install from Source

Clone the repository and install locally:

```bash
cd <repo-dir>
pip install .
```

For editable development work:

```bash
cd <repo-dir>
poetry install --with dev --extras wgcna
```