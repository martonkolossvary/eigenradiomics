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
sensitivity diagnostic in `compute_batch_effects`. It requires Python >= 3.11;
on Python 3.10 the ComBat step is skipped with a warning. You can combine
extras, e.g. `pip install 'eigenradiomics[wgcna,combat]'`.

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