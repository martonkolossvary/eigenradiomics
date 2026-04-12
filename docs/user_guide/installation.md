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