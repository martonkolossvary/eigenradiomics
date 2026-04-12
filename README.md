# eigenradiomics

[![CI](https://github.com/martonkolossvary/eigenradiomics/actions/workflows/ci.yml/badge.svg)](https://github.com/martonkolossvary/eigenradiomics/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://martonkolossvary.github.io/eigenradiomics/)
[![PyPI](https://img.shields.io/pypi/v/eigenradiomics)](https://pypi.org/project/eigenradiomics/)
[![Python](https://img.shields.io/pypi/pyversions/eigenradiomics)](https://pypi.org/project/eigenradiomics/)
[![Downloads](https://img.shields.io/pypi/dm/eigenradiomics)](https://pypi.org/project/eigenradiomics/)
[![License](https://img.shields.io/github/license/martonkolossvary/eigenradiomics)](https://github.com/martonkolossvary/eigenradiomics/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/martonkolossvary/eigenradiomics/graph/badge.svg)](https://codecov.io/gh/martonkolossvary/eigenradiomics)
[![Ruff](https://img.shields.io/badge/ruff-0%20issues-261230.svg)](https://github.com/astral-sh/ruff)
[![Mypy](https://img.shields.io/badge/mypy-0%20errors-blue.svg)](https://mypy-lang.org/)

**eigenradiomics** is a modular, scikit-learn-compatible dimensionality reduction framework for high-dimensional feature matrices, with an initial focus on wide radiomics outputs such as the tables produced by [Pictologics](https://github.com/martonkolossvary/pictologics).

Documentation (User Guide, API, Recipes): https://martonkolossvary.github.io/eigenradiomics/

## Goals

- Accept generic `n_samples x n_features` matrices from numpy arrays or pandas DataFrames.
- Support training once and transforming unseen data later with the same fitted reducer.
- Preserve scikit-learn ergonomics: pipelines, parameter grids, feature-name checks, and estimator-style APIs.
- Stay modular so new reducers can be added without changing the package architecture.

## Current Components

Currently, the package is focused on evaluating the WGCNA implementation to serve as a stable scaffolding and UI model before expanding.

- `WGCNAReducer`: module-based reduction using PyWGCNA network construction plus stored SVD loadings for transform-time projection.

Use standard sklearn-compatible preprocessors upstream in a normal `Pipeline`.

## Installation

Base package:

```bash
pip install eigenradiomics
```

With WGCNA support:

```bash
pip install 'eigenradiomics[wgcna]'
```

For local development:

```bash
poetry install --with dev --extras wgcna
```

## Quick Start

```python
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eigenradiomics import WGCNAReducer

X = pd.read_csv("radiomics_features.csv")

pipe = Pipeline(
	[
		("impute", SimpleImputer(strategy="median")),
		("var", VarianceThreshold(threshold=0.0)),
		("scale", StandardScaler()),
		("reduce", WGCNAReducer(soft_power="auto", min_module_size=30)),
	]
)

Y = pipe.fit_transform(X)
```

## Design Notes

- Reducers inherit from a common base class and use reducer-specific output names such as `wgcna_0`, `wgcna_1`, ...
- numpy inputs are supported; synthetic feature names are generated internally so fitted reducers can still validate future inputs safely.
- The package framework is designed as a stable scaffolding for future add-ins such as PCA, sparse PCA, dictionary learning, NMF, and encoder-based reducers. (Currently, only WGCNA is implemented for initial evaluation.)

## Quality Checks

GitHub Actions runs the core quality gates on every push and pull request across Python 3.10, 3.11, and 3.12:

- `ruff check eigenradiomics tests`
- `pytest -q`
- `sklearn.utils.estimator_checks.check_estimator(...)` for all public estimators

The same checks can be run locally after installing the dev dependencies:

```bash
poetry install --with dev --extras wgcna
poetry run ruff check eigenradiomics tests
poetry run pytest -q
```

## Documentation

The project now includes a full MkDocs site with a Pictologics-style layout: a tutorial-first
user guide, API reference pages powered by docstrings, and a generated quality report.

To build the documentation locally:

```bash
poetry install --with dev --extras wgcna
poetry run python dev/generate_docs.py
poetry run mkdocs serve
```

The site is built in CI and deployed to GitHub Pages from `main`.
