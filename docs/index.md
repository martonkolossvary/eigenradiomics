# Welcome to eigenradiomics

[![CI](https://github.com/martonkolossvary/eigenradiomics/actions/workflows/ci.yml/badge.svg)](https://github.com/martonkolossvary/eigenradiomics/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://martonkolossvary.github.io/eigenradiomics/)
[![PyPI](https://img.shields.io/pypi/v/eigenradiomics)](https://pypi.org/project/eigenradiomics/)
[![Python](https://img.shields.io/pypi/pyversions/eigenradiomics)](https://pypi.org/project/eigenradiomics/)
[![Downloads](https://img.shields.io/pepy/dt/eigenradiomics)](https://pypi.org/project/eigenradiomics/)
[![License](https://img.shields.io/github/license/martonkolossvary/eigenradiomics)](https://github.com/martonkolossvary/eigenradiomics/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/martonkolossvary/eigenradiomics/graph/badge.svg)](https://codecov.io/gh/martonkolossvary/eigenradiomics)
[![Ruff](https://img.shields.io/badge/ruff-0%20issues-261230.svg)](https://github.com/astral-sh/ruff)
[![Mypy](https://img.shields.io/badge/mypy-0%20errors-blue.svg)](https://mypy-lang.org/)

![eigenradiomics icon](assets/logo.svg){ align=right width=180 }

**eigenradiomics** is a modular, scikit-learn-compatible dimensionality reduction framework for wide feature matrices, with an initial focus on radiomics tables such as the outputs produced by [Pictologics](https://github.com/martonkolossvary/pictologics).

The package is built around a simple contract: reduce a wide feature matrix \(X \in \mathbb{R}^{n \times m}\) to a compact representation \(Y \in \mathbb{R}^{n \times k}\) with \(k \ll m\), where \(n\) is the number of samples. Each reducer learns its mapping (factors, clusters, loadings) from the training data only and stores it, so the same transformation can later be applied deterministically to new, unseen samples without re-fitting — avoiding leakage between cohorts. A single architecture hosts multiple reducers (currently WGCNA; PCA and NMF are planned).

## Why eigenradiomics?

- **🧩 Modular framework**: Add new dimensionality-reduction methods without changing the pipeline architecture.
- **⚙️ Scikit-learn-native**: Works out of the box with `sklearn.pipeline.Pipeline`, `GridSearchCV`, and cross-validation; every reducer parameter is exposed for tuning.
- **🛡️ Safe handling of wide tables**: Validates DataFrame feature names and order across `fit` and `transform`, so column misalignment is caught instead of silently producing wrong results.
- **📊 Reducer-specific diagnostics**: Estimators expose their internals — for example, `WGCNAReducer` provides dendrograms, module sizes, and soft-power diagnostics.
- **🚀 Reproducible by design**: A fitted reducer maps new data deterministically, giving identical outputs across runs and evaluation splits.

## Current Components

- **`WGCNAReducer`**: Module-based reduction using [WGCNA](https://pubmed.ncbi.nlm.nih.gov/19114008/) network construction (via [PyWGCNA](https://pubmed.ncbi.nlm.nih.gov/37399090/)), representing each module of correlated features by its eigengene.

## Documentation Map

1. Start with the [Quick Start](user_guide/quick_start.md) guide for the shortest path from input table to reduced representation.
2. Read [Input Data Model](user_guide/input_data_model.md) to understand feature identity guarantees and transform semantics.
3. Use [Pipelines and Grid Search](user_guide/pipelines_and_grid_search.md) for sklearn preprocessing and integration patterns.
4. Dive into [WGCNA Reducer](reducers/wgcna.md) for the current main backend.
5. Refer to the [API Reference](api/reducers.md) for full object and method documentation.



## Getting Started

- [Installation](user_guide/installation.md)
- [Quick Start](user_guide/quick_start.md)
- [WGCNA Reducer](reducers/wgcna.md)