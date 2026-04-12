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

The package is designed around a simple, robust mathematical contract: reducing wide feature arrays \(X \in \mathbb{R}^{n \times m}\) down to a highly constrained dense subspace \(Y \in \mathbb{R}^{n \times k}\), where \(n\) represents the number of samples and the preserved feature count is massively compressed (\(k \ll m\)). By universally supporting multiple reducers (e.g., WGCNA, and planned methods like PCA or NMF) under a single architecture, the framework explicitly preserves latent metrics—spanning mathematical factors, clusters, and loadings—strictly preventing training data leakage. This enables seamless transfer of the dimensionality reduction to new, unseen data, guaranteeing deterministic mapping for completely independent predictive cohorts downstream.

## Why eigenradiomics?

- **🧩 Modular framework**: Aggregate distinct dimensionality methods effortlessly without altering fundamental pipeline models or architectures locally.
- **⚙️ Scikit-learn-native behavior**: Fully exploits native execution logic allowing parameter bindings, hyper-parameter GridSearch scanning (`GridSearchCV`), clustering cross-validations, and `sklearn.pipeline.Pipeline` integrations cleanly natively out of the box globally. Every specific argument of a given reducer inherently stays fully exposed fundamentally.
- **🛡️ Safe handling of wide tables**: Evaluates and isolates DataFrame structures correctly avoiding silent misalignment failures frequently present during validation merges automatically.
- **📊 Reducer-specific diagnostics**: Empowers specific estimators universally (e.g. `WGCNAReducer` exposes specific internal dendrograms, cluster densities, and topologies actively independently).
- **🚀 Designed for reliable deployments**: Secure validation execution guarantees natively predicting outputs exactly identically across separate evaluation splits statically.

## Current Components

- **`WGCNAReducer`**: Native functional encapsulation of [WGCNA](https://pubmed.ncbi.nlm.nih.gov/19114008/) driven via the highly optimized [PyWGCNA](https://pubmed.ncbi.nlm.nih.gov/37399090/) backbone algorithm mathematically extracting networks reliably.

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