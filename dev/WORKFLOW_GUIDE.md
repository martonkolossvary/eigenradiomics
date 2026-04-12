# eigenradiomics Developer Workflow Guide

Step-by-step instructions for building, testing, documenting, and releasing eigenradiomics.

---

## Prerequisites

- **Python 3.10+** (locally tested on 3.12)
- **Poetry** for dependency management
- **Git** (repository: `github.com/martonkolossvary/eigenradiomics`)

```bash
poetry install --with dev --extras wgcna
```

---

## Command Line Reference

### pre_push.py — Pre-Push Validation

```bash
# Full check (runs everything)
poetry run python dev/pre_push.py

# Quick development check (skip docs and build)
poetry run python dev/pre_push.py --skip-docs --skip-build

# Skip options
poetry run python dev/pre_push.py --skip-mypy
poetry run python dev/pre_push.py --skip-cache-clear

# Clear caches only
poetry run python dev/pre_push.py --cache-clear-only
```

### generate_docs.py — Documentation Generation

```bash
# Generate the quality report page (docs/quality.md)
poetry run python dev/generate_docs.py
```

### check_estimators.py — sklearn Estimator Validation

```bash
# Run sklearn's check_estimator on all public estimators
poetry run python dev/check_estimators.py
```

---

## VS Code Build Tasks

Press **Cmd+Shift+B** (Mac) or **Ctrl+Shift+B** (Windows/Linux) for the task list.

### Individual Checks

| Task | Runs |
|------|------|
| Run Tests | `pytest -q --cov` |
| Ruff: Check | Linter (check only) |
| Ruff: Fix | Linter with auto-fix |
| Mypy: Type Check | Static type analysis |
| Estimator Checks | `sklearn.check_estimator` for all reducers |

### Pre-Push Workflows

| Task | Runs |
|------|------|
| Pre-Push: Quick Check | Ruff + Mypy + Pytest (skip docs & build) |
| Pre-Push: Full Check | Ruff + Mypy + Pytest + Estimator checks + Docs + Build |

### Documentation

| Task | Runs |
|------|------|
| Generate Docs | Regenerate `docs/quality.md` |
| Serve Docs Locally | `mkdocs serve` on localhost:8000 |

### Build & Cleanup

| Task | Runs |
|------|------|
| Build Package | `poetry build` |
| Clean Caches | Remove all cache directories |

---

## Quality Gates

Every push/PR triggers CI (`ci.yml`) which runs across Python 3.10, 3.11, 3.12:

| Gate | Tool | Config |
|------|------|--------|
| Linting | `ruff check eigenradiomics tests` | `pyproject.toml [tool.ruff]` |
| Type checking | `mypy eigenradiomics` | `pyproject.toml [tool.mypy]` (strict) |
| Tests + Coverage | `pytest --cov` | `pyproject.toml [tool.pytest]`, `[tool.coverage]` |
| Coverage upload | Codecov (Python 3.12 only) | `CODECOV_TOKEN` secret |
| Estimator checks | `dev/check_estimators.py` | `sklearn.check_estimator` |
| Docs build | `mkdocs build` | `mkdocs.yml` |
| Docs deploy | GitHub Pages (main only) | `peaceiris/actions-gh-pages` |

---

## CI/CD Workflows

### `ci.yml` — Continuous Integration

- **Trigger**: Every push and pull request.
- **Jobs**: `test` (matrix: 3.10, 3.11, 3.12) → `docs` (on main only).
- Installs via `snok/install-poetry@v1` with venv caching.

### `publish-pypi.yml` — Release to PyPI

- **Trigger**: GitHub release published (non-prerelease).
- Syncs package version from the release tag.
- Uses **Trusted Publisher (OIDC)** — no API token needed.
- Also rebuilds and deploys docs to GitHub Pages.

### `publish-testpypi.yml` — Release to TestPyPI

- **Trigger**: GitHub pre-release or manual `workflow_dispatch`.
- Same flow as PyPI but targets `https://test.pypi.org/legacy/`.

---

## Version Bumping

Update in **two** places:

1. `pyproject.toml` line 3: `version = "X.Y.Z"`
2. `eigenradiomics/_version.py`: `__version__ = "X.Y.Z"`

Follow [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

---

## Changelog Management

Managed via [towncrier](https://towncrier.readthedocs.io/).

1. Add a fragment to `changelog.d/`:
   ```
   changelog.d/<issue-number>.<type>.md
   ```
   Types: `added`, `changed`, `fixed`, `removed`, `deprecated`.

2. Build changelog before release:
   ```bash
   poetry run towncrier build --version X.Y.Z
   ```
   This updates `docs/changelog.md` and removes the fragments.

---

## Release Checklist

1. Run full pre-push checks:
   ```bash
   poetry run python dev/pre_push.py
   ```
2. Bump version in `pyproject.toml` and `eigenradiomics/_version.py`.
3. Build changelog:
   ```bash
   poetry run towncrier build --version X.Y.Z
   ```
4. Regenerate docs:
   ```bash
   poetry run python dev/generate_docs.py
   ```
5. Commit and push:
   ```bash
   git add -A && git commit -m "release: vX.Y.Z"
   git push origin main
   ```
6. Create a GitHub release with tag `vX.Y.Z` → triggers `publish-pypi.yml`.

### Pre-Release (TestPyPI)

Same as above but create a **pre-release** on GitHub (tag `vX.Y.Za1`) → triggers `publish-testpypi.yml`.

---

## Trusted Publisher Setup (One-Time)

### PyPI

1. Go to https://pypi.org/manage/account/publishing/
2. Add: Project = `eigenradiomics`, Owner = `martonkolossvary`, Repo = `eigenradiomics`, Workflow = `publish-pypi.yml`, Environment = `pypi`

### TestPyPI

1. Go to https://test.pypi.org/manage/account/publishing/
2. Add: Project = `eigenradiomics`, Owner = `martonkolossvary`, Repo = `eigenradiomics`, Workflow = `publish-testpypi.yml`, Environment = `testpypi`

---

## Project Structure

```
eigenradiomics/
├── .github/workflows/        # CI/CD pipelines
│   ├── ci.yml                # Test + lint + docs
│   ├── publish-pypi.yml      # Release publish
│   └── publish-testpypi.yml  # Pre-release publish
├── changelog.d/              # Towncrier fragments
├── dev/                      # Developer scripts (not shipped)
│   ├── WORKFLOW_GUIDE.md     # This file
│   ├── check_estimators.py   # sklearn estimator validation
│   ├── generate_docs.py      # Quality report generation
│   └── pre_push.py           # Pre-push quality workflow
├── docs/                     # MkDocs source
├── eigenradiomics/           # Package source
│   ├── __init__.py
│   ├── _version.py           # Single version source
│   ├── _utils.py             # Shared utilities
│   └── reducers/             # Reducer implementations
│       ├── _base.py          # BaseReducer ABC
│       └── _wgcna.py         # WGCNAReducer
├── tests/                    # Test suite
├── mkdocs.yml                # Documentation config
├── pyproject.toml            # Poetry + tool config
└── poetry.lock               # Locked dependencies
```

---

## Troubleshooting

- **Tests fail locally**: Run `poetry run python dev/pre_push.py --cache-clear-only` then `poetry install --with dev --extras wgcna`.
- **Mypy errors in test files**: The `[[tool.mypy.overrides]]` for `tests.*` disables `disallow_untyped_defs`.
- **CI workflow fails**: Check the Actions tab; verify Codecov token and Trusted Publisher config.
- **Docs won't build**: Ensure `dev/generate_docs.py` runs first (generates `docs/quality.md`).
