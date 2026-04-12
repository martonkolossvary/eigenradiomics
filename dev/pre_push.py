#!/usr/bin/env python3
"""Pre-push workflow script for eigenradiomics.

Runs the local quality gates in a Pictologics-style workflow:
1. Clear caches
2. Run ruff
3. Run mypy
4. Run pytest with coverage
5. Run sklearn estimator checks
6. Generate docs quality page
7. Build MkDocs site
8. Build the package
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-push validation for eigenradiomics.")
    parser.add_argument(
        "--cache-clear-only",
        action="store_true",
        help="Clear caches and exit without running any checks.",
    )
    parser.add_argument(
        "--skip-cache-clear",
        action="store_true",
        help="Skip clearing cache directories before running checks.",
    )
    parser.add_argument(
        "--skip-mypy",
        action="store_true",
        help="Skip mypy type checking.",
    )
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Skip docs generation and MkDocs build.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip package build.",
    )
    return parser.parse_args()


def print_step(message: str) -> None:
    print(f"\n==> {message}")


def run_command(command: list[str], *, capture: bool = False) -> tuple[int, str, str]:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=capture,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def clear_caches() -> bool:
    print_step("Clearing cache directories")
    cache_paths = [
        PROJECT_ROOT / "__pycache__",
        PROJECT_ROOT / ".pytest_cache",
        PROJECT_ROOT / ".mypy_cache",
        PROJECT_ROOT / ".ruff_cache",
        PROJECT_ROOT / ".coverage",
        PROJECT_ROOT / "coverage.xml",
        PROJECT_ROOT / "site",
        PROJECT_ROOT / "eigenradiomics" / "__pycache__",
        PROJECT_ROOT / "eigenradiomics" / "reducers" / "__pycache__",
        PROJECT_ROOT / "tests" / "__pycache__",
        PROJECT_ROOT / "dev" / "__pycache__",
    ]

    for path in cache_paths:
        if not path.exists():
            continue
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)

    print("Caches cleared")
    return True


def run_ruff() -> bool:
    print_step("Running ruff")
    code, _, _ = run_command(["poetry", "run", "ruff", "check", "eigenradiomics", "tests", "dev"])
    print("Ruff passed" if code == 0 else "Ruff failed")
    return code == 0


def run_mypy() -> bool:
    print_step("Running mypy")
    code, _, _ = run_command(["poetry", "run", "mypy", "eigenradiomics"])
    print("Mypy passed" if code == 0 else "Mypy failed")
    return code == 0


def run_pytest() -> bool:
    print_step("Running pytest with coverage")
    code, _, _ = run_command(
        [
            "poetry",
            "run",
            "pytest",
            "-q",
            "--cov=eigenradiomics",
            "--cov-report=term-missing",
            "--cov-report=xml",
        ]
    )
    print("Pytest passed" if code == 0 else "Pytest failed")
    return code == 0


def run_estimator_checks() -> bool:
    print_step("Running sklearn estimator checks")
    code, _, _ = run_command(["poetry", "run", "python", "dev/check_estimators.py"])
    print("Estimator checks passed" if code == 0 else "Estimator checks failed")
    return code == 0


def run_docs() -> bool:
    print_step("Generating documentation pages")
    code_docs, _, _ = run_command(["poetry", "run", "python", "dev/generate_docs.py"])
    if code_docs != 0:
        print("Docs generation failed")
        return False

    print_step("Building MkDocs site")
    code_build, _, _ = run_command(["poetry", "run", "mkdocs", "build"])
    print("Docs build passed" if code_build == 0 else "Docs build failed")
    return code_build == 0


def run_package_build() -> bool:
    print_step("Building package")
    code, _, _ = run_command(["poetry", "build"])
    print("Package build passed" if code == 0 else "Package build failed")
    return code == 0


def main() -> int:
    args = parse_args()

    if args.cache_clear_only:
        clear_caches()
        return 0

    steps: list[tuple[str, bool]] = []

    if not args.skip_cache_clear:
        steps.append(("cache clear", clear_caches()))

    steps.append(("ruff", run_ruff()))

    if not args.skip_mypy:
        steps.append(("mypy", run_mypy()))

    steps.append(("pytest", run_pytest()))
    steps.append(("estimator checks", run_estimator_checks()))

    if not args.skip_docs:
        steps.append(("docs", run_docs()))

    if not args.skip_build:
        steps.append(("package build", run_package_build()))

    print_step("Summary")
    failed = [name for name, ok in steps if not ok]
    for name, ok in steps:
        print(f"- {name}: {'PASS' if ok else 'FAIL'}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
