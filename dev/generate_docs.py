#!/usr/bin/env python3
"""Generate documentation report pages for eigenradiomics."""

from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from dev.check_estimators import run_estimator_checks
except ModuleNotFoundError:  # pragma: no cover - script execution path
    from check_estimators import run_estimator_checks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
QUALITY_PATH = DOCS_DIR / "quality.md"


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    # Try poetry first; if it fails, drop the "poetry run" prefix.
    import shutil

    executable = command[0]
    if executable == "poetry" and shutil.which("poetry") is None:
        # Strip "poetry run" and use the actual underlying tool
        command = command[2:]

    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def summarize_pytest() -> tuple[bool, str, str]:
    result = run_command(
        [
            "poetry",
            "run",
            "pytest",
            "-q",
            "--cov=eigenradiomics",
            "--cov-report=xml",
            "--cov-report=term-missing",
        ]
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    summary = next(
        (line.strip() for line in reversed(output.splitlines()) if line.strip()),
        "No pytest output captured.",
    )
    return result.returncode == 0, summary, output


def summarize_coverage() -> tuple[bool, str, str]:
    coverage_file = PROJECT_ROOT / "coverage.xml"
    if not coverage_file.exists():
        return False, "coverage.xml missing", "Coverage report was not generated."

    try:
        tree = ET.parse(coverage_file)
        root = tree.getroot()
        percent = float(root.attrib.get("line-rate", 0.0)) * 100
    except (ET.ParseError, ValueError) as exc:
        return False, "Coverage parsing failed", str(exc)

    return True, f"{percent:.2f}% Coverage", f"coverage={percent:.2f}%"


def summarize_ruff() -> tuple[bool, str, str]:
    result = run_command(
        [
            "poetry",
            "run",
            "ruff",
            "check",
            "eigenradiomics",
            "tests",
            "dev",
            "--output-format",
            "json",
        ]
    )
    if result.returncode == 0:
        return True, "0 issues", result.stdout

    try:
        issues = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False, "Ruff reported issues", result.stdout + result.stderr

    return False, f"{len(issues)} issues", result.stdout


def summarize_mypy() -> tuple[bool, str, str]:
    result = run_command(["poetry", "run", "mypy", "eigenradiomics"])
    output = (result.stdout + "\n" + result.stderr).strip()

    if result.returncode == 0:
        return True, "0 Errors", output

    error_count = output.count("error:")
    if error_count == 0:
        error_count = 1
    return False, f"{error_count} Errors", output


def summarize_estimator_checks() -> tuple[bool, list[tuple[str, bool, str]]]:
    return run_estimator_checks()


def summarize_build() -> tuple[bool, str, str]:
    result = run_command(["poetry", "build"])
    output = (result.stdout + "\n" + result.stderr).strip()
    summary = next(
        (line.strip() for line in reversed(output.splitlines()) if line.strip()),
        "No build output captured.",
    )
    return result.returncode == 0, summary, output


def format_status(ok: bool, detail: str) -> str:
    icon = "✅" if ok else "❌"
    word = "Pass" if ok else "Fail"
    return f"**Status:** {icon} {word} ({detail})"


def render_quality_page() -> tuple[str, bool]:
    pytest_ok, pytest_summary, pytest_output = summarize_pytest()
    coverage_ok, coverage_summary, coverage_output = summarize_coverage()
    mypy_ok, mypy_summary, mypy_output = summarize_mypy()
    ruff_ok, ruff_summary, ruff_output = summarize_ruff()
    estimator_ok, estimator_results = summarize_estimator_checks()
    build_ok, build_summary, build_output = summarize_build()

    estimator_rows = "\n".join(
        f"| {name} | {'✅ Pass' if ok else '❌ Fail'} | {detail} |"
        for name, ok, detail in estimator_results
    )

    parts = [
        "# Code Quality Report",
        "",
        "This page is generated automatically by `dev/generate_docs.py`.",
        "",
        "## Test Coverage",
        "",
        format_status(pytest_ok and coverage_ok, coverage_summary),
        "",
        "## Static Type Checking (Mypy)",
        "",
        format_status(mypy_ok, mypy_summary),
        "",
        "## Linting (Ruff)",
        "",
        format_status(ruff_ok, ruff_summary),
        "",
        "## sklearn Estimator Checks",
        "",
        format_status(
            estimator_ok,
            (
                f"{sum(ok for _, ok, _ in estimator_results)}/"
                f"{len(estimator_results)} estimators passed"
            ),
        ),
        "",
        "| Estimator | Status | Detail |",
        "|:--|:--:|:--|",
        estimator_rows,
        "",
        "## Package Build",
        "",
        format_status(build_ok, build_summary),
    ]

    if not pytest_ok:
        parts.extend(["", "```text", pytest_output.strip(), "```"])
    elif not coverage_ok:
        parts.extend(["", "```text", coverage_output.strip(), "```"])
    if not mypy_ok:
        parts.extend(["", "```text", mypy_output.strip(), "```"])
    if not ruff_ok:
        parts.extend(["", "```text", ruff_output.strip(), "```"])
    if not build_ok:
        parts.extend(["", "```text", build_output.strip(), "```"])

    page = "\n".join(parts) + "\n"
    ok = pytest_ok and coverage_ok and mypy_ok and ruff_ok and estimator_ok and build_ok
    return page, ok


def main() -> int:
    page, ok = render_quality_page()
    QUALITY_PATH.write_text(page, encoding="utf-8")
    print(f"Wrote quality report to {QUALITY_PATH}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
