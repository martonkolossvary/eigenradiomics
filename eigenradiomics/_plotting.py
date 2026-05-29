"""Shared matplotlib styling for eigenradiomics figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_science_style(figure_titlesize: int = 13) -> None:
    """Apply the SciencePlots accessible style with a safe sans-serif fallback.

    Uses the ``["science", "no-latex"]`` style when available and falls back to
    a clean built-in style otherwise, then forces high-readability sans-serif
    typography.

    Parameters
    ----------
    figure_titlesize : int
        Font size for figure suptitles (differs slightly between plot types).
    """
    try:
        plt.style.use(["science", "no-latex"])
    except Exception:
        plt.style.use(
            "seaborn-v0_8-whitegrid"
            if "seaborn-v0_8-whitegrid" in plt.style.available
            else "default"
        )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": figure_titlesize,
        }
    )
