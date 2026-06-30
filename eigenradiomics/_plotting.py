"""Shared matplotlib styling for eigenradiomics figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

#: The Okabe-Ito colour-vision-deficiency-safe qualitative palette (Okabe & Ito,
#: 2008). Eight colours chosen to remain distinguishable under the common forms
#: of colour blindness; used where colour must carry categorical meaning.
OKABE_ITO: list[str] = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
]

#: Distinct filled marker shapes for redundant (shape + colour) categorical
#: encoding in scatter plots, so groups stay separable without colour.
CVD_MARKERS: list[str] = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p"]


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
