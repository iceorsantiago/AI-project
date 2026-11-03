"""Matplotlib helpers: mean curve + shaded 95% CI across seeds, step-based x-axes, paper styling."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to files
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from common.stats import mean_ci

# Categorical palette, used in this FIXED order (colour follows the config, never its rank).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "legend.frameon": False,
})


def _kfmt(x, _pos):
    return f"{x / 1e6:g}M" if abs(x) >= 1e6 else (f"{x / 1e3:g}k" if abs(x) >= 1e3 else f"{x:g}")


def shaded(ax, steps, matrix, label, color, alpha=0.15, ls="-"):
    """Mean across seeds (rows) with a shaded 95% t-CI band; NaN columns are ignored."""
    matrix = np.asarray(matrix, dtype=float)
    keep = ~np.isnan(matrix).any(axis=0)
    steps, matrix = np.asarray(steps)[keep], matrix[:, keep]
    m, h = mean_ci(matrix)
    ax.plot(steps, m, color=color, label=f"{label} (n={matrix.shape[0]})", ls=ls)
    ax.fill_between(steps, m - h, m + h, color=color, alpha=alpha, linewidth=0)


def style_steps_axis(ax, xlabel="Cumulative environment steps"):
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(FuncFormatter(_kfmt))


def hline(ax, y, text):
    ax.axhline(y, color=TEXT2, lw=0.8, ls=":")
    ax.annotate(text, (1.0, y), xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=7, color=TEXT2)


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  wrote {path.with_suffix('.pdf').name} / .png")
