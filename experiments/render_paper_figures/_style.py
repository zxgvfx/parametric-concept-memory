"""Shared matplotlib style + output paths for paper figures.

Separated so per-figure modules can ``from ._style import ...`` without
pulling in the heavy bundle-cache module.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


__all__ = [
    "DEVICE",
    "OUT_DIR",
    "OUTPUTS",
    "CMAP_COS",
    "CMAP_SEQ",
    "DOMAIN_COLORS",
    "savefig",
]


# Paper figure style — uniform across all F2..F8.
plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Serif",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("mind/docs/research/figures")
OUTPUTS = Path("outputs")

# Color palettes
CMAP_COS = "RdBu_r"
CMAP_SEQ = "viridis"
DOMAIN_COLORS = {
    "number": "#1f77b4",
    "color":  "#d62728",
    "space":  "#2ca02c",
    "phoneme": "#9467bd",
}


def savefig(fig: plt.Figure, name: str) -> None:
    """Persist a figure as both PDF and PNG under :data:`OUT_DIR`."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"{name}.{ext}"
        fig.savefig(p)
        print(f"  saved  {p}")
    plt.close(fig)
