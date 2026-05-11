"""F16 — space-domain length extrapolation visualization (PAPER §7.5-space).

Reads ``experiments.sleep_space_extrapolate`` summary JSON and
renders a three-panel figure mirroring F13 (number length-OOD)
but for the 2-D grid: in-range / mixed-OOD / outer-OOD.

Usage::

    python -m experiments.render_paper_figures.F16_space_extrapolate \\
        --in outputs/space_extrap_5_7_5seed/summary.json \\
        --out docs/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ORDER = [
    "A_baseline",
    "B_cardinal",
    "C_centerbias",
    "D_rowindex",
    "BCD_combined",
]
LABELS = {
    "A_baseline":   "A\nbaseline",
    "B_cardinal":   "B\n+cardinal\ncentroid",
    "C_centerbias": "C\n+center-bias\nsampling",
    "D_rowindex":   "D\n+row-index\nhead",
    "BCD_combined": "B+C+D",
}
COLORS = ["#7F8C8D", "#3498DB", "#27AE60", "#E74C3C", "#8E44AD"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="inp", type=Path,
        default=Path("outputs/space_extrap_5_7_5seed/summary.json"),
    )
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    ap.add_argument(
        "--prefix", default="F16_space_extrapolate",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = json.loads(args.inp.read_text(encoding="utf-8"))
    n_seeds = summary["config"]["n_seeds"]
    n_train_r = summary["config"]["n_rows_train"]
    n_total_r = summary["config"]["n_rows_total"]
    by_cond = summary["by_condition"]
    conds = [c for c in DEFAULT_ORDER if c in by_cond]
    if not conds:
        conds = list(by_cond.keys())

    labels = [LABELS.get(c, c) for c in conds]
    colors = COLORS[: len(conds)]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    x = np.arange(len(conds))
    chance = 0.20  # 1 / 5 for 5-way direction classification

    panels = [
        ("test_random_in_range",
         f"In-range OOD ({n_train_r}×{n_train_r} random hold-out)",
         "in-range OOD acc"),
        ("test_mixed_OOD",
         "Mixed OOD (one inner, one outer cell)",
         "mixed-OOD acc"),
        ("test_outer_OOD",
         f"Outer OOD (both cells on outer ring; {n_total_r}×{n_total_r} − {n_train_r}×{n_train_r})",
         "outer-OOD acc"),
    ]

    for ax_idx, (key, title, ylabel) in enumerate(panels):
        means = [by_cond[c][key]["mean"] for c in conds]
        stds = [by_cond[c][key]["std"] for c in conds]
        bars = axes[ax_idx].bar(
            x, means, yerr=stds, capsize=4,
            color=colors, edgecolor="black", linewidth=0.8,
        )
        for b, v in zip(bars, means):
            axes[ax_idx].text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.02,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9,
            )
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels(labels, fontsize=9)
        axes[ax_idx].set_ylim(0, 1.08)
        axes[ax_idx].set_ylabel(f"{ylabel} ({n_seeds} seeds)")
        axes[ax_idx].set_title(title, fontsize=10)
        axes[ax_idx].axhline(
            y=chance, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
            label=f"chance = {chance}",
        )
        axes[ax_idx].grid(axis="y", linestyle=":", alpha=0.4)
        axes[ax_idx].legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "PAPER §7.5-space — spatial length extrapolation\n"
        "B (cardinal centroid) drives outer-OOD from chance to ~3×; "
        "mixed-OOD strict 0/25 reveals an input-distribution ceiling",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out_path = args.out / f"{args.prefix}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
