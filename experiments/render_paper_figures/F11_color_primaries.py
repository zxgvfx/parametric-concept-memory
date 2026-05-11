"""F11 — 5-condition ablation visualization for PAPER §6.8.

Reads the summary JSON from
``experiments.sleep_color_primaries`` and renders one figure with
two panels:

* **F11a** — strict-equidistant rate per condition (cyclic-symmetry
  break by the supervised geometry layer)
* **F11b** — red-wedge anchor rate per condition (cyclic-symmetry
  break by the task-driven layer)

Usage::

    python -m experiments.render_paper_figures.F11_color_primaries \
        --in outputs/primaries_5cond_8seed/summary.json \
        --out docs/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Ordered by ablation depth: A < B < C < D < BCD
DEFAULT_ORDER = [
    "A_baseline",
    "B_lms",
    "C_greenpeak",
    "D_ripehead",
    "BCD_combined",
]
LABELS = {
    "A_baseline":   "A\nbaseline",
    "B_lms":        "B\n+LMS",
    "C_greenpeak":  "C\n+green",
    "D_ripehead":   "D\n+ripe",
    "BCD_combined": "B+C+D",
}
COLORS = ["#7F8C8D", "#3498DB", "#27AE60", "#E74C3C", "#8E44AD"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="inp", type=Path,
        default=Path("outputs/primaries_5cond_8seed/summary.json"),
    )
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    ap.add_argument(
        "--prefix", default="F11_color_primaries",
        help="filename stem; outputs <prefix>.png",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = json.loads(args.inp.read_text(encoding="utf-8"))
    n_seeds = summary["config"]["n_seeds"]
    by_cond = summary["by_condition"]
    conds = [c for c in DEFAULT_ORDER if c in by_cond]
    if not conds:
        conds = list(by_cond.keys())

    equi_rate = [by_cond[c]["fraction_equidistant"] for c in conds]
    redw_rate = [by_cond[c]["fraction_anchor_in_red_wedge"] for c in conds]
    rgb_hit = [
        by_cond[c]["perceptual_prior_match_counts"].get("RGB", 0) / n_seeds
        for c in conds
    ]
    labels = [LABELS.get(c, c) for c in conds]
    colors = COLORS[: len(conds)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(conds))

    bars = axes[0].bar(
        x, equi_rate, color=colors, edgecolor="black", linewidth=0.8,
    )
    for b, v in zip(bars, equi_rate):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.02,
                     f"{int(round(v * n_seeds))}/{n_seeds}",
                     ha="center", va="bottom", fontsize=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("strict-equidistant rate (8 seeds)")
    axes[0].set_title("Cyclic-symmetry break via supervised geometry\n"
                      "(EQUI = anchor spacings = [4,4,4])")
    axes[0].grid(axis="y", linestyle=":", alpha=0.4)

    bars = axes[1].bar(
        x, redw_rate, color=colors, edgecolor="black", linewidth=0.8,
    )
    for b, v in zip(bars, redw_rate):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.02,
                     f"{int(round(v * n_seeds))}/{n_seeds}",
                     ha="center", va="bottom", fontsize=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=10)
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("red-wedge anchor rate (8 seeds)")
    axes[1].set_title("Cyclic-symmetry break via task asymmetry\n"
                      "(≥1 anchor lands on hue ∈ {0, 1, 11})")
    axes[1].grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(
        "PAPER §6.8 — three causal layers of perceptual primaries\n"
        "(B = biological prior, C = ecological statistics, D = task-driven)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = args.out / f"{args.prefix}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
