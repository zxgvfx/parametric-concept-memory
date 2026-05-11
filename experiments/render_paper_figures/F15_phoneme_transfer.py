"""F15 — phoneme cross-language transfer visualization (PAPER §6.9).

Reads ``experiments.sleep_phoneme_transfer`` summary JSON and
renders one figure with three panels (one per axis).

Each panel: 5-condition bar chart of target-language accuracy
with chance baseline drawn as a dashed line. The expected
pattern: A flat-line at or below chance, B alone soaring on V/M
(slightly less on P), D alone only transferring its consumed
voice facet, B+C+D close to B.

Usage::

    python -m experiments.render_paper_figures.F15_phoneme_transfer \\
        --in outputs/phoneme_transfer_5seed/summary.json \\
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
    "B_articulator",
    "C_zipfsource",
    "D_minpair",
    "BCD_combined",
]
LABELS = {
    "A_baseline":    "A\nbaseline",
    "B_articulator": "B\n+articulator\ncentroid",
    "C_zipfsource":  "C\n+Zipf\nsource",
    "D_minpair":     "D\n+minimal-\npair head",
    "BCD_combined":  "B+C+D",
}
COLORS = ["#7F8C8D", "#3498DB", "#27AE60", "#E74C3C", "#8E44AD"]
AXIS_TITLES = {
    "v": ("Voice axis (binary)", 0.5),
    "m": ("Manner axis (4-class)", 0.25),
    "p": ("Place axis (4-class)", 0.25),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="inp", type=Path,
        default=Path("outputs/phoneme_transfer_5seed/summary.json"),
    )
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    ap.add_argument(
        "--prefix", default="F15_phoneme_transfer",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = json.loads(args.inp.read_text(encoding="utf-8"))
    n_seeds = summary["config"]["n_seeds"]
    n_target = summary["config"]["n_target"]
    by_cond = summary["by_condition"]
    conds = [c for c in DEFAULT_ORDER if c in by_cond]
    if not conds:
        conds = list(by_cond.keys())

    labels = [LABELS.get(c, c) for c in conds]
    colors = COLORS[: len(conds)]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    x = np.arange(len(conds))

    for ax_idx, axis in enumerate(("v", "m", "p")):
        title, chance = AXIS_TITLES[axis]
        means = [by_cond[c][f"target_{axis}"]["mean"] for c in conds]
        stds = [by_cond[c][f"target_{axis}"]["std"] for c in conds]
        bars = axes[ax_idx].bar(
            x, means, yerr=stds, capsize=4,
            color=colors, edgecolor="black", linewidth=0.8,
        )
        for b, v in zip(bars, means):
            axes[ax_idx].text(
                b.get_x() + b.get_width() / 2, b.get_height() + 0.025,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9,
            )
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels(labels, fontsize=8.5)
        axes[ax_idx].set_ylim(0, 1.12)
        axes[ax_idx].set_ylabel(
            f"target-language acc, axis {axis} ({n_seeds} seeds)"
        )
        axes[ax_idx].axhline(
            y=chance, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
            label=f"chance = {chance:.2f}",
        )
        axes[ax_idx].set_title(title)
        axes[ax_idx].grid(axis="y", linestyle=":", alpha=0.4)
        axes[ax_idx].legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"PAPER §6.9 — phoneme cross-language transfer "
        f"(n_target={n_target} held-out phonemes, V/M/P heads only see source)\n"
        "B (articulator centroid) is the dominant layer here, in contrast to "
        "colour and number domains where D (task-driven) is dominant",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out_path = args.out / f"{args.prefix}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
