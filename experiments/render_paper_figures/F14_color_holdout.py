"""F14 — colour hue holdout visualization (PAPER §7.5-color).

Reads the summary JSON from
``experiments.sleep_color_holdout`` and renders one figure with
two panels:

* **F14a** — train accuracy per condition (B / BCD show LMS-centroid
  trade-off: cone-overlapping centroids make the closed mixing
  task harder).
* **F14b** — held-out target-hue OOD accuracy per condition; all
  conditions strictly 0.000, well below the 1/12 ≈ 0.083 chance
  baseline. The "closed-output-set" ceiling.

Usage::

    python -m experiments.render_paper_figures.F14_color_holdout \\
        --in outputs/color_holdout_h5_5seed/summary.json \\
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
        default=Path("outputs/color_holdout_h5_5seed/summary.json"),
    )
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    ap.add_argument(
        "--prefix", default="F14_color_holdout",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = json.loads(args.inp.read_text(encoding="utf-8"))
    n_seeds = summary["config"]["n_seeds"]
    holdout_hue = summary["config"]["holdout_hue"]
    by_cond = summary["by_condition"]
    conds = [c for c in DEFAULT_ORDER if c in by_cond]
    if not conds:
        conds = list(by_cond.keys())

    train_mean = [by_cond[c]["train_acc"]["mean"] for c in conds]
    train_std = [by_cond[c]["train_acc"]["std"] for c in conds]
    ood_mean = [by_cond[c]["ood_acc_holdout_hue"]["mean"] for c in conds]
    ood_std = [by_cond[c]["ood_acc_holdout_hue"]["std"] for c in conds]
    labels = [LABELS.get(c, c) for c in conds]
    colors = COLORS[: len(conds)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    x = np.arange(len(conds))

    bars = axes[0].bar(
        x, train_mean, yerr=train_std, capsize=4,
        color=colors, edgecolor="black", linewidth=0.8,
    )
    for b, v in zip(bars, train_mean):
        axes[0].text(
            b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel(f"train accuracy ({n_seeds} seeds)")
    axes[0].set_title(
        "Mixing accuracy on TRAINED triples (c ≠ holdout)\n"
        "B / BCD: LMS centroids overlap → closed-task slightly harder"
    )
    axes[0].grid(axis="y", linestyle=":", alpha=0.4)

    bars = axes[1].bar(
        x, ood_mean, yerr=ood_std, capsize=4,
        color=colors, edgecolor="black", linewidth=0.8,
    )
    for b, v in zip(bars, ood_mean):
        axes[1].text(
            b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=10)
    axes[1].set_ylim(0, max(0.12, max(ood_mean) * 1.5 + 0.02))
    axes[1].set_ylabel(f"hue-{holdout_hue} OOD accuracy ({n_seeds} seeds)")
    chance_y = 1 / 12
    axes[1].axhline(
        y=chance_y, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
        label=f"chance ≈ 1/12 = {chance_y:.3f}",
    )
    axes[1].set_title(
        f"Hue-{holdout_hue} HELD OUT as mixing target\n"
        f"all 5 conditions × {n_seeds} seeds = 0.000: closed-output-set ceiling"
    )
    axes[1].grid(axis="y", linestyle=":", alpha=0.4)
    axes[1].legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"PAPER §7.5-color — hue {holdout_hue} target-holdout: "
        "head never predicts an output class it was never trained on,\n"
        "regardless of any prior injection (mirrors §7.5-number length-OOD ceiling)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out_path = args.out / f"{args.prefix}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
