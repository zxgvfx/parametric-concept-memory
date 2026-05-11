"""F13 — number length extrapolation visualization (PAPER §7.5).

Reads the summary JSON from
``experiments.sleep_number_extrapolate`` and renders one figure
with two panels:

* **F13a** — in-range OOD accuracy per condition (replicates §7.4
  trade-off: BCD < A on random hold-out within training range).
* **F13b** — length-100 OOD accuracy per condition (the key §7.5
  test: only D / BCD significantly above chance ≈ 0.051; absolute
  effect small but statistically robust given std ≈ 0.001-0.003).

Usage::

    python -m experiments.render_paper_figures.F13_number_extrapolate \\
        --in outputs/extrap_30_100_5seed/summary.json \\
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
    "B_decimal",
    "C_roundbias",
    "D_lastdigit",
    "BCD_combined",
]
LABELS = {
    "A_baseline":   "A\nbaseline",
    "B_decimal":    "B\n+decimal\ncones",
    "C_roundbias":  "C\n+round-num\nbias",
    "D_lastdigit":  "D\n+last-digit\nhead",
    "BCD_combined": "B+C+D",
}
COLORS = ["#7F8C8D", "#3498DB", "#27AE60", "#E74C3C", "#8E44AD"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="inp", type=Path,
        default=Path("outputs/extrap_30_100_5seed/summary.json"),
    )
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    ap.add_argument(
        "--prefix", default="F13_number_extrapolate",
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

    in_range_mean = [by_cond[c]["test_random_in_range"]["mean"] for c in conds]
    in_range_std = [by_cond[c]["test_random_in_range"]["std"] for c in conds]
    len_mean = [by_cond[c]["test_length_100"]["mean"] for c in conds]
    len_std = [by_cond[c]["test_length_100"]["std"] for c in conds]
    train_mean = [by_cond[c]["train_acc"]["mean"] for c in conds]
    labels = [LABELS.get(c, c) for c in conds]
    colors = COLORS[: len(conds)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    x = np.arange(len(conds))

    bars = axes[0].bar(
        x, in_range_mean, yerr=in_range_std, capsize=4,
        color=colors, edgecolor="black", linewidth=0.8,
    )
    for b, v in zip(bars, in_range_mean):
        axes[0].text(
            b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("in-range OOD accuracy (5 seeds)")
    axes[0].set_title(
        "In-range OOD (a, b ≤ 30, random hold-out)\n"
        "trade-off: BCD geometric compression mildly hurts interpolation"
    )
    axes[0].grid(axis="y", linestyle=":", alpha=0.4)
    train_line = axes[0].axhline(
        y=np.mean(train_mean), color="black", linestyle=":",
        linewidth=0.8, label=f"avg train acc ≈ {np.mean(train_mean):.2f}",
    )
    axes[0].legend(loc="lower left", fontsize=8)

    bars = axes[1].bar(
        x, len_mean, yerr=len_std, capsize=4,
        color=colors, edgecolor="black", linewidth=0.8,
    )
    for b, v in zip(bars, len_mean):
        axes[1].text(
            b.get_x() + b.get_width() / 2, b.get_height() + 0.001,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylim(0, max(len_mean) * 1.4 + 0.005)
    axes[1].set_ylabel("length-100 OOD accuracy (5 seeds)")
    chance_line = axes[1].axhline(
        y=0.051, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
        label="chance ≈ 0.051 (head's systematic bias on OOD)",
    )
    axes[1].set_title(
        "Length OOD (a OR b ∈ [31, 100])\n"
        "PCM-architectural ceiling: D / BCD detectable above chance, "
        "but small absolute"
    )
    axes[1].grid(axis="y", linestyle=":", alpha=0.4)
    axes[1].legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "PAPER §7.5 — length extrapolation hits an architectural ceiling\n"
        "task-asymmetric priors lift OOD by +1.1pp but cannot reach\n"
        "non-trivial extrapolation; D93a slot generators required for full crossing",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = args.out / f"{args.prefix}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
