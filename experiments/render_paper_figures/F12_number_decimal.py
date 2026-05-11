"""F12 — number-domain three-causal-layer ablation visualization (PAPER §7.4).

Mirrors the F11 colour figure but for the base-10 reverse claim.
Reads the summary JSON from
``experiments.sleep_number_decimal`` and renders one figure with
two panels:

* **F12a** — ``spike_10`` per condition (positive value = base-10
  10-period structure detected in the bundle cosine matrix).
* **F12b** — ``last_digit_cluster_purity`` per condition (purity
  of sleep k=10 anchors against the 10 last-digit equivalence
  classes; 1.0 = perfect base-10 column structure, 0.1 = chance).

Usage::

    python -m experiments.render_paper_figures.F12_number_decimal \\
        --in outputs/decimal_5cond_8seed/summary.json \\
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
        default=Path("outputs/decimal_5cond_8seed/summary.json"),
    )
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    ap.add_argument(
        "--prefix", default="F12_number_decimal",
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

    spike10_mean = [by_cond[c]["spike_10"]["mean"] for c in conds]
    spike10_std = [by_cond[c]["spike_10"]["std"] for c in conds]
    purity_mean = [by_cond[c]["last_digit_purity"]["mean"] for c in conds]
    purity_std = [by_cond[c]["last_digit_purity"]["std"] for c in conds]
    units_gap_mean = [by_cond[c]["units_gap"]["mean"] for c in conds]
    units_gap_std = [by_cond[c]["units_gap"]["std"] for c in conds]
    labels = [LABELS.get(c, c) for c in conds]
    colors = COLORS[: len(conds)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(conds))

    bars = axes[0].bar(
        x, spike10_mean, yerr=spike10_std, capsize=4,
        color=colors, edgecolor="black", linewidth=0.8,
    )
    for b, v in zip(bars, spike10_mean):
        axes[0].text(
            b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
            f"{v:+.3f}", ha="center", va="bottom", fontsize=9,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylim(0, max(spike10_mean) * 1.25 + 0.05)
    axes[0].set_ylabel("spike$_{10}$ (8 seeds)")
    axes[0].set_title(
        "10-period spike\n"
        "spike$_{10}$ = avg cos(n,n+10) − ½(cos(n,n+9)+cos(n,n+11))"
    )
    axes[0].grid(axis="y", linestyle=":", alpha=0.4)
    axes[0].axhline(0, color="black", linewidth=0.5)

    bars = axes[1].bar(
        x, units_gap_mean, yerr=units_gap_std, capsize=4,
        color=colors, edgecolor="black", linewidth=0.8,
    )
    for b, v in zip(bars, units_gap_mean):
        axes[1].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + (0.02 if v >= 0 else -0.06),
            f"{v:+.3f}", ha="center",
            va=("bottom" if v >= 0 else "top"), fontsize=9,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    y_min = min(0, min(units_gap_mean) - 0.1)
    y_max = max(units_gap_mean) + 0.15
    axes[1].set_ylim(y_min, y_max)
    axes[1].set_ylabel("units gap (8 seeds)")
    axes[1].set_title(
        "Sign-flip in units geometry\n"
        "cos[+10] − cos[+1]: linear → digit-aligned"
    )
    axes[1].grid(axis="y", linestyle=":", alpha=0.4)
    axes[1].axhline(0, color="black", linewidth=0.5)

    bars = axes[2].bar(
        x, purity_mean, yerr=purity_std, capsize=4,
        color=colors, edgecolor="black", linewidth=0.8,
    )
    for b, v in zip(bars, purity_mean):
        axes[2].text(
            b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9,
        )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=9)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("last-digit cluster purity (8 seeds)")
    axes[2].set_title(
        "Sleep k=10 anchor purity\n"
        "(1.0 = anchors align with 10 last-digit classes)"
    )
    axes[2].grid(axis="y", linestyle=":", alpha=0.4)
    axes[2].axhline(0.1, color="black", linestyle="--", alpha=0.5,
                    label="chance = 1/10")
    axes[2].legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "PAPER §7.4 — three causal layers reverse the §7 base-10 negative\n"
        "(B = decimal cone supervision, C = round-number sampling, D = last-digit head)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out_path = args.out / f"{args.prefix}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
