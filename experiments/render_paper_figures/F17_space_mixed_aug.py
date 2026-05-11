"""F17 — space mixed-pair augmentation ablation visualization (PAPER §7.5-space addendum).

Reads ``experiments.sleep_space_mixed_augment`` summary JSON and
renders one figure with two panels:

* **F17a** — mixed-test OOD accuracy vs augmentation rate
  (shows the ceiling break: 0.000 at rate=0 jumps to ~0.6 at any
  non-zero rate).
* **F17b** — outer OOD accuracy vs augmentation rate
  (shows the trade-off: cardinal-prior-driven transfer
  monotonically degrades as fc1 fits the mixed-pair distribution).

Usage::

    python -m experiments.render_paper_figures.F17_space_mixed_aug \\
        --in outputs/space_aug_5seed/summary.json \\
        --out docs/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="inp", type=Path,
        default=Path("outputs/space_aug_5seed/summary.json"),
    )
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    ap.add_argument(
        "--prefix", default="F17_space_mixed_aug",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = json.loads(args.inp.read_text(encoding="utf-8"))
    n_seeds = summary["config"]["n_seeds"]
    rates = sorted(float(k) for k in summary["by_rate"].keys())

    mixed_means = [summary["by_rate"][f"{r:.2f}"]["mixed_test_OOD"]["mean"]
                   for r in rates]
    mixed_stds = [summary["by_rate"][f"{r:.2f}"]["mixed_test_OOD"]["std"]
                  for r in rates]
    outer_means = [summary["by_rate"][f"{r:.2f}"]["outer_OOD"]["mean"]
                   for r in rates]
    outer_stds = [summary["by_rate"][f"{r:.2f}"]["outer_OOD"]["std"]
                  for r in rates]
    in_means = [summary["by_rate"][f"{r:.2f}"]["test_random_in_range"]["mean"]
                for r in rates]
    in_stds = [summary["by_rate"][f"{r:.2f}"]["test_random_in_range"]["std"]
               for r in rates]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    chance = 0.20
    x = np.array(rates)

    axes[0].errorbar(
        x, mixed_means, yerr=mixed_stds, fmt="o-", color="#E74C3C",
        capsize=4, linewidth=2, markersize=8, label="mixed-test OOD",
    )
    axes[0].errorbar(
        x, in_means, yerr=in_stds, fmt="s--", color="#7F8C8D",
        capsize=4, linewidth=1.5, markersize=6, alpha=0.7,
        label="in-range OOD (sanity)",
    )
    axes[0].axhline(
        y=chance, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
        label=f"chance = {chance}",
    )
    axes[0].set_xlabel(f"mixed-pair augmentation rate per training batch")
    axes[0].set_ylabel(f"accuracy ({n_seeds} seeds)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title(
        "Ceiling break: 0% → 5% augmentation\n"
        "lifts mixed-OOD from 0.000 to ~0.60"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=9)
    for xi, m, s in zip(x, mixed_means, mixed_stds):
        axes[0].text(xi, m + 0.04, f"{m:.2f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")

    axes[1].errorbar(
        x, outer_means, yerr=outer_stds, fmt="o-", color="#8E44AD",
        capsize=4, linewidth=2, markersize=8, label="outer-OOD",
    )
    axes[1].axhline(
        y=chance, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
        label=f"chance = {chance}",
    )
    axes[1].set_xlabel("mixed-pair augmentation rate per training batch")
    axes[1].set_ylabel(f"outer-OOD accuracy ({n_seeds} seeds)")
    axes[1].set_ylim(0, 0.85)
    axes[1].set_title(
        "Trade-off: cardinal-prior transfer degrades\n"
        "as fc1 fits the mixed input-distribution"
    )
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=9)
    for xi, m, s in zip(x, outer_means, outer_stds):
        axes[1].text(xi, m + 0.025, f"{m:.2f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")

    fig.suptitle(
        "PAPER §7.5-space addendum (F37) — mixed-OOD ceiling is fc1-distribution-coverage,\n"
        "not PCM-fundamental: 5% augmentation lifts mixed-OOD by 60 pp; "
        "outer-OOD pays a 26 pp trade-off cost",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out_path = args.out / f"{args.prefix}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
