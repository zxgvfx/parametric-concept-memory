"""F9 — Tier-G sleep 4 域对比图.

读取 ``outputs/f5_5seed_ood30/summary.json`` (由
``experiments.sleep_ablation_four_domain --ood-ratio 0.3 --n-seeds 5``
产出), 渲染两张图:

* **F9a** — 4 域 ρ_geometry mean ± std, A vs C bar chart
* **F9b** — 同样布局, task accuracy / OOD accuracy mean ± std

跑法::

    python -m experiments.render_paper_figures.F9_sleep_four_domain \
        --in outputs/f5_5seed_ood30/summary.json \
        --out docs/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "A": "#7F8C8D",  # baseline grey
    "C": "#2E86C1",  # sleep blue
}


def _bar_pair(ax, labels, means_a, stds_a, means_c, stds_c,
              title: str, ylabel: str, ylim=None) -> None:
    x = np.arange(len(labels))
    w = 0.35
    bars_a = ax.bar(
        x - w / 2, means_a, w, yerr=stds_a, capsize=4,
        color=PALETTE["A"], label="A: no sleep",
        edgecolor="black", linewidth=0.6,
    )
    bars_c = ax.bar(
        x + w / 2, means_c, w, yerr=stds_c, capsize=4,
        color=PALETTE["C"], label="C: sleep + soft abstract",
        edgecolor="black", linewidth=0.6,
    )
    for b in list(bars_a) + list(bars_c):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="inp", type=Path,
        default=Path("outputs/f5_5seed_ood30/summary.json"),
    )
    ap.add_argument(
        "--out", type=Path, default=Path("docs/figures"),
    )
    ap.add_argument(
        "--prefix", default="F9_sleep_four_domain",
        help="filename stem; outputs <prefix>_rho.png and <prefix>_acc.png",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = json.loads(args.inp.read_text(encoding="utf-8"))
    domain_order = ["number", "color", "space", "phoneme"]
    domains = [d for d in domain_order if d in summary["by_domain"]]

    rho_a_mean = [summary["by_domain"][d]["rho_A"]["mean"] for d in domains]
    rho_a_std = [summary["by_domain"][d]["rho_A"]["std"] for d in domains]
    rho_c_mean = [summary["by_domain"][d]["rho_C"]["mean"] for d in domains]
    rho_c_std = [summary["by_domain"][d]["rho_C"]["std"] for d in domains]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _bar_pair(
        ax, domains, rho_a_mean, rho_a_std, rho_c_mean, rho_c_std,
        title=("Tier-G sleep abstraction · 4-domain ρ (5 seeds, OOD=0.30)"),
        ylabel="ρ_geometry (per-domain primary metric)",
        ylim=(0, 1.05),
    )
    fig.tight_layout()
    rho_path = args.out / f"{args.prefix}_rho.png"
    fig.savefig(rho_path, dpi=180)
    plt.close(fig)
    print(f"wrote {rho_path}")

    task_a_mean = [summary["by_domain"][d]["task_acc_A"]["mean"]
                   for d in domains]
    task_a_std = [summary["by_domain"][d]["task_acc_A"]["std"]
                  for d in domains]
    task_c_mean = [summary["by_domain"][d]["task_acc_C"]["mean"]
                   for d in domains]
    task_c_std = [summary["by_domain"][d]["task_acc_C"]["std"]
                  for d in domains]

    has_ood = all(summary["by_domain"][d].get("ood_acc_A") is not None
                  for d in domains if d != "phoneme")
    if has_ood:
        ood_domains = [d for d in domains if d != "phoneme"]
        ood_a_mean = [summary["by_domain"][d]["ood_acc_A"]["mean"]
                      for d in ood_domains]
        ood_a_std = [summary["by_domain"][d]["ood_acc_A"]["std"]
                     for d in ood_domains]
        ood_c_mean = [summary["by_domain"][d]["ood_acc_C"]["mean"]
                      for d in ood_domains]
        ood_c_std = [summary["by_domain"][d]["ood_acc_C"]["std"]
                     for d in ood_domains]
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        _bar_pair(
            axes[0], domains, task_a_mean, task_a_std, task_c_mean, task_c_std,
            title="Train task accuracy (5 seeds)",
            ylabel="accuracy", ylim=(0, 1.08),
        )
        _bar_pair(
            axes[1], ood_domains, ood_a_mean, ood_a_std,
            ood_c_mean, ood_c_std,
            title="OOD task accuracy (held-out 30% triples)",
            ylabel="OOD accuracy", ylim=(0, 1.0),
        )
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        _bar_pair(
            ax, domains, task_a_mean, task_a_std, task_c_mean, task_c_std,
            title="Train task accuracy (5 seeds)",
            ylabel="accuracy", ylim=(0, 1.08),
        )
        fig.tight_layout()
    acc_path = args.out / f"{args.prefix}_acc.png"
    fig.savefig(acc_path, dpi=180)
    plt.close(fig)
    print(f"wrote {acc_path}")


if __name__ == "__main__":
    main()
