"""F6 — Counterfactual bundle swap double-dissociation bars.

Reads ``outputs/counterfactual_swap/summary.json`` and renders two
side-by-side bar charts (number domain + color domain) showing the
canonical 4-condition × 4-bar pattern of the §B swap experiment.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from ._style import OUTPUTS, savefig


__all__ = ["render_F6_swap_dissociation"]


def _avg(per_seed: list[dict], path: list[str]) -> float:
    xs = []
    for row in per_seed:
        v = row
        for k in path:
            v = v[k]
        xs.append(v)
    return float(np.mean(xs))


def render_F6_swap_dissociation() -> None:
    js = json.loads((OUTPUTS / "counterfactual_swap" / "summary.json").read_text())

    cond_labels = ["baseline", "swap facet A only", "swap facet B only", "swap both"]

    def build_rows(dom_key: str, head_a: str, head_b: str) -> dict:
        """Return dict of condition → 4 accuracies: A-inv, A-not, B-inv, B-not."""
        per_seed = js[dom_key]["per_seed"]
        conds_num = ["baseline"] + (
            ["swap_arith_only", "swap_ord_only", "swap_both"]
            if dom_key == "number_domain"
            else ["swap_mix_only", "swap_adj_only", "swap_both"]
        )
        rows = {}
        for cond in conds_num:
            rows[cond] = {
                "A_inv": _avg(per_seed, [cond, head_a, "involving_swap", "acc"]),
                "A_not": _avg(per_seed, [cond, head_a, "not_involving", "acc"]),
                "B_inv": _avg(per_seed, [cond, head_b, "involving_swap", "acc"]),
                "B_not": _avg(per_seed, [cond, head_b, "not_involving", "acc"]),
            }
        return rows

    num_rows = build_rows("number_domain", "add", "cmp")
    col_rows = build_rows("color_domain", "mix", "adj")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    def _draw(ax, rows, cond_keys, head_a_name, head_b_name, domain):
        conds = cond_keys
        x = np.arange(len(conds))
        w = 0.18
        ax.bar(x - 1.5 * w, [rows[c]["A_inv"] for c in conds],
               width=w, color="#d62728", edgecolor="black", linewidth=0.4,
               label=f"{head_a_name}  (inv)")
        ax.bar(x - 0.5 * w, [rows[c]["A_not"] for c in conds],
               width=w, color="#ff9896", edgecolor="black", linewidth=0.4,
               label=f"{head_a_name}  (not-inv)")
        ax.bar(x + 0.5 * w, [rows[c]["B_inv"] for c in conds],
               width=w, color="#1f77b4", edgecolor="black", linewidth=0.4,
               label=f"{head_b_name}  (inv)")
        ax.bar(x + 1.5 * w, [rows[c]["B_not"] for c in conds],
               width=w, color="#aec7e8", edgecolor="black", linewidth=0.4,
               label=f"{head_b_name}  (not-inv)")
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels)
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="gray", lw=0.5, ls=":")
        ax.set_title(f"{domain}  (N seeds = "
                     f"{len(js[domain.lower() + '_domain']['per_seed'])})")
        if ax is axes[0]:
            ax.set_ylabel("accuracy")
        ax.legend(loc="lower left", ncol=2, fontsize=7.5, framealpha=0.9)

    _draw(
        axes[0], num_rows,
        ["baseline", "swap_arith_only", "swap_ord_only", "swap_both"],
        "AddHead", "CmpHead", "Number",
    )
    _draw(
        axes[1], col_rows,
        ["baseline", "swap_mix_only", "swap_adj_only", "swap_both"],
        "MixHead", "AdjHead", "Color",
    )

    fig.suptitle(
        "F6  Post-hoc bundle swap: textbook double dissociation  |  "
        "target-facet collapses only on involved-pair accuracy of the consuming muscle",
        y=1.01, fontsize=10,
    )
    fig.tight_layout()
    savefig(fig, "F6_swap_dissociation")
