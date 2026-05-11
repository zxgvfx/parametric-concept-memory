"""F7 — H5″ four-domain cross-facet alignment bars.

Bar chart with permutation-null shaded band, showing six cross-facet
ρ values across four domains; algebra-compatible pairs (number,
color) align, incompatible (space) and orthogonal (phoneme) ones don't.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from ._style import DOMAIN_COLORS, OUTPUTS, savefig


__all__ = ["render_F7_h5pp_alignment_schema"]


def _extract(per_seed: list[dict], key: str) -> tuple[float, float]:
    xs = [r[key] for r in per_seed]
    return float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def render_F7_h5pp_alignment_schema() -> None:
    robust = json.loads((OUTPUTS / "robustness" / "summary.json").read_text())
    color  = json.loads((OUTPUTS / "color_full" / "summary.json").read_text())
    space  = json.loads((OUTPUTS / "space_concept" / "summary.json").read_text())
    phon   = json.loads((OUTPUTS / "phoneme_concept" / "summary.json").read_text())

    m_num, s_num = _extract(robust["E1_multi_seed"]["per_seed"], "dual_cross_facet_align")
    m_col, s_col = _extract(color["E1_multi_seed"]["per_seed"], "dual_cross_facet_align")
    m_spc, s_spc = _extract(space["E1_multi_seed"]["per_seed"], "dual_cross_facet_align")
    phon_rows = phon["E1_multi_seed"]["per_seed"]
    m_vm = float(np.mean([r["cross_facet_align"]["v_m"] for r in phon_rows]))
    s_vm = float(np.std([r["cross_facet_align"]["v_m"] for r in phon_rows], ddof=1))
    m_vp = float(np.mean([r["cross_facet_align"]["v_p"] for r in phon_rows]))
    s_vp = float(np.std([r["cross_facet_align"]["v_p"] for r in phon_rows], ddof=1))
    m_mp = float(np.mean([r["cross_facet_align"]["m_p"] for r in phon_rows]))
    s_mp = float(np.std([r["cross_facet_align"]["m_p"] for r in phon_rows], ddof=1))

    pvals = {
        "number": 0.003,   # robustness permutation result
        "color":  0.016,
        "space":  0.77,
        "phon_vm": 0.052,
        "phon_vp": 0.991,
        "phon_mp": 0.037,
    }

    labels = [
        "number\narith ↔ ord",
        "color\nmix ↔ adj",
        "space\nmotion ↔ L1",
        "phoneme\nvoice ↔ manner",
        "phoneme\nvoice ↔ place",
        "phoneme\nmanner ↔ place",
    ]
    means = [m_num, m_col, m_spc, m_vm, m_vp, m_mp]
    stds  = [s_num, s_col, s_spc, s_vm, s_vp, s_mp]
    ps    = [pvals["number"], pvals["color"], pvals["space"],
             pvals["phon_vm"], pvals["phon_vp"], pvals["phon_mp"]]
    predict_align = [True, True, False, False, False, False]
    colors = [
        DOMAIN_COLORS["number"], DOMAIN_COLORS["color"],
        DOMAIN_COLORS["space"],
        DOMAIN_COLORS["phoneme"], DOMAIN_COLORS["phoneme"], DOMAIN_COLORS["phoneme"],
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    x = np.arange(len(labels))

    # vertical separators / algebra region shading (draw FIRST so bars are on top)
    ax.axvspan(-0.5, 1.5, color=DOMAIN_COLORS["number"], alpha=0.06,
               label="same facet-algebra → align  (H5″ 'if')")
    ax.axvspan(1.5, 2.5, color=DOMAIN_COLORS["space"], alpha=0.07,
               label="same domain, vector vs scalar algebra → null")
    ax.axvspan(2.5, 5.5, color=DOMAIN_COLORS["phoneme"], alpha=0.06,
               label="orthogonal categorical axes → null")

    ax.axhline(0, color="gray", lw=0.6, ls="-")
    ax.fill_between([-0.5, len(labels) - 0.5], -0.10, +0.10,
                    color="gray", alpha=0.13,
                    label="permutation-null band (|ρ| < 0.10)")

    ax.bar(x, means, yerr=stds, capsize=3, color=colors,
           edgecolor="black", linewidth=0.5,
           error_kw={"elinewidth": 0.8})

    for xi, (m, s, p, pred) in enumerate(zip(means, stds, ps, predict_align)):
        kind = "align" if pred else "null"
        sig = "p < 0.01" if p < 0.01 else "p < 0.05" if p < 0.05 else f"p = {p:.2f}"
        if m >= 0:
            y = m + s + 0.05
            va = "bottom"
        else:
            y = m - s - 0.05
            va = "top"
        ax.text(xi, y, f"{m:+.2f}\n{sig}\n[pred: {kind}]", ha="center",
                fontsize=7.8, color="black", va=va)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("cross-facet alignment  ρ (Spearman)")
    ax.set_ylim(-0.35, 1.18)
    ax.set_title(
        "F7  H5″  four-domain schema  —  alignment is gated by facet-level "
        "algebraic compatibility"
    )
    ax.legend(loc="upper right", fontsize=7.8, framealpha=0.95)
    fig.tight_layout()
    savefig(fig, "F7_h5pp_alignment_schema")
