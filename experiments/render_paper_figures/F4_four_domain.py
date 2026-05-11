"""F4 — Four-domain universality panel.

Renders the headline 2×2 figure of the paper:

(a) Numbers — linear cos heatmap with ρ(cos, −|Δn|).
(b) Colors — circular MDS, points coloured by true hue, with
    ``ρ_circular``.
(c) Space — 2-D Procrustes-aligned MDS with disparity.
(d) Phonemes — manner-block-sorted cos heatmap with class block lines.

All four sub-panels read from the in-process bundle caches.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from scipy.spatial import procrustes
from scipy.stats import spearmanr

from ._bundles import (
    cos_matrix_from_bundles,
    get_color_bundles,
    get_number_bundles,
    get_phoneme_bundles,
    get_space_bundles,
    mds_2d,
)
from ._style import CMAP_COS, CMAP_SEQ, DOMAIN_COLORS, savefig


__all__ = ["render_F4_four_domain_panel"]


def render_F4_four_domain_panel() -> None:
    # (a) number — linear
    nb = get_number_bundles()
    ns = list(range(nb["n_min"], nb["n_max"] + 1))
    ncids = [f"concept:ans:{n}" for n in ns]
    cos_num = cos_matrix_from_bundles(nb["bundle_state"], ncids, nb["facet_add"])
    triu = np.triu_indices(len(ns), k=1)
    rho_num = spearmanr(
        cos_num[triu],
        -np.abs(np.subtract.outer(ns, ns))[triu],
    )[0]

    # (b) color — circular
    cb = get_color_bundles()
    K = cb["n_colors"]
    ccids = [f"concept:color:{i}" for i in range(K)]
    cos_col = cos_matrix_from_bundles(cb["bundle_state"], ccids, cb["facet_mix"])
    coords_col = mds_2d(cos_col, seed=0)
    coords_col -= coords_col.mean(0, keepdims=True)
    coords_col /= np.linalg.norm(coords_col, axis=1).max() + 1e-9

    # (c) space — 2D grid
    sb = get_space_bundles()
    sids = [sb["cid_of"](r, c) for r in range(sb["n_rows"]) for c in range(sb["n_cols"])]
    cos_spc = cos_matrix_from_bundles(sb["bundle_state"], sids, sb["facet_motion"])
    coords_spc = mds_2d(cos_spc, seed=0)
    gt = np.array(
        [[r, c] for r in range(sb["n_rows"]) for c in range(sb["n_cols"])],
        dtype=float,
    )
    gt_n, coords_n, disparity = procrustes(gt, coords_spc)

    # (d) phoneme — cos heatmap with manner block lines
    pb = get_phoneme_bundles()
    phons = pb["phonemes"]
    pcids = [pb["cid_of"](i) for i in range(len(phons))]
    sort_idx = sorted(
        range(len(phons)),
        key=lambda i: (phons[i][2], phons[i][3], phons[i][1]),  # manner, place, voice
    )
    pcids_sorted = [pcids[i] for i in sort_idx]
    labels_sorted = [phons[i][0] for i in sort_idx]
    cos_phon = cos_matrix_from_bundles(
        pb["bundle_state"], pcids_sorted, "manner_bias"
    )
    manner_of = [phons[i][2] for i in sort_idx]
    block_starts = [0]
    for k in range(1, len(manner_of)):
        if manner_of[k] != manner_of[k - 1]:
            block_starts.append(k)
    block_starts.append(len(manner_of))

    # ── Render 2×2 panel ──
    fig = plt.figure(figsize=(10.5, 9.0))

    # (a)
    ax_a = fig.add_subplot(2, 2, 1)
    im = ax_a.imshow(cos_num, cmap=CMAP_COS, vmin=-1, vmax=1, aspect="equal")
    ax_a.set_title(f"(a) Numbers 1–{ns[-1]}   ρ(cos, −|Δn|) = {rho_num:+.3f}",
                   color=DOMAIN_COLORS["number"])
    ax_a.set_xlabel("n"); ax_a.set_ylabel("n")
    ticks = list(range(0, len(ns), max(1, len(ns) // 6)))
    ax_a.set_xticks(ticks); ax_a.set_yticks(ticks)
    ax_a.set_xticklabels([str(ns[i]) for i in ticks])
    ax_a.set_yticklabels([str(ns[i]) for i in ticks])
    fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)

    # (b) color ring — points coloured by true hue
    ax_b = fig.add_subplot(2, 2, 2)
    hues = np.array([(i / K) for i in range(K)])
    rgb = hsv_to_rgb(np.stack([hues, np.ones(K), np.ones(K)], axis=-1))
    theta = np.linspace(0, 2 * np.pi, 256)
    ax_b.plot(np.cos(theta), np.sin(theta), color="gray", lw=0.7, ls="--", alpha=0.6)
    ax_b.scatter(coords_col[:, 0], coords_col[:, 1], c=rgb, s=120,
                 edgecolor="black", linewidth=0.6, zorder=3)
    for i, (x, y) in enumerate(coords_col):
        ax_b.text(x * 1.15, y * 1.15, str(i), ha="center", va="center",
                  fontsize=8, color="black")
    from experiments.color_concept_study import _rho_circular
    rho_circ = _rho_circular(torch.tensor(cos_col))
    ax_b.set_title(f"(b) Colors 12 hues   ρ_circular = {rho_circ:+.3f}",
                   color=DOMAIN_COLORS["color"])
    ax_b.set_aspect("equal", "box")
    ax_b.set_xlim(-1.4, 1.4); ax_b.set_ylim(-1.4, 1.4)
    ax_b.set_xticks([]); ax_b.set_yticks([])
    ax_b.spines["left"].set_visible(False); ax_b.spines["bottom"].set_visible(False)

    # (c) space grid — Procrustes-aligned MDS + GT grid
    ax_c = fig.add_subplot(2, 2, 3)
    for r in range(sb["n_rows"]):
        ax_c.plot(gt_n[r * sb["n_cols"]:(r + 1) * sb["n_cols"], 0],
                  gt_n[r * sb["n_cols"]:(r + 1) * sb["n_cols"], 1],
                  color="lightgray", lw=1, zorder=1)
    for c in range(sb["n_cols"]):
        ax_c.plot(gt_n[c::sb["n_cols"], 0], gt_n[c::sb["n_cols"], 1],
                  color="lightgray", lw=1, zorder=1)
    idx = np.arange(sb["n_cells"])
    color_vals = idx / max(sb["n_cells"] - 1, 1)
    ax_c.scatter(coords_n[:, 0], coords_n[:, 1], c=color_vals, cmap=CMAP_SEQ,
                 s=90, edgecolor="black", linewidth=0.4, zorder=3)
    for i in range(sb["n_cells"]):
        ax_c.plot([gt_n[i, 0], coords_n[i, 0]], [gt_n[i, 1], coords_n[i, 1]],
                  color="red", lw=0.5, alpha=0.6, zorder=2)
    ax_c.set_title(f"(c) Space 5×5 grid   Procrustes disp = {disparity:.3f}",
                   color=DOMAIN_COLORS["space"])
    ax_c.set_aspect("equal", "box")
    ax_c.set_xticks([]); ax_c.set_yticks([])
    ax_c.spines["left"].set_visible(False); ax_c.spines["bottom"].set_visible(False)

    # (d) phoneme heatmap with manner-block boundaries
    ax_d = fig.add_subplot(2, 2, 4)
    im = ax_d.imshow(cos_phon, cmap=CMAP_COS, vmin=-1, vmax=1, aspect="equal")
    manner_names = ["STOP", "FRIC", "NAS", "APR"]
    for s in block_starts[1:-1]:
        ax_d.axhline(s - 0.5, color="black", lw=1.2)
        ax_d.axvline(s - 0.5, color="black", lw=1.2)
    ax_d.set_xticks(range(len(labels_sorted)))
    ax_d.set_yticks(range(len(labels_sorted)))
    ax_d.set_xticklabels(labels_sorted, fontsize=6)
    ax_d.set_yticklabels(labels_sorted, fontsize=6)
    for b_i in range(len(block_starts) - 1):
        mid = (block_starts[b_i] + block_starts[b_i + 1] - 1) / 2
        ax_d.text(mid, -2.0, manner_names[manner_of[block_starts[b_i]]],
                  ha="center", fontsize=8, color="black")
    ax_d.set_title("(d) Phonemes 20   manner_bias cos, block = manner class",
                   color=DOMAIN_COLORS["phoneme"])
    fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)

    fig.suptitle(
        "F4  Four-Domain Universality of Bundle Geometry  "
        "(linear · circular · 2-D lattice · categorical; same framework, no change)",
        y=1.00, fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, "F4_four_domain_universality")
