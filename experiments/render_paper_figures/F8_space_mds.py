"""F8 — Space grid MDS overlay (trained vs shuffle counterfactual).

Two side-by-side Procrustes-aligned MDS panels for the space §6.2 task:
left panel uses true cid identity, right panel uses a shuffled
identity. The disparity gap visualises how the lattice geometry
dissolves when concept identity is broken.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes

from ._bundles import cos_matrix_from_bundles, get_space_bundles, mds_2d
from ._style import CMAP_SEQ, savefig


__all__ = ["render_F8_space_mds_trained_vs_shuffle"]


def _compute(sb_obj):
    cids = [sb_obj["cid_of"](r, c)
            for r in range(sb_obj["n_rows"])
            for c in range(sb_obj["n_cols"])]
    cos = cos_matrix_from_bundles(sb_obj["bundle_state"], cids,
                                  sb_obj["facet_motion"])
    coords = mds_2d(cos, seed=0)
    gt = np.array(
        [[r, c] for r in range(sb_obj["n_rows"])
         for c in range(sb_obj["n_cols"])],
        dtype=float,
    )
    gt_n, coords_n, disparity = procrustes(gt, coords)
    return gt_n, coords_n, disparity


def render_F8_space_mds_trained_vs_shuffle() -> None:
    sb = get_space_bundles(seed=1000, shuffled=False)
    sb_sh = get_space_bundles(seed=1000, shuffled=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    for ax, sb_obj, title in [
        (axes[0], sb,    "(a) trained on true identity"),
        (axes[1], sb_sh, "(b) trained on shuffled identity (counterfactual)"),
    ]:
        gt_n, coords_n, disp = _compute(sb_obj)
        R, C = sb_obj["n_rows"], sb_obj["n_cols"]
        for r in range(R):
            idx = np.arange(r * C, (r + 1) * C)
            ax.plot(gt_n[idx, 0], gt_n[idx, 1], color="lightgray", lw=1, zorder=1)
        for c in range(C):
            idx = np.arange(c, R * C, C)
            ax.plot(gt_n[idx, 0], gt_n[idx, 1], color="lightgray", lw=1, zorder=1)
        for i in range(R * C):
            ax.plot([gt_n[i, 0], coords_n[i, 0]], [gt_n[i, 1], coords_n[i, 1]],
                    color="red", lw=0.4, alpha=0.55, zorder=2)
        cols = np.arange(R * C) / (R * C - 1)
        ax.scatter(coords_n[:, 0], coords_n[:, 1], c=cols, cmap=CMAP_SEQ,
                   s=85, edgecolor="black", linewidth=0.4, zorder=3)
        ax.set_title(f"{title}\nProcrustes disp = {disp:.3f}")
        ax.set_aspect("equal", "box")
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    fig.suptitle(
        "F8  Space grid MDS: trained bundle recovers the 5×5 lattice; "
        "under shuffled-identity training, the lattice dissolves",
        y=1.02, fontsize=10,
    )
    fig.tight_layout()
    savefig(fig, "F8_space_mds_trained_vs_shuffle")
