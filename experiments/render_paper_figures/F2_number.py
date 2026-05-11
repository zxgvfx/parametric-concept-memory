"""F2 — Number bundle cos heatmaps (arithmetic_bias + ordinal_offset).

Two side-by-side cos heatmaps over ``concept:ans:n`` for n in
[n_min..n_max], with the cross-facet alignment ρ printed in the
suptitle. Reproduces the F2 panel of the paper.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from ._bundles import cos_matrix_from_bundles, get_number_bundles
from ._style import CMAP_COS, savefig


__all__ = ["render_F2_number_cos_heatmaps"]


def render_F2_number_cos_heatmaps() -> None:
    b = get_number_bundles()
    bs = b["bundle_state"]
    ns = list(range(b["n_min"], b["n_max"] + 1))
    cids = [f"concept:ans:{n}" for n in ns]
    cos_add = cos_matrix_from_bundles(bs, cids, b["facet_add"])
    cos_ord = cos_matrix_from_bundles(bs, cids, b["facet_ord"])

    triu = np.triu_indices(len(ns), k=1)
    delta_n = -np.abs(np.subtract.outer(ns, ns))[triu]
    rho_add = spearmanr(cos_add[triu], delta_n)[0]
    rho_ord = spearmanr(cos_ord[triu], delta_n)[0]
    align = spearmanr(cos_add[triu], cos_ord[triu])[0]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, cos, rho, title in [
        (axes[0], cos_add, rho_add, "arithmetic_bias (AddHead)"),
        (axes[1], cos_ord, rho_ord, "ordinal_offset (CmpHead)"),
    ]:
        im = ax.imshow(cos, cmap=CMAP_COS, vmin=-1, vmax=1, aspect="equal")
        ax.set_title(f"{title}\nρ(cos, −|Δn|) = {rho:+.3f}")
        ax.set_xticks(range(0, len(ns), max(1, len(ns) // 7)))
        ax.set_yticks(range(0, len(ns), max(1, len(ns) // 7)))
        ax.set_xticklabels([str(ns[i]) for i in ax.get_xticks()])
        ax.set_yticklabels([str(ns[i]) for i in ax.get_yticks()])
        ax.set_xlabel("concept ID (n)")
        ax.set_ylabel("concept ID (n)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cos")

    fig.suptitle(
        f"F2  Number bundle cos heatmaps (N={len(ns)}, dual muscle)  |  "
        f"cross-facet align ρ = {align:+.3f}",
        y=1.02, fontsize=10,
    )
    fig.tight_layout()
    savefig(fig, "F2_number_cos_heatmaps")
