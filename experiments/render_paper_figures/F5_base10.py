"""F5 — Base-10 spike null.

Plots ``avg cos(n, n+k)`` vs shift ``k`` for N ∈ {50, 100} from the
emergent_base10 study summary. Visually demonstrates that no peak
appears at k=10 — the negative result anchoring paper §7.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from ._style import OUTPUTS, savefig


__all__ = ["render_F5_base10_spike_null"]


def render_F5_base10_spike_null() -> None:
    js = json.loads((OUTPUTS / "emergent_base10_full" / "summary.json").read_text())
    configs = js["configs"]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for c in configs:
        N = c["N"]
        shifts = []
        for row in c["per_seed"]:
            shifts.append(row["shift_stats"])
        keys = sorted({int(k) for d in shifts for k in d.keys()})
        means = []
        for k in keys:
            vs = [d[str(k)] for d in shifts if str(k) in d]
            means.append(np.mean(vs) if vs else np.nan)
        ax.plot(keys, means, "-o", label=f"N={N}  (n={len(shifts)} seeds)",
                markersize=3, linewidth=1)
        if 10 in keys:
            ax.axvline(10, color="red", ls="--", lw=0.8, alpha=0.4,
                       label="_expected base-10 peak_" if N == configs[0]["N"] else None)

    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("shift k (|n_a − n_b|)")
    ax.set_ylabel("avg cos(n, n+k)")
    ax.set_title("F5  Pure base-10 emergence null  —  cos(shift) is monotone linear, "
                 "no peak at k=10")
    ax.annotate("no peak at k=10\n(spike₁₀ ≈ 0.001,  p = 0.44)",
                xy=(10, 0), xytext=(14, 0.35),
                fontsize=8, color="red",
                arrowprops={"arrowstyle": "->", "color": "red", "lw": 0.8})
    ax.legend(loc="lower left")
    fig.tight_layout()
    savefig(fig, "F5_base10_spike_null")
