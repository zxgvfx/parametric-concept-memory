"""Per-experiment runners (E1 / E2 / E4) for the color study."""
from __future__ import annotations

import math
import random
import time

import torch
from scipy.stats import spearmanr, ttest_ind

from ._config import EMBED_DIM, FACET_ADJ, FACET_MIX, N_COLORS
from .graph_builder import make_random_orthogonal_centroids
from .metrics import (
    _cos_matrix,
    _cross_facet_align,
    _rho_circular,
    _rho_linear,
)
from .train import train_one


__all__ = [
    "run_e1_multi_seed",
    "run_e2_shuffled",
    "run_e4_permutation",
]


def _stats(xs):
    xs = list(xs)
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
    return {"mean": m, "std": sd, "min": min(xs), "max": max(xs)}


def run_e1_multi_seed(n_seeds: int, seed_base: int = 1000) -> dict:
    rows = []
    for i in range(n_seeds):
        seed = seed_base + i
        centroids = make_random_orthogonal_centroids(N_COLORS, EMBED_DIM, seed)
        t0 = time.time()
        s = train_one("single", seed, centroids)
        d = train_one("dual",   seed, centroids)
        rho_s_c = _rho_circular(_cos_matrix(s["bundle_state"], FACET_MIX))
        rho_s_l = _rho_linear(_cos_matrix(s["bundle_state"], FACET_MIX))
        rho_d_c = _rho_circular(_cos_matrix(d["bundle_state"], FACET_MIX))
        rho_d_adj = _rho_circular(_cos_matrix(d["bundle_state"], FACET_ADJ))
        align = _cross_facet_align(d["bundle_state"], FACET_MIX, FACET_ADJ)
        dt = time.time() - t0
        print(f"[E1 seed={seed}] single mix_acc={s['mix_acc']:.3f}  "
              f"ρ_circ={rho_s_c:+.3f}  ρ_lin={rho_s_l:+.3f}  | "
              f"dual mix_acc={d['mix_acc']:.3f} adj_acc={d['adj_acc']:.3f}  "
              f"ρ_mix_circ={rho_d_c:+.3f}  ρ_adj_circ={rho_d_adj:+.3f}  "
              f"align={align:+.3f}  ({dt:.1f}s)")
        rows.append({
            "seed": seed,
            "single_mix_acc": s["mix_acc"],
            "single_rho_circular": rho_s_c,
            "single_rho_linear": rho_s_l,
            "dual_mix_acc": d["mix_acc"],
            "dual_adj_acc": d["adj_acc"],
            "dual_rho_mix_circular": rho_d_c,
            "dual_rho_adj_circular": rho_d_adj,
            "dual_cross_facet_align": align,
            "wall_s": dt,
        })

    circ_s = [r["single_rho_circular"] for r in rows]
    circ_d = [r["dual_rho_mix_circular"] for r in rows]
    tstat, pval = ttest_ind(circ_s, circ_d, equal_var=False)
    return {
        "n_seeds": n_seeds,
        "per_seed": rows,
        "single_rho_circular": _stats(circ_s),
        "single_rho_linear":   _stats([r["single_rho_linear"] for r in rows]),
        "dual_rho_mix_circular": _stats(circ_d),
        "dual_rho_adj_circular": _stats([r["dual_rho_adj_circular"] for r in rows]),
        "dual_cross_facet_align": _stats([r["dual_cross_facet_align"] for r in rows]),
        "welch_single_vs_dual": {"t": float(tstat), "p_value": float(pval)},
    }


def run_e2_shuffled(n_seeds: int, seed_base: int = 2000) -> dict:
    rows = []
    for i in range(n_seeds):
        seed = seed_base + i
        centroids = make_random_orthogonal_centroids(N_COLORS, EMBED_DIM, seed)
        perm = list(range(N_COLORS))
        random.Random(seed).shuffle(perm)
        sm = {k: perm[k] for k in range(N_COLORS)}
        t0 = time.time()
        s = train_one("single", seed, centroids, shuffle_map=sm)
        rho_c = _rho_circular(_cos_matrix(s["bundle_state"], FACET_MIX))
        rho_l = _rho_linear(_cos_matrix(s["bundle_state"], FACET_MIX))
        dt = time.time() - t0
        print(f"[E2 seed={seed} shuffled] mix_acc={s['mix_acc']:.3f}  "
              f"ρ_circ={rho_c:+.3f} ρ_lin={rho_l:+.3f}  ({dt:.1f}s)")
        rows.append({
            "seed": seed, "shuffle_map": sm,
            "mix_acc": s["mix_acc"],
            "rho_circular_raw_order": rho_c,
            "rho_linear_raw_order": rho_l,
            "wall_s": dt,
        })

    abs_circ = [abs(r["rho_circular_raw_order"]) for r in rows]
    return {
        "n_seeds": n_seeds,
        "per_seed": rows,
        "abs_rho_circular_stats": _stats(abs_circ),
        "notes": "训练时 shuffle concept_id→bundle. ρ 是对 RAW circular order 测. "
                 "若 shuffle 破坏身份, |ρ| 应 ≈ 0.",
    }


def run_e4_permutation(dual_bundle: dict, n_perm: int = 1000) -> dict:
    cos_m = _cos_matrix(dual_bundle, FACET_MIX)
    cos_a = _cos_matrix(dual_bundle, FACET_ADJ)
    mask = ~torch.eye(N_COLORS, dtype=torch.bool)
    om = cos_m[mask].numpy()
    oa = cos_a[mask].numpy()
    observed = float(spearmanr(om, oa)[0])

    rng = random.Random(0)
    ge = 0
    null = []
    for _ in range(n_perm):
        perm = list(range(N_COLORS))
        rng.shuffle(perm)
        cos_a_perm = cos_a[perm][:, perm]
        oap = cos_a_perm[mask].numpy()
        r = float(spearmanr(om, oap)[0])
        null.append(r)
        if abs(r) >= abs(observed):
            ge += 1
    p = (ge + 1) / (n_perm + 1)
    return {
        "observed_cross_facet_rho": observed,
        "n_permutations": n_perm,
        "p_value": p,
        "null_mean": sum(null) / len(null),
        "null_std": math.sqrt(sum((r - sum(null) / len(null)) ** 2 for r in null)
                              / max(len(null) - 1, 1)),
        "conclusion": ("significant (p<0.01)" if p < 0.01
                       else "significant (p<0.05)" if p < 0.05
                       else "not significant"),
    }
