"""Per-experiment runners (E1 / E2 / E4) for the spatial study."""
from __future__ import annotations

import math
import random
import time

import torch
from scipy.stats import spearmanr

from ._config import EPOCHS, FACET_DIST, FACET_MOVE, N_CELLS, STEPS_PER_EPOCH
from .metrics import (
    _cos_matrix,
    _cross_facet_align,
    _mds_grid_fit,
    _rho_col_within,
    _rho_L1,
    _rho_linear_flat,
    _rho_row_within,
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


def run_e1_multi_seed(
    n_seeds: int,
    seed_base: int = 1000,
    epochs: int = EPOCHS,
    steps_per_epoch: int = STEPS_PER_EPOCH,
) -> dict:
    rows = []
    for i in range(n_seeds):
        seed = seed_base + i
        t0 = time.time()
        s = train_one("single", seed, epochs=epochs, steps_per_epoch=steps_per_epoch)
        d = train_one("dual",   seed, epochs=epochs, steps_per_epoch=steps_per_epoch)
        rho_L1_s = _rho_L1(_cos_matrix(s["bundle_state"], FACET_MOVE))
        rho_lin_s = _rho_linear_flat(_cos_matrix(s["bundle_state"], FACET_MOVE))
        rho_row_s = _rho_row_within(_cos_matrix(s["bundle_state"], FACET_MOVE))
        rho_col_s = _rho_col_within(_cos_matrix(s["bundle_state"], FACET_MOVE))

        rho_L1_d = _rho_L1(_cos_matrix(d["bundle_state"], FACET_MOVE))
        rho_L1_d_dist = _rho_L1(_cos_matrix(d["bundle_state"], FACET_DIST))
        align = _cross_facet_align(d["bundle_state"], FACET_MOVE, FACET_DIST)

        mds_fit_single = _mds_grid_fit(s["bundle_state"], FACET_MOVE, seed=seed)
        mds_fit_dual_move = _mds_grid_fit(d["bundle_state"], FACET_MOVE, seed=seed)

        dt = time.time() - t0
        print(f"[E1 seed={seed}] single move_acc={s['move_acc']:.3f}  "
              f"ρ_L1={rho_L1_s:+.3f}  ρ_lin={rho_lin_s:+.3f}  "
              f"ρ_row={rho_row_s:+.3f}  ρ_col={rho_col_s:+.3f}  "
              f"MDS_disp={mds_fit_single['disparity']:.3f}  | "
              f"dual move_acc={d['move_acc']:.3f} dist_acc={d['dist_acc']:.3f}  "
              f"ρ_mov={rho_L1_d:+.3f}  ρ_dst={rho_L1_d_dist:+.3f}  "
              f"align={align:+.3f}  ({dt:.1f}s)")
        rows.append({
            "seed": seed,
            "single_move_acc": s["move_acc"],
            "single_rho_L1": rho_L1_s,
            "single_rho_linear_flat": rho_lin_s,
            "single_rho_row_within": rho_row_s,
            "single_rho_col_within": rho_col_s,
            "single_mds_fit": mds_fit_single,
            "dual_move_acc": d["move_acc"],
            "dual_dist_acc": d["dist_acc"],
            "dual_rho_motion_L1": rho_L1_d,
            "dual_rho_distance_L1": rho_L1_d_dist,
            "dual_cross_facet_align": align,
            "dual_mds_fit_motion": mds_fit_dual_move,
            "wall_s": dt,
        })

    return {
        "n_seeds": n_seeds,
        "per_seed": rows,
        "single_rho_L1": _stats([r["single_rho_L1"] for r in rows]),
        "single_rho_linear_flat": _stats([r["single_rho_linear_flat"] for r in rows]),
        "single_rho_row_within": _stats([r["single_rho_row_within"] for r in rows]),
        "single_rho_col_within": _stats([r["single_rho_col_within"] for r in rows]),
        "single_mds_disparity": _stats(
            [r["single_mds_fit"]["disparity"] for r in rows]),
        "dual_rho_motion_L1": _stats([r["dual_rho_motion_L1"] for r in rows]),
        "dual_rho_distance_L1": _stats([r["dual_rho_distance_L1"] for r in rows]),
        "dual_cross_facet_align": _stats([r["dual_cross_facet_align"] for r in rows]),
        "dual_mds_disparity_motion": _stats(
            [r["dual_mds_fit_motion"]["disparity"] for r in rows]),
    }


def run_e2_shuffled(
    n_seeds: int,
    seed_base: int = 2000,
    epochs: int = EPOCHS,
    steps_per_epoch: int = STEPS_PER_EPOCH,
) -> dict:
    """Shuffle concept_id → bundle 映射, ρ 应大幅塌缩."""
    rows = []
    for i in range(n_seeds):
        seed = seed_base + i
        perm = list(range(N_CELLS))
        random.Random(seed).shuffle(perm)
        sm = {k: perm[k] for k in range(N_CELLS)}
        t0 = time.time()
        s = train_one("single", seed, shuffle_map=sm,
                      epochs=epochs, steps_per_epoch=steps_per_epoch)
        rho_L1 = _rho_L1(_cos_matrix(s["bundle_state"], FACET_MOVE))
        rho_row = _rho_row_within(_cos_matrix(s["bundle_state"], FACET_MOVE))
        rho_col = _rho_col_within(_cos_matrix(s["bundle_state"], FACET_MOVE))
        mds_fit = _mds_grid_fit(s["bundle_state"], FACET_MOVE, seed=seed)
        dt = time.time() - t0
        print(f"[E2 seed={seed} shuffled] move_acc={s['move_acc']:.3f}  "
              f"ρ_L1={rho_L1:+.3f} ρ_row={rho_row:+.3f} ρ_col={rho_col:+.3f}  "
              f"MDS_disp={mds_fit['disparity']:.3f}  ({dt:.1f}s)")
        rows.append({
            "seed": seed, "shuffle_map": sm,
            "move_acc": s["move_acc"],
            "rho_L1_raw_order": rho_L1,
            "rho_row_raw_order": rho_row,
            "rho_col_raw_order": rho_col,
            "mds_fit": mds_fit,
            "wall_s": dt,
        })

    abs_xs = [abs(r["rho_L1_raw_order"]) for r in rows]
    return {
        "n_seeds": n_seeds,
        "per_seed": rows,
        "abs_rho_L1_stats": _stats(abs_xs),
        "notes": "shuffle_map 破坏 concept_id→bundle 身份映射, |ρ_L1| 应 ≈ 0.",
    }


def run_e4_permutation(dual_bundle: dict, n_perm: int = 1000) -> dict:
    cos_m = _cos_matrix(dual_bundle, FACET_MOVE)
    cos_d = _cos_matrix(dual_bundle, FACET_DIST)
    mask = ~torch.eye(N_CELLS, dtype=torch.bool)
    om = cos_m[mask].numpy()
    od = cos_d[mask].numpy()
    observed = float(spearmanr(om, od)[0])

    rng = random.Random(0)
    ge = 0
    null = []
    for _ in range(n_perm):
        perm = list(range(N_CELLS))
        rng.shuffle(perm)
        cos_d_perm = cos_d[perm][:, perm]
        odp = cos_d_perm[mask].numpy()
        r = float(spearmanr(om, odp)[0])
        null.append(r)
        if abs(r) >= abs(observed):
            ge += 1
    p = (ge + 1) / (n_perm + 1)
    return {
        "observed_cross_facet_rho": observed,
        "n_permutations": n_perm,
        "p_value": p,
        "null_mean": sum(null) / len(null),
        "null_std": math.sqrt(
            sum((r - sum(null) / len(null)) ** 2 for r in null)
            / max(len(null) - 1, 1)
        ),
        "conclusion": (
            "significant (p<0.01)" if p < 0.01
            else "significant (p<0.05)" if p < 0.05
            else "not significant"
        ),
    }
