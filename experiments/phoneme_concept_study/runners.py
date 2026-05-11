"""E1 / E2 per-experiment runners for the phoneme study."""
from __future__ import annotations

import math
import random
import time

from ._config import EPOCHS, FACET_M, FACET_P, FACET_V, STEPS_PER_EPOCH
from .inventory import N_PH
from .metrics import (
    _cos_matrix,
    _cross_facet_align,
    _intra_vs_inter_gap,
    _rho_hamming_total,
    _rho_same_axis,
)
from .train import train_one


__all__ = ["run_e1_multi_seed", "run_e2_shuffled"]


def _stats(xs):
    xs = list(xs)
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
    return {"mean": m, "std": sd, "min": min(xs), "max": max(xs)}


def run_e1_multi_seed(n_seeds: int, seed_base: int = 1000,
                      epochs: int = EPOCHS,
                      steps_per_epoch: int = STEPS_PER_EPOCH) -> dict:
    rows = []
    for i in range(n_seeds):
        seed = seed_base + i
        t0 = time.time()
        t = train_one("triple", seed, epochs=epochs,
                      steps_per_epoch=steps_per_epoch)
        bs = t["bundle_state"]

        # per-facet geometry: ρ on its own axis vs other axes
        rho_same_v = _rho_same_axis(_cos_matrix(bs, FACET_V), axis=0)
        rho_same_m = _rho_same_axis(_cos_matrix(bs, FACET_M), axis=1)
        rho_same_p = _rho_same_axis(_cos_matrix(bs, FACET_P), axis=2)
        rho_v_on_m = _rho_same_axis(_cos_matrix(bs, FACET_V), axis=1)
        rho_m_on_v = _rho_same_axis(_cos_matrix(bs, FACET_M), axis=0)
        rho_v_on_p = _rho_same_axis(_cos_matrix(bs, FACET_V), axis=2)
        rho_p_on_v = _rho_same_axis(_cos_matrix(bs, FACET_P), axis=0)
        rho_m_on_p = _rho_same_axis(_cos_matrix(bs, FACET_M), axis=2)
        rho_p_on_m = _rho_same_axis(_cos_matrix(bs, FACET_P), axis=1)

        gap_v = _intra_vs_inter_gap(_cos_matrix(bs, FACET_V), axis=0)
        gap_m = _intra_vs_inter_gap(_cos_matrix(bs, FACET_M), axis=1)
        gap_p = _intra_vs_inter_gap(_cos_matrix(bs, FACET_P), axis=2)

        align_vm = _cross_facet_align(bs, FACET_V, FACET_M)
        align_vp = _cross_facet_align(bs, FACET_V, FACET_P)
        align_mp = _cross_facet_align(bs, FACET_M, FACET_P)

        rho_hm_v = _rho_hamming_total(_cos_matrix(bs, FACET_V))
        rho_hm_m = _rho_hamming_total(_cos_matrix(bs, FACET_M))
        rho_hm_p = _rho_hamming_total(_cos_matrix(bs, FACET_P))

        dt = time.time() - t0
        print(f"[E1 seed={seed}] accs(v/m/p)={t['accs']['v']:.2f}/"
              f"{t['accs']['m']:.2f}/{t['accs']['p']:.2f}  "
              f"ρ_same(v/m/p)={rho_same_v:+.3f}/{rho_same_m:+.3f}/{rho_same_p:+.3f}  "
              f"align(v-m/v-p/m-p)={align_vm:+.3f}/{align_vp:+.3f}/{align_mp:+.3f}  "
              f"({dt:.1f}s)")

        rows.append({
            "seed": seed,
            "accs": t["accs"],
            "rho_same_axis": {"v": rho_same_v, "m": rho_same_m, "p": rho_same_p},
            "rho_leakage": {
                "v_on_m": rho_v_on_m, "m_on_v": rho_m_on_v,
                "v_on_p": rho_v_on_p, "p_on_v": rho_p_on_v,
                "m_on_p": rho_m_on_p, "p_on_m": rho_p_on_m,
            },
            "intra_inter_gap": {"v": gap_v, "m": gap_m, "p": gap_p},
            "cross_facet_align": {
                "v_m": align_vm, "v_p": align_vp, "m_p": align_mp,
            },
            "rho_hamming_total": {"v": rho_hm_v, "m": rho_hm_m, "p": rho_hm_p},
            "wall_s": dt,
        })

    return {
        "n_seeds": n_seeds,
        "per_seed": rows,
        "rho_same_axis_v": _stats([r["rho_same_axis"]["v"] for r in rows]),
        "rho_same_axis_m": _stats([r["rho_same_axis"]["m"] for r in rows]),
        "rho_same_axis_p": _stats([r["rho_same_axis"]["p"] for r in rows]),
        "cross_facet_align_vm": _stats([r["cross_facet_align"]["v_m"] for r in rows]),
        "cross_facet_align_vp": _stats([r["cross_facet_align"]["v_p"] for r in rows]),
        "cross_facet_align_mp": _stats([r["cross_facet_align"]["m_p"] for r in rows]),
        "gap_v": _stats([r["intra_inter_gap"]["v"]["gap"] for r in rows]),
        "gap_m": _stats([r["intra_inter_gap"]["m"]["gap"] for r in rows]),
        "gap_p": _stats([r["intra_inter_gap"]["p"]["gap"] for r in rows]),
    }


def run_e2_shuffled(n_seeds: int, seed_base: int = 2000,
                    epochs: int = EPOCHS,
                    steps_per_epoch: int = STEPS_PER_EPOCH) -> dict:
    rows = []
    for i in range(n_seeds):
        seed = seed_base + i
        perm = list(range(N_PH))
        random.Random(seed).shuffle(perm)
        sm = {k: perm[k] for k in range(N_PH)}
        t0 = time.time()
        t = train_one("triple", seed, shuffle_map=sm,
                      epochs=epochs, steps_per_epoch=steps_per_epoch)
        bs = t["bundle_state"]
        rho_v = _rho_same_axis(_cos_matrix(bs, FACET_V), axis=0)
        rho_m = _rho_same_axis(_cos_matrix(bs, FACET_M), axis=1)
        rho_p = _rho_same_axis(_cos_matrix(bs, FACET_P), axis=2)
        dt = time.time() - t0
        print(f"[E2 seed={seed} shuffled] accs(v/m/p)="
              f"{t['accs']['v']:.2f}/{t['accs']['m']:.2f}/{t['accs']['p']:.2f}  "
              f"ρ_same(v/m/p)={rho_v:+.3f}/{rho_m:+.3f}/{rho_p:+.3f}  "
              f"({dt:.1f}s)")
        rows.append({
            "seed": seed, "shuffle_map": sm,
            "accs": t["accs"],
            "rho_same_raw_order_v": rho_v,
            "rho_same_raw_order_m": rho_m,
            "rho_same_raw_order_p": rho_p,
            "wall_s": dt,
        })

    abs_xs_v = [abs(r["rho_same_raw_order_v"]) for r in rows]
    abs_xs_m = [abs(r["rho_same_raw_order_m"]) for r in rows]
    abs_xs_p = [abs(r["rho_same_raw_order_p"]) for r in rows]
    return {
        "n_seeds": n_seeds,
        "per_seed": rows,
        "abs_rho_same_v_stats": _stats(abs_xs_v),
        "abs_rho_same_m_stats": _stats(abs_xs_m),
        "abs_rho_same_p_stats": _stats(abs_xs_p),
        "notes": "shuffle concept_id→bundle 映射后, 单 facet 在 raw 轴上的 ρ 应塌缩",
    }
