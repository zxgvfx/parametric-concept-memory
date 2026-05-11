"""PAPER §7.4 — three-causal-layer ablation for the number domain.

Mirror of ``experiments.sleep_color_primaries`` but for numbers
1..N. Tests whether the §6.8 layered-causal recipe — biological
prior + ecological statistics + task-driven asymmetry — flips the
§7 base-10 negative result into a positive one.

Five conditions, 8 seeds each, k = 3 sleep:

* **A** baseline — random orthogonal centroids, uniform sampling,
  no auxiliary head. Replicates §7's null on `spike_10`.
* **B** + decimal cones — :func:`make_decimal_cone_centroids`
  injects 10 unit + 10 tens orthogonal axes; supervision now
  carries digit identity.
* **C** + round-number sampling — multiples of 10 sampled 5×
  more often (`round_number_weights`); models human Zipf-style
  bias toward round numbers.
* **D** + last-digit head — :class:`LastDigitHead` consumes the
  same `arithmetic_bias` facet, predicting `n % 10`. The strongest
  base-10 task signal: shared loss across same-units numbers.
* **B+C+D** — all three layers stacked; mirrors the §6.8
  `BCD_combined` condition that yielded both EQUI = 4/8 and
  red-wedge = 1.00 in the colour study.

Diagnostics replicate PAPER §7.1:

* `spike_k` for k ∈ {5, 10}: positive `spike_10` with flat
  neighbours = base-10 periodicity emergence.
* `last_digit_cluster_purity`: when sleep runs k = 10, do the
  10 anchors align with the 10 last-digit equivalence classes?
* `bundle_cos_by_units`: average cos(n, n+10) − cos(n, n+1) gap.

Usage::

    python -m experiments.sleep_number_decimal \\
        --n-seeds 8 --N 30 --epochs 30 --steps 240 \\
        --out outputs/decimal_5cond_8seed
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.quad_study import (
    DEVICE,
    enumerate_triples,
    eval_on_triples,
    rho_variants_on_grid,
    train_quad,
    value_to_concept_id,
)
from experiments.number_decimal_priors import (
    round_number_weights,
)
from pcm.sleep import PROTO_CID_TEMPLATE


# ────────────────────────────────────────────────────────────────────────
# Diagnostic metrics (replicate PAPER §7.1)
# ────────────────────────────────────────────────────────────────────────


def _bundle_cos_matrix(bundle_by_idx: dict[int, torch.Tensor]) -> torch.Tensor:
    ns = sorted(bundle_by_idx.keys())
    M = F.normalize(torch.stack([bundle_by_idx[n] for n in ns]), dim=-1)
    return M @ M.t()  # (N, N)


def _spike_k(cos_mat: torch.Tensor, k: int) -> float:
    """avg cos(n, n+k) − ½(avg cos(n, n+k−1) + avg cos(n, n+k+1))."""
    N = cos_mat.shape[0]
    if k + 1 >= N:
        return float("nan")

    def _avg_offset(off: int) -> float:
        vals = []
        for i in range(N - off):
            vals.append(float(cos_mat[i, i + off].item()))
        return sum(vals) / len(vals) if vals else float("nan")

    a_k = _avg_offset(k)
    a_km1 = _avg_offset(k - 1)
    a_kp1 = _avg_offset(k + 1)
    return a_k - 0.5 * (a_km1 + a_kp1)


def _avg_cos_at_offset(cos_mat: torch.Tensor, off: int) -> float:
    N = cos_mat.shape[0]
    vals = []
    for i in range(N - off):
        vals.append(float(cos_mat[i, i + off].item()))
    return sum(vals) / len(vals) if vals else float("nan")


def _last_digit_cluster_purity(
    bundle_by_idx: dict[int, torch.Tensor],
    proto_rows: list[torch.Tensor],
) -> dict:
    """Run hard k-means style assignment from each number to its nearest
    prototype, then compute purity = max-class-frequency averaged over
    clusters. If anchors align with last-digit equivalence classes,
    purity → 1.0; if anchors are distance-aligned (linear), purity ≈ 1/k.
    """
    if not proto_rows:
        return {"purity": float("nan"), "n_clusters": 0,
                "by_cluster_top_digit": {}}
    ns = sorted(bundle_by_idx.keys())
    M = F.normalize(torch.stack([bundle_by_idx[n] for n in ns]), dim=-1)
    A = F.normalize(torch.stack(proto_rows), dim=-1)
    sims = M @ A.t()  # (N, k)
    assignments = sims.argmax(dim=-1).tolist()

    cluster_to_digits: dict[int, Counter[int]] = {}
    for i, n in enumerate(ns):
        c = int(assignments[i])
        d = n % 10
        cluster_to_digits.setdefault(c, Counter())[d] += 1
    purities: list[float] = []
    by_cluster_top: dict[int, dict] = {}
    for c, digits in cluster_to_digits.items():
        most = digits.most_common(1)[0]
        size = sum(digits.values())
        purity = most[1] / max(size, 1)
        purities.append(purity)
        by_cluster_top[c] = {
            "size": size, "top_digit": most[0], "top_count": most[1],
            "digit_hist": dict(digits),
        }
    return {
        "purity": sum(purities) / len(purities) if purities else 0.0,
        "n_clusters": len(cluster_to_digits),
        "by_cluster_top_digit": by_cluster_top,
    }


# ────────────────────────────────────────────────────────────────────────
# Per-condition runner
# ────────────────────────────────────────────────────────────────────────


from pcm.diagnostics import (
    AblationCondition, AblationLayers, run_causal_ablation,
)


CONDITIONS = (
    AblationCondition("A_baseline", AblationLayers()),
    AblationCondition("B_decimal", AblationLayers(B="decimal_cones")),
    AblationCondition("C_roundbias", AblationLayers(C="round_number")),
    AblationCondition("D_lastdigit", AblationLayers(D=True)),
    AblationCondition("BCD_combined",
                      AblationLayers(B="decimal_cones",
                                     C="round_number", D=True)),
)


def _split_triples(N: int, step: float, ood_ratio: float, seed: int):
    all_triples = enumerate_triples(N, step)
    rng = random.Random(seed)
    train_triples = []
    test_triples = []
    for op, trips in all_triples.items():
        trips = list(trips)
        rng.shuffle(trips)
        n_test = max(1, int(len(trips) * ood_ratio))
        for (a, b, c) in trips[:n_test]:
            test_triples.append((a, b, op, c))
        for (a, b, c) in trips[n_test:]:
            train_triples.append((a, b, op, c))
    return train_triples, test_triples


def _build_weights(mode: str | None, N: int) -> list[float] | None:
    if mode is None:
        return None
    if mode == "round_number":
        return round_number_weights(N, boost=5.0)
    raise ValueError(f"unknown sample mode {mode!r}")


def _run_one_seed(
    seed: int, layers: AblationLayers, *,
    N: int, epochs: int, steps_per_epoch: int,
    ood_ratio: float,
    sleep_warmup: int, sleep_every: int, sleep_k: int,
) -> dict:
    train_triples, test_triples = _split_triples(N, 1.0, ood_ratio, seed)
    weights = _build_weights(layers.C, N)

    t0 = time.time()
    r = train_quad(
        N, 1.0, seed,
        train_triples=train_triples,
        epochs=epochs, steps_per_epoch=steps_per_epoch,
        centroid_mode=layers.B if layers.B else "random",
        digit_sample_weight=weights,
        enable_last_digit_head=bool(layers.is_active("D")),
        sleep_every=sleep_every, sleep_warmup=sleep_warmup,
        sleep_k_clusters=sleep_k,
        sleep_assignment="hard",
        use_abstract=False,
    )

    train_acc = eval_on_triples(
        r["head"], r["cg"], r["centroids"], train_triples, 1.0
    )
    ood_acc = eval_on_triples(
        r["head"], r["cg"], r["centroids"], test_triples, 1.0
    )
    rho = rho_variants_on_grid(r["bundle_by_idx"])

    cos_mat = _bundle_cos_matrix(r["bundle_by_idx"])
    spike10 = _spike_k(cos_mat, 10)
    spike5 = _spike_k(cos_mat, 5)
    avg_cos1 = _avg_cos_at_offset(cos_mat, 1)
    avg_cos10 = _avg_cos_at_offset(cos_mat, 10)
    units_gap = avg_cos10 - avg_cos1

    proto_rows: list[torch.Tensor] = []
    for c in range(sleep_k):
        proto_id = PROTO_CID_TEMPLATE.format(facet="arithmetic_bias", k=c)
        cg = r["cg"]
        if proto_id in cg.cid_to_slot:
            slot = cg.cid_to_slot[proto_id]
            proto_rows.append(cg.bundle_pool["arithmetic_bias"]
                              .data[slot].detach().cpu().clone())
    purity = _last_digit_cluster_purity(r["bundle_by_idx"], proto_rows)

    return {
        "centroid_mode": layers.B,
        "sample_mode": layers.C,
        "last_digit_head": bool(layers.is_active("D")),
        "wall_s": time.time() - t0,
        "train_acc_overall": sum(train_acc.values()) / max(len(train_acc), 1),
        "ood_acc_overall": sum(ood_acc.values()) / max(len(ood_acc), 1),
        "rho_log": rho.get("rho_log"),
        "rho_lin": rho.get("rho_linear"),
        "spike_10": spike10,
        "spike_5": spike5,
        "avg_cos_offset1": avg_cos1,
        "avg_cos_offset10": avg_cos10,
        "units_gap": units_gap,
        "last_digit_purity": purity["purity"],
        "purity_n_clusters": purity["n_clusters"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=84000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=240)
    ap.add_argument("--ood-ratio", type=float, default=0.15)
    ap.add_argument("--sleep-warmup", type=int, default=15)
    ap.add_argument("--sleep-every", type=int, default=5)
    ap.add_argument("--sleep-k", type=int, default=10,
                    help="cluster count for the sleep diagnostic; 10 lets us "
                         "test if anchors align with last-digit equivalence")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/decimal_primaries"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"  PAPER §7.4 number decimal-primary ablation: N={args.N}, "
          f"n_seeds={args.n_seeds}, epochs={args.epochs}, "
          f"steps={args.steps_per_epoch}")
    print(f"  sleep k={args.sleep_k}, warmup={args.sleep_warmup}, "
          f"every={args.sleep_every}, ood={args.ood_ratio}")
    print(f"  device={DEVICE}; protocol=pcm.diagnostics.run_causal_ablation")
    print("=" * 76)

    summary = run_causal_ablation(
        _run_one_seed,
        n_seeds=args.n_seeds,
        seed_base=args.seed_base,
        conditions=CONDITIONS,
        primary_metrics=(
            "train_acc_overall", "ood_acc_overall", "rho_log",
            "spike_10", "spike_5", "units_gap", "last_digit_purity",
        ),
        N=args.N, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
        ood_ratio=args.ood_ratio,
        sleep_warmup=args.sleep_warmup, sleep_every=args.sleep_every,
        sleep_k=args.sleep_k,
    )
    summary["config"] |= vars(args) | {"out": str(args.out)}

    # Backward-compat aliases for existing F12 figure / paper sections.
    for cond_name, cs in summary["by_condition"].items():
        if "train_acc_overall" in cs and "train_acc" not in cs:
            cs["train_acc"] = cs["train_acc_overall"]
        if "ood_acc_overall" in cs and "ood_acc" not in cs:
            cs["ood_acc"] = cs["ood_acc_overall"]

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    # Cross-condition table
    print("\n" + "═" * 76)
    print("  cross-condition summary:")
    print(f"  {'condition':<16s} {'spike10':>10s} {'spike5':>10s} "
          f"{'units_gap':>10s} {'purity':>8s} {'OOD':>7s}")
    for cond in CONDITIONS:
        cs = summary["by_condition"][cond.name]
        print(
            f"  {cond.name:<16s} "
            f"{cs['spike_10']['mean']:>+8.4f}±{cs['spike_10']['std']:.3f} "
            f"{cs['spike_5']['mean']:>+8.4f}±{cs['spike_5']['std']:.3f} "
            f"{cs['units_gap']['mean']:>+8.3f}±{cs['units_gap']['std']:.3f} "
            f"{cs['last_digit_purity']['mean']:>5.3f}±{cs['last_digit_purity']['std']:.3f} "
            f"{cs['ood_acc_overall']['mean']:>5.3f}±{cs['ood_acc_overall']['std']:.3f}"
        )
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
