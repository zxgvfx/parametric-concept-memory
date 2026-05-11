"""PAPER §7.5-space — three-causal-layer ablation for spatial length extrapolation.

Mirror of ``experiments.sleep_number_extrapolate`` for the 2-D grid.
Tests whether the §6.8 / §7.4 / §7.5 / §6.9 layered priors
(cardinal-axis centroid + center-bias sampling + row-index head)
let PCM **extrapolate** movement classification from a 5×5
training subgrid to a 7×7 outer-ring test set.

Setup:

* Concept registry: 7 × 7 = 49 cells (every cell up to the test
  range has a bundle row).
* :class:`MoveHead` trains on triples whose **both** cells lie in
  the 5×5 subgrid (top-left corner; r, c ∈ [0, 5)).
* :class:`RowIndexHead` (D / BCD condition) sees the full 7×7
  inventory, so outer-ring cells' bundle rows receive
  row-identity gradient even though they never participate in
  any movement triple.

Three test splits per condition:

* **T1 random** — random hold-out from in-range 5×5 triples;
  baseline interpolation accuracy.
* **T2 mixed-OOD** — triples where exactly one of (a, b) is in
  outer ring; tests whether the system can predict direction
  *involving* a held-out cell.
* **T3 outer-OOD** — triples where **both** (a, b) are in outer
  ring; the hardest length-extrapolation test.

Predicted pattern (parallels §7.5 number ceiling):

* T1: A high (interpolation works fine in-range).
* T2 / T3: A near chance (1/5 = 0.20); D / BCD lift slightly via
  row-identity prior, but absolute gain expected ~+1–5 pp due
  to the same input-side architectural ceiling as §7.5.

Usage::

    python -m experiments.sleep_space_extrapolate \\
        --n-rows-train 5 --n-rows-total 7 --n-seeds 5 \\
        --epochs 20 --steps 240 \\
        --out outputs/space_extrap_5_7
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcm.concept_graph import ConceptGraph

from experiments.space_concept_study._config import (
    BATCH_SIZE, DEVICE, EMBED_DIM, LR, MOTION_DIM,
)
from experiments.space_concept_study.heads import MoveHead
from experiments.space_concept_study.topology import (
    move_class as _move_class_5x5,
)
from experiments.space_cardinal_priors import (
    RowIndexHead, center_bias_weights, make_cardinal_axis_centroids,
)


# ────────────────────────────────────────────────────────────────────────
# Grid + triple helpers (parametric over (n_rows, n_cols))
# ────────────────────────────────────────────────────────────────────────


def cid_of(r: int, c: int) -> str:
    return f"concept:space:{r}_{c}"


def _move_class(a, b):
    """5-class direction. Reuses the §6.2 5x5 logic since direction
    semantics are independent of grid size."""
    return _move_class_5x5(a, b)


def enumerate_move_triples_grid(n_rows: int, n_cols: int):
    out = []
    for r1 in range(n_rows):
        for c1 in range(n_cols):
            for r2 in range(n_rows):
                for c2 in range(n_cols):
                    cls = _move_class((r1, c1), (r2, c2))
                    if cls is not None:
                        out.append(((r1, c1), (r2, c2), cls))
    return out


def _is_inner(rc: tuple[int, int], n_rows_train: int, n_cols_train: int) -> bool:
    return rc[0] < n_rows_train and rc[1] < n_cols_train


def build_splits(
    n_rows_train: int, n_cols_train: int,
    n_rows_total: int, n_cols_total: int,
    ood_ratio: float, seed: int,
):
    """Return train + 3 test splits."""
    full = enumerate_move_triples_grid(n_rows_total, n_cols_total)
    train_pool = [t for t in full
                  if _is_inner(t[0], n_rows_train, n_cols_train)
                  and _is_inner(t[1], n_rows_train, n_cols_train)]
    rng = random.Random(seed)
    rng.shuffle(train_pool)
    n_test = max(1, int(len(train_pool) * ood_ratio))
    test_random = train_pool[:n_test]
    train = train_pool[n_test:]

    test_mixed = []
    test_outer = []
    for t in full:
        a_in = _is_inner(t[0], n_rows_train, n_cols_train)
        b_in = _is_inner(t[1], n_rows_train, n_cols_train)
        if a_in and b_in:
            continue
        if (not a_in) and (not b_in):
            test_outer.append(t)
        else:
            test_mixed.append(t)
    return {
        "train": train,
        "test_random": test_random,
        "test_mixed_OOD": test_mixed,
        "test_outer_OOD": test_outer,
    }


# ────────────────────────────────────────────────────────────────────────
# Conditions
# ────────────────────────────────────────────────────────────────────────


from pcm.diagnostics import (
    AblationCondition, AblationLayers, run_causal_ablation,
)


CONDITIONS = (
    AblationCondition("A_baseline", AblationLayers()),
    AblationCondition("B_cardinal", AblationLayers(B="cardinal")),
    AblationCondition("C_centerbias", AblationLayers(C="center_bias")),
    AblationCondition("D_rowindex", AblationLayers(D=True)),
    AblationCondition("BCD_combined",
                      AblationLayers(B="cardinal", C="center_bias", D=True)),
)


def _build_centroids(mode: str, n_rows: int, n_cols: int, seed: int):
    n_total = n_rows * n_cols
    if mode == "random":
        g = torch.Generator().manual_seed(seed)
        A = torch.randn(EMBED_DIM, n_total, generator=g)
        Q, _ = torch.linalg.qr(A)
        return F.normalize(Q.t(), dim=-1)
    if mode == "cardinal":
        return make_cardinal_axis_centroids(n_rows, n_cols, EMBED_DIM, seed)
    raise ValueError(f"unknown centroid mode {mode!r}")


def _build_weights(mode, n_rows_train: int, n_cols_train: int):
    if mode is None:
        return None
    if mode == "center_bias":
        return center_bias_weights(n_rows_train, n_cols_train, sigma=1.2)
    raise ValueError(f"unknown sample mode {mode!r}")


# ────────────────────────────────────────────────────────────────────────
# Per-seed runner
# ────────────────────────────────────────────────────────────────────────


def _run_one(
    seed: int, layers: AblationLayers, *,
    n_rows_train: int, n_cols_train: int,
    n_rows_total: int, n_cols_total: int,
    epochs: int, steps_per_epoch: int, ood_ratio: float,
) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    splits = build_splits(
        n_rows_train, n_cols_train, n_rows_total, n_cols_total,
        ood_ratio, seed,
    )

    cg = ConceptGraph(feat_dim=EMBED_DIM)
    for r in range(n_rows_total):
        for c in range(n_cols_total):
            cg.register_concept(
                node_id=cid_of(r, c),
                label=f"SPACE_{r}_{c}",
                scope="BASE",
                provenance=f"space_extrap:r={r},c={c}",
            )
    centroid_mode = layers.B if layers.B else "random"
    centroids = _build_centroids(
        centroid_mode, n_rows_total, n_cols_total, seed,
    ).to(DEVICE)

    head_move = MoveHead(facet_dim=MOTION_DIM).to(DEVICE)
    row_head: RowIndexHead | None = None
    if layers.is_active("D"):
        row_head = RowIndexHead(
            n_rows=n_rows_total, facet_dim=MOTION_DIM,
        ).to(DEVICE)

    with torch.no_grad():
        for r in range(n_rows_total):
            for c in range(n_cols_total):
                cn = cg.concepts[cid_of(r, c)]
                cn.collapse(
                    "MoveHead", "motion_bias", (MOTION_DIM,),
                    tick=0, device=DEVICE, init="normal_small",
                )
        # Inject B-layer cardinal centroids into bundle pool (motion_bias).
        if centroid_mode == "cardinal":
            pool = cg.bundle_pool["motion_bias"]
            target_dim = pool.shape[-1]
            cents_trim = centroids[..., :target_dim]
            for r in range(n_rows_total):
                for c in range(n_cols_total):
                    slot = cg.cid_to_slot[cid_of(r, c)]
                    pool.data[slot] = cents_trim[r * n_cols_total + c].to(
                        pool.device,
                    )
    cg.bundles_to(torch.device(DEVICE))

    # C-layer: per-cell sampling weights over the train pool.
    weights_cells = _build_weights(layers.C, n_rows_train, n_cols_train)
    if weights_cells is not None:
        train_weights = []
        for t in splits["train"]:
            r1, c1 = t[0]
            r2, c2 = t[1]
            train_weights.append(
                float(weights_cells[r1 * n_cols_train + c1])
                * float(weights_cells[r2 * n_cols_train + c2])
            )
    else:
        train_weights = None

    params = list(head_move.parameters())
    if row_head is not None:
        params += list(row_head.parameters())
    params += list(cg.iter_bundle_parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    train_pool = splits["train"]
    n_train = len(train_pool)
    if n_train == 0:
        raise RuntimeError(f"empty train pool for {cond_name}")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        head_move.train()
        if row_head is not None:
            row_head.train()
        for step_i in range(steps_per_epoch):
            if train_weights is not None:
                batch = rng.choices(train_pool, weights=train_weights,
                                    k=BATCH_SIZE)
            else:
                batch = [train_pool[rng.randrange(n_train)]
                         for _ in range(BATCH_SIZE)]
            ids_a = [cid_of(*t[0]) for t in batch]
            ids_b = [cid_of(*t[1]) for t in batch]
            tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
            pred = head_move(ids_a, ids_b, cg, tick=epoch * 10000 + step_i)
            loss = F.cross_entropy(pred, tgt)

            if row_head is not None:
                # Sample uniformly over all 49 cells; predict row index.
                rh_idx = [rng.randrange(n_rows_total * n_cols_total)
                          for _ in range(BATCH_SIZE)]
                rh_ids = [cid_of(idx // n_cols_total, idx % n_cols_total)
                          for idx in rh_idx]
                rh_tgt = torch.tensor(
                    [idx // n_cols_total for idx in rh_idx],
                    device=DEVICE,
                )
                rh_logits = row_head(
                    rh_ids, cg, tick=epoch * 10000 + step_i,
                )
                loss = loss + F.cross_entropy(rh_logits, rh_tgt)

            opt.zero_grad(); loss.backward(); opt.step()

    head_move.eval()
    if row_head is not None:
        row_head.eval()

    @torch.no_grad()
    def _eval(triples):
        if not triples:
            return float("nan")
        hits = 0
        for i in range(0, len(triples), 64):
            batch = triples[i:i + 64]
            ids_a = [cid_of(*t[0]) for t in batch]
            ids_b = [cid_of(*t[1]) for t in batch]
            tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
            pred = head_move(ids_a, ids_b, cg)
            hits += pred.argmax(-1).eq(tgt).sum().item()
        return hits / len(triples)

    train_acc = _eval(splits["train"])
    test_random = _eval(splits["test_random"])
    test_mixed = _eval(splits["test_mixed_OOD"])
    test_outer = _eval(splits["test_outer_OOD"])

    return {
        "centroid_mode": layers.B,
        "sample_mode": layers.C,
        "row_head": bool(layers.is_active("D")),
        "wall_s": time.time() - t0,
        "train_acc": train_acc,
        "test_random_in_range": test_random,
        "test_mixed_OOD": test_mixed,
        "test_outer_OOD": test_outer,
        "n_train_triples": len(splits["train"]),
        "n_random_triples": len(splits["test_random"]),
        "n_mixed_triples": len(splits["test_mixed_OOD"]),
        "n_outer_triples": len(splits["test_outer_OOD"]),
    }


# ────────────────────────────────────────────────────────────────────────
# Entrypoint
# ────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows-train", type=int, default=5)
    ap.add_argument("--n-cols-train", type=int, default=5)
    ap.add_argument("--n-rows-total", type=int, default=7)
    ap.add_argument("--n-cols-total", type=int, default=7)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=92000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=240)
    ap.add_argument("--ood-ratio", type=float, default=0.15)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/space_extrap"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"  PAPER §7.5-space length extrapolation: "
          f"train_grid={args.n_rows_train}x{args.n_cols_train}, "
          f"total_grid={args.n_rows_total}x{args.n_cols_total}, "
          f"n_seeds={args.n_seeds}")
    print(f"  device={DEVICE}; protocol=pcm.diagnostics.run_causal_ablation")
    print("=" * 80)

    summary = run_causal_ablation(
        _run_one,
        n_seeds=args.n_seeds,
        seed_base=args.seed_base,
        conditions=CONDITIONS,
        primary_metrics=(
            "train_acc",
            "test_random_in_range",
            "test_mixed_OOD",
            "test_outer_OOD",
        ),
        n_rows_train=args.n_rows_train,
        n_cols_train=args.n_cols_train,
        n_rows_total=args.n_rows_total,
        n_cols_total=args.n_cols_total,
        epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
        ood_ratio=args.ood_ratio,
    )
    summary["config"] |= vars(args) | {"out": str(args.out)}

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "═" * 80)
    print("  cross-condition summary (chance ≈ 1/5 = 0.200):")
    print(f"  {'condition':<16s} {'in_range':>16s} {'mixed_OOD':>16s} "
          f"{'outer_OOD':>16s}")
    for cond in CONDITIONS:
        cs = summary["by_condition"][cond.name]
        ir = cs["test_random_in_range"]
        mx = cs["test_mixed_OOD"]
        ot = cs["test_outer_OOD"]
        print(
            f"  {cond.name:<16s} "
            f"{ir['mean']:>8.3f}±{ir['std']:.3f}   "
            f"{mx['mean']:>8.3f}±{mx['std']:.3f}   "
            f"{ot['mean']:>8.3f}±{ot['std']:.3f}"
        )
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
