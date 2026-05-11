"""PAPER §15 (S3) — Causal-JEPA-style mask-infer head for spatial OOD.

Mirrors :mod:`experiments.sleep_space_extrapolate` but swaps /
adds ``MaskInferHead`` as the D-layer task. The hypothesis (S3
in ``docs/2026_LITERATURE_AND_PLANS.md``) is that a head whose
inputs are **neighbour bundle rows** of a target cell rotates
every cell through the input-distribution slot, breaking the
``MoveHead.fc1`` joint-distribution coverage ceiling that left
§7.5-space ``mixed_OOD = 0.000`` universally — without per-task
data augmentation.

Conditions (4 × N_seeds, paired with §7.5-space's BCD baselines):

* ``A_baseline`` — random centroids, uniform sampling, no D
  head. Reproduces the §7.5-space A row.
* ``BCD_RowIndex`` — the existing strongest §7.5-space baseline
  (cardinal centroids + center bias + RowIndexHead D layer).
* ``BCD_MaskInfer`` — replace RowIndexHead with the new
  MaskInferHead. Tests whether mask-infer ALONE can break
  ``mixed_OOD`` ceiling.
* ``BCD_Both`` — keep RowIndexHead AND add MaskInferHead. Tests
  whether the two D-layer signals are additive.

Usage::

    python -m experiments.sleep_space_mask_infer \\
        --n-rows-train 5 --n-rows-total 7 --n-seeds 5 \\
        --epochs 20 --steps 240 \\
        --out outputs/space_mask_infer_5_7
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from pcm.concept_graph import ConceptGraph
from pcm.diagnostics import (
    AblationCondition, AblationLayers, run_causal_ablation,
)

from experiments.space_concept_study._config import (
    BATCH_SIZE, DEVICE, EMBED_DIM, LR, MOTION_DIM,
)
from experiments.space_concept_study.heads import MoveHead
from experiments.space_cardinal_priors import (
    InverseMoveHead, MaskInferHead, RowIndexHead,
    center_bias_weights, enumerate_inverse_move_samples,
    enumerate_mask_infer_samples, make_cardinal_axis_centroids,
)
from experiments.sleep_space_extrapolate import (
    build_splits, cid_of,
)


# ─────────────────────────────────────────────────────────────────
# Conditions — 4 cells of the §7.5-space cube + S3 mask-infer
# ─────────────────────────────────────────────────────────────────
# We re-use AblationLayers semantics:
#   B = "cardinal" → cardinal-axis centroids
#   C = "center_bias" → C-layer non-uniform sampling
#   D = True       → some D-layer head is active
# We additionally read condition.name to decide WHICH D head:
#   "BCD_RowIndex" → RowIndexHead only
#   "BCD_MaskInfer" → MaskInferHead only
#   "BCD_Both" → RowIndexHead + MaskInferHead


CONDITIONS = (
    AblationCondition("A_baseline", AblationLayers()),
    AblationCondition(
        "BCD_RowIndex",
        AblationLayers(B="cardinal", C="center_bias", D="row_index"),
    ),
    AblationCondition(
        "BCD_MaskInfer",
        AblationLayers(B="cardinal", C="center_bias", D="mask_infer"),
    ),
    AblationCondition(
        "BCD_InverseMove",
        AblationLayers(B="cardinal", C="center_bias", D="inverse_move"),
    ),
    AblationCondition(
        "BCD_All",
        AblationLayers(B="cardinal", C="center_bias", D="all_three"),
    ),
)


# ─────────────────────────────────────────────────────────────────
# Per-seed runner
# ─────────────────────────────────────────────────────────────────


def _build_centroids(mode, n_rows: int, n_cols: int, seed: int):
    if mode is None:
        g = torch.Generator().manual_seed(seed)
        n_total = n_rows * n_cols
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


def _run_one(
    seed: int, layers: AblationLayers, *,
    n_rows_train: int, n_cols_train: int,
    n_rows_total: int, n_cols_total: int,
    epochs: int, steps_per_epoch: int, ood_ratio: float,
) -> dict:
    """One seed × one condition. D-layer head selection is encoded
    in ``layers.D``: "row_index", "mask_infer", or "row_and_mask".
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)

    splits = build_splits(
        n_rows_train, n_cols_train, n_rows_total, n_cols_total,
        ood_ratio, seed,
    )

    cg = ConceptGraph(feat_dim=EMBED_DIM)
    n_total_cells = n_rows_total * n_cols_total
    for r in range(n_rows_total):
        for c in range(n_cols_total):
            cg.register_concept(
                node_id=cid_of(r, c),
                label=f"SPACE_{r}_{c}",
                scope="BASE",
                provenance=f"space_mask_infer:r={r},c={c}",
            )

    centroids = _build_centroids(
        layers.B, n_rows_total, n_cols_total, seed,
    ).to(DEVICE)

    head_move = MoveHead(facet_dim=MOTION_DIM).to(DEVICE)

    d_kind = layers.D if isinstance(layers.D, str) else None
    use_row = d_kind in ("row_index", "row_and_mask", "all_three")
    use_mask = d_kind in ("mask_infer", "row_and_mask", "all_three")
    use_inverse = d_kind in ("inverse_move", "all_three")

    row_head: RowIndexHead | None = None
    if use_row:
        row_head = RowIndexHead(
            n_rows=n_rows_total, facet_dim=MOTION_DIM,
        ).to(DEVICE)

    mask_head: MaskInferHead | None = None
    mask_samples = None
    if use_mask:
        mask_head = MaskInferHead(
            n_cells=n_total_cells, facet_dim=MOTION_DIM,
        ).to(DEVICE)
        mask_samples = enumerate_mask_infer_samples(
            n_rows_total, n_cols_total,
        )

    inverse_head: InverseMoveHead | None = None
    inverse_samples = None
    if use_inverse:
        inverse_head = InverseMoveHead(
            n_cells=n_total_cells, facet_dim=MOTION_DIM,
        ).to(DEVICE)
        inverse_samples = enumerate_inverse_move_samples(
            n_rows_total, n_cols_total,
        )

    with torch.no_grad():
        for r in range(n_rows_total):
            for c in range(n_cols_total):
                cn = cg.concepts[cid_of(r, c)]
                cn.collapse(
                    "MoveHead", "motion_bias", (MOTION_DIM,),
                    tick=0, device=DEVICE, init="normal_small",
                )
        if layers.B == "cardinal":
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
    if mask_head is not None:
        params += list(mask_head.parameters())
    if inverse_head is not None:
        params += list(inverse_head.parameters())
    params += list(cg.iter_bundle_parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    train_pool = splits["train"]
    n_train = len(train_pool)
    if n_train == 0:
        raise RuntimeError("empty train pool")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        head_move.train()
        if row_head is not None:
            row_head.train()
        if mask_head is not None:
            mask_head.train()
        if inverse_head is not None:
            inverse_head.train()
        for step_i in range(steps_per_epoch):
            if train_weights is not None:
                batch = rng.choices(
                    train_pool, weights=train_weights, k=BATCH_SIZE,
                )
            else:
                batch = [
                    train_pool[rng.randrange(n_train)]
                    for _ in range(BATCH_SIZE)
                ]
            ids_a = [cid_of(*t[0]) for t in batch]
            ids_b = [cid_of(*t[1]) for t in batch]
            tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
            pred = head_move(ids_a, ids_b, cg, tick=epoch * 10000 + step_i)
            loss = F.cross_entropy(pred, tgt)

            if row_head is not None:
                rh_idx = [
                    rng.randrange(n_total_cells)
                    for _ in range(BATCH_SIZE)
                ]
                rh_ids = [
                    cid_of(idx // n_cols_total, idx % n_cols_total)
                    for idx in rh_idx
                ]
                rh_tgt = torch.tensor(
                    [idx // n_cols_total for idx in rh_idx],
                    device=DEVICE,
                )
                rh_logits = row_head(
                    rh_ids, cg, tick=epoch * 10000 + step_i,
                )
                loss = loss + F.cross_entropy(rh_logits, rh_tgt)

            if mask_head is not None:
                samples = [
                    mask_samples[rng.randrange(len(mask_samples))]
                    for _ in range(BATCH_SIZE)
                ]
                neighbor_ids_per_sample = [
                    [
                        cid_of(nidx // n_cols_total, nidx % n_cols_total)
                        for nidx in s[1]
                    ]
                    for s in samples
                ]
                mask_tgt = torch.tensor(
                    [s[0] for s in samples], device=DEVICE,
                )
                mask_logits = mask_head(
                    neighbor_ids_per_sample, cg,
                    tick=epoch * 10000 + step_i,
                )
                loss = loss + F.cross_entropy(mask_logits, mask_tgt)

            if inverse_head is not None:
                inv_samples = [
                    inverse_samples[rng.randrange(len(inverse_samples))]
                    for _ in range(BATCH_SIZE)
                ]
                inv_a_ids = [
                    cid_of(s[0] // n_cols_total, s[0] % n_cols_total)
                    for s in inv_samples
                ]
                inv_dir = torch.tensor(
                    [s[1] for s in inv_samples], device=DEVICE,
                )
                inv_tgt = torch.tensor(
                    [s[2] for s in inv_samples], device=DEVICE,
                )
                inv_logits = inverse_head(
                    inv_a_ids, inv_dir, cg, tick=epoch * 10000 + step_i,
                )
                loss = loss + F.cross_entropy(inv_logits, inv_tgt)

            opt.zero_grad(); loss.backward(); opt.step()

    head_move.eval()
    if row_head is not None:
        row_head.eval()
    if mask_head is not None:
        mask_head.eval()
    if inverse_head is not None:
        inverse_head.eval()

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
        "d_kind": d_kind,
        "use_row_head": use_row,
        "use_mask_head": use_mask,
        "use_inverse_head": use_inverse,
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


# ─────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows-train", type=int, default=5)
    ap.add_argument("--n-cols-train", type=int, default=5)
    ap.add_argument("--n-rows-total", type=int, default=7)
    ap.add_argument("--n-cols-total", type=int, default=7)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=99100)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=240)
    ap.add_argument("--ood-ratio", type=float, default=0.15)
    ap.add_argument(
        "--out", type=Path,
        default=Path("outputs/space_mask_infer"),
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(
        f"  PAPER §15 (S3) mask-infer ablation: "
        f"train={args.n_rows_train}x{args.n_cols_train}, "
        f"total={args.n_rows_total}x{args.n_cols_total}, "
        f"n_seeds={args.n_seeds}"
    )
    print(f"  device={DEVICE}; protocol=pcm.diagnostics.run_causal_ablation")
    print("=" * 80)

    # The protocol passes condition.name to _run_one as
    # ``condition_name`` so that the same _run_one body can branch
    # on which D head(s) are active.
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
    print(
        f"  {'condition':<18s} {'in_range':>16s} {'mixed_OOD':>16s} "
        f"{'outer_OOD':>16s}"
    )
    for cond in CONDITIONS:
        cs = summary["by_condition"][cond.name]
        ir = cs["test_random_in_range"]
        mx = cs["test_mixed_OOD"]
        ot = cs["test_outer_OOD"]
        print(
            f"  {cond.name:<18s} "
            f"{ir['mean']:>8.3f}±{ir['std']:.3f}   "
            f"{mx['mean']:>8.3f}±{mx['std']:.3f}   "
            f"{ot['mean']:>8.3f}±{ot['std']:.3f}"
        )
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
