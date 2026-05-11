"""PAPER §7.5-space addendum — does training-time mixed-pair augmentation
break the input-distribution interaction ceiling?

§7.5-space found that all five A/B/C/D/B+C+D conditions give
strictly mixed_OOD = 0.000 (25 / 25 runs). The diagnosis: the
``MoveHead.fc1`` layer is trained only on the inner × inner
joint distribution; (inner, outer) pairs land in an OOD joint
even when each cell's individual bundle is prior-injected.

This experiment probes the *mechanism* of that ceiling. We add a
``--mixed-aug-frac`` knob: at training time, a fraction of every
batch is drawn from a held-out portion of the *mixed-pair* pool
(triples with exactly one cell on the outer ring). The
remaining mixed pairs stay held out and become the OOD test.

If the ceiling is purely a joint-distribution-coverage issue,
*any* non-zero fraction of mixed pairs in training should bring
mixed_OOD above 0.000. If it is something deeper (e.g.
fc1's representation needs to compositionally generalise, not
just see new joint distributions), augmentation may not help.

Setup:

* Total grid 7 × 7, training inner subgrid 5 × 5 (same as §7.5-space).
* All cells registered; BCD_combined layer activations
  (cardinal centroid + center-bias sampling + row-index head).
* For each ``mixed_aug_frac ∈ {0.0, 0.05, 0.15, 0.30}``:
  * Take the full mixed-pair pool, randomly split it 50/50 per
    seed into (mixed_train_pool, mixed_test_pool).
  * Each training batch contains
    ``(1 - aug_frac) * BATCH_SIZE`` inner pairs plus
    ``aug_frac * BATCH_SIZE`` mixed pairs sampled from
    mixed_train_pool.
  * Evaluation: in_range / mixed_train_pool / mixed_test_pool /
    outer_OOD.

Predicted patterns:

* aug_frac = 0.0: replicate §7.5-space (mixed_test = 0.000).
* aug_frac small (5–15 %): if ceiling is coverage-only,
  mixed_test should rise sharply; if compositional, may stay 0.
* aug_frac large (30 %): mixed_test approaches in-distribution
  performance.

Result interpretation: a *sharp* lift at any non-zero
augmentation rate confirms the ceiling is fc1-distribution-
specific and not a deeper PCM-architectural limit.

Usage::

    python -m experiments.sleep_space_mixed_augment --n-seeds 5 \\
        --rates 0.0 0.05 0.15 0.30 \\
        --epochs 20 --steps-per-epoch 200 \\
        --out outputs/space_mixed_aug
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from pcm.concept_graph import ConceptGraph

from experiments.space_concept_study._config import (
    BATCH_SIZE, DEVICE, EMBED_DIM, LR, MOTION_DIM,
)
from experiments.space_concept_study.heads import MoveHead
from experiments.sleep_space_extrapolate import (
    build_splits, cid_of, enumerate_move_triples_grid,
)
from experiments.space_cardinal_priors import (
    RowIndexHead, center_bias_weights, make_cardinal_axis_centroids,
)


def _split_mixed_pool(
    splits: dict, seed: int,
) -> tuple[list, list]:
    """Halve the mixed-OOD pool per seed: half goes to training-aug
    pool, half remains the held-out test set."""
    rng = random.Random(seed + 9999)
    mixed = list(splits["test_mixed_OOD"])
    rng.shuffle(mixed)
    half = len(mixed) // 2
    return mixed[:half], mixed[half:]


def _run_one(
    seed: int, mixed_aug_frac: float, *,
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
    mixed_train_pool, mixed_test_pool = _split_mixed_pool(splits, seed)

    cg = ConceptGraph(feat_dim=EMBED_DIM)
    for r in range(n_rows_total):
        for c in range(n_cols_total):
            cg.register_concept(
                node_id=cid_of(r, c),
                label=f"SPACE_{r}_{c}",
                scope="BASE",
                provenance=f"space_extrap:r={r},c={c}",
            )
    centroids = make_cardinal_axis_centroids(
        n_rows_total, n_cols_total, EMBED_DIM, seed,
    ).to(DEVICE)

    head_move = MoveHead(facet_dim=MOTION_DIM).to(DEVICE)
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

    weights_cells = center_bias_weights(n_rows_train, n_cols_train, sigma=1.2)
    train_pool = splits["train"]
    train_weights = []
    for t in train_pool:
        r1, c1 = t[0]
        r2, c2 = t[1]
        train_weights.append(
            float(weights_cells[r1 * n_cols_train + c1])
            * float(weights_cells[r2 * n_cols_train + c2])
        )

    params = (list(head_move.parameters())
              + list(row_head.parameters())
              + list(cg.iter_bundle_parameters()))
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    n_aug = int(round(BATCH_SIZE * mixed_aug_frac))
    n_inner = BATCH_SIZE - n_aug

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        head_move.train()
        row_head.train()
        for step_i in range(steps_per_epoch):
            # Inner-pool sample with center-bias weights.
            if n_inner > 0:
                inner_batch = rng.choices(
                    train_pool, weights=train_weights, k=n_inner,
                )
            else:
                inner_batch = []

            # Mixed-pool augmentation (uniform over training-aug pool).
            if n_aug > 0 and mixed_train_pool:
                mixed_batch = [
                    mixed_train_pool[rng.randrange(len(mixed_train_pool))]
                    for _ in range(n_aug)
                ]
            else:
                mixed_batch = []

            batch = inner_batch + mixed_batch
            if not batch:
                continue
            ids_a = [cid_of(*t[0]) for t in batch]
            ids_b = [cid_of(*t[1]) for t in batch]
            tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
            pred = head_move(ids_a, ids_b, cg, tick=epoch * 10000 + step_i)
            loss = F.cross_entropy(pred, tgt)

            # D-layer row-index head on full inventory.
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

    return {
        "seed": seed,
        "mixed_aug_frac": mixed_aug_frac,
        "n_inner_per_batch": n_inner,
        "n_aug_per_batch": n_aug,
        "n_train_inner": len(train_pool),
        "n_mixed_train_pool": len(mixed_train_pool),
        "n_mixed_test_pool": len(mixed_test_pool),
        "wall_s": time.time() - t0,
        "train_inner_acc": _eval(train_pool),
        "test_random_in_range": _eval(splits["test_random"]),
        "mixed_train_acc": _eval(mixed_train_pool),
        "mixed_test_OOD": _eval(mixed_test_pool),
        "outer_OOD": _eval(splits["test_outer_OOD"]),
    }


def _stats(xs):
    xs = [x for x in xs
          if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
    return {"mean": m, "std": sd, "min": min(xs), "max": max(xs), "n": len(xs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows-train", type=int, default=5)
    ap.add_argument("--n-cols-train", type=int, default=5)
    ap.add_argument("--n-rows-total", type=int, default=7)
    ap.add_argument("--n-cols-total", type=int, default=7)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=94000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--ood-ratio", type=float, default=0.15)
    ap.add_argument("--rates", type=float, nargs="+",
                    default=[0.0, 0.05, 0.15, 0.30],
                    help="mixed-pair augmentation rates per training batch")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/space_mixed_aug"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"  PAPER §7.5-space addendum: mixed-pair augmentation ablation")
    print(f"  train_grid={args.n_rows_train}x{args.n_cols_train}, "
          f"total_grid={args.n_rows_total}x{args.n_cols_total}, "
          f"n_seeds={args.n_seeds}, rates={args.rates}")
    print(f"  device={DEVICE}; condition fixed: BCD_combined")
    print("=" * 80)

    summary: dict = {
        "config": vars(args) | {"out": str(args.out)},
        "by_rate": {},
    }
    for rate in args.rates:
        print(f"\n── rate = {rate:.2f} ──")
        rows = []
        for si in range(args.n_seeds):
            seed = args.seed_base + si
            r = _run_one(
                seed, rate,
                n_rows_train=args.n_rows_train,
                n_cols_train=args.n_cols_train,
                n_rows_total=args.n_rows_total,
                n_cols_total=args.n_cols_total,
                epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
                ood_ratio=args.ood_ratio,
            )
            rows.append(r)
            print(
                f"  [seed={seed}] "
                f"train={r['train_inner_acc']:.3f} "
                f"in_range={r['test_random_in_range']:.3f}  "
                f"mix_train={r['mixed_train_acc']:.3f}  "
                f"mix_test={r['mixed_test_OOD']:.3f}  "
                f"outer={r['outer_OOD']:.3f}  "
                f"({r['wall_s']:.1f}s)"
            )

        cs = {
            "rate": rate,
            "per_seed": rows,
            "train_inner_acc":
                _stats([r["train_inner_acc"] for r in rows]),
            "test_random_in_range":
                _stats([r["test_random_in_range"] for r in rows]),
            "mixed_train_acc":
                _stats([r["mixed_train_acc"] for r in rows]),
            "mixed_test_OOD":
                _stats([r["mixed_test_OOD"] for r in rows]),
            "outer_OOD":
                _stats([r["outer_OOD"] for r in rows]),
        }
        summary["by_rate"][f"{rate:.2f}"] = cs
        mt = cs["mixed_test_OOD"]
        ot = cs["outer_OOD"]
        print(
            f"  → mix_test = {mt['mean']:.3f}±{mt['std']:.3f}  "
            f"outer = {ot['mean']:.3f}±{ot['std']:.3f}"
        )

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "═" * 80)
    print("  cross-rate summary (chance ≈ 1/5 = 0.200):")
    print(f"  {'rate':>8s} {'in_range':>16s} {'mixed_train':>16s} "
          f"{'mixed_test_OOD':>20s} {'outer_OOD':>16s}")
    for rate in args.rates:
        cs = summary["by_rate"][f"{rate:.2f}"]
        ir = cs["test_random_in_range"]
        mt = cs["mixed_train_acc"]
        mtest = cs["mixed_test_OOD"]
        ot = cs["outer_OOD"]
        print(
            f"  {rate:>8.2f} "
            f"{ir['mean']:>8.3f}±{ir['std']:.3f}   "
            f"{mt['mean']:>8.3f}±{mt['std']:.3f}   "
            f"{mtest['mean']:>10.3f}±{mtest['std']:.3f}   "
            f"{ot['mean']:>8.3f}±{ot['std']:.3f}"
        )
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
