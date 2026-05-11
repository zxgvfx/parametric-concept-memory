"""PCM v3 E4 — sleep-cache distillation (System 2 → System 1).

Builds on `experiments/number_dual_process_poc.py` (E1 / E2 / E3
infrastructure) and adds a sleep phase that distils the cook's
procedural predictions on OOD displacements into the RPE table.

The goal is the falsifiability core of E4 in the design doc:

    Year-1 procedural performance becomes Year-3 conceptual fact.

Operationally, after the standard v3 training (cook saturates `K=99`
while RPE saturates only `|Δ| ≤ 19`), we run a **sleep pass** in
which the cook acts as oracle on OOD pairs, supplying targets for
RPE training. After the sleep pass we re-evaluate and expect:

  * RPE acc on `K=99` rises from ~0 → > 0.5 (procedural → cached).
  * Cook usage under :func:`pcm.dual_process.route_diff` drops as
    the routing threshold can be widened.
  * Cook accuracy itself is unchanged (the sleep pass touches only
    RPE parameters).

These are the four E4 measurements reported by this script.

Usage::

    python -m experiments.number_dual_process_sleep_poc \\
        --n-seeds 3 --epochs 15 --steps-per-epoch 200 \\
        --n-total 100 --train-max 19 \\
        --distill-steps 400 \\
        --out outputs/v3_sleep_cache
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
from pcm.dual_channel import (
    RelativePositionEmbedding,
    collapse_dual_channel,
    register_dual_channel_facet,
)
from pcm.dual_process import (
    IterativeDiffCook,
    SuccessorHead,
    calibrate_rpe_coverage,
    distill_cook_to_rpe,
    route_diff,
)


__all__ = ["main"]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SLOT_DIM = 16
ATTR_DIM = 8
BASE_FACET = "num"


# ─────────────────────────────────────────────────────────────────
# RPE wrapper (same as in number_dual_process_poc, kept local to
# avoid cross-script dependency)
# ─────────────────────────────────────────────────────────────────


class RPEDiffHead(nn.Module):
    def __init__(self, n_total: int, embed_dim: int = 16):
        super().__init__()
        self.n_total = n_total
        n_classes = 2 * n_total - 1
        self.rpe = RelativePositionEmbedding(
            ranges=[(-(n_total - 1), n_total - 1)], embed_dim=embed_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, n_classes),
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.rpe(delta))


def _build_cg(n_total: int) -> tuple[ConceptGraph, list[str]]:
    cg = ConceptGraph(feat_dim=SLOT_DIM)
    cids = []
    for i in range(n_total):
        cid = f"concept:num:{i}"
        cg.register_concept(node_id=cid, label=f"NUM_{i}", scope="BASE",
                            provenance=f"v3-sleep:{i}")
        cids.append(cid)
    register_dual_channel_facet(
        cg, BASE_FACET, slot_dim=SLOT_DIM, attr_dim=ATTR_DIM,
    )
    return cg, cids


# ─────────────────────────────────────────────────────────────────
# Per-seed runner
# ─────────────────────────────────────────────────────────────────


def _run_one(
    seed: int, *,
    n_total: int, train_max: int,
    epochs: int, steps_per_epoch: int,
    distill_steps: int, distill_pairs: int,
    batch_size: int = 64, lr: float = 5e-3,
    max_step: int = 1,
) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    cg, cids = _build_cg(n_total)
    with torch.no_grad():
        collapse_dual_channel(
            cg, caller="warmup", base_facet=BASE_FACET, concept_ids=cids,
            slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
            tick=0, device=DEVICE,
        )

    successor = SuccessorHead(
        slot_dim=SLOT_DIM, attr_dim=ATTR_DIM,
        max_step=max_step, hidden=64,
    ).to(DEVICE)
    rpe_head = RPEDiffHead(n_total=n_total, embed_dim=16).to(DEVICE)

    main_opt = torch.optim.AdamW(
        list(successor.parameters())
        + list(rpe_head.parameters())
        + list(cg.iter_bundle_parameters()),
        lr=lr, weight_decay=1e-4,
    )

    def _lookup(idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cid = cids[idx]
        slot, attr = collapse_dual_channel(
            cg, caller="cook-lookup", base_facet=BASE_FACET,
            concept_ids=[cid],
            slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
            tick=99999, device=DEVICE,
        )
        return slot[0], attr[0]

    # ─── PHASE 1: standard v3 training ───
    n_max = max_step
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        successor.train(); rpe_head.train()
        for step_i in range(steps_per_epoch):
            batch_a, batch_b, batch_step = [], [], []
            for _ in range(batch_size):
                a = rng.randrange(n_total)
                d = rng.randint(-train_max, train_max)
                b = max(0, min(n_total - 1, a + d))
                actual_d = b - a
                if actual_d > 0:
                    s = min(actual_d, n_max)
                elif actual_d < 0:
                    s = max(actual_d, -n_max)
                else:
                    s = 0
                batch_a.append(a); batch_b.append(b); batch_step.append(s)
            ids_a = [cids[a] for a in batch_a]
            ids_b = [cids[b] for b in batch_b]
            slot_a, attr_a = collapse_dual_channel(
                cg, caller="succ-a", base_facet=BASE_FACET,
                concept_ids=ids_a,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=epoch * 10000 + step_i, device=DEVICE,
            )
            slot_b, attr_b = collapse_dual_channel(
                cg, caller="succ-b", base_facet=BASE_FACET,
                concept_ids=ids_b,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=epoch * 10000 + step_i + 1, device=DEVICE,
            )
            tgt = torch.tensor(
                [s + max_step for s in batch_step], device=DEVICE,
            )
            logits = successor(slot_a, slot_b, attr_a, attr_b)
            loss_succ = F.cross_entropy(logits, tgt)

            batch_rpe_a, batch_rpe_b = [], []
            for _ in range(batch_size):
                a = rng.randrange(n_total)
                d = rng.randint(-train_max, train_max)
                b = max(0, min(n_total - 1, a + d))
                batch_rpe_a.append(a); batch_rpe_b.append(b)
            delta = torch.tensor(
                [b - a for a, b in zip(batch_rpe_a, batch_rpe_b)],
                device=DEVICE, dtype=torch.long,
            )
            tgt_rpe = torch.tensor(
                [(b - a) + (n_total - 1)
                 for a, b in zip(batch_rpe_a, batch_rpe_b)],
                device=DEVICE,
            )
            loss_rpe = F.cross_entropy(rpe_head(delta), tgt_rpe)

            loss = loss_succ + loss_rpe
            main_opt.zero_grad(); loss.backward(); main_opt.step()

    successor.eval()
    phase1_wall = time.time() - t0

    # ─── Eval @ end of phase 1 ───
    cook = IterativeDiffCook(
        successor_head=successor, identity_lookup=_lookup,
        max_iters=2 * n_total // max(max_step, 1),
        cursor_min=0, cursor_max=n_total - 1,
        with_attr=True,
    )

    def _rpe_predict(a: int, b: int) -> int:
        with torch.no_grad():
            d = torch.tensor([b - a], device=DEVICE, dtype=torch.long)
            cls = int(rpe_head(d).argmax(-1).item())
        return cls - (n_total - 1)

    def _rpe_step_fn(deltas: torch.Tensor) -> torch.Tensor:
        return rpe_head(deltas)

    eval_Ks = [1, 5, 10, 20, 30, 50, 70, 99]
    eval_Ks = [k for k in eval_Ks if k <= n_total - 1]

    def _eval_at_phase(
        label: str, *, route_threshold: int | None = None,
    ) -> dict:
        """Evaluate RPE / cook accuracy per K, plus routing usage
        with the supplied threshold (defaults to train_max). The
        adaptive case overrides this with the calibrated value
        after phase 2."""
        if route_threshold is None:
            route_threshold = train_max
        rpe_acc, cook_acc = {}, {}
        cook_route_count, rpe_route_count = 0, 0
        for K in eval_Ks:
            valid_starts = [a for a in range(n_total)
                            if 0 <= a + K < n_total]
            sample = valid_starts[:min(40, len(valid_starts))]
            rpe_h, cook_h = 0, 0
            for a in sample:
                b = a + K
                if _rpe_predict(a, b) == K:
                    rpe_h += 1
                diff_c, _ = cook(a, b)
                if diff_c == K:
                    cook_h += 1
                # Routing usage with the (possibly adaptive) threshold.
                _, route = route_diff(
                    a, b,
                    rpe_predict=_rpe_predict, cook=cook,
                    train_max_abs_delta=route_threshold,
                )
                if route == "cook":
                    cook_route_count += 1
                else:
                    rpe_route_count += 1
            rpe_acc[K] = rpe_h / len(sample)
            cook_acc[K] = cook_h / len(sample)
        n_total_route = cook_route_count + rpe_route_count
        return {
            "label": label,
            "route_threshold": route_threshold,
            "rpe_acc_per_K": rpe_acc,
            "cook_acc_per_K": cook_acc,
            "rpe_OOD_K_max": rpe_acc[max(eval_Ks)],
            "rpe_in_range_K20": rpe_acc.get(20, float("nan")),
            "cook_route_fraction": cook_route_count
            / max(n_total_route, 1),
        }

    phase1_eval = _eval_at_phase("phase1")

    # ─── PHASE 2: sleep cache distillation ───
    # Sample pairs across the cook's OOD success range. We bias the
    # distribution to OOD K values (|Δ| > train_max) since the
    # whole point is to expand the RPE's coverage.
    #
    # **Stratified sampling**: each |Δ| in (train_max, n_total) gets
    # a balanced quota. Without this, large K (K=99 has only 1 valid
    # pair vs K=20 with 80) get under-represented in the distillation
    # batches and the RPE never sees those displacements at training
    # time. This was the F53 N=100 failure mode in the un-stratified
    # sample (RPE K=99 acc stayed 0.000 despite 4× more total pairs
    # than the smoke).
    by_abs_K: dict[int, list[tuple[int, int]]] = {}
    for K in range(train_max + 1, n_total):
        bucket = []
        for a in range(n_total):
            b = a + K
            if 0 <= b < n_total:
                bucket.append((a, b))
            b_neg = a - K
            if 0 <= b_neg < n_total:
                bucket.append((a, b_neg))
        if bucket:
            by_abs_K[K] = bucket
    if not by_abs_K:
        sample_pairs = []
    else:
        n_buckets = len(by_abs_K)
        per_bucket = max(1, distill_pairs // n_buckets)
        sample_pairs = []
        for K, bucket in by_abs_K.items():
            rng.shuffle(bucket)
            # Repeat-with-replacement when the bucket is smaller
            # than the per-bucket quota (e.g. K=99 only has 1 pair
            # natively).
            if len(bucket) >= per_bucket:
                sample_pairs.extend(bucket[:per_bucket])
            else:
                sample_pairs.extend(bucket)
                sample_pairs.extend(
                    rng.choices(bucket, k=per_bucket - len(bucket))
                )
        rng.shuffle(sample_pairs)

    # Distillation uses a fresh optimiser (RPE only) so we don't
    # disturb successor / bundle parameters.
    distill_opt = torch.optim.AdamW(rpe_head.parameters(), lr=lr)
    distill_t0 = time.time()

    def _device_aware_step_fn(deltas: torch.Tensor) -> torch.Tensor:
        # Make sure deltas live on the same device as the RPE head.
        if deltas.device != next(rpe_head.parameters()).device:
            deltas = deltas.to(next(rpe_head.parameters()).device)
        return rpe_head(deltas)

    def _delta_to_idx(d: int) -> int:
        return d + (n_total - 1)

    distill_report = distill_cook_to_rpe(
        cook=cook,
        rpe_step_fn=_device_aware_step_fn,
        rpe_parameters=list(rpe_head.parameters()),
        sample_pairs=sample_pairs,
        optimizer=distill_opt,
        n_steps=distill_steps,
        batch_size=batch_size,
        delta_to_idx=_delta_to_idx,
    )
    distill_wall = time.time() - distill_t0

    # ─── Eval @ end of phase 2 (with original threshold) ───
    phase2_eval = _eval_at_phase("phase2")

    # ─── F54: adaptive routing — re-calibrate threshold post-distill ───
    new_threshold = calibrate_rpe_coverage(
        rpe_predict=_rpe_predict, n_total=n_total,
        threshold=0.95, sample_size=30, rng_seed=seed,
    )
    phase2_adaptive = _eval_at_phase(
        "phase2_adaptive", route_threshold=new_threshold,
    )

    return {
        "seed": seed,
        "phase1_wall_s": phase1_wall,
        "phase1": phase1_eval,
        "phase2": phase2_eval,
        "phase2_adaptive": phase2_adaptive,
        "distill": {
            "n_pairs": distill_report.n_pairs_distilled,
            "n_steps": distill_report.n_steps,
            "initial_loss": distill_report.initial_loss,
            "final_loss": distill_report.final_loss,
            "cook_oracle_acc": distill_report.cook_oracle_acc,
            "wall_s": distill_wall,
        },
        "calibrated_threshold": new_threshold,
        "rpe_OOD_uplift": phase2_eval["rpe_OOD_K_max"]
        - phase1_eval["rpe_OOD_K_max"],
        "cook_route_fraction_drop_static": (
            phase1_eval["cook_route_fraction"]
            - phase2_eval["cook_route_fraction"]
        ),
        "cook_route_fraction_drop_adaptive": (
            phase1_eval["cook_route_fraction"]
            - phase2_adaptive["cook_route_fraction"]
        ),
    }


# ─────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-total", type=int, default=100)
    ap.add_argument("--train-max", type=int, default=19)
    ap.add_argument("--max-step", type=int, default=1)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=98000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--distill-steps", type=int, default=400)
    ap.add_argument("--distill-pairs", type=int, default=2000)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/v3_sleep_cache"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"  PCM v3 E4 sleep-cache: N={args.n_total}, "
          f"train_max=|Δ|≤{args.train_max}, "
          f"distill_steps={args.distill_steps}")
    print("=" * 76)

    rows: list[dict] = []
    for si in range(args.n_seeds):
        seed = args.seed_base + si
        r = _run_one(
            seed,
            n_total=args.n_total, train_max=args.train_max,
            epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
            distill_steps=args.distill_steps,
            distill_pairs=args.distill_pairs,
            max_step=args.max_step,
        )
        rows.append(r)
        p1 = r["phase1"]; p2 = r["phase2"]; p2a = r["phase2_adaptive"]
        print(
            f"  [seed={seed}] RPE_K99 p1→p2={p1['rpe_OOD_K_max']:.3f}→"
            f"{p2['rpe_OOD_K_max']:.3f}  "
            f"distill {r['distill']['initial_loss']:.2f}→"
            f"{r['distill']['final_loss']:.2f}  "
            f"cook_route p1→p2(static)={p1['cook_route_fraction']:.2f}→"
            f"{p2['cook_route_fraction']:.2f}  "
            f"adaptive_thr={r['calibrated_threshold']} → "
            f"cook_route={p2a['cook_route_fraction']:.2f}"
        )

    def _stats(key: str, root: list[dict]) -> dict:
        vals = []
        for r in root:
            v = r
            for k in key.split("."):
                v = v.get(k, None) if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, (int, float)) and not (
                isinstance(v, float) and math.isnan(v)
            ):
                vals.append(v)
        if not vals:
            return {"n": 0}
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((x - m) ** 2 for x in vals)
                       / max(len(vals) - 1, 1))
        return {"mean": m, "std": sd, "min": min(vals), "max": max(vals),
                "n": len(vals)}

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "per_seed": rows,
        "phase1_rpe_OOD_K_max": _stats("phase1.rpe_OOD_K_max", rows),
        "phase2_rpe_OOD_K_max": _stats("phase2.rpe_OOD_K_max", rows),
        "phase1_cook_route_fraction": _stats(
            "phase1.cook_route_fraction", rows,
        ),
        "phase2_cook_route_fraction": _stats(
            "phase2.cook_route_fraction", rows,
        ),
        "phase2_adaptive_cook_route_fraction": _stats(
            "phase2_adaptive.cook_route_fraction", rows,
        ),
        "calibrated_threshold": _stats("calibrated_threshold", rows),
        "rpe_OOD_uplift": _stats("rpe_OOD_uplift", rows),
        "cook_route_fraction_drop_static": _stats(
            "cook_route_fraction_drop_static", rows,
        ),
        "cook_route_fraction_drop_adaptive": _stats(
            "cook_route_fraction_drop_adaptive", rows,
        ),
        "distill_initial_loss": _stats("distill.initial_loss", rows),
        "distill_final_loss": _stats("distill.final_loss", rows),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "═" * 76)
    print("  E4 sleep-cache summary (n_seeds={n}):".format(
        n=args.n_seeds,
    ))
    p1m = summary["phase1_rpe_OOD_K_max"]
    p2m = summary["phase2_rpe_OOD_K_max"]
    uplift = summary["rpe_OOD_uplift"]
    drop_s = summary["cook_route_fraction_drop_static"]
    drop_a = summary["cook_route_fraction_drop_adaptive"]
    p1cm = summary["phase1_cook_route_fraction"]
    p2cm = summary["phase2_cook_route_fraction"]
    p2acm = summary["phase2_adaptive_cook_route_fraction"]
    cthr = summary["calibrated_threshold"]
    def _f(d, k):
        return d.get(k, float("nan"))
    print(f"  phase1 RPE K_max OOD acc:        {_f(p1m, 'mean'):+.3f}"
          f"+-{_f(p1m, 'std'):.3f}")
    print(f"  phase2 RPE K_max OOD acc:        {_f(p2m, 'mean'):+.3f}"
          f"+-{_f(p2m, 'std'):.3f}")
    print(f"  RPE OOD uplift (E4):             {_f(uplift, 'mean'):+.3f}"
          f"+-{_f(uplift, 'std'):.3f}")
    print(f"  cook_route_fraction p1:          {_f(p1cm, 'mean'):+.3f}"
          f"+-{_f(p1cm, 'std'):.3f}")
    print(f"  cook_route_fraction p2 (static): {_f(p2cm, 'mean'):+.3f}"
          f"+-{_f(p2cm, 'std'):.3f}")
    print(f"  cook_route drop static (Δ):      {_f(drop_s, 'mean'):+.3f}"
          f"+-{_f(drop_s, 'std'):.3f}")
    print(f"  calibrated threshold (F54):      "
          f"{_f(cthr, 'mean'):.1f}+-{_f(cthr, 'std'):.1f}  "
          f"(was static train_max={args.train_max})")
    print(f"  cook_route_fraction p2 (adaptive): "
          f"{_f(p2acm, 'mean'):+.3f}+-{_f(p2acm, 'std'):.3f}")
    print(f"  cook_route drop adaptive (Δ):    {_f(drop_a, 'mean'):+.3f}"
          f"+-{_f(drop_a, 'std'):.3f}")

    e4_pass = uplift.get("mean", 0.0) >= 0.30
    f54_pass = drop_a.get("mean", 0.0) >= 0.30
    print(
        f"\n  E4 verdict:  "
        f"{'[PASS]' if e4_pass else '[FAIL]'} (target uplift ≥ 0.30)"
    )
    print(
        f"  F54 verdict: "
        f"{'[PASS]' if f54_pass else '[FAIL]'} "
        f"(target adaptive cook_route drop ≥ 0.30)"
    )
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
