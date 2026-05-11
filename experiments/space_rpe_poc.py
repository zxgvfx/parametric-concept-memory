"""V3 follow-up — relative-position-encoding (RPE) ablation.

Motivation: D4 (oracle-attr) diagnostic showed that perfect
absolute-position attr rows did **not** unlock mixed_OOD;
``mixed_OOD`` actually dropped from 0.240 (trained attr) to 0.000
(oracle attr + ReLU head). The displacement ``b - a`` between two
absolute embeddings is geometrically more varied than the
displacement between their one-hot row/col index pair, so a
small head cannot reliably read direction from ``attr_a - attr_b``
when ``attr_a`` and ``attr_b`` lie in different regions of the
absolute-position embedding space.

A more principled fix, suggested by reviewer feedback and
mirroring Shaw et al. 2018 / Su et al. 2021 RoPE / Press et al.
2022 ALiBi, is to give the head a **learned embedding of
(Δr, Δc)** directly:

* The 5×5 inner training set already exhausts every displacement
  ``(Δr, Δc) ∈ {-4..4} × {-4..4}``.
* The 5×5/7×7 mixed_OOD test set's pairs all have the same
  displacement support — ``mixed_OOD = 0.000`` is therefore a
  *coverage* bug at the head level, not at the bundle level.
* An RPE head that classifies direction from a lookup keyed on
  ``(Δr, Δc)`` should saturate mixed_OOD, since the same
  displacement that occurs in training occurs in test.

This script tests that hypothesis directly, in three flavours:

* ``rpe_only``  — direction = classifier(RPE(Δr, Δc)). Head
  ignores slot/attr facets entirely. Upper bound for v2.1.
* ``rpe_plus_attn`` — v2 translation-invariant slot attention
  + RPE classifier (logit sum). Tests whether attention adds
  anything on top of RPE.

Usage::

    python -m experiments.space_rpe_poc --variant rpe_only \\
        --n-seeds 5 --epochs 20 --steps-per-epoch 240 \\
        --out outputs/v3_rpe_only
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
    collapse_dual_channel, register_dual_channel_facet,
)

from experiments.sleep_space_extrapolate import build_splits, cid_of


__all__ = ["main", "RelativePositionMoveHead"]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SLOT_DIM = 32
ATTR_DIM = 16
HIDDEN = 64
BASE_FACET = "motion"


def _parse_cid_to_rc(cid: str) -> tuple[int, int]:
    """Parse ``concept:space:R_C`` → (R, C)."""
    suffix = cid.rsplit(":", 1)[-1]
    r_s, c_s = suffix.split("_")
    return int(r_s), int(c_s)


# ─────────────────────────────────────────────────────────────────
# RPE head variants
# ─────────────────────────────────────────────────────────────────


class RelativePositionMoveHead(nn.Module):
    """Direction prediction from ``(Δr, Δc)`` lookup only.

    Args:
        n_rows, n_cols: grid extent (used to size the lookup table
            to ``(2*(n_rows-1)+1) × (2*(n_cols-1)+1)`` displacement
            slots, covering every plausible inner→inner / inner→
            outer / outer→outer pair).
        embed_dim: per-displacement embedding dim.
        n_classes: 5 for the move task (SAME, UP, DOWN, LEFT, RIGHT).
        slot_dim, hidden: only used when an attention path is
            stacked alongside (variant=``rpe_plus_attn``).
        variant: ``"rpe_only"`` or ``"rpe_plus_attn"``.
    """

    def __init__(
        self,
        n_rows: int, n_cols: int,
        embed_dim: int = 32,
        n_classes: int = 5,
        *,
        slot_dim: int = SLOT_DIM,
        hidden: int = HIDDEN,
        variant: str = "rpe_only",
    ) -> None:
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.max_dr = n_rows - 1
        self.max_dc = n_cols - 1
        self.n_dr = 2 * self.max_dr + 1
        self.n_dc = 2 * self.max_dc + 1
        self.variant = variant
        self.rpe = nn.Embedding(self.n_dr * self.n_dc, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )
        if variant == "rpe_plus_attn":
            self.q_proj = nn.Linear(slot_dim, hidden, bias=False)
            self.k_proj = nn.Linear(slot_dim, hidden, bias=False)
            self.v_proj = nn.Linear(slot_dim, hidden, bias=False)
            self.slot_out = nn.Linear(hidden, n_classes)

    def _rpe_lookup(
        self, dr: torch.Tensor, dc: torch.Tensor,
    ) -> torch.Tensor:
        idx = (dr + self.max_dr) * self.n_dc + (dc + self.max_dc)
        return self.rpe(idx)

    def forward(
        self,
        dr: torch.Tensor, dc: torch.Tensor,
        slot_a: torch.Tensor | None = None,
        slot_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rpe = self._rpe_lookup(dr, dc)
        rpe_logits = self.classifier(rpe)
        if self.variant == "rpe_only":
            return rpe_logits
        # rpe_plus_attn
        slot_diff = slot_b - slot_a
        q = self.q_proj(slot_diff)
        k = self.k_proj(slot_diff)
        v = self.v_proj(slot_diff)
        scale = 1.0 / float(q.shape[-1]) ** 0.5
        score = (q * k).sum(dim=-1, keepdim=True) * scale
        attn = torch.tanh(score)
        slot_logits = self.slot_out(attn * v)
        return rpe_logits + slot_logits


# ─────────────────────────────────────────────────────────────────
# Per-seed runner
# ─────────────────────────────────────────────────────────────────


def _run_one(
    seed: int, *,
    n_rows_train: int, n_cols_train: int,
    n_rows_total: int, n_cols_total: int,
    epochs: int, steps_per_epoch: int, ood_ratio: float,
    variant: str,
    batch_size: int = 64, lr: float = 5e-3,
) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    splits = build_splits(
        n_rows_train, n_cols_train, n_rows_total, n_cols_total,
        ood_ratio, seed,
    )
    train_pool = splits["train"]

    cg = ConceptGraph(feat_dim=SLOT_DIM)
    for r in range(n_rows_total):
        for c in range(n_cols_total):
            cg.register_concept(
                node_id=cid_of(r, c), label=f"S_{r}_{c}",
                scope="BASE", provenance="rpe-poc",
            )
    if variant == "rpe_plus_attn":
        register_dual_channel_facet(
            cg, BASE_FACET, slot_dim=SLOT_DIM, attr_dim=ATTR_DIM,
        )
        all_cids = [
            cid_of(r, c) for r in range(n_rows_total) for c in range(n_cols_total)
        ]
        with torch.no_grad():
            collapse_dual_channel(
                cg, caller="warmup", base_facet=BASE_FACET,
                concept_ids=all_cids,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=0, device=DEVICE, attr_init="normal",
            )

    head = RelativePositionMoveHead(
        n_rows=n_rows_total, n_cols=n_cols_total,
        embed_dim=32, n_classes=5,
        slot_dim=SLOT_DIM, hidden=HIDDEN,
        variant=variant,
    ).to(DEVICE)

    params = list(head.parameters())
    if variant == "rpe_plus_attn":
        params += list(cg.iter_bundle_parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    def _displacement(batch):
        dr = []
        dc = []
        for t in batch:
            ra, ca = t[0]
            rb, cb = t[1]
            dr.append(rb - ra)
            dc.append(cb - ca)
        return (
            torch.tensor(dr, device=DEVICE, dtype=torch.long),
            torch.tensor(dc, device=DEVICE, dtype=torch.long),
        )

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        head.train()
        for step_i in range(steps_per_epoch):
            batch = [train_pool[rng.randrange(len(train_pool))]
                     for _ in range(batch_size)]
            dr, dc = _displacement(batch)
            tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
            if variant == "rpe_only":
                logits = head(dr, dc)
            else:
                ids_a = [cid_of(*t[0]) for t in batch]
                ids_b = [cid_of(*t[1]) for t in batch]
                slot_a, _ = collapse_dual_channel(
                    cg, caller="rpe-a", base_facet=BASE_FACET,
                    concept_ids=ids_a,
                    slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                    tick=epoch * 10000 + step_i, device=DEVICE,
                )
                slot_b, _ = collapse_dual_channel(
                    cg, caller="rpe-b", base_facet=BASE_FACET,
                    concept_ids=ids_b,
                    slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                    tick=epoch * 10000 + step_i + 1, device=DEVICE,
                )
                logits = head(dr, dc, slot_a, slot_b)
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()

    head.eval()

    @torch.no_grad()
    def _eval(triples):
        if not triples:
            return float("nan")
        hits = 0
        for i in range(0, len(triples), 64):
            batch = triples[i:i + 64]
            dr, dc = _displacement(batch)
            tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
            if variant == "rpe_only":
                logits = head(dr, dc)
            else:
                ids_a = [cid_of(*t[0]) for t in batch]
                ids_b = [cid_of(*t[1]) for t in batch]
                slot_a, _ = collapse_dual_channel(
                    cg, caller="ev-a", base_facet=BASE_FACET,
                    concept_ids=ids_a,
                    slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                    tick=999, device=DEVICE,
                )
                slot_b, _ = collapse_dual_channel(
                    cg, caller="ev-b", base_facet=BASE_FACET,
                    concept_ids=ids_b,
                    slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                    tick=999, device=DEVICE,
                )
                logits = head(dr, dc, slot_a, slot_b)
            hits += logits.argmax(-1).eq(tgt).sum().item()
        return hits / len(triples)

    return {
        "seed": seed,
        "wall_s": time.time() - t0,
        "variant": variant,
        "train_acc": _eval(splits["train"]),
        "test_random_in_range": _eval(splits["test_random"]),
        "test_mixed_OOD": _eval(splits["test_mixed_OOD"]),
        "test_outer_OOD": _eval(splits["test_outer_OOD"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant", choices=["rpe_only", "rpe_plus_attn"],
        default="rpe_only",
    )
    ap.add_argument("--n-rows-train", type=int, default=5)
    ap.add_argument("--n-cols-train", type=int, default=5)
    ap.add_argument("--n-rows-total", type=int, default=7)
    ap.add_argument("--n-cols-total", type=int, default=7)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=95000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=240)
    ap.add_argument("--ood-ratio", type=float, default=0.15)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/v3_rpe"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"  V3 RPE poc: variant={args.variant}, "
          f"train={args.n_rows_train}x{args.n_cols_train}, "
          f"total={args.n_rows_total}x{args.n_cols_total}, "
          f"n_seeds={args.n_seeds}")
    print("=" * 76)

    rows: list[dict] = []
    for si in range(args.n_seeds):
        seed = args.seed_base + si
        r = _run_one(
            seed,
            n_rows_train=args.n_rows_train,
            n_cols_train=args.n_cols_train,
            n_rows_total=args.n_rows_total,
            n_cols_total=args.n_cols_total,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            ood_ratio=args.ood_ratio,
            variant=args.variant,
        )
        rows.append(r)
        print(
            f"  [seed={seed}] train={r['train_acc']:.3f}  "
            f"in_range={r['test_random_in_range']:.3f}  "
            f"mixed_OOD={r['test_mixed_OOD']:.3f}  "
            f"outer_OOD={r['test_outer_OOD']:.3f}  "
            f"({r['wall_s']:.1f}s)"
        )

    def _stats(key):
        vals = [r[key] for r in rows
                if isinstance(r.get(key), (int, float))
                and not (isinstance(r.get(key), float) and math.isnan(r[key]))]
        if not vals:
            return {"n": 0}
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1))
        return {"mean": m, "std": sd, "min": min(vals), "max": max(vals),
                "n": len(vals)}

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "per_seed": rows,
        "train_acc": _stats("train_acc"),
        "test_random_in_range": _stats("test_random_in_range"),
        "test_mixed_OOD": _stats("test_mixed_OOD"),
        "test_outer_OOD": _stats("test_outer_OOD"),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print(f"  RPE-{args.variant} summary (chance = 0.20):")
    for k in ("train_acc", "test_random_in_range",
              "test_mixed_OOD", "test_outer_OOD"):
        s = summary[k]
        m = s.get("mean", float("nan"))
        sd = s.get("std", float("nan"))
        print(f"    {k:<24s} {m:+.3f}+-{sd:.3f}")

    mx = summary["test_mixed_OOD"].get("mean", float("nan"))
    if mx >= 0.80:
        print(f"\n  -> mixed_OOD = {mx:.3f} >= 0.80 — RPE saturates the "
              "ceiling. Architectural lever confirmed.")
    elif mx >= 0.30:
        print(f"\n  -> mixed_OOD = {mx:.3f} >= 0.30 — RPE clears the "
              "design-doc V3 target.")
    elif mx >= 0.10:
        print(f"\n  -> mixed_OOD = {mx:.3f} — RPE partial; below "
              "0.30 strict.")
    else:
        print(f"\n  -> mixed_OOD = {mx:.3f} — RPE alone fails. "
              "Bottleneck is elsewhere (training data, optimizer, etc).")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
