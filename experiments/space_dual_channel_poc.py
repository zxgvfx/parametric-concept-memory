"""PCM v2 V3 — space mixed_OOD with DualChannelMoveHead.

The §7.5-space ``mixed_OOD = 0.000`` ceiling held against every
single-cell D head we tried in S3 (RowIndex / MaskInfer /
InverseMove). The PCM_V2_DUAL_CHANNEL_DESIGN §V3 hypothesis is
that pair-attention over slot embeddings + attribute-difference
projection breaks this ceiling because:

* attention's ``q=slot_a, k=v=slot_b`` does not couple the joint
  pair to the train-time joint distribution: each test-time pair
  only needs slot_a and slot_b to be **individually** in the
  range of slot embeddings, not their joint;
* the attribute difference ``attr_a − attr_b`` is the v2 of
  vector analogy: under successor-consistency it produces a
  monotone embedding of the row/column index, so direction
  classification falls out of a *linear* projection of the diff
  rather than a learned MLP over a concatenation.

Falsifiability: ``mixed_OOD ≥ 0.30`` on the §7.5-space 5×5/7×7
grid breaks the ceiling. ``mixed_OOD < 0.10`` falsifies V3 and
triggers the abort branch in PCM_V2_DUAL_CHANNEL_DESIGN §7.

Usage::

    python -m experiments.space_dual_channel_poc \\
        --n-seeds 3 --epochs 20 --steps-per-epoch 240 \\
        --out outputs/v2_space_v3
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
    successor_consistency_loss,
)

from experiments.sleep_space_extrapolate import build_splits, cid_of


__all__ = ["main", "DualChannelMoveHead"]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SLOT_DIM = 32
ATTR_DIM = 16
HIDDEN = 64
BASE_FACET = "motion"


# ─────────────────────────────────────────────────────────────────
# DualChannelMoveHead — 5-class direction prediction
# ─────────────────────────────────────────────────────────────────


class DualChannelMoveHead(nn.Module):
    """v2 spatial head — replaces v1 MoveHead's concat-then-MLP fc1
    pipeline. Two parallel paths:

    * **slot path**: pair attention via :func:`pair_attention_logits`
      over (slot_a, slot_b). The attention output is a function of
      (slot_a, slot_b) but is structurally invariant to the joint
      distribution they were trained on, which is the property
      MoveHead.fc1 lacked (and which produced the §7.5-space
      mixed_OOD = 0.000 ceiling).
    * **attribute path**: linear projection of ``attr_a − attr_b``
      to direction logits. Under successor-consistency, this is a
      linear function of the displacement and therefore generalises
      to outer-ring cells without any inner-outer joint coverage.

    The two paths' logits are summed; auxiliary supervision on the
    slot path alone keeps slot_facet learning cell identity in
    parallel to the move task.
    """

    def __init__(
        self,
        slot_dim: int = SLOT_DIM,
        attr_dim: int = ATTR_DIM,
        hidden: int = HIDDEN,
        n_classes: int = 5,
    ) -> None:
        super().__init__()
        # Slot attention path
        self.q_proj = nn.Linear(slot_dim, hidden, bias=False)
        self.k_proj = nn.Linear(slot_dim, hidden, bias=False)
        self.v_proj = nn.Linear(slot_dim, hidden, bias=False)
        self.slot_out = nn.Linear(hidden, n_classes)
        # Attribute difference path
        self.attr_diff = nn.Linear(attr_dim, n_classes)

    def forward(
        self,
        slot_a: torch.Tensor, slot_b: torch.Tensor,
        attr_a: torch.Tensor, attr_b: torch.Tensor,
    ) -> torch.Tensor:
        # Translation-invariant: direction(a, b) is a function of
        # (b - a) only, never of a or b individually. v1 MoveHead's
        # fc1 over a 2*facet_dim concat could memorise pair
        # fingerprints; v2's slot path is therefore *also* fed the
        # difference, not the pair, eliminating the train-acc=1.0
        # but in_range=0.156 over-fitting failure mode observed in
        # the v3-tuned smoke (PCM_V2_DUAL_CHANNEL_DESIGN §V3
        # debug log).
        slot_diff = slot_b - slot_a
        q = self.q_proj(slot_diff)
        k = self.k_proj(slot_diff)
        v = self.v_proj(slot_diff)
        scale = 1.0 / float(q.shape[-1]) ** 0.5
        score = (q * k).sum(dim=-1, keepdim=True) * scale
        attn = torch.tanh(score)
        slot_logits = self.slot_out(attn * v)
        attr_logits = self.attr_diff(attr_a - attr_b)
        return slot_logits + attr_logits


class SlotIdentityHead(nn.Module):
    """Auxiliary head: predict cell flat index from slot row.
    Plays the role of LastDigitHead / RowIndexHead in v1 — drives
    the slot facet to learn cluster identity."""

    def __init__(self, slot_dim: int, n_cells: int, hidden: int = HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(slot_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_cells)

    def forward(self, slot_rows: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(slot_rows)))


# ─────────────────────────────────────────────────────────────────
# Per-seed runner
# ─────────────────────────────────────────────────────────────────


def _train_one_seed(
    seed: int, *,
    n_rows_train: int, n_cols_train: int,
    n_rows_total: int, n_cols_total: int,
    epochs: int, steps_per_epoch: int, ood_ratio: float,
    batch_size: int = 64, lr: float = 5e-3,
    succ_weight: float = 5.0,
    aux_weight: float = 1.0,
) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    splits = build_splits(
        n_rows_train, n_cols_train, n_rows_total, n_cols_total,
        ood_ratio, seed,
    )
    train_pool = splits["train"]
    if not train_pool:
        raise RuntimeError("empty train pool")

    n_total_cells = n_rows_total * n_cols_total
    cg = ConceptGraph(feat_dim=SLOT_DIM)
    for r in range(n_rows_total):
        for c in range(n_cols_total):
            cg.register_concept(
                node_id=cid_of(r, c), label=f"S_{r}_{c}",
                scope="BASE", provenance=f"v2-v3:r={r},c={c}",
            )
    register_dual_channel_facet(
        cg, BASE_FACET, slot_dim=SLOT_DIM, attr_dim=ATTR_DIM,
    )
    all_cids = [
        cid_of(r, c) for r in range(n_rows_total) for c in range(n_cols_total)
    ]

    with torch.no_grad():
        collapse_dual_channel(
            cg, caller="warmup", base_facet=BASE_FACET, concept_ids=all_cids,
            slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
            tick=0, device=DEVICE, attr_init="normal",
        )

    head = DualChannelMoveHead(
        slot_dim=SLOT_DIM, attr_dim=ATTR_DIM, n_classes=5,
    ).to(DEVICE)
    aux = SlotIdentityHead(slot_dim=SLOT_DIM, n_cells=n_total_cells).to(DEVICE)

    params = (
        list(head.parameters())
        + list(aux.parameters())
        + list(cg.iter_bundle_parameters())
    )
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    # Successor-consistency idx pairs across the row index 0..n_rows-1
    # and the column index 0..n_cols-1 — both should produce a monotone
    # attr embedding in their respective principal directions. We supply
    # *both* sets to successor_consistency_loss by stacking pair lists.
    row_pairs: list[tuple[int, int]] = []
    col_pairs: list[tuple[int, int]] = []
    for r in range(n_rows_total):
        for c in range(n_cols_total - 1):
            i = r * n_cols_total + c
            j = r * n_cols_total + c + 1
            col_pairs.append((i, j))
    for c in range(n_cols_total):
        for r in range(n_rows_total - 1):
            i = r * n_cols_total + c
            j = (r + 1) * n_cols_total + c
            row_pairs.append((i, j))

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        head.train(); aux.train()
        for step_i in range(steps_per_epoch):
            batch = [train_pool[rng.randrange(len(train_pool))]
                     for _ in range(batch_size)]
            ids_a = [cid_of(*t[0]) for t in batch]
            ids_b = [cid_of(*t[1]) for t in batch]
            tgt = torch.tensor([t[2] for t in batch], device=DEVICE)

            slot_a, attr_a = collapse_dual_channel(
                cg, caller="v3-move-a", base_facet=BASE_FACET,
                concept_ids=ids_a,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=epoch * 10000 + step_i, device=DEVICE,
            )
            slot_b, attr_b = collapse_dual_channel(
                cg, caller="v3-move-b", base_facet=BASE_FACET,
                concept_ids=ids_b,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=epoch * 10000 + step_i + 1, device=DEVICE,
            )

            logits = head(slot_a, slot_b, attr_a, attr_b)
            loss_move = F.cross_entropy(logits, tgt)

            # Aux: slot identity over the FULL inventory.
            aux_idx = [rng.randrange(n_total_cells) for _ in range(batch_size)]
            aux_cids = [
                cid_of(idx // n_cols_total, idx % n_cols_total)
                for idx in aux_idx
            ]
            aux_slot, _ = collapse_dual_channel(
                cg, caller="v3-aux", base_facet=BASE_FACET,
                concept_ids=aux_cids,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=epoch * 10000 + step_i + 2, device=DEVICE,
            )
            aux_logits = aux(aux_slot)
            aux_tgt = torch.tensor(aux_idx, device=DEVICE)
            loss_aux = F.cross_entropy(aux_logits, aux_tgt)

            # Successor consistency on attr facet (both axes).
            _, attr_full = collapse_dual_channel(
                cg, caller="v3-succ", base_facet=BASE_FACET,
                concept_ids=all_cids,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=epoch * 10000 + step_i + 3, device=DEVICE,
            )
            l_s_row = successor_consistency_loss(attr_full, row_pairs)
            l_s_col = successor_consistency_loss(attr_full, col_pairs)
            loss_succ = l_s_row + l_s_col

            total = (
                loss_move
                + aux_weight * loss_aux
                + succ_weight * loss_succ
            )
            opt.zero_grad(); total.backward(); opt.step()

    head.eval(); aux.eval()

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
            slot_a, attr_a = collapse_dual_channel(
                cg, caller="eval-a", base_facet=BASE_FACET,
                concept_ids=ids_a,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=999000, device=DEVICE,
            )
            slot_b, attr_b = collapse_dual_channel(
                cg, caller="eval-b", base_facet=BASE_FACET,
                concept_ids=ids_b,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=999001, device=DEVICE,
            )
            pred = head(slot_a, slot_b, attr_a, attr_b)
            hits += pred.argmax(-1).eq(tgt).sum().item()
        return hits / len(triples)

    train_acc = _eval(splits["train"])
    test_random = _eval(splits["test_random"])
    test_mixed = _eval(splits["test_mixed_OOD"])
    test_outer = _eval(splits["test_outer_OOD"])

    return {
        "seed": seed,
        "wall_s": time.time() - t0,
        "train_acc": train_acc,
        "test_random_in_range": test_random,
        "test_mixed_OOD": test_mixed,
        "test_outer_OOD": test_outer,
        "n_train": len(splits["train"]),
        "n_mixed": len(splits["test_mixed_OOD"]),
        "n_outer": len(splits["test_outer_OOD"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows-train", type=int, default=5)
    ap.add_argument("--n-cols-train", type=int, default=5)
    ap.add_argument("--n-rows-total", type=int, default=7)
    ap.add_argument("--n-cols-total", type=int, default=7)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=93000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=240)
    ap.add_argument("--ood-ratio", type=float, default=0.15)
    ap.add_argument("--succ-weight", type=float, default=5.0)
    ap.add_argument("--aux-weight", type=float, default=1.0)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/v2_space_v3"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"  PCM v2 V3: train={args.n_rows_train}x{args.n_cols_train}, "
          f"total={args.n_rows_total}x{args.n_cols_total}, "
          f"n_seeds={args.n_seeds}")
    print(f"  device={DEVICE}; succ_w={args.succ_weight}, "
          f"aux_w={args.aux_weight}")
    print("  V3 target: mixed_OOD >= 0.30 (vs v1 baseline 0.000)")
    print("=" * 76)

    rows: list[dict] = []
    for si in range(args.n_seeds):
        seed = args.seed_base + si
        r = _train_one_seed(
            seed,
            n_rows_train=args.n_rows_train, n_cols_train=args.n_cols_train,
            n_rows_total=args.n_rows_total, n_cols_total=args.n_cols_total,
            epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
            ood_ratio=args.ood_ratio,
            succ_weight=args.succ_weight, aux_weight=args.aux_weight,
        )
        rows.append(r)
        print(
            f"  [seed={seed}] train={r['train_acc']:.3f}  "
            f"in_range={r['test_random_in_range']:.3f}  "
            f"mixed_OOD={r['test_mixed_OOD']:.3f}  "
            f"outer_OOD={r['test_outer_OOD']:.3f}  "
            f"({r['wall_s']:.1f}s)"
        )

    def _stats(key: str) -> dict:
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
    print("  PCM v2 V3 summary (chance = 0.20):")
    for k in ("train_acc", "test_random_in_range",
              "test_mixed_OOD", "test_outer_OOD"):
        s = summary[k]
        m = s.get("mean", float("nan"))
        sd = s.get("std", float("nan"))
        print(f"    {k:<24s} {m:+.3f}+-{sd:.3f}")

    mx = summary["test_mixed_OOD"].get("mean", float("nan"))
    if mx >= 0.30:
        print(f"\n  -> V3 PASS: mixed_OOD = {mx:.3f} >= 0.30")
        print("     v2 architecture breaks the §7.5-space ceiling.")
    elif mx >= 0.10:
        print(f"\n  -> V3 PARTIAL: mixed_OOD = {mx:.3f} above v1 = 0.000 "
              "but below 0.30 target.")
    else:
        print(f"\n  -> V3 FAIL: mixed_OOD = {mx:.3f} < 0.10. "
              "v2 attention path does not break the ceiling.")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
