"""V3 boundary diagnostics — what is holding mixed_OOD at 0.240?

Three diagnostic experiments, each isolating a different
suspected cause:

* **D1 (attr-PCA)**: after standard training, run PCA on the
  learned attr_table and check whether PC1, PC2 correspond to
  row, column. Reports NMI(PC1 quartile, row_id) and
  NMI(PC2 quartile, col_id). Tells us whether the attr facet is
  actually learning a 2-D linear embedding under the
  succ_consistency loss applied along both axes.

* **D2 (asymmetric mixed)**: split mixed_OOD into
  (inner, outer) and (outer, inner) sub-buckets and report
  separately. Asymmetric numbers reveal a slot/attr symmetry-
  breaking that current head architecture can't undo.

* **D4 (oracle-attr)**: write a perfect attr table
  ``attr_i = onehot(row_i) ⊕ onehot(col_i)`` directly into the
  pool, freeze the attr facet, and train the head + slot facet
  + aux only. Reports mixed_OOD under perfect attr. This pins
  down the **architectural ceiling** of v2 with current head:
  if oracle attr unlocks mixed_OOD >> 0.30, the bottleneck is
  attr-loss design (we know how to fix that); if oracle attr
  only gets us to ~0.40, the head is the next bottleneck.

Usage::

    python -m experiments.space_v3_diagnostics --diag d4 --n-seeds 3
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
import torch.nn as nn
import torch.nn.functional as F

from pcm.concept_graph import ConceptGraph
from pcm.dual_channel import (
    collapse_dual_channel, register_dual_channel_facet,
    successor_consistency_loss,
)

from experiments.sleep_space_extrapolate import build_splits, cid_of
from experiments.space_dual_channel_poc import (
    ATTR_DIM, BASE_FACET, DEVICE, DualChannelMoveHead,
    HIDDEN, SLOT_DIM, SlotIdentityHead,
)


# ─────────────────────────────────────────────────────────────────
# D1 — PCA of trained attr_table
# ─────────────────────────────────────────────────────────────────


def _nmi(a: list[int], b: list[int]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    n = len(a)
    ca = Counter(a); cb = Counter(b); cab = Counter(zip(a, b))

    def _h(c: Counter) -> float:
        h = 0.0
        for v in c.values():
            p = v / n
            if p > 0:
                h -= p * math.log2(p)
        return h
    h_a, h_b = _h(ca), _h(cb)
    if h_a == 0 or h_b == 0:
        return 0.0
    h_ab = _h(cab)
    return (h_a + h_b - h_ab) / math.sqrt(h_a * h_b)


def attr_pca_diagnostics(
    attr_table: torch.Tensor, n_rows: int, n_cols: int,
) -> dict[str, float]:
    """Compute NMI between top-2 PCA components of attr_table and
    (row_id, col_id) labels.

    A v2 attr facet that has correctly learned a 2-D ordinal
    embedding should give NMI(PC1, row_or_col) > 0.5 for one of
    the axes and NMI(PC2, the_other) > 0.5 for the other.
    """
    table = attr_table.detach().cpu().float()
    centred = table - table.mean(dim=0, keepdim=True)
    u, s, v = torch.svd(centred)
    pc = (u * s).detach()  # (N, K)
    pc1 = pc[:, 0].tolist()
    pc2 = pc[:, 1].tolist()

    # Bucket into n_rows / n_cols quantiles
    def _bucket(vals: list[float], n_buckets: int) -> list[int]:
        sorted_vals = sorted(enumerate(vals), key=lambda x: x[1])
        out = [0] * len(vals)
        for rank, (idx, _) in enumerate(sorted_vals):
            out[idx] = min(rank * n_buckets // len(vals), n_buckets - 1)
        return out

    rows = [i // n_cols for i in range(n_rows * n_cols)]
    cols = [i % n_cols for i in range(n_rows * n_cols)]
    pc1_b = _bucket(pc1, n_rows)
    pc2_b = _bucket(pc2, n_cols)
    pc1_c = _bucket(pc1, n_cols)
    pc2_r = _bucket(pc2, n_rows)
    return {
        "pc1_var_explained": float((s[0] ** 2 / (s ** 2).sum()).item()),
        "pc2_var_explained": float((s[1] ** 2 / (s ** 2).sum()).item()),
        "nmi_pc1_row": _nmi(pc1_b, rows),
        "nmi_pc1_col": _nmi(pc1_c, cols),
        "nmi_pc2_row": _nmi(pc2_r, rows),
        "nmi_pc2_col": _nmi(pc2_b, cols),
    }


# ─────────────────────────────────────────────────────────────────
# D2 — asymmetric mixed_OOD split
# ─────────────────────────────────────────────────────────────────


def _split_mixed_by_direction(
    mixed_triples, n_rows_train: int, n_cols_train: int,
):
    inner_outer = []
    outer_inner = []
    for t in mixed_triples:
        a_in = t[0][0] < n_rows_train and t[0][1] < n_cols_train
        b_in = t[1][0] < n_rows_train and t[1][1] < n_cols_train
        if a_in and not b_in:
            inner_outer.append(t)
        elif b_in and not a_in:
            outer_inner.append(t)
    return inner_outer, outer_inner


# ─────────────────────────────────────────────────────────────────
# D4 — oracle attr injection
# ─────────────────────────────────────────────────────────────────


def _make_oracle_attr_table(
    n_rows: int, n_cols: int, attr_dim: int, *, scale: float = 5.0,
) -> torch.Tensor:
    """Construct attr_i = scale * (one_hot_row(r_i) ⊕ one_hot_col(c_i))
    in the first ``n_rows + n_cols`` dims; rest zero. This is a
    perfect 2-D ordinal embedding — vector analogy on it gives
    100 % top-1 inside any axis-aligned subset.
    """
    if attr_dim < n_rows + n_cols:
        raise ValueError(
            f"attr_dim ({attr_dim}) too small for oracle "
            f"(needs >= n_rows+n_cols = {n_rows + n_cols})"
        )
    table = torch.zeros(n_rows * n_cols, attr_dim)
    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c
            table[i, r] = scale
            table[i, n_rows + c] = scale
    return table


def _run_d4(
    seed: int, *,
    n_rows_train: int, n_cols_train: int,
    n_rows_total: int, n_cols_total: int,
    epochs: int, steps_per_epoch: int, ood_ratio: float,
    batch_size: int = 64, lr: float = 5e-3,
    aux_weight: float = 1.0,
) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    splits = build_splits(
        n_rows_train, n_cols_train, n_rows_total, n_cols_total,
        ood_ratio, seed,
    )
    train_pool = splits["train"]

    n_total_cells = n_rows_total * n_cols_total
    cg = ConceptGraph(feat_dim=SLOT_DIM)
    for r in range(n_rows_total):
        for c in range(n_cols_total):
            cg.register_concept(
                node_id=cid_of(r, c), label=f"S_{r}_{c}",
                scope="BASE", provenance="d4",
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

    # Inject oracle attr table.
    oracle = _make_oracle_attr_table(
        n_rows_total, n_cols_total, ATTR_DIM,
    ).to(DEVICE)
    attr_pool = cg.bundle_pool[f"{BASE_FACET}_attr"]
    with torch.no_grad():
        for i, cid in enumerate(all_cids):
            slot = cg.cid_to_slot[cid]
            attr_pool.data[slot] = oracle[i]
    attr_pool.requires_grad_(False)

    head = DualChannelMoveHead(
        slot_dim=SLOT_DIM, attr_dim=ATTR_DIM, n_classes=5,
    ).to(DEVICE)
    aux = SlotIdentityHead(slot_dim=SLOT_DIM, n_cells=n_total_cells).to(DEVICE)

    # Train only head + aux + slot pool.
    slot_pool = cg.bundle_pool[f"{BASE_FACET}_slot"]
    params = (
        list(head.parameters())
        + list(aux.parameters())
        + [slot_pool]
    )
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

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
                cg, caller="d4-a", base_facet=BASE_FACET, concept_ids=ids_a,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=epoch * 10000 + step_i, device=DEVICE,
            )
            slot_b, attr_b = collapse_dual_channel(
                cg, caller="d4-b", base_facet=BASE_FACET, concept_ids=ids_b,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=epoch * 10000 + step_i + 1, device=DEVICE,
            )
            logits = head(slot_a, slot_b, attr_a, attr_b)
            loss_move = F.cross_entropy(logits, tgt)
            aux_idx = [rng.randrange(n_total_cells) for _ in range(batch_size)]
            aux_cids = [
                cid_of(idx // n_cols_total, idx % n_cols_total)
                for idx in aux_idx
            ]
            aux_slot, _ = collapse_dual_channel(
                cg, caller="d4-aux", base_facet=BASE_FACET,
                concept_ids=aux_cids,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=epoch * 10000 + step_i + 2, device=DEVICE,
            )
            aux_logits = aux(aux_slot)
            aux_tgt = torch.tensor(aux_idx, device=DEVICE)
            loss_aux = F.cross_entropy(aux_logits, aux_tgt)
            total = loss_move + aux_weight * loss_aux
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
            sa, aa = collapse_dual_channel(
                cg, caller="d4-eva", base_facet=BASE_FACET,
                concept_ids=ids_a,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=999, device=DEVICE,
            )
            sb, ab = collapse_dual_channel(
                cg, caller="d4-evb", base_facet=BASE_FACET,
                concept_ids=ids_b,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=999, device=DEVICE,
            )
            pred = head(sa, sb, aa, ab)
            hits += pred.argmax(-1).eq(tgt).sum().item()
        return hits / len(triples)

    inner_outer, outer_inner = _split_mixed_by_direction(
        splits["test_mixed_OOD"], n_rows_train, n_cols_train,
    )
    return {
        "seed": seed,
        "wall_s": time.time() - t0,
        "train_acc": _eval(splits["train"]),
        "test_random_in_range": _eval(splits["test_random"]),
        "test_mixed_OOD": _eval(splits["test_mixed_OOD"]),
        "test_outer_OOD": _eval(splits["test_outer_OOD"]),
        "mixed_inner_outer": _eval(inner_outer),
        "mixed_outer_inner": _eval(outer_inner),
    }


# ─────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", choices=["d4"], default="d4")
    ap.add_argument("--n-rows-train", type=int, default=5)
    ap.add_argument("--n-cols-train", type=int, default=5)
    ap.add_argument("--n-rows-total", type=int, default=7)
    ap.add_argument("--n-cols-total", type=int, default=7)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=94000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=240)
    ap.add_argument("--ood-ratio", type=float, default=0.15)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/v3_diag"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(
        f"  V3 diagnostic {args.diag.upper()}: "
        f"train={args.n_rows_train}x{args.n_cols_train}, "
        f"total={args.n_rows_total}x{args.n_cols_total}, "
        f"n_seeds={args.n_seeds}"
    )
    print("=" * 76)

    rows: list[dict] = []
    for si in range(args.n_seeds):
        seed = args.seed_base + si
        if args.diag == "d4":
            r = _run_d4(
                seed,
                n_rows_train=args.n_rows_train,
                n_cols_train=args.n_cols_train,
                n_rows_total=args.n_rows_total,
                n_cols_total=args.n_cols_total,
                epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch,
                ood_ratio=args.ood_ratio,
            )
        else:
            raise ValueError(args.diag)
        rows.append(r)
        print(
            f"  [seed={seed}] train={r['train_acc']:.3f}  "
            f"in_range={r['test_random_in_range']:.3f}  "
            f"mixed={r['test_mixed_OOD']:.3f}  "
            f"(I->O={r['mixed_inner_outer']:.3f}, "
            f"O->I={r['mixed_outer_inner']:.3f})  "
            f"outer={r['test_outer_OOD']:.3f}"
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
        "mixed_inner_outer": _stats("mixed_inner_outer"),
        "mixed_outer_inner": _stats("mixed_outer_inner"),
        "test_outer_OOD": _stats("test_outer_OOD"),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print(f"  D{args.diag[1:]} summary:")
    for k in ("train_acc", "test_random_in_range",
              "test_mixed_OOD", "mixed_inner_outer", "mixed_outer_inner",
              "test_outer_OOD"):
        s = summary[k]
        m = s.get("mean", float("nan"))
        sd = s.get("std", float("nan"))
        print(f"    {k:<24s} {m:+.3f}+-{sd:.3f}")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
