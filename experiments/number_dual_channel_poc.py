"""PCM v2 MVP — number N=10 dual-channel proof-of-concept.

See ``docs/PCM_V2_DUAL_CHANNEL_DESIGN.md`` §7 for the contract this
script tests:

* **V1 (slot purity)** — Tier-G k-means on the *slot* facet recovers
  last-digit identity at NMI ≥ 0.85.
* **V2 (vector analogy)** — `attr_a + attr_c − attr_b` nearest
  neighbour matches `d` at top-1 ≥ 0.30 (vs S6 baseline 0.079).

V3 (mixed_OOD) is tested in `experiments/space_dual_channel_poc.py`.

Usage::

    python -m experiments.number_dual_channel_poc \\
        --n-seeds 5 --epochs 30 --steps-per-epoch 200 \\
        --out outputs/v2_number_poc
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
    arithmetic_consistency_loss, collapse_dual_channel,
    info_nce_loss, register_dual_channel_facet, spread_regularizer,
    successor_consistency_loss,
)
from pcm.sleep import (
    SleepConfig, attach_sleep, run_sleep_pass,
)


__all__ = ["main", "v1_slot_purity_nmi", "v2_vector_analogy_top1"]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 10
SLOT_DIM = 16
ATTR_DIM = 16
BASE_FACET = "arith"
HIDDEN = 64


# ─────────────────────────────────────────────────────────────────
# Tiny model: LastDigitHead reads slot facet only.
# ─────────────────────────────────────────────────────────────────


class SlotLastDigitHead(nn.Module):
    """Reads the slot facet, predicts last-digit class.

    The whole point of the v2 split is that this head does *not* see
    the attr facet — slot is for identity, attr is for arithmetic.
    """

    def __init__(self, slot_dim: int = SLOT_DIM, n_classes: int = N,
                 hidden: int = HIDDEN) -> None:
        super().__init__()
        self.fc1 = nn.Linear(slot_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, slot_rows: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(slot_rows))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


# ─────────────────────────────────────────────────────────────────
# V1 / V2 evaluation
# ─────────────────────────────────────────────────────────────────


def _nmi(labels_a: list[int], labels_b: list[int]) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0
    n = len(labels_a)
    ca = Counter(labels_a)
    cb = Counter(labels_b)
    cab = Counter(zip(labels_a, labels_b))

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
    mi = h_a + h_b - h_ab
    return mi / math.sqrt(h_a * h_b)


def v1_slot_purity_nmi(
    slot_rows: torch.Tensor, n_total: int, *, seed: int,
    k: int | None = None,
) -> dict[str, float]:
    """Cluster slot rows with k-means, compute NMI vs last_digit."""
    from pcm.sleep import _kmeans
    if k is None:
        k = min(n_total, 10)
    cfg = SleepConfig(k_clusters=k, kmeans_iters=64, distance="cosine",
                      init="kmeans++", seed=seed)
    centroids, assignments = _kmeans(slot_rows, k, cfg=cfg)
    labels = [int(a) for a in assignments.tolist()]
    last_digit = [n % 10 for n in range(n_total)]
    return {
        "v1_nmi": _nmi(labels, last_digit),
        "v1_k": k,
        "v1_unique_clusters": len(set(labels)),
    }


def v2_vector_analogy_top1(
    attr_rows: torch.Tensor, n_total: int, *,
    max_triples: int = 4000, seed: int = 0,
) -> dict[str, float]:
    """Sample (a, b, c, d=a+c-b) quadruples with d in range and d != c,
    measure how often `attr_a + attr_c - attr_b` has `attr_d` as
    nearest neighbour."""
    rng = random.Random(seed)
    table = attr_rows.detach().cpu().float()
    triples: list[tuple[int, int, int, int]] = []
    for a in range(n_total):
        for b in range(n_total):
            if a == b:
                continue
            for c in range(n_total):
                d = a + c - b
                if 0 <= d < n_total and d != c:
                    triples.append((a, b, c, d))
    rng.shuffle(triples)
    triples = triples[:max_triples]
    n = len(triples)
    if n == 0:
        return {"v2_top1": float("nan"), "v2_top3": float("nan"), "n": 0}
    hits1 = 0
    hits3 = 0
    for a, b, c, d in triples:
        query = table[a] + table[c] - table[b]
        dist = (table - query.unsqueeze(0)).pow(2).sum(dim=-1)
        topk = dist.topk(3, largest=False).indices.tolist()
        if topk[0] == d:
            hits1 += 1
        if d in topk:
            hits3 += 1
    return {
        "v2_top1": hits1 / n, "v2_top3": hits3 / n,
        "v2_n_triples": n, "v2_chance": 1.0 / n_total,
    }


# ─────────────────────────────────────────────────────────────────
# Single-seed training
# ─────────────────────────────────────────────────────────────────


def _enumerate_quadruples(n_total: int) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for a in range(n_total):
        for b in range(n_total):
            if a == b:
                continue
            for c in range(n_total):
                d = a + c - b
                if 0 <= d < n_total and d != c:
                    out.append((a, b, c, d))
    return out


def _train_one_seed(
    seed: int, *,
    epochs: int, steps_per_epoch: int,
    batch_size: int = 64,
    lr: float = 5e-3,
    info_nce_weight: float = 0.0,
    arith_weight: float = 1.0,
    spread_weight: float = 0.0,
    succ_weight: float = 5.0,
    normalize_attr: bool = False,
) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    cg = ConceptGraph(feat_dim=SLOT_DIM)
    cids = []
    for i in range(N):
        cid = f"concept:num:{i}"
        cg.register_concept(node_id=cid, label=f"NUM_{i}", scope="BASE",
                            provenance=f"v2-poc:n={i}")
        cids.append(cid)
    register_dual_channel_facet(
        cg, BASE_FACET, slot_dim=SLOT_DIM, attr_dim=ATTR_DIM,
    )

    head = SlotLastDigitHead(slot_dim=SLOT_DIM, n_classes=N).to(DEVICE)

    # Lazy facet allocation by collapsing once per facet.
    # Use init="normal" (unit variance) on the attr facet so the v2
    # MVP doesn't immediately collapse to attr ≈ 0 (the diagnostic
    # finding that motivated normalize_attr + spread_regularizer).
    with torch.no_grad():
        s, a = collapse_dual_channel(
            cg, caller="warmup", base_facet=BASE_FACET, concept_ids=cids,
            slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
            tick=0, device=DEVICE, init="normal_small",
            attr_init="normal",
        )
    cg.bundles_to(torch.device(DEVICE))

    quadruples = _enumerate_quadruples(N)

    params = list(head.parameters()) + list(cg.iter_bundle_parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    t0 = time.time()
    last_loss = {"sup": 0.0, "info": 0.0, "arith": 0.0}
    for epoch in range(1, epochs + 1):
        head.train()
        for step_i in range(steps_per_epoch):
            # Sample a batch of (cid, last_digit) pairs.
            batch_idx = [rng.randrange(N) for _ in range(batch_size)]
            batch_cids = [cids[i] for i in batch_idx]
            tgt = torch.tensor(batch_idx, device=DEVICE)

            slot_rows, attr_rows = collapse_dual_channel(
                cg, caller="v2_poc", base_facet=BASE_FACET,
                concept_ids=batch_cids,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=epoch * 10000 + step_i, device=DEVICE,
                normalize_attr=normalize_attr,
            )

            logits = head(slot_rows)
            loss_sup = F.cross_entropy(logits, tgt)

            # InfoNCE on attr; positives = same true digit.
            loss_info = info_nce_loss(
                attr_rows, slot_labels=tgt, temperature=0.5,
            )

            # Arithmetic consistency on the WHOLE table this step
            # (collapse all 10 concepts into a fresh attr table).
            _, attr_table = collapse_dual_channel(
                cg, caller="v2_arith", base_facet=BASE_FACET,
                concept_ids=cids,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=epoch * 10000 + step_i + 1, device=DEVICE,
                normalize_attr=normalize_attr,
            )
            qb = min(len(quadruples), 64)
            quad_sample = [
                quadruples[rng.randrange(len(quadruples))] for _ in range(qb)
            ]
            ia = torch.tensor([q[0] for q in quad_sample], device=DEVICE)
            ib = torch.tensor([q[1] for q in quad_sample], device=DEVICE)
            ic = torch.tensor([q[2] for q in quad_sample], device=DEVICE)
            id_ = torch.tensor([q[3] for q in quad_sample], device=DEVICE)
            loss_arith = arithmetic_consistency_loss(
                attr_table, ia, ib, ic, id_,
            )
            loss_spread = spread_regularizer(
                attr_table, target_min_dist=0.5,
            )
            loss_succ = successor_consistency_loss(attr_table)

            total = (
                loss_sup
                + info_nce_weight * loss_info
                + arith_weight * loss_arith
                + spread_weight * loss_spread
                + succ_weight * loss_succ
            )
            opt.zero_grad()
            total.backward()
            opt.step()

            last_loss = {
                "sup": float(loss_sup.item()),
                "info": float(loss_info.item()),
                "arith": float(loss_arith.item()),
                "spread": float(loss_spread.item()),
                "succ": float(loss_succ.item()),
            }

    head.eval()

    # Build final tables on CPU for evaluation.
    with torch.no_grad():
        slot_rows, attr_rows = collapse_dual_channel(
            cg, caller="v2_eval", base_facet=BASE_FACET, concept_ids=cids,
            slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
            tick=99999, device=DEVICE,
            normalize_attr=normalize_attr,
        )
    slot_table = slot_rows.detach().cpu()
    attr_table = attr_rows.detach().cpu()

    # Train-acc on last-digit classification.
    with torch.no_grad():
        logits = head(slot_rows)
        train_acc = float(logits.argmax(-1).eq(
            torch.arange(N, device=DEVICE)
        ).float().mean().item())

    # V1 / V2 invariants.
    v1 = v1_slot_purity_nmi(slot_table, N, seed=seed)
    v2 = v2_vector_analogy_top1(attr_table, N, seed=seed)

    # Sleep pass: confirm Tier-G also reaches similar V1.
    attach_sleep(cg, facets=[
        f"{BASE_FACET}_slot",  # cluster slot only
    ])
    sleep_report = run_sleep_pass(
        cg, facets=[f"{BASE_FACET}_slot"],
        config=SleepConfig(k_clusters=N, replay_steps=0, seed=seed),
        tick=200000,
    )
    sleep_facets = [f.facet for f in sleep_report.facets]

    return {
        "seed": seed,
        "wall_s": time.time() - t0,
        "train_acc": train_acc,
        **v1,
        **v2,
        "loss_sup": last_loss["sup"],
        "loss_info": last_loss["info"],
        "loss_arith": last_loss["arith"],
        "sleep_facets_clustered": sleep_facets,
    }


# ─────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=92000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--info-nce-weight", type=float, default=0.0)
    ap.add_argument("--arith-weight", type=float, default=1.0)
    ap.add_argument("--spread-weight", type=float, default=0.0)
    ap.add_argument("--succ-weight", type=float, default=5.0)
    ap.add_argument("--normalize-attr", action="store_true",
                    help="enable F.normalize on attr (default: disabled "
                         "since normalize destroys vector arithmetic; "
                         "see _v2_diag.py sanity check)")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/v2_number_poc"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"  PCM v2 MVP (number N={N}): n_seeds={args.n_seeds}, "
          f"epochs={args.epochs}, steps={args.steps_per_epoch}")
    print(f"  device={DEVICE}, info_nce_w={args.info_nce_weight}, "
          f"arith_w={args.arith_weight}")
    print("  V1 target NMI ≥ 0.85 / V2 target top1 ≥ 0.30 "
          f"(chance {1.0 / N:.3f})")
    print("=" * 76)

    rows: list[dict] = []
    for si in range(args.n_seeds):
        seed = args.seed_base + si
        r = _train_one_seed(
            seed,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            info_nce_weight=args.info_nce_weight,
            arith_weight=args.arith_weight,
            spread_weight=args.spread_weight,
            succ_weight=args.succ_weight,
            normalize_attr=args.normalize_attr,
        )
        rows.append(r)
        print(
            f"  [seed={seed}] train_acc={r['train_acc']:.3f}  "
            f"V1_NMI={r['v1_nmi']:+.3f}  "
            f"V2_top1={r['v2_top1']:+.3f}  V2_top3={r['v2_top3']:+.3f}  "
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
        "v1_nmi": _stats("v1_nmi"),
        "v2_top1": _stats("v2_top1"),
        "v2_top3": _stats("v2_top3"),
        "train_acc": _stats("train_acc"),
    }

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "═" * 76)
    print("  PCM v2 MVP summary:")
    print(
        f"    train_acc:   {summary['train_acc'].get('mean', float('nan')):+.3f}"
        f"±{summary['train_acc'].get('std', float('nan')):.3f}"
    )
    v1m = summary["v1_nmi"].get("mean", float("nan"))
    v1s = summary["v1_nmi"].get("std", float("nan"))
    v1_pass = v1m >= 0.85
    print(
        f"    V1_NMI:      {v1m:+.3f}+-{v1s:.3f}"
        f"   {'[PASS]' if v1_pass else '[FAIL]'} (target >= 0.85)"
    )
    v2m = summary["v2_top1"].get("mean", float("nan"))
    v2s = summary["v2_top1"].get("std", float("nan"))
    v2_pass = v2m >= 0.30
    print(
        f"    V2_top1:     {v2m:+.3f}+-{v2s:.3f}"
        f"   {'[PASS]' if v2_pass else '[FAIL]'} (target >= 0.30,"
        f" chance {1.0 / N:.3f})"
    )
    if v1_pass and v2_pass:
        print("\n  → V1 + V2 GREEN: proceed to V3 (space domain)")
    elif not v1_pass and not v2_pass:
        print("\n  → both V1 + V2 FAILED: design needs revision")
    elif not v2_pass:
        print("\n  → V1 OK, V2 FAILED: arithmetic loss insufficient,"
              " consider weight schedule or longer training")
    else:
        print("\n  → V2 OK, V1 FAILED: slot facet not learning identity")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
