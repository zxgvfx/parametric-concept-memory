"""V3 follow-up — Relative Position Embedding across number / colour /
phoneme domains.

The §7.5-space mixed_OOD = 0.000 ceiling was fully broken by an
RPE-only head (commit ``a6e6dec``); a follow-up λ-sweep showed
slot attention was pure overhead on the spatial direction task.
This script asks the next obvious question: **does RPE-as-prior
generalise to other PCM domains where the task answer also depends
only on a displacement?**

Three task variants are tested with the same pcm.dual_channel
RelativePositionEmbedding API:

* ``number_succ`` — predict ``b - a`` class on integer pairs in
  ``[0, N)``. Trained on inner pairs ``a, b ∈ [0, N_train)``,
  tested on **outer-OOD** pairs spanning the held-out range
  ``[N_train, N_total)``. RPE range is ``(-(N_total-1), N_total-1)``.
  Connects to S6 (vector analogy collapsed to chance under v1
  attr; we now ask whether RPE recovers it).

* ``color_cyclic`` — predict the 12-class hue-step ``(b - a) mod 12``
  on a 12-hue ring. RPE table is cyclic-by-construction (only
  12 distinct displacements). Connects to §6.8 / §7.5-color.

* ``phoneme_diff`` — predict V/M/P-feature differences on phoneme
  pairs (signed integer triples). Tests whether RPE generalises
  to multi-axis discrete displacements without invoking the full
  PCM phoneme study scaffolding. Connects to §6.9 cross-language
  transfer.

For each variant we compare:
* **A (none)**: legacy MLP `Linear(2*concat) -> ReLU -> Linear`,
  the v1-style head archetype.
* **B (RPE-only)**: lookup `(Δ) -> embed -> ReLU -> classes`.

Falsifiability: B (RPE) should saturate the OOD splits where v1
collapses, mirroring §7.5-space mixed_OOD = 0.000 -> 1.000.

Usage::

    python -m experiments.rpe_cross_domain --domain number_succ \\
        --n-seeds 5 --epochs 20 --steps-per-epoch 240 \\
        --out outputs/rpe_number
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

from pcm.dual_channel import RelativePositionEmbedding


__all__ = ["main"]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 16
HIDDEN = 64


# ─────────────────────────────────────────────────────────────────
# Heads
# ─────────────────────────────────────────────────────────────────


class ConcatMLPHead(nn.Module):
    """v1-style baseline: concat absolute-position embeddings of a, b
    and run a 2-layer ReLU MLP. The same fc1-over-pair archetype
    that produced the §7.5-space mixed_OOD = 0.000 ceiling and the
    S6 number vector analogy < chance result.
    """

    def __init__(
        self, n_items: int, embed_dim: int, n_classes: int,
        hidden: int = HIDDEN,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_items, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ea = self.embed(a)
        eb = self.embed(b)
        return self.mlp(torch.cat([ea, eb], dim=-1))


class RPEHead(nn.Module):
    """RPE-only head. Generic over k displacement axes."""

    def __init__(
        self, ranges: list[tuple[int, int]], embed_dim: int,
        n_classes: int, hidden: int = HIDDEN,
    ) -> None:
        super().__init__()
        self.rpe = RelativePositionEmbedding(ranges, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, *deltas: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.rpe(*deltas))


# ─────────────────────────────────────────────────────────────────
# Domain — number (1-d ordinal)
# ─────────────────────────────────────────────────────────────────


def _number_succ_setup(
    seed: int, n_train: int, n_total: int, ood_ratio: float,
):
    """Triples ``(a, b, target_class)`` where target_class encodes
    the signed difference ``b - a + (n_total - 1)`` so it's a
    ``2 * n_total - 1`` class problem.
    Returns train, test_in_range, test_outer_ood splits."""
    rng = random.Random(seed)
    inner_pairs = [
        (a, b) for a in range(n_train) for b in range(n_train)
    ]
    outer_pairs = [
        (a, b) for a in range(n_total) for b in range(n_total)
        if not (a < n_train and b < n_train)
    ]
    rng.shuffle(inner_pairs)
    n_test = max(1, int(len(inner_pairs) * ood_ratio))
    test_in_range = inner_pairs[:n_test]
    train = inner_pairs[n_test:]
    return train, test_in_range, outer_pairs


def _number_target(a: int, b: int, n_total: int) -> int:
    return b - a + (n_total - 1)


def _run_number(
    seed: int, *, head_type: str,
    n_train: int, n_total: int, ood_ratio: float,
    epochs: int, steps_per_epoch: int,
    batch_size: int = 64, lr: float = 5e-3,
) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train, in_range, outer = _number_succ_setup(
        seed, n_train, n_total, ood_ratio,
    )
    n_classes = 2 * n_total - 1

    if head_type == "concat":
        head = ConcatMLPHead(
            n_items=n_total, embed_dim=EMBED_DIM, n_classes=n_classes,
        ).to(DEVICE)
    elif head_type == "rpe":
        head = RPEHead(
            ranges=[(-(n_total - 1), n_total - 1)],
            embed_dim=EMBED_DIM, n_classes=n_classes,
        ).to(DEVICE)
    else:
        raise ValueError(head_type)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        head.train()
        for _ in range(steps_per_epoch):
            batch = [train[rng.randrange(len(train))]
                     for _ in range(batch_size)]
            a = torch.tensor([p[0] for p in batch], device=DEVICE,
                             dtype=torch.long)
            b = torch.tensor([p[1] for p in batch], device=DEVICE,
                             dtype=torch.long)
            tgt = torch.tensor(
                [_number_target(p[0], p[1], n_total) for p in batch],
                device=DEVICE,
            )
            if head_type == "concat":
                logits = head(a, b)
            else:
                logits = head(b - a)
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()

    head.eval()

    @torch.no_grad()
    def _eval(pairs):
        if not pairs:
            return float("nan")
        hits = 0
        for i in range(0, len(pairs), 256):
            batch = pairs[i:i + 256]
            a = torch.tensor([p[0] for p in batch], device=DEVICE,
                             dtype=torch.long)
            b = torch.tensor([p[1] for p in batch], device=DEVICE,
                             dtype=torch.long)
            tgt = torch.tensor(
                [_number_target(p[0], p[1], n_total) for p in batch],
                device=DEVICE,
            )
            if head_type == "concat":
                logits = head(a, b)
            else:
                logits = head(b - a)
            hits += logits.argmax(-1).eq(tgt).sum().item()
        return hits / len(pairs)

    return {
        "seed": seed, "head_type": head_type,
        "wall_s": time.time() - t0,
        "train_acc": _eval(train),
        "test_in_range": _eval(in_range),
        "test_outer_OOD": _eval(outer),
    }


# ─────────────────────────────────────────────────────────────────
# Domain — colour (12-hue cyclic)
# ─────────────────────────────────────────────────────────────────


def _color_setup(seed: int, n_colors: int, ood_ratio: float):
    rng = random.Random(seed)
    pairs = [(a, b) for a in range(n_colors) for b in range(n_colors)]
    rng.shuffle(pairs)
    # Hold out a fraction of *target hues* (b cells) in test_outer.
    test_hues = set(rng.sample(range(n_colors), max(2, n_colors // 4)))
    test_outer = [(a, b) for (a, b) in pairs if b in test_hues]
    train_pool = [(a, b) for (a, b) in pairs if b not in test_hues]
    rng.shuffle(train_pool)
    n_test = max(1, int(len(train_pool) * ood_ratio))
    test_in_range = train_pool[:n_test]
    train = train_pool[n_test:]
    return train, test_in_range, test_outer


def _color_target(a: int, b: int, n_colors: int) -> int:
    return (b - a) % n_colors


def _run_color(
    seed: int, *, head_type: str, n_colors: int,
    ood_ratio: float, epochs: int, steps_per_epoch: int,
    batch_size: int = 64, lr: float = 5e-3,
) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train, in_range, outer = _color_setup(seed, n_colors, ood_ratio)

    n_classes = n_colors

    if head_type == "concat":
        head = ConcatMLPHead(
            n_items=n_colors, embed_dim=EMBED_DIM, n_classes=n_classes,
        ).to(DEVICE)
    elif head_type == "rpe":
        # Cyclic: range (0, n_colors - 1) on the modular delta.
        head = RPEHead(
            ranges=[(0, n_colors - 1)],
            embed_dim=EMBED_DIM, n_classes=n_classes,
        ).to(DEVICE)
    else:
        raise ValueError(head_type)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        head.train()
        for _ in range(steps_per_epoch):
            batch = [train[rng.randrange(len(train))]
                     for _ in range(batch_size)]
            a = torch.tensor([p[0] for p in batch], device=DEVICE,
                             dtype=torch.long)
            b = torch.tensor([p[1] for p in batch], device=DEVICE,
                             dtype=torch.long)
            tgt = torch.tensor(
                [_color_target(p[0], p[1], n_colors) for p in batch],
                device=DEVICE,
            )
            if head_type == "concat":
                logits = head(a, b)
            else:
                logits = head((b - a) % n_colors)
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()

    @torch.no_grad()
    def _eval(pairs):
        if not pairs:
            return float("nan")
        hits = 0
        for i in range(0, len(pairs), 256):
            batch = pairs[i:i + 256]
            a = torch.tensor([p[0] for p in batch], device=DEVICE,
                             dtype=torch.long)
            b = torch.tensor([p[1] for p in batch], device=DEVICE,
                             dtype=torch.long)
            tgt = torch.tensor(
                [_color_target(p[0], p[1], n_colors) for p in batch],
                device=DEVICE,
            )
            if head_type == "concat":
                logits = head(a, b)
            else:
                logits = head((b - a) % n_colors)
            hits += logits.argmax(-1).eq(tgt).sum().item()
        return hits / len(pairs)

    return {
        "seed": seed, "head_type": head_type,
        "wall_s": time.time() - t0,
        "n_train": len(train), "n_outer": len(outer),
        "train_acc": _eval(train),
        "test_in_range": _eval(in_range),
        "test_holdout_hue": _eval(outer),
    }


# ─────────────────────────────────────────────────────────────────
# Domain — phoneme (V/M/P 3-axis discrete)
# ─────────────────────────────────────────────────────────────────


def _phoneme_setup(seed: int, n_v: int, n_m: int, n_p: int,
                   ood_ratio: float):
    rng = random.Random(seed)
    items = [(v, m, p)
             for v in range(n_v) for m in range(n_m) for p in range(n_p)]
    n_total = len(items)
    pairs = [(i, j) for i in range(n_total) for j in range(n_total)]
    rng.shuffle(pairs)

    # Hold out a fraction of source-language items (i_a) — those
    # become test_outer (cross-language transfer analogue).
    held_a = set(rng.sample(range(n_total), max(2, n_total // 4)))
    test_outer = [(i, j) for (i, j) in pairs if i in held_a]
    train_pool = [(i, j) for (i, j) in pairs if i not in held_a]
    rng.shuffle(train_pool)
    n_test = max(1, int(len(train_pool) * ood_ratio))
    test_in_range = train_pool[:n_test]
    train = train_pool[n_test:]
    return train, test_in_range, test_outer, items


def _phoneme_classes(n_v: int, n_m: int, n_p: int) -> int:
    # Joint class id: encode (Δv, Δm, Δp) as a single index.
    return (2 * n_v - 1) * (2 * n_m - 1) * (2 * n_p - 1)


def _phoneme_target(item_a, item_b, n_v, n_m, n_p) -> int:
    dv = item_b[0] - item_a[0] + (n_v - 1)
    dm = item_b[1] - item_a[1] + (n_m - 1)
    dp = item_b[2] - item_a[2] + (n_p - 1)
    n_dv = 2 * n_v - 1
    n_dm = 2 * n_m - 1
    n_dp = 2 * n_p - 1
    return dv * (n_dm * n_dp) + dm * n_dp + dp


def _run_phoneme(
    seed: int, *, head_type: str,
    n_v: int = 2, n_m: int = 4, n_p: int = 4,
    ood_ratio: float = 0.15,
    epochs: int = 20, steps_per_epoch: int = 240,
    batch_size: int = 64, lr: float = 5e-3,
) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train, in_range, outer, items = _phoneme_setup(
        seed, n_v, n_m, n_p, ood_ratio,
    )
    n_total = len(items)
    n_classes = _phoneme_classes(n_v, n_m, n_p)

    if head_type == "concat":
        head = ConcatMLPHead(
            n_items=n_total, embed_dim=EMBED_DIM, n_classes=n_classes,
            hidden=128,
        ).to(DEVICE)
    elif head_type == "rpe":
        head = RPEHead(
            ranges=[
                (-(n_v - 1), n_v - 1),
                (-(n_m - 1), n_m - 1),
                (-(n_p - 1), n_p - 1),
            ],
            embed_dim=EMBED_DIM, n_classes=n_classes,
        ).to(DEVICE)
    else:
        raise ValueError(head_type)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    def _deltas(batch):
        ia, ib = zip(*batch)
        items_a = [items[i] for i in ia]
        items_b = [items[i] for i in ib]
        dv = torch.tensor(
            [b[0] - a[0] for a, b in zip(items_a, items_b)],
            device=DEVICE, dtype=torch.long,
        )
        dm = torch.tensor(
            [b[1] - a[1] for a, b in zip(items_a, items_b)],
            device=DEVICE, dtype=torch.long,
        )
        dp = torch.tensor(
            [b[2] - a[2] for a, b in zip(items_a, items_b)],
            device=DEVICE, dtype=torch.long,
        )
        return dv, dm, dp, items_a, items_b

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        head.train()
        for _ in range(steps_per_epoch):
            batch = [train[rng.randrange(len(train))]
                     for _ in range(batch_size)]
            a = torch.tensor([p[0] for p in batch], device=DEVICE,
                             dtype=torch.long)
            b = torch.tensor([p[1] for p in batch], device=DEVICE,
                             dtype=torch.long)
            dv, dm, dp, items_a, items_b = _deltas(batch)
            tgt = torch.tensor(
                [_phoneme_target(items_a[k], items_b[k], n_v, n_m, n_p)
                 for k in range(len(batch))],
                device=DEVICE,
            )
            if head_type == "concat":
                logits = head(a, b)
            else:
                logits = head(dv, dm, dp)
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()

    @torch.no_grad()
    def _eval(pairs):
        if not pairs:
            return float("nan")
        hits = 0
        for i in range(0, len(pairs), 256):
            batch = pairs[i:i + 256]
            a = torch.tensor([p[0] for p in batch], device=DEVICE,
                             dtype=torch.long)
            b = torch.tensor([p[1] for p in batch], device=DEVICE,
                             dtype=torch.long)
            dv, dm, dp, items_a, items_b = _deltas(batch)
            tgt = torch.tensor(
                [_phoneme_target(items_a[k], items_b[k], n_v, n_m, n_p)
                 for k in range(len(batch))],
                device=DEVICE,
            )
            if head_type == "concat":
                logits = head(a, b)
            else:
                logits = head(dv, dm, dp)
            hits += logits.argmax(-1).eq(tgt).sum().item()
        return hits / len(pairs)

    return {
        "seed": seed, "head_type": head_type,
        "wall_s": time.time() - t0,
        "train_acc": _eval(train),
        "test_in_range": _eval(in_range),
        "test_outer_OOD": _eval(outer),
    }


# ─────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--domain",
        choices=["number_succ", "color_cyclic", "phoneme_diff"],
        default="number_succ",
    )
    ap.add_argument("--head", choices=["concat", "rpe", "both"],
                    default="both")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=96000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=240)
    ap.add_argument("--ood-ratio", type=float, default=0.15)
    ap.add_argument("--n-train", type=int, default=20,
                    help="number_succ: |inner range|")
    ap.add_argument("--n-total", type=int, default=30,
                    help="number_succ: |full range|")
    ap.add_argument("--n-colors", type=int, default=12,
                    help="color_cyclic: |hue ring|")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/rpe_cross_domain"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    head_types = (
        ["concat", "rpe"] if args.head == "both" else [args.head]
    )

    print("=" * 76)
    print(f"  V3 cross-domain RPE: domain={args.domain}, "
          f"heads={head_types}, n_seeds={args.n_seeds}")
    print("=" * 76)

    all_rows: list[dict] = []
    for head_type in head_types:
        print(f"\n-- head={head_type} --")
        for si in range(args.n_seeds):
            seed = args.seed_base + si
            if args.domain == "number_succ":
                r = _run_number(
                    seed, head_type=head_type,
                    n_train=args.n_train, n_total=args.n_total,
                    ood_ratio=args.ood_ratio,
                    epochs=args.epochs,
                    steps_per_epoch=args.steps_per_epoch,
                )
                key = "test_outer_OOD"
            elif args.domain == "color_cyclic":
                r = _run_color(
                    seed, head_type=head_type, n_colors=args.n_colors,
                    ood_ratio=args.ood_ratio,
                    epochs=args.epochs,
                    steps_per_epoch=args.steps_per_epoch,
                )
                key = "test_holdout_hue"
            else:
                r = _run_phoneme(
                    seed, head_type=head_type,
                    ood_ratio=args.ood_ratio,
                    epochs=args.epochs,
                    steps_per_epoch=args.steps_per_epoch,
                )
                key = "test_outer_OOD"
            all_rows.append(r)
            print(
                f"  [{head_type} seed={seed}] "
                f"train={r['train_acc']:.3f}  "
                f"in_range={r['test_in_range']:.3f}  "
                f"OOD_key({key})={r[key]:.3f}  ({r['wall_s']:.1f}s)"
            )

    # Aggregate
    def _stats(rows, key):
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
        "per_seed": all_rows,
    }
    keys = (
        ["train_acc", "test_in_range", "test_outer_OOD"]
        if args.domain != "color_cyclic"
        else ["train_acc", "test_in_range", "test_holdout_hue"]
    )
    by_head: dict[str, dict[str, dict]] = {}
    for head_type in head_types:
        rows = [r for r in all_rows if r["head_type"] == head_type]
        by_head[head_type] = {k: _stats(rows, k) for k in keys}
    summary["by_head"] = by_head
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print(f"  RPE cross-domain ({args.domain}, n_seeds={args.n_seeds}):")
    for k in keys:
        line = f"    {k:<22s}"
        for head_type in head_types:
            s = by_head[head_type][k]
            line += f"  {head_type}={s.get('mean', float('nan')):+.3f}+-{s.get('std', float('nan')):.3f}"
        print(line)
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
