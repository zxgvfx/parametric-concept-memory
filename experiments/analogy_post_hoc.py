"""PAPER §19 (S6) — post-hoc vector-analogy evaluation.

Tests whether PCM's bundle rows, learned without any explicit
analogy supervision, support **vector analogy** of the form

    bundle_d ≈ bundle_a + bundle_c − bundle_b   ⟺   d == a + c − b

i.e. the classic king − man + woman ≈ queen pattern (Mikolov et
al. 2013) extended to ordinal numbers / hue rings. Per Feature
Resemblance (arxiv 2603.05143, 2026) and Emergent Analogy in
Transformers (OpenReview aFCoTBGM4M, 2026), analogical reasoning
emerges **after** similarity structure is learned; we ask whether
PCM's BCD condition produces a bundle geometry that already
satisfies the vector-analogy criterion across domains.

This script is a **post-hoc analyser** — it expects an existing
trained graph and reads only ``cg.bundle_pool[facet]`` rows. No
training, no head fitting.

Falsifiability:

* ``analogy_acc`` ≥ 0.30 (vs chance 1/N) → PCM bundle geometry
  is "linearly compositional" for that domain.
* ``analogy_acc`` ≤ chance + 1σ → S6 falsified for that domain;
  vector analogy needs an explicit head (D-layer extension).

Usage::

    python -m experiments.analogy_post_hoc \\
        --domain number --N 10 --n-seeds 5 \\
        --epochs 30 --steps-per-epoch 200 \\
        --out outputs/analogy_number_10
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


__all__ = ["analogy_acc_grid", "analogy_acc_color", "main"]


# ─────────────────────────────────────────────────────────────────
# Vector-analogy metric on a fixed table of bundle rows.
# ─────────────────────────────────────────────────────────────────


def _topk_match(query: torch.Tensor, bank: torch.Tensor,
                target: int, k: int = 1) -> bool:
    """Whether ``target`` is in the top-k nearest rows of ``bank``
    (by L2 distance) to ``query``.
    """
    d = (bank - query.unsqueeze(0)).pow(2).sum(dim=-1)
    topk = d.topk(k, largest=False).indices.tolist()
    return target in topk


def analogy_acc_grid(
    bundle_rows: torch.Tensor,
    *,
    n_total: int,
    max_triples: int = 4000,
    seed: int = 0,
    topk: int = 1,
) -> dict[str, float]:
    """Vector-analogy accuracy over ordinal triples on
    ``bundle_rows``.

    Generates triples ``(a, b, c)`` with ``a != b``, computes
    ``query = bundle_a + bundle_c − bundle_b`` and counts the
    fraction whose nearest neighbour in the same bundle table
    equals ``a + c − b`` (when in-range).

    Returns dict with ``acc_top1``, ``acc_top3``, ``n_evaluated``,
    ``chance``.
    """
    rng = random.Random(seed)
    bundle_rows = bundle_rows.detach().cpu().float()
    if bundle_rows.shape[0] < n_total:
        return {
            "acc_top1": float("nan"), "acc_top3": float("nan"),
            "n_evaluated": 0, "chance": 1.0 / n_total,
        }
    table = bundle_rows[:n_total]
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
        return {
            "acc_top1": float("nan"), "acc_top3": float("nan"),
            "n_evaluated": 0, "chance": 1.0 / n_total,
        }
    hits1 = 0
    hits3 = 0
    for a, b, c, d in triples:
        query = table[a] + table[c] - table[b]
        if _topk_match(query, table, d, k=1):
            hits1 += 1
        if _topk_match(query, table, d, k=3):
            hits3 += 1
    return {
        "acc_top1": hits1 / n,
        "acc_top3": hits3 / n,
        "n_evaluated": n,
        "chance": 1.0 / n_total,
    }


def analogy_acc_color(
    bundle_rows: torch.Tensor,
    *,
    n_colors: int = 12,
    max_triples: int = 4000,
    seed: int = 0,
) -> dict[str, float]:
    """Vector-analogy on cyclic hue ring.

    Triples ``(a, b, c) → d = (a + c − b) mod n_colors``. Same
    metric, but target is mod-n_colors so all triples are valid.
    """
    rng = random.Random(seed)
    bundle_rows = bundle_rows.detach().cpu().float()
    if bundle_rows.shape[0] < n_colors:
        return {
            "acc_top1": float("nan"), "acc_top3": float("nan"),
            "n_evaluated": 0, "chance": 1.0 / n_colors,
        }
    table = bundle_rows[:n_colors]
    triples: list[tuple[int, int, int, int]] = []
    for a in range(n_colors):
        for b in range(n_colors):
            if a == b:
                continue
            for c in range(n_colors):
                d = (a + c - b) % n_colors
                if d != c:
                    triples.append((a, b, c, d))
    rng.shuffle(triples)
    triples = triples[:max_triples]
    n = len(triples)
    hits1 = 0
    hits3 = 0
    for a, b, c, d in triples:
        query = table[a] + table[c] - table[b]
        if _topk_match(query, table, d, k=1):
            hits1 += 1
        if _topk_match(query, table, d, k=3):
            hits3 += 1
    return {
        "acc_top1": hits1 / n,
        "acc_top3": hits3 / n,
        "n_evaluated": n,
        "chance": 1.0 / n_colors,
    }


# ─────────────────────────────────────────────────────────────────
# Domain runners that fresh-train a small graph then run analogy.
# ─────────────────────────────────────────────────────────────────


def _number_one_seed(
    seed: int, *, N: int, epochs: int, steps_per_epoch: int,
    bcd: bool,
) -> dict:
    from experiments.number_decimal_priors import zipf_weights
    from experiments.quad_study import enumerate_triples, train_quad

    weights = zipf_weights(N) if bcd else None
    centroid_mode = "decimal_cones" if bcd else "random"

    all_triples = enumerate_triples(N, 1.0)
    rng = random.Random(seed)
    train_triples = []
    for op, trips in all_triples.items():
        trips = list(trips)
        rng.shuffle(trips)
        n_test = max(1, int(len(trips) * 0.15))
        for (a, b, c) in trips[n_test:]:
            train_triples.append((a, b, op, c))

    t0 = time.time()
    r = train_quad(
        N, 1.0, seed,
        train_triples=train_triples,
        epochs=epochs, steps_per_epoch=steps_per_epoch,
        centroid_mode=centroid_mode,
        digit_sample_weight=weights,
        enable_last_digit_head=bcd,
        n_total=N,
        sleep_every=None,
    )
    cg = r["cg"]
    facet = "arithmetic_bias"
    pool = cg.bundle_pool[facet].detach().cpu()
    rows: list[torch.Tensor] = []
    for i in range(N):
        cid = f"concept:num:{i}"
        if cid in cg.cid_to_slot:
            slot = cg.cid_to_slot[cid]
            rows.append(pool.data[slot].clone())
        else:
            rows.append(torch.zeros_like(pool.data[0]))
    table = torch.stack(rows)
    return {
        "wall_s": time.time() - t0,
        **analogy_acc_grid(table, n_total=N, seed=seed),
    }


def _color_one_seed(
    seed: int, *, n_colors: int, epochs: int, steps_per_epoch: int,
    bcd: bool,
) -> dict:
    from experiments.color_concept_study._config import (
        EMBED_DIM, FACET_MIX, N_COLORS,
    )
    from experiments.color_concept_study.graph_builder import (
        make_lms_like_centroids, make_random_orthogonal_centroids,
    )
    from experiments.color_concept_study.train import train_one
    from experiments.sleep_color_primaries import _green_peak_weights

    centroids = (
        make_lms_like_centroids(N_COLORS, EMBED_DIM, seed) if bcd
        else make_random_orthogonal_centroids(N_COLORS, EMBED_DIM, seed)
    )
    weights = _green_peak_weights() if bcd else None

    t0 = time.time()
    r = train_one(
        "single", seed, centroids,
        epochs=epochs, steps_per_epoch=steps_per_epoch,
        mix_sample_weight=weights,
        enable_ripe_head=bcd,
        sleep_every=None,
    )
    bs = r["bundle_state"]
    rows = torch.stack([
        bs[f"concept:color:{i}"][f"params.{FACET_MIX}"]
        for i in range(N_COLORS)
    ])
    return {
        "wall_s": time.time() - t0,
        **analogy_acc_color(rows, n_colors=N_COLORS, seed=seed),
    }


# ─────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["number", "color"], default="number")
    ap.add_argument("--N", type=int, default=10,
                    help="number domain only: |numbers|")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=89000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--bcd", action="store_true",
                    help="enable B+C+D causal layers (default: A baseline)")
    ap.add_argument(
        "--out", type=Path,
        default=Path("outputs/analogy_post_hoc"),
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(
        f"  PAPER §19 (S6) post-hoc vector analogy: domain={args.domain}, "
        f"BCD={args.bcd}, n_seeds={args.n_seeds}"
    )
    print("=" * 72)

    rows: list[dict] = []
    for si in range(args.n_seeds):
        seed = args.seed_base + si
        if args.domain == "number":
            r = _number_one_seed(
                seed, N=args.N, epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch, bcd=args.bcd,
            )
        else:
            from experiments.color_concept_study._config import N_COLORS
            r = _color_one_seed(
                seed, n_colors=N_COLORS, epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch, bcd=args.bcd,
            )
        r["seed"] = seed
        rows.append(r)
        print(
            f"  [seed={seed}] top1={r.get('acc_top1', float('nan')):.3f} "
            f"top3={r.get('acc_top3', float('nan')):.3f} "
            f"chance={r.get('chance', float('nan')):.3f} "
            f"({r.get('wall_s', float('nan')):.1f}s)"
        )

    def _stats(key: str) -> dict:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))
                and not math.isnan(r[key])]
        if not vals:
            return {"n": 0}
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1))
        return {"mean": m, "std": sd, "min": min(vals), "max": max(vals),
                "n": len(vals)}

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "per_seed": rows,
        "acc_top1": _stats("acc_top1"),
        "acc_top3": _stats("acc_top3"),
        "chance": rows[0].get("chance") if rows else None,
    }

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "═" * 72)
    print(
        f"  acc_top1: {summary['acc_top1'].get('mean', float('nan')):+.3f}"
        f"±{summary['acc_top1'].get('std', float('nan')):.3f}"
        f"  (chance {summary['chance']:.3f})"
    )
    print(
        f"  acc_top3: {summary['acc_top3'].get('mean', float('nan')):+.3f}"
        f"±{summary['acc_top3'].get('std', float('nan')):.3f}"
    )
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
