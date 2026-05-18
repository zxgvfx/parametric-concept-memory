"""F75 — Episodic memory + sleep consolidation + "snake at 10".

Tests the F75 episodic-memory architecture with five falsifiable
invariants. The headline is **E5** — the "bitten by a snake at
age 10, afraid of snakes at age 60" test — which demonstrates
that a single rare high-salience event persists in long-term
memory across orders-of-magnitude more bland routine events.

Setup: synthesise a 1000-step "lifetime" of agent experience
where each step is a random slot vector plus a small noise
salience (mean ~0.2). At step 10 we inject a high-|salience|
"snake bite" event with salience=10. After the lifetime, we
test whether:

* the buffer (capacity=50) has long forgotten the snake bite,
  but
* the long-term salience-gated trace (capacity=20, threshold=2)
  still holds it, and
* a query similar to the snake event at step 800 retrieves
  the original event via :meth:`LongTermEpisodicTrace.biased_recall`.

Five falsifiable invariants:

* **E1** Similarity recall: a query slot retrieves the closest
  stored slot (top-1 accuracy ≥ 0.95 on a synthetic test set).
* **E2** Temporal recall: ``recall_time_range`` returns
  precisely the episodes within the requested window.
* **E3** Sleep consolidation: cluster centroids returned by
  :func:`consolidate_to_concept_graph` recover ground-truth
  cluster means (mean cosine ≥ 0.95 after Hungarian matching).
* **E4** FIFO forgetting curve: after appending ``capacity ×
  K`` episodes (K ≥ 5), retrieval of episodes that were >
  capacity steps ago fails; recent ones still found.
* **E5** Long-term emotional retention ("snake at 10"): at
  step 800 (790 steps after the snake event), the snake event
  is still imprinted in :class:`LongTermEpisodicTrace` and is
  the top-1 result of :meth:`biased_recall` for a query similar
  to the original snake slot.

Usage::

    python -m experiments.episodic_memory_f75 \\
        --buffer-capacity 50 --long-capacity 20 \\
        --out outputs/f75_full
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from pcm import (
    EpisodicBuffer,
    LongTermEpisodicTrace,
    consolidate_to_concept_graph,
)


__all__ = ["main"]

DEVICE = "cpu"  # episodic memory is pure data structure; CPU is fine


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


def _hungarian_match_mean_cosine(
    pred_centroids: torch.Tensor, true_centroids: torch.Tensor,
) -> float:
    """Greedy bipartite matching by cosine similarity; return
    the mean matched cosine."""
    n = min(pred_centroids.shape[0], true_centroids.shape[0])
    if n == 0:
        return float("nan")
    pred_n = F.normalize(pred_centroids, dim=-1)
    true_n = F.normalize(true_centroids, dim=-1)
    cos = pred_n @ true_n.t()
    matched_true: set[int] = set()
    matched_pred: set[int] = set()
    pairs = []
    for _ in range(n):
        # Find best remaining pair
        best_val = -2.0
        best_pair = None
        for i in range(cos.shape[0]):
            if i in matched_pred:
                continue
            for j in range(cos.shape[1]):
                if j in matched_true:
                    continue
                v = float(cos[i, j].item())
                if v > best_val:
                    best_val = v
                    best_pair = (i, j)
        if best_pair is None:
            break
        matched_pred.add(best_pair[0])
        matched_true.add(best_pair[1])
        pairs.append(best_val)
    return sum(pairs) / len(pairs) if pairs else float("nan")


# ─────────────────────────────────────────────────────────────────
# E1 — Similarity recall
# ─────────────────────────────────────────────────────────────────


def test_e1_similarity_recall(
    *, slot_dim: int, n_episodes: int, rng_seed: int = 0,
) -> dict:
    torch.manual_seed(rng_seed)
    buf = EpisodicBuffer(capacity=n_episodes, slot_dim=slot_dim)
    # Use orthogonal-ish slots so each is uniquely identifiable
    slots = torch.randn(n_episodes, slot_dim)
    slots = F.normalize(slots, dim=-1)
    for t, slot in enumerate(slots):
        buf.append(slot, timestamp=t)
    # For each, query the buffer with the original slot; top-1
    # should return the same timestamp
    n_correct = 0
    for t, slot in enumerate(slots):
        results = buf.recall_by_similarity(slot, k=1)
        if results and results[0].timestamp == t:
            n_correct += 1
    return {
        "n_episodes": n_episodes,
        "n_correct_top1": n_correct,
        "accuracy": n_correct / n_episodes,
    }


# ─────────────────────────────────────────────────────────────────
# E2 — Temporal recall
# ─────────────────────────────────────────────────────────────────


def test_e2_temporal_recall(
    *, slot_dim: int, n_episodes: int = 100, rng_seed: int = 1,
) -> dict:
    torch.manual_seed(rng_seed)
    buf = EpisodicBuffer(capacity=n_episodes, slot_dim=slot_dim)
    for t in range(n_episodes):
        buf.append(torch.randn(slot_dim), timestamp=t * 3)  # stride 3
    # Query several windows
    queries = [
        (0, 9),                    # first 4 episodes (ts 0, 3, 6, 9)
        (30, 39),                  # ts 30, 33, 36, 39
        (75, 90),                  # ts 75, 78, 81, 84, 87, 90
        (-5, 5),                   # ts 0, 3
        (1000, 9999),              # empty range
    ]
    n_correct = 0
    n_total = 0
    for lo, hi in queries:
        out = buf.recall_time_range(lo, hi)
        expected = [t * 3 for t in range(n_episodes)
                    if lo <= t * 3 <= hi]
        got = [r.timestamp for r in out]
        n_total += 1
        if got == expected:
            n_correct += 1
    return {
        "n_queries": n_total,
        "n_correct": n_correct,
        "accuracy": n_correct / n_total,
    }


# ─────────────────────────────────────────────────────────────────
# E3 — Sleep consolidation: episodic → semantic clusters
# ─────────────────────────────────────────────────────────────────


def test_e3_consolidation(
    *, slot_dim: int, n_clusters: int = 5,
    points_per_cluster: int = 50, noise: float = 0.15,
    rng_seed: int = 2,
) -> dict:
    torch.manual_seed(rng_seed)
    buf = EpisodicBuffer(
        capacity=n_clusters * points_per_cluster, slot_dim=slot_dim,
    )
    # Generate well-separated true centroids on a hypersphere
    true_centroids = F.normalize(torch.randn(n_clusters, slot_dim), dim=-1)
    timestamp = 0
    for c in range(n_clusters):
        for _ in range(points_per_cluster):
            slot = true_centroids[c] + noise * torch.randn(slot_dim)
            buf.append(slot, timestamp=timestamp)
            timestamp += 1
    res = consolidate_to_concept_graph(
        buf, n_clusters=n_clusters, n_iter=30,
    )
    mean_cos = _hungarian_match_mean_cosine(
        res["centroids"], true_centroids,
    )
    return {
        "n_clusters": n_clusters,
        "n_episodes": int(res["n_episodes"]),
        "inertia": res["inertia"],
        "matched_mean_cosine_to_truth": mean_cos,
    }


# ─────────────────────────────────────────────────────────────────
# E4 — FIFO forgetting curve
# ─────────────────────────────────────────────────────────────────


def test_e4_fifo_forgetting(
    *, slot_dim: int, capacity: int = 50,
    n_appends: int = 250, rng_seed: int = 3,
) -> dict:
    torch.manual_seed(rng_seed)
    buf = EpisodicBuffer(capacity=capacity, slot_dim=slot_dim)
    slots = [F.normalize(torch.randn(slot_dim), dim=-1) for _ in range(n_appends)]
    for t, s in enumerate(slots):
        buf.append(s, timestamp=t)
    expected_remaining = list(range(n_appends - capacity, n_appends))
    actual = sorted([r.timestamp for r in buf.records()])
    forgot_correctly = (actual == expected_remaining)
    # Spot-check: oldest episode (timestamp 0) cannot be recalled
    old_query = slots[0]
    results = buf.recall_by_similarity(old_query, k=1)
    old_returned = (results[0].timestamp == 0) if results else False
    # Recent episode (timestamp n_appends-1) can be recalled
    recent_query = slots[n_appends - 1]
    recent_results = buf.recall_by_similarity(recent_query, k=1)
    recent_returned = (
        recent_results[0].timestamp == n_appends - 1
        if recent_results else False
    )
    return {
        "n_appends": n_appends,
        "capacity": capacity,
        "forgot_oldest_correctly": forgot_correctly,
        "old_query_returned_old": old_returned,        # should be False
        "recent_query_returned_recent": recent_returned,  # should be True
    }


# ─────────────────────────────────────────────────────────────────
# E5 — "Bitten by a snake at 10, still afraid at 60"
# ─────────────────────────────────────────────────────────────────


def test_e5_snake_at_10(
    *, slot_dim: int,
    buffer_capacity: int = 50,
    long_capacity: int = 20,
    salience_threshold: float = 2.0,
    snake_step: int = 10,
    snake_salience: float = 10.0,
    routine_salience_mean: float = 0.2,
    lifetime: int = 1000,
    query_step: int = 800,
    rng_seed: int = 7,
) -> dict:
    """The headline test.

    Simulate a ``lifetime``-step agent life. Routine events at
    every step have small random salience (mean
    ``routine_salience_mean``). At ``snake_step`` we inject one
    high-salience event (salience = ``snake_salience``) with a
    distinctive slot vector (the "snake" concept). At
    ``query_step`` (long after the buffer has overflowed), we
    query with a slot similar to the snake event.

    Pass criteria:
    * The buffer no longer contains the snake event (it has been
      FIFO-evicted hundreds of steps ago).
    * The long-term trace still contains the snake event.
    * Biased recall on the long-term trace returns the snake
      event as top-1 for the snake-similar query.
    """
    torch.manual_seed(rng_seed)
    buf = EpisodicBuffer(capacity=buffer_capacity, slot_dim=slot_dim)
    trace = LongTermEpisodicTrace(
        capacity=long_capacity, slot_dim=slot_dim,
        salience_threshold=salience_threshold,
    )
    # "snake" slot — a deliberately recognisable vector
    snake_slot = torch.zeros(slot_dim)
    snake_slot[0] = 1.0
    snake_slot[1] = 1.0
    snake_slot = F.normalize(snake_slot, dim=-1)
    snake_imprinted_at = None
    for t in range(lifetime):
        if t == snake_step:
            slot = snake_slot.clone()
            sal = snake_salience
            md = {"event": "SNAKE_BITE"}
        else:
            slot = F.normalize(torch.randn(slot_dim), dim=-1)
            # Routine salience: small positive noise
            sal = abs(routine_salience_mean
                       + 0.05 * float(torch.randn(1).item()))
            md = {"event": "routine", "step": t}
        buf.append(slot, timestamp=t, salience=sal, metadata=md)
        imprinted = trace.maybe_imprint(slot, timestamp=t,
                                         salience=sal, metadata=md)
        if imprinted and t == snake_step:
            snake_imprinted_at = t

    # At query_step: buffer no longer has snake
    buf_records = buf.records()
    buf_has_snake = any(
        r.metadata.get("event") == "SNAKE_BITE" for r in buf_records
    )
    # Long-term trace still has snake
    trace_records = trace.records()
    trace_has_snake = any(
        r.metadata.get("event") == "SNAKE_BITE" for r in trace_records
    )
    # Query at query_step with a slot similar to snake
    query = snake_slot + 0.05 * torch.randn(slot_dim)
    query = F.normalize(query, dim=-1)
    top_results = trace.biased_recall(query, k=3, salience_weight=0.5)
    top1_is_snake = (
        len(top_results) > 0
        and top_results[0].metadata.get("event") == "SNAKE_BITE"
    )
    return {
        "lifetime": lifetime,
        "snake_step": snake_step,
        "query_step": query_step,
        "buffer_size_at_end": len(buf),
        "trace_size_at_end": len(trace),
        "trace_writes_attempted": trace.writes_attempted,
        "trace_writes_committed": trace.writes_committed,
        "snake_imprinted_in_long_term_at": snake_imprinted_at,
        "buffer_still_has_snake": buf_has_snake,
        "long_term_still_has_snake": trace_has_snake,
        "top1_recall_at_query_is_snake": top1_is_snake,
        "top3_recall_event_labels": [
            r.metadata.get("event") for r in top_results
        ],
    }


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot-dim", type=int, default=32)
    ap.add_argument("--buffer-capacity", type=int, default=50)
    ap.add_argument("--long-capacity", type=int, default=20)
    ap.add_argument("--salience-threshold", type=float, default=2.0)
    ap.add_argument("--lifetime", type=int, default=1000)
    ap.add_argument("--query-step", type=int, default=800)
    ap.add_argument("--snake-step", type=int, default=10)
    ap.add_argument("--snake-salience", type=float, default=10.0)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f75_full"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"  F75 episodic memory + sleep consolidation + "
          f"'snake at 10' (slot_dim={args.slot_dim})")
    print("=" * 76)

    t0 = time.time()
    e1 = test_e1_similarity_recall(
        slot_dim=args.slot_dim, n_episodes=100,
    )
    print(f"\n[E1] similarity recall:    "
          f"top-1 accuracy = {e1['accuracy']:.3f}  "
          f"({e1['n_correct_top1']}/{e1['n_episodes']})")

    e2 = test_e2_temporal_recall(slot_dim=args.slot_dim)
    print(f"[E2] temporal recall:      "
          f"accuracy = {e2['accuracy']:.3f}  "
          f"({e2['n_correct']}/{e2['n_queries']} window queries)")

    e3 = test_e3_consolidation(
        slot_dim=args.slot_dim, n_clusters=5,
        points_per_cluster=50, noise=0.15,
    )
    print(f"[E3] consolidation:        "
          f"matched mean cos to ground-truth centroids = "
          f"{e3['matched_mean_cosine_to_truth']:.3f}  "
          f"(inertia={e3['inertia']:.3f}, n={e3['n_episodes']})")

    e4 = test_e4_fifo_forgetting(
        slot_dim=args.slot_dim, capacity=args.buffer_capacity,
        n_appends=args.buffer_capacity * 5,
    )
    print(f"[E4] FIFO forgetting:      "
          f"forgot_oldest={e4['forgot_oldest_correctly']}  "
          f"old_query_returned_old={e4['old_query_returned_old']}  "
          f"recent_returned_recent={e4['recent_query_returned_recent']}")

    e5 = test_e5_snake_at_10(
        slot_dim=args.slot_dim,
        buffer_capacity=args.buffer_capacity,
        long_capacity=args.long_capacity,
        salience_threshold=args.salience_threshold,
        snake_step=args.snake_step,
        snake_salience=args.snake_salience,
        lifetime=args.lifetime,
        query_step=args.query_step,
    )
    print(f"\n[E5] 'snake at 10' (snake injected at step "
          f"{args.snake_step}, queried at step "
          f"{args.query_step}):")
    print(f"     buffer (cap {args.buffer_capacity}) still has snake : "
          f"{e5['buffer_still_has_snake']}  (expected False)")
    print(f"     long-term trace still has snake               : "
          f"{e5['long_term_still_has_snake']}  (expected True)")
    print(f"     biased recall top-1 is snake                  : "
          f"{e5['top1_recall_at_query_is_snake']}  (expected True)")
    print(f"     trace writes: {e5['trace_writes_committed']}/"
          f"{e5['trace_writes_attempted']} attempted")

    wall = time.time() - t0
    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "wall_seconds": wall,
        "E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5,
    }
    summary["verdict"] = {
        "E1_similarity_recall_pass": e1["accuracy"] >= 0.95,
        "E2_temporal_recall_pass": e2["accuracy"] >= 1.0,
        "E3_consolidation_pass":
            e3["matched_mean_cosine_to_truth"] >= 0.95,
        "E4_fifo_forgetting_pass": (
            e4["forgot_oldest_correctly"]
            and not e4["old_query_returned_old"]
            and e4["recent_query_returned_recent"]
        ),
        "E5_snake_at_10_pass": (
            not e5["buffer_still_has_snake"]
            and e5["long_term_still_has_snake"]
            and e5["top1_recall_at_query_is_snake"]
        ),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print("  F75 verdict:")
    print("=" * 76)
    v = summary["verdict"]
    print(f"  E1 similarity recall (>=0.95)    : "
          f"{e1['accuracy']:.3f}  "
          f"[{'PASS' if v['E1_similarity_recall_pass'] else 'FAIL'}]")
    print(f"  E2 temporal recall (==1.0)       : "
          f"{e2['accuracy']:.3f}  "
          f"[{'PASS' if v['E2_temporal_recall_pass'] else 'FAIL'}]")
    print(f"  E3 consolidation (cos >= 0.95)   : "
          f"{e3['matched_mean_cosine_to_truth']:.3f}  "
          f"[{'PASS' if v['E3_consolidation_pass'] else 'FAIL'}]")
    print(f"  E4 FIFO forgetting all 3 sub     : "
          f"[{'PASS' if v['E4_fifo_forgetting_pass'] else 'FAIL'}]")
    print(f"  E5 'snake at 10' all 3 sub       : "
          f"[{'PASS' if v['E5_snake_at_10_pass'] else 'FAIL'}]")
    print(f"\n  wall = {wall:.1f}s, wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
