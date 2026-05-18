"""Unit tests for F75 episodic memory."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from pcm import (
    EpisodeRecord,
    EpisodicBuffer,
    LongTermEpisodicTrace,
    consolidate_to_concept_graph,
)


# ─────────────────────────────────────────────────────────────────
# EpisodicBuffer basic API
# ─────────────────────────────────────────────────────────────────


def test_buffer_basic_append_and_len():
    b = EpisodicBuffer(capacity=8, slot_dim=4)
    assert len(b) == 0
    assert not b.is_full
    for t in range(5):
        b.append(torch.randn(4), timestamp=t)
    assert len(b) == 5
    assert not b.is_full
    for t in range(5, 10):
        b.append(torch.randn(4), timestamp=t)
    assert len(b) == 8
    assert b.is_full


def test_buffer_fifo_eviction():
    """Once full, oldest timestamp is evicted on next append."""
    b = EpisodicBuffer(capacity=4, slot_dim=2)
    for t in range(10):
        b.append(torch.randn(2), timestamp=t)
    ts = sorted([r.timestamp for r in b.records()])
    assert ts == [6, 7, 8, 9]


def test_buffer_recall_by_similarity_self_returns_self():
    torch.manual_seed(0)
    b = EpisodicBuffer(capacity=16, slot_dim=8)
    slots = F.normalize(torch.randn(16, 8), dim=-1)
    for t, s in enumerate(slots):
        b.append(s, timestamp=t)
    # Query with the original slot — top-1 should be itself
    for t in range(16):
        out = b.recall_by_similarity(slots[t], k=1)
        assert len(out) == 1
        assert out[0].timestamp == t


def test_buffer_recall_time_range_inclusive():
    b = EpisodicBuffer(capacity=8, slot_dim=2)
    for t in [0, 5, 10, 15, 20]:
        b.append(torch.randn(2), timestamp=t)
    out = b.recall_time_range(5, 15)
    assert [r.timestamp for r in out] == [5, 10, 15]
    # Boundary: empty range returns []
    assert b.recall_time_range(100, 200) == []


def test_buffer_recall_most_recent_order():
    b = EpisodicBuffer(capacity=8, slot_dim=2)
    for t in range(7):
        b.append(torch.randn(2), timestamp=t * 10)
    out = b.recall_most_recent(3)
    assert [r.timestamp for r in out] == [40, 50, 60]


def test_buffer_recall_most_salient():
    b = EpisodicBuffer(capacity=8, slot_dim=2)
    for t in range(5):
        b.append(torch.randn(2), timestamp=t, salience=float(t))
    out = b.recall_most_salient(2)
    assert {r.salience for r in out} == {3.0, 4.0}


def test_buffer_metadata_round_trip():
    b = EpisodicBuffer(capacity=4, slot_dim=2)
    b.append(torch.randn(2), timestamp=7,
             metadata={"event": "test", "k": 42})
    out = b.recall_most_recent(1)
    assert out[0].metadata["event"] == "test"
    assert out[0].metadata["k"] == 42


def test_buffer_rejects_wrong_slot_shape():
    b = EpisodicBuffer(capacity=2, slot_dim=4)
    try:
        b.append(torch.randn(3), timestamp=0)
    except ValueError:
        return
    assert False, "expected ValueError"


def test_buffer_records_order_after_wrap():
    """After wrap-around, records() returns in insertion order
    (oldest first, most recent last)."""
    b = EpisodicBuffer(capacity=4, slot_dim=2)
    for t in range(7):
        b.append(torch.randn(2), timestamp=t)
    ts = [r.timestamp for r in b.records()]
    assert ts == [3, 4, 5, 6]


# ─────────────────────────────────────────────────────────────────
# LongTermEpisodicTrace
# ─────────────────────────────────────────────────────────────────


def test_long_term_salience_gate_rejects_low():
    t = LongTermEpisodicTrace(
        capacity=4, slot_dim=2, salience_threshold=2.0,
    )
    accepted = t.maybe_imprint(torch.randn(2), timestamp=0, salience=0.5)
    assert not accepted
    assert len(t) == 0
    assert t.writes_attempted == 1
    assert t.writes_committed == 0


def test_long_term_salience_gate_accepts_high():
    t = LongTermEpisodicTrace(
        capacity=4, slot_dim=2, salience_threshold=2.0,
    )
    assert t.maybe_imprint(torch.randn(2), timestamp=0, salience=5.0)
    assert len(t) == 1


def test_long_term_eviction_by_lowest_salience():
    t = LongTermEpisodicTrace(
        capacity=3, slot_dim=2, salience_threshold=1.0,
    )
    t.maybe_imprint(torch.randn(2), timestamp=0, salience=5.0)
    t.maybe_imprint(torch.randn(2), timestamp=1, salience=3.0)
    t.maybe_imprint(torch.randn(2), timestamp=2, salience=7.0)
    assert len(t) == 3
    # Full; a new event with salience=4 should evict the
    # salience=3 entry (the lowest)
    assert t.maybe_imprint(torch.randn(2), timestamp=3, salience=4.0)
    saliences = sorted([r.salience for r in t.records()])
    assert saliences == [4.0, 5.0, 7.0]
    # A new event with salience=2 should be rejected (lower than min)
    assert not t.maybe_imprint(torch.randn(2), timestamp=4, salience=2.0)


def test_long_term_biased_recall_prefers_salient():
    """When two stored episodes have similar slot but different
    saliences, biased recall prefers the higher-salience one."""
    torch.manual_seed(42)
    t = LongTermEpisodicTrace(
        capacity=4, slot_dim=4, salience_threshold=0.0,
    )
    s1 = F.normalize(torch.tensor([1.0, 0.0, 0.0, 0.0]), dim=-1)
    s2 = F.normalize(torch.tensor([1.0, 0.05, 0.0, 0.0]), dim=-1)
    # s1 has high salience, s2 has low salience
    t.maybe_imprint(s1, timestamp=0, salience=10.0,
                     metadata={"label": "salient"})
    t.maybe_imprint(s2, timestamp=1, salience=0.1,
                     metadata={"label": "bland"})
    # Query with a slot closer to s2 but biased recall should
    # still rank s1 high due to salience
    query = s2.clone()
    out = t.biased_recall(query, k=1, salience_weight=2.0)
    assert out[0].metadata["label"] == "salient"
    # Sanity: without salience bias, plain similarity would pick s2
    out_no_bias = t.biased_recall(query, k=1, salience_weight=0.0)
    assert out_no_bias[0].metadata["label"] == "bland"


# ─────────────────────────────────────────────────────────────────
# Consolidation
# ─────────────────────────────────────────────────────────────────


def test_consolidate_recovers_known_clusters():
    """K-means with restarts should recover well-separated
    clusters at high mean cosine."""
    torch.manual_seed(0)
    n_clusters = 3
    points = 40
    noise = 0.1
    centroids = F.normalize(torch.eye(n_clusters, 8), dim=-1)
    buf = EpisodicBuffer(capacity=n_clusters * points, slot_dim=8)
    for c in range(n_clusters):
        for _ in range(points):
            buf.append(centroids[c] + noise * torch.randn(8),
                        timestamp=0)
    res = consolidate_to_concept_graph(
        buf, n_clusters=n_clusters, n_restarts=5,
    )
    pred = F.normalize(res["centroids"], dim=-1)
    cos = pred @ centroids.t()
    # Greedy match
    matched = []
    used_t = set()
    used_p = set()
    for _ in range(n_clusters):
        best_v = -2.0
        best_p, best_t = None, None
        for i in range(n_clusters):
            if i in used_p:
                continue
            for j in range(n_clusters):
                if j in used_t:
                    continue
                if float(cos[i, j]) > best_v:
                    best_v = float(cos[i, j])
                    best_p, best_t = i, j
        matched.append(best_v)
        used_p.add(best_p)
        used_t.add(best_t)
    assert min(matched) >= 0.9, f"matched cos {matched} too low"


def test_consolidate_empty_buffer_raises():
    buf = EpisodicBuffer(capacity=4, slot_dim=2)
    try:
        consolidate_to_concept_graph(buf, n_clusters=2)
    except ValueError:
        return
    assert False, "expected ValueError on empty buffer"
