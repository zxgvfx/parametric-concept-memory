"""Unit tests for ``pcm.agent.episodic_agent`` (F78)."""
from __future__ import annotations

import torch

from pcm.agent import (
    AgentAttractorHead,
    EpisodicAgent,
    EpisodicAgentDecision,
    MultiArgActionTransitionHead,
    MultiArgPolicyHead,
    SlotStateEncoder,
)
from pcm.agent.envs import MTC_ARG_DIM


def _make_agent(*, S: int = 5, dim: int = 8,
                buffer_capacity: int = 20,
                recall_threshold: float = 0.95,
                p_success_threshold: float = 0.8) -> EpisodicAgent:
    torch.manual_seed(0)
    n_states = 2 * S + 1
    enc = SlotStateEncoder(n_states=n_states, dim=dim)
    trans = MultiArgActionTransitionHead(
        dim=dim, n_tools=6, arg_dim=MTC_ARG_DIM,
    )
    pol = MultiArgPolicyHead(
        dim=dim, n_tools=6, arg_dim=MTC_ARG_DIM,
    )
    attr = AgentAttractorHead(dim=dim)
    return EpisodicAgent(
        enc, trans, pol, attr,
        buffer_capacity=buffer_capacity,
        recall_threshold=recall_threshold,
        p_success_threshold=p_success_threshold,
    )


def test_empty_buffer_returns_s1_or_s2() -> None:
    agent = _make_agent()
    d = agent.decide(state_idx=2, goal_idx=7)
    assert isinstance(d, EpisodicAgentDecision)
    assert d.route in ("S1", "S2")
    assert d.recall_sim is None


def test_remember_writes_only_on_success() -> None:
    agent = _make_agent()
    written = agent.remember(2, 7, tool_id=1, args=(0.5, 0.0),
                             n_steps=3, success=False)
    assert not written
    assert len(agent.buffer) == 0

    written = agent.remember(2, 7, tool_id=1, args=(0.5, 0.0),
                             n_steps=3, success=True)
    assert written
    assert len(agent.buffer) == 1


def test_exact_revisit_triggers_recall() -> None:
    agent = _make_agent(recall_threshold=0.99)
    agent.remember(2, 7, tool_id=3, args=(0.0, 0.0),
                   n_steps=2, success=True)
    d = agent.decide(state_idx=2, goal_idx=7)
    assert d.route == "RECALL"
    assert d.tool_id == 3
    assert d.args == (0.0, 0.0)
    assert d.recall_sim is not None
    assert d.recall_sim > 0.99


def test_dissimilar_query_does_not_recall() -> None:
    agent = _make_agent(recall_threshold=0.99)
    agent.remember(0, 1, tool_id=2, args=(0.0, 0.0),
                   n_steps=1, success=True)
    d = agent.decide(state_idx=9, goal_idx=10)
    # The two queries are different (state and goal pair); the
    # encoder embeds them differently → cosine should be < 0.99
    # → recall should not fire.
    assert d.route in ("S1", "S2")


def test_recall_disabled_threshold_above_one() -> None:
    agent = _make_agent(recall_threshold=1.01)
    agent.remember(2, 7, tool_id=3, args=(0.0, 0.0),
                   n_steps=2, success=True)
    d = agent.decide(state_idx=2, goal_idx=7)
    assert d.route in ("S1", "S2")
    assert d.recall_sim is None


def test_fifo_eviction() -> None:
    agent = _make_agent(buffer_capacity=3)
    for i in range(5):
        agent.remember(i, i + 1, tool_id=0, args=(0.1 * i, 0.0),
                       n_steps=2, success=True)
    assert len(agent.buffer) == 3
    md = [r.metadata for r in agent.buffer.records()]
    state_idxs = [m["state_idx"] for m in md]
    assert sorted(state_idxs) == [2, 3, 4]


def test_diagnostics_count_routes() -> None:
    agent = _make_agent(recall_threshold=0.99)
    agent.remember(2, 7, tool_id=3, args=(0.0, 0.0),
                   n_steps=2, success=True)
    agent.decide(2, 7)
    agent.decide(0, 9)
    diag = agent.diagnostics()
    assert diag["buffer_size"] == 1
    assert diag["n_recalls"] >= 1
    assert diag["n_recalls"] + diag["n_s1"] + diag["n_s2"] == 2


def test_salience_inverse_steps() -> None:
    agent = _make_agent()
    agent.remember(0, 1, tool_id=0, args=(0.0, 0.0),
                   n_steps=1, success=True)
    agent.remember(0, 2, tool_id=0, args=(0.0, 0.0),
                   n_steps=5, success=True)
    records = agent.buffer.records()
    assert abs(records[0].salience - 1.0) < 1e-5
    assert abs(records[1].salience - 0.2) < 1e-5


def test_buffer_grows_with_distinct_episodes() -> None:
    agent = _make_agent(buffer_capacity=50)
    for s in range(0, 10):
        for g in range(0, 10):
            if s == g:
                continue
            agent.remember(s, g, tool_id=0, args=(0.0, 0.0),
                           n_steps=2, success=True)
    assert len(agent.buffer) == 50  # capped


def test_reset_counters_clears_route_counts() -> None:
    agent = _make_agent(recall_threshold=0.99)
    agent.remember(2, 7, tool_id=3, args=(0.0, 0.0),
                   n_steps=2, success=True)
    agent.decide(2, 7)
    agent.decide(2, 7)
    assert agent.n_recalls == 2
    agent.reset_counters()
    assert agent.n_recalls == 0
    assert agent.n_s1 == 0
    assert agent.n_s2 == 0
