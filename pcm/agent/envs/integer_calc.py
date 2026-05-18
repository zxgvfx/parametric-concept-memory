"""pcm.agent.envs.integer_calc — mixed-arity tool-call calculator.

Performance note: ``bfs_optimal_action`` / ``bfs_optimal_steps``
cache the full BFS distance table per ``(goal, S, arg_grid,
max_steps)`` tuple via ``functools.lru_cache``. This is critical
for BC training where the same oracle is called O(10^5) times.

Environment for F66 (PCM v6.2-followup) PoC. Tests typed-arg tool
calls: one tool consumes a continuous argument, two tools are
nullary.

State space: integer ``s ∈ [-S, S]`` (default S=20, so 41 states).
Goal: integer ``g`` in the same range.

Tool set (3 tools):

* 0 ``ADD_K(arg)`` — ``s := s + round(arg · S)``, clipped to
  ``[-S, S]``. ``arg ∈ [-1, 1]`` continuous; effective integer
  shift is in ``[-S, S]``.
* 1 ``NEG``         — ``s := -s`` (nullary).
* 2 ``HALVE``       — ``s := s // 2`` (Python floor division;
  nullary).

Reward / termination: ``+1`` and ``done`` when ``s == g``;
``done`` also when budget exhausted.

Oracle (``optimal_action``): BFS over the state graph with arg
quantised to a small set ``{-S/2, …, S/2}`` (so the action space
for BFS is finite). Returns the next ``(tool_id, arg)`` along
the shortest path; if multiple shortest paths exist, picks the
lowest tool index, then the smallest |arg|.
"""
from __future__ import annotations

import math
from collections import deque
from functools import lru_cache


__all__ = [
    "TOOLS",
    "IntegerCalcEnv",
    "tool_takes_arg",
    "apply_tool",
    "bfs_optimal_action",
    "bfs_optimal_steps",
]


# Tool name table (informational only — env uses indices).
TOOLS = ("ADD_K", "NEG", "HALVE")


def tool_takes_arg(tool_id: int) -> bool:
    """Whether ``tool_id`` consumes a continuous arg."""
    return tool_id == 0


def apply_tool(
    s: int, tool_id: int, arg: float, *, S: int,
) -> int:
    """Apply tool to state ``s`` and return the new state.

    Args:
        s: current state, integer in ``[-S, S]``.
        tool_id: 0/1/2 — ADD_K / NEG / HALVE.
        arg: continuous in ``[-1, 1]``; ignored for nullary tools.
        S: state range half-width.
    """
    if tool_id == 0:  # ADD_K
        shift = int(round(max(-1.0, min(1.0, float(arg))) * S))
        s_new = s + shift
    elif tool_id == 1:  # NEG
        s_new = -s
    elif tool_id == 2:  # HALVE
        s_new = s // 2
    else:
        raise ValueError(f"unknown tool_id {tool_id}")
    return max(-S, min(S, int(s_new)))


class IntegerCalcEnv:
    """Mixed-arity calculator environment on integer state.

    Args:
        S: half-width of the state range. State and goal lie in
            ``[-S, S]``.
        goal: target integer (set externally per episode).
        max_steps: hard episode budget.
    """

    def __init__(
        self, S: int = 20, *,
        goal: int = 0, max_steps: int = 16,
    ) -> None:
        if S <= 0:
            raise ValueError(f"S must be positive, got {S}")
        self.S = S
        self.goal = max(-S, min(S, int(goal)))
        self.max_steps = max_steps
        self.state: int = 0
        self.t: int = 0

    @property
    def n_tools(self) -> int:
        return len(TOOLS)

    @property
    def n_states(self) -> int:
        return 2 * self.S + 1

    def state_to_idx(self, s: int) -> int:
        """Map signed state ``s ∈ [-S, S]`` to ``[0, 2S+1)``
        index for use as a slot-bundle key."""
        return int(s) + self.S

    def idx_to_state(self, idx: int) -> int:
        return int(idx) - self.S

    def set_goal(self, goal: int) -> None:
        self.goal = max(-self.S, min(self.S, int(goal)))

    def reset(self, state: int | None = None) -> int:
        self.state = (
            max(-self.S, min(self.S, int(state)))
            if state is not None else 0
        )
        self.t = 0
        return self.state_to_idx(self.state)

    def step(
        self, tool_id: int, arg: float,
    ) -> tuple[int, float, bool]:
        """Apply ``(tool_id, arg)``; return ``(next_state_idx,
        reward, done)``. Reward is +1 iff the new state equals
        the goal."""
        self.state = apply_tool(
            self.state, tool_id, arg, S=self.S,
        )
        self.t += 1
        reached = self.state == self.goal
        done = reached or self.t >= self.max_steps
        reward = 1.0 if reached else 0.0
        return self.state_to_idx(self.state), reward, done


# ─────────────────────────────────────────────────────────────────
# BFS oracle (quantised arg)
# ─────────────────────────────────────────────────────────────────


def _enumerate_actions(S: int, arg_grid: int):
    """Enumerate ``(tool_id, arg)`` pairs for BFS.

    ``arg_grid`` is the number of distinct arg values used for
    ADD_K. We sample uniformly from ``[-1, 1]`` with ``arg_grid``
    points (including endpoints). For nullary tools, arg = 0.
    """
    actions: list[tuple[int, float]] = []
    # ADD_K with quantised args. Skip arg=0 (no-op).
    for i in range(arg_grid):
        arg = -1.0 + 2.0 * i / max(arg_grid - 1, 1)
        shift = int(round(arg * S))
        if shift == 0:
            continue
        actions.append((0, arg))
    actions.append((1, 0.0))   # NEG
    actions.append((2, 0.0))   # HALVE
    return actions


def _default_arg_grid(S: int) -> int:
    """Choose ``arg_grid`` so ``arg · S`` lands on every integer
    shift in ``[-S, S]`` exactly. With ``arg_grid = 2S + 1`` the
    quantised args are ``{-1, -1+1/S, …, 0, …, 1}``."""
    return 2 * S + 1


@lru_cache(maxsize=512)
def _bfs_cached(
    goal: int, S: int, arg_grid: int, max_steps: int,
) -> tuple[tuple[int, int, int, float], ...]:
    """Cached BFS — returns a tuple-of-tuples ``(state, dist,
    tool_id, arg)`` for every reachable state, suitable as a dict
    key. Hashable types throughout."""
    table = _bfs(goal, S, arg_grid=arg_grid, max_steps=max_steps)
    return tuple(
        (s, d, t, a) for s, (d, t, a) in table.items()
    )


def _bfs(
    goal: int, S: int, *, arg_grid: int | None = None,
    max_steps: int = 16,
) -> dict[int, tuple[int, int, float]]:
    """Backwards BFS from ``goal``: for every reachable state
    ``s``, return ``(distance, tool_id, arg)`` — the *first*
    action along a shortest path from ``s`` to ``goal``.

    We BFS forwards from ``goal`` along *reverse* transitions, but
    HALVE / NEG are not symmetric in this env so we instead BFS
    forwards from every starting state. The trick: build the
    distance table by BFS forwards from the goal under forward
    transitions, then for each state we pick the action whose
    next-state has the smallest distance-to-goal.
    """
    if arg_grid is None:
        arg_grid = _default_arg_grid(S)
    actions = _enumerate_actions(S, arg_grid)

    # Step 1: compute forward-reachable distances *to* goal by
    # multi-source BFS where we expand backwards: starting from
    # the set ``{goal}``, repeatedly find all states ``s`` that
    # can reach an already-known state in one step.
    dist: dict[int, int] = {goal: 0}
    # Build predecessor table: for each (state, action) compute
    # next-state. Then for each state s' in the frontier, its
    # predecessors are all s such that apply_tool(s, action) = s'.
    successors: dict[int, list[int]] = {
        s: [apply_tool(s, t, a, S=S) for (t, a) in actions]
        for s in range(-S, S + 1)
    }
    # Invert to predecessor table
    predecessors: dict[int, list[int]] = {
        s: [] for s in range(-S, S + 1)
    }
    for s in range(-S, S + 1):
        for s_next in successors[s]:
            if s != s_next:
                predecessors[s_next].append(s)
    frontier = deque([goal])
    while frontier:
        s_next = frontier.popleft()
        d_next = dist[s_next]
        if d_next >= max_steps:
            continue
        for s_pred in predecessors[s_next]:
            if s_pred not in dist:
                dist[s_pred] = d_next + 1
                frontier.append(s_pred)

    # Step 2: for each state, pick the next-action whose
    # successor has the smallest distance-to-goal.
    out: dict[int, tuple[int, int, float]] = {}
    for s in range(-S, S + 1):
        if s == goal:
            out[s] = (0, 0, 0.0)
            continue
        if s not in dist:
            continue
        best = None
        best_succ_dist = max_steps + 1
        for (t, a) in actions:
            s_next = apply_tool(s, t, a, S=S)
            if s_next == s:
                continue
            d_succ = dist.get(s_next, max_steps + 1)
            if d_succ < best_succ_dist:
                best_succ_dist = d_succ
                best = (t, a)
        if best is not None:
            out[s] = (dist[s], best[0], best[1])
    return out


def _table_for(
    goal: int, S: int, arg_grid: int | None, max_steps: int,
) -> dict[int, tuple[int, int, float]]:
    """Public wrapper around the cached BFS table."""
    if arg_grid is None:
        arg_grid = _default_arg_grid(S)
    rows = _bfs_cached(goal, S, arg_grid, max_steps)
    return {s: (d, t, a) for (s, d, t, a) in rows}


def bfs_optimal_action(
    state: int, goal: int, S: int, *,
    arg_grid: int | None = None, max_steps: int = 16,
) -> tuple[int, float]:
    """Return the optimal next ``(tool_id, arg)`` from ``state``
    to ``goal`` under BFS with quantised args.

    If the goal is unreachable within ``max_steps`` (shouldn't
    happen for the F66 default config: ``S=20`` is fully
    connected in ≤ 6 steps with the given tool set), returns
    ``(0, 0.0)`` (a no-op ADD_K with arg=0).

    Uses a process-wide LRU cache (``_bfs_cached``) so repeated
    calls with the same ``(goal, S, arg_grid, max_steps)`` reuse
    the precomputed table — essential for BC training speed.
    """
    table = _table_for(goal, S, arg_grid, max_steps)
    if state == goal:
        return 0, 0.0
    if state not in table:
        return 0, 0.0
    _, t, a = table[state]
    return t, a


def bfs_optimal_steps(
    state: int, goal: int, S: int, *,
    arg_grid: int | None = None, max_steps: int = 16,
) -> int:
    table = _table_for(goal, S, arg_grid, max_steps)
    if state == goal:
        return 0
    if state not in table:
        return max_steps
    return table[state][0]
