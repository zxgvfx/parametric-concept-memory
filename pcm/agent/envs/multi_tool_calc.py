"""pcm.agent.envs.multi_tool_calc — multi-tool calculator with
binary continuous-arg operations.

Extends F66's IntegerCalcEnv to test the v6 agent base on a
richer, more *agent-realistic* tool interface:

* **6 tools** (vs F66's 3), spanning all three F66 axes —
  nullary, unary-continuous, *and* binary-continuous.
* **One genuinely 2-arg tool** (``LERP(α, β)``) that the agent
  must learn to drive with two scalar arguments simultaneously.
* **Tool redundancy** — several tools can solve the same
  sub-problem (e.g. NEG vs LERP(-1, 0); HALVE vs LERP(0.5, 0))
  forcing the agent to pick among equivalent options.

State: integer ``s ∈ [-S, S]`` (S=20 → 41 states).

Tool set (6 tools, action_dim ≤ 2):

* 0 ``SET(arg)``       — ``s := round(arg · S)``           [unary]
* 1 ``ADD_K(arg)``     — ``s := clamp(s + round(arg · S/2))`` [unary]
* 2 ``NEG``            — ``s := -s``                       [nullary]
* 3 ``HALVE``          — ``s := s // 2``                   [nullary]
* 4 ``DOUBLE``         — ``s := clamp(s · 2)``             [nullary]
* 5 ``LERP(α, β)``     — ``s := clamp(round(α·s + β·S))``  [**binary**]

The arg representation is a fixed-length vector of length
``arg_dim = 2``: arg[0] is the first arg, arg[1] is the second
(used only by LERP); for unary tools, arg[1] is ignored.

BFS oracle: ``bfs_optimal_action`` returns the next
``(tool_id, arg_vector)`` along a shortest path with arg
quantised to ``arg_grid_per_dim`` values per dimension. For
LERP this gives ``arg_grid² = 1681`` possible argument pairs
(at ``arg_grid = 41``), still tractable for BFS at S=20.
"""
from __future__ import annotations

import math
from collections import deque
from functools import lru_cache


__all__ = [
    "TOOLS",
    "TOOL_ARITY",
    "ARG_DIM",
    "MultiToolCalcEnv",
    "apply_tool",
    "tool_arity",
    "bfs_optimal_action",
    "bfs_optimal_steps",
]


# Tool name table + arity (number of continuous args used).
TOOLS = ("SET", "ADD_K", "NEG", "HALVE", "DOUBLE", "LERP")
TOOL_ARITY = (1, 1, 0, 0, 0, 2)
ARG_DIM = 2  # max number of continuous args per tool


def tool_arity(tool_id: int) -> int:
    return TOOL_ARITY[tool_id]


def apply_tool(s: int, tool_id: int, args, *, S: int) -> int:
    """Apply tool to integer state ``s``. ``args`` is a sequence
    of length ``ARG_DIM`` (extras ignored for low-arity tools)."""
    a0 = max(-1.0, min(1.0, float(args[0]))) if len(args) > 0 else 0.0
    a1 = max(-1.0, min(1.0, float(args[1]))) if len(args) > 1 else 0.0
    if tool_id == 0:  # SET
        s_new = int(round(a0 * S))
    elif tool_id == 1:  # ADD_K
        s_new = s + int(round(a0 * S / 2))
    elif tool_id == 2:  # NEG
        s_new = -s
    elif tool_id == 3:  # HALVE
        s_new = s // 2
    elif tool_id == 4:  # DOUBLE
        s_new = s * 2
    elif tool_id == 5:  # LERP
        s_new = int(round(a0 * s + a1 * S))
    else:
        raise ValueError(f"unknown tool_id {tool_id}")
    return max(-S, min(S, int(s_new)))


class MultiToolCalcEnv:
    """Multi-tool calculator on integer state in ``[-S, S]``."""

    def __init__(
        self, S: int = 20, *,
        goal: int = 0, max_steps: int = 12,
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
        self, tool_id: int, args,
    ) -> tuple[int, float, bool]:
        self.state = apply_tool(self.state, tool_id, args, S=self.S)
        self.t += 1
        reached = self.state == self.goal
        done = reached or self.t >= self.max_steps
        reward = 1.0 if reached else 0.0
        return self.state_to_idx(self.state), reward, done


# ─────────────────────────────────────────────────────────────────
# BFS oracle (quantised args)
# ─────────────────────────────────────────────────────────────────


def _default_arg_grid(S: int) -> int:
    return 2 * S + 1


def _enumerate_actions(S: int, arg_grid: int):
    """All (tool_id, args_tuple) pairs with arg components on a
    uniform grid in [-1, 1]. Skips no-ops (action that leaves
    every state unchanged) heuristically by not enumerating
    duplicate arg vectors for nullary tools.
    """
    actions: list[tuple[int, tuple[float, float]]] = []
    # Unary tools: SET, ADD_K — sweep arg[0] only
    for i in range(arg_grid):
        a0 = -1.0 + 2.0 * i / max(arg_grid - 1, 1)
        actions.append((0, (a0, 0.0)))                # SET
        actions.append((1, (a0, 0.0)))                # ADD_K
    # Nullary tools: NEG, HALVE, DOUBLE — single arg vector each
    actions.append((2, (0.0, 0.0)))   # NEG
    actions.append((3, (0.0, 0.0)))   # HALVE
    actions.append((4, (0.0, 0.0)))   # DOUBLE
    # Binary tool: LERP — sweep arg[0] × arg[1] but coarsen to
    # avoid combinatorial blow-up. For S=20 we use 9×9=81 grid.
    coarse = max(5, arg_grid // 5)
    for i in range(coarse):
        a0 = -1.0 + 2.0 * i / max(coarse - 1, 1)
        for j in range(coarse):
            a1 = -1.0 + 2.0 * j / max(coarse - 1, 1)
            actions.append((5, (a0, a1)))
    return actions


@lru_cache(maxsize=64)
def _bfs_cached(
    goal: int, S: int, arg_grid: int, max_steps: int,
) -> tuple:
    """Cached BFS table; returns tuple-of-tuples ``(state, dist,
    tool_id, a0, a1)``."""
    actions = _enumerate_actions(S, arg_grid)
    successors = {
        s: [apply_tool(s, t, a, S=S) for (t, a) in actions]
        for s in range(-S, S + 1)
    }
    predecessors = {s: [] for s in range(-S, S + 1)}
    for s in range(-S, S + 1):
        for s_next in successors[s]:
            if s_next != s:
                predecessors[s_next].append(s)
    dist = {goal: 0}
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
    out = []
    for s in range(-S, S + 1):
        if s == goal:
            out.append((s, 0, 0, 0.0, 0.0))
            continue
        if s not in dist:
            continue
        best = None
        best_dist = max_steps + 1
        for (t, a) in actions:
            s_next = apply_tool(s, t, a, S=S)
            if s_next == s:
                continue
            d = dist.get(s_next, max_steps + 1)
            if d < best_dist:
                best_dist = d
                best = (t, a)
        if best is not None:
            out.append((s, dist[s], best[0], best[1][0], best[1][1]))
    return tuple(out)


def _table_for(
    goal: int, S: int, arg_grid: int | None, max_steps: int,
) -> dict[int, tuple[int, int, float, float]]:
    if arg_grid is None:
        arg_grid = _default_arg_grid(S)
    rows = _bfs_cached(goal, S, arg_grid, max_steps)
    return {s: (d, t, a0, a1) for (s, d, t, a0, a1) in rows}


def bfs_optimal_action(
    state: int, goal: int, S: int, *,
    arg_grid: int | None = None, max_steps: int = 12,
) -> tuple[int, tuple[float, float]]:
    """Return next ``(tool_id, args_tuple)`` along shortest path."""
    table = _table_for(goal, S, arg_grid, max_steps)
    if state == goal:
        return 0, (0.0, 0.0)
    if state not in table:
        return 0, (0.0, 0.0)
    _, t, a0, a1 = table[state]
    return t, (a0, a1)


def bfs_optimal_steps(
    state: int, goal: int, S: int, *,
    arg_grid: int | None = None, max_steps: int = 12,
) -> int:
    table = _table_for(goal, S, arg_grid, max_steps)
    if state == goal:
        return 0
    if state not in table:
        return max_steps
    return table[state][0]
