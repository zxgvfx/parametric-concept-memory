"""pcm.agent.envs.cyclic_nav — 1-D cyclic navigation toy.

State space: ``ℤ_N`` (N integers arranged in a ring).
Action space: ``{+1, -1, +5, -5}`` modulo N.
Episode: agent picks actions one at a time. Episode terminates
when the agent reaches the goal state, or when ``max_steps`` is
exhausted. Reward is +1 on success, 0 elsewhere.

This is the smallest non-trivial agentic env and is structurally
identical to the F62 cyclic-group ℤ_N task plus a *policy* layer
(action selection). If F64 succeeds on this env, the F62
universal operator carries over to active-interaction settings
without architectural change.

Optimal trajectory:
    Δ = (goal - state) mod N, signed to lie in ``(-N/2, N/2]``.
    Repeatedly take the largest-magnitude action that does not
    overshoot, until ``state == goal``.
"""
from __future__ import annotations

from typing import Sequence


__all__ = ["ACTION_DELTAS", "CyclicNavEnv", "optimal_action",
           "optimal_trajectory"]


# Action set: index → integer displacement on ℤ_N
# 0: +1   1: -1   2: +5   3: -5
ACTION_DELTAS: tuple[int, ...] = (+1, -1, +5, -5)


class CyclicNavEnv:
    """1-D cyclic navigation environment on ℤ_N.

    Args:
        N: ring size.
        goal: target state index. Goal is set externally for each
            episode (so the same env instance can serve different
            goal distributions).
        max_steps: hard budget per episode.
        action_deltas: per-action integer displacements; defaults
            to ``ACTION_DELTAS`` (``(+1, -1, +5, -5)``).
    """

    def __init__(
        self, N: int = 20, *,
        goal: int = 0,
        max_steps: int = 32,
        action_deltas: Sequence[int] = ACTION_DELTAS,
    ) -> None:
        if N <= 0:
            raise ValueError(f"N must be positive, got {N}")
        self.N = N
        self.goal = int(goal) % N
        self.max_steps = max_steps
        self.action_deltas = tuple(action_deltas)
        self.state: int = 0
        self.t: int = 0

    @property
    def n_actions(self) -> int:
        return len(self.action_deltas)

    def set_goal(self, goal: int) -> None:
        self.goal = int(goal) % self.N

    def reset(self, state: int | None = None) -> int:
        """Reset env to ``state`` (or 0 by default). Returns the
        initial state index."""
        self.state = (int(state) if state is not None else 0) % self.N
        self.t = 0
        return self.state

    def step(self, action_idx: int) -> tuple[int, float, bool]:
        """Apply ``action_idx``, return ``(next_state, reward,
        done)``. ``reward = 1.0`` iff the agent reaches the goal
        on this step."""
        if not 0 <= action_idx < len(self.action_deltas):
            raise ValueError(
                f"action {action_idx} out of range "
                f"[0, {len(self.action_deltas)})"
            )
        delta = self.action_deltas[action_idx]
        self.state = (self.state + delta) % self.N
        self.t += 1
        reached = self.state == self.goal
        done = reached or self.t >= self.max_steps
        reward = 1.0 if reached else 0.0
        return self.state, reward, done


def optimal_action(
    state: int, goal: int, N: int,
    action_deltas: Sequence[int] = ACTION_DELTAS,
) -> int:
    """Return the index of the optimal next action.

    Strategy: compute signed displacement
    ``Δ = (goal - state + N//2) % N - N//2``, take whichever
    action has the largest magnitude that does not overshoot
    ``Δ`` (so we never increase ``|Δ|`` after the action). Ties
    are broken by action index for determinism.
    """
    if state == goal:
        # Convention: when already at goal, pick action 0; the env
        # will time out without giving extra reward but the
        # optimal label is well-defined.
        return 0
    half = N // 2
    delta_signed = (goal - state + half) % N - half  # in (-N/2, N/2]
    # We want to *reduce* |delta_signed|. An action with
    # ``a_delta`` reduces |Δ| iff sign(a_delta) == sign(Δ) AND
    # ``|a_delta| ≤ |Δ|`` (no overshoot).
    best_idx = 0
    best_progress = -1
    for idx, a_delta in enumerate(action_deltas):
        if a_delta == 0:
            continue
        if (a_delta > 0) != (delta_signed > 0):
            continue  # wrong direction
        if abs(a_delta) > abs(delta_signed):
            continue  # would overshoot
        if abs(a_delta) > best_progress:
            best_progress = abs(a_delta)
            best_idx = idx
    return best_idx


def optimal_trajectory(
    start: int, goal: int, N: int,
    action_deltas: Sequence[int] = ACTION_DELTAS,
    max_steps: int = 32,
) -> list[int]:
    """Generate the greedy oracle action sequence from ``start``
    to ``goal`` on ℤ_N. Returns the list of action indices.

    NB: this is the *greedy* oracle (always pick the largest
    non-overshooting action). For some action sets greedy is
    not globally optimal — e.g. for ``{±1, ±5}`` on ``ℤ_20`` the
    greedy 0→9 takes 5 steps (5+1+1+1+1) while BFS finds
    3 steps (5+5-1). Use :func:`shortest_path_length` for
    BFS-true-shortest distance.
    """
    state = start % N
    goal = goal % N
    actions: list[int] = []
    for _ in range(max_steps):
        if state == goal:
            break
        a = optimal_action(state, goal, N, action_deltas)
        actions.append(a)
        state = (state + action_deltas[a]) % N
    return actions


def shortest_path_length(
    start: int, goal: int, N: int,
    action_deltas: Sequence[int] = ACTION_DELTAS,
    max_steps: int = 64,
) -> int:
    """BFS true shortest-path length from ``start`` to ``goal``
    on ``ℤ_N`` under ``action_deltas``. Used by F64's U6
    multi-step-horizon evaluation so the ratio "agent steps /
    optimal steps" is interpretable."""
    return _bfs_distances(goal, N, action_deltas, max_steps).get(
        start % N, max_steps,
    )


def _bfs_distances(
    goal: int, N: int,
    action_deltas: Sequence[int] = ACTION_DELTAS,
    max_steps: int = 64,
) -> dict[int, int]:
    """Backwards BFS from ``goal``: return dict mapping each
    reachable state to its shortest-path distance to ``goal``.

    We BFS *backwards* — from each state ``s``, predecessors are
    ``s - a`` for each action delta ``a``. This lets a single BFS
    serve queries from every starting state to a fixed ``goal``.
    """
    goal = goal % N
    dist: dict[int, int] = {goal: 0}
    frontier = [goal]
    while frontier:
        next_frontier = []
        for s in frontier:
            d = dist[s]
            if d >= max_steps:
                continue
            for a in action_deltas:
                pred = (s - a) % N
                if pred not in dist or dist[pred] > d + 1:
                    dist[pred] = d + 1
                    next_frontier.append(pred)
        frontier = next_frontier
    return dist


def bfs_optimal_action(
    state: int, goal: int, N: int,
    action_deltas: Sequence[int] = ACTION_DELTAS,
    max_steps: int = 64,
) -> int:
    """BFS-true-optimal next action from ``state`` toward
    ``goal``. Picks the action whose successor has the smallest
    distance-to-goal (ties broken by action index for
    determinism).
    """
    state = state % N
    goal = goal % N
    if state == goal:
        return 0
    dist = _bfs_distances(goal, N, action_deltas, max_steps)
    best_idx = 0
    best_dist = max_steps
    for idx, a in enumerate(action_deltas):
        succ = (state + a) % N
        d = dist.get(succ, max_steps)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    return best_idx
