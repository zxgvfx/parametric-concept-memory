"""pcm.agent.envs.continuous_nav — continuous-action S¹ navigation.

Environment for F65 (PCM v6.2) PoC. Mirrors
``cyclic_nav.CyclicNavEnv`` but with a **continuous scalar
action**: at each step the agent commits a scalar
``a ∈ [-1, 1]``, the env advances by a fraction of a fixed
maximum step size in that direction.

* state space: ``ℤ_N`` — the discretisation of S¹ used for
  goal-success matching and for the slot bundle in F65.
* action space: ``a ∈ [-1, 1]`` continuous scalar (clipped).
* dynamics: ``θ_next = (θ + a · max_step) mod 2π`` where
  ``max_step`` is the env's per-step angular budget.
* success: bin-quantised ``θ_next`` equals ``goal`` index.

This is the smallest possible test of the F65 claim: the F62
universal operator ``UniversalCombiner(slot, RoPE(action))``
handles continuous scalar actions on an S¹ state space, the
*continuous-action mirror* of F62c's continuous Lie-group
operator on S¹.

The companion oracle ``optimal_continuous_action`` returns the
optimal continuous action a single step out from any state under
greedy "advance toward goal as fast as possible without
overshooting" (used as BC labels). Multi-step optimal step counts
are computed by the same greedy rule (this env's action set is
amply expressive — every state is reachable in
``ceil(π / max_step)`` steps — so greedy is BFS-optimal).
"""
from __future__ import annotations

import math


__all__ = [
    "ContinuousCyclicNavEnv",
    "optimal_continuous_action",
    "optimal_continuous_steps",
]


class ContinuousCyclicNavEnv:
    """1-D cyclic navigation env with a continuous scalar action.

    Args:
        N: state-bin count (S¹ discretisation).
        max_step: maximum per-step displacement in *radians*. The
            agent's continuous action ``a ∈ [-1, 1]`` is mapped
            to a displacement of ``a · max_step`` rad.
        goal: target state-bin index (set externally per episode).
        max_steps: hard episode budget.
    """

    def __init__(
        self, N: int = 100, *,
        max_step: float = math.pi / 4,
        goal: int = 0,
        max_steps: int = 32,
    ) -> None:
        if N <= 0:
            raise ValueError(f"N must be positive, got {N}")
        self.N = N
        self.max_step = float(max_step)
        self.goal = int(goal) % N
        self.max_steps = max_steps
        self.bin_size = 2 * math.pi / N
        self.theta: float = 0.0
        self.t: int = 0

    @property
    def state(self) -> int:
        """Current state bin index (closest bin to ``theta``)."""
        return int(round(self.theta / self.bin_size)) % self.N

    def set_goal(self, goal: int) -> None:
        self.goal = int(goal) % self.N

    def reset(self, state: int | None = None) -> int:
        """Reset the env to ``state`` (or 0). The internal
        continuous angle is set to the bin's centre. Returns the
        bin index."""
        idx = (int(state) if state is not None else 0) % self.N
        self.theta = idx * self.bin_size
        self.t = 0
        return self.state

    def step(self, action: float) -> tuple[int, float, bool]:
        """Apply continuous ``action`` (clipped to ``[-1, 1]``).

        Returns ``(next_state_idx, reward, done)``. Reward is
        +1.0 iff the bin-quantised next state equals the goal.
        """
        a = max(-1.0, min(1.0, float(action)))
        self.theta = (self.theta + a * self.max_step) % (2 * math.pi)
        self.t += 1
        s_next = self.state
        reached = s_next == self.goal
        done = reached or self.t >= self.max_steps
        reward = 1.0 if reached else 0.0
        return s_next, reward, done


def _signed_delta(state_idx: int, goal_idx: int, N: int) -> float:
    """Signed angular distance from ``state`` to ``goal``, in
    radians, in ``(-π, π]``."""
    bin_size = 2 * math.pi / N
    theta_s = state_idx * bin_size
    theta_g = goal_idx * bin_size
    d = (theta_g - theta_s + math.pi) % (2 * math.pi) - math.pi
    return d


def optimal_continuous_action(
    state_idx: int, goal_idx: int, N: int,
    *, max_step: float = math.pi / 4,
) -> float:
    """Greedy oracle for the continuous-action env.

    If the goal is reachable within one max-step of the current
    bin, the optimal action is the *exact* fractional step that
    centres the next ``θ`` on the goal bin. Else, take a full
    step (±1) in the shortest direction.

    Returns the scalar in ``[-1, 1]``.
    """
    if state_idx == goal_idx:
        return 0.0
    delta = _signed_delta(state_idx, goal_idx, N)
    if abs(delta) <= max_step:
        return max(-1.0, min(1.0, delta / max_step))
    return 1.0 if delta > 0 else -1.0


def optimal_continuous_steps(
    state_idx: int, goal_idx: int, N: int,
    *, max_step: float = math.pi / 4,
) -> int:
    """Number of greedy-oracle steps to reach the goal.

    Equivalent to ``ceil(|signed_delta(state, goal)| / max_step)``
    rounded up to whole steps. The greedy oracle is BFS-optimal
    here because the action set is convex (any sub-step is
    available), so this is also the true minimum step count.
    """
    if state_idx == goal_idx:
        return 0
    delta = _signed_delta(state_idx, goal_idx, N)
    return max(1, math.ceil(abs(delta) / max_step))
