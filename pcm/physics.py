"""pcm.physics — PCM v4 physics-as-procedural-cook architecture.

See ``docs/PCM_V4_PHYSICS_COOK_DESIGN.md`` for the full design
rationale and the four-line literature evidence chain
(IGNS / Causal-JEPA / Battaglia interaction nets / DMP).

Public API:

* :class:`PhysicsStateHead` — single-step state-space transition
  predictor. Takes (state, force) → Δstate. Trained on 1-step
  transitions; cook accumulates them into arbitrary-horizon
  trajectories.
* :class:`PhysicsCook` — iterative state-space rollout. Mirrors
  :class:`pcm.dual_process.IterativeDiffCook` but for continuous
  state ℝ^D rather than discrete ordinal indices.
* :class:`RolloutReport` — diagnostics for one cook rollout
  (n_iters, terminal state, wall_seconds).

The module is opt-in: importing it has no effect on existing v1,
v2, or v3 callers. Use it when:

* the task involves predicting a state at horizon K from
  (initial_state, force_sequence), AND
* the test horizon exceeds the training horizon (a v4-style
  length-OOD on continuous dynamics), AND
* the dynamics decompose into a 1-step transition function that
  the SuccessorHead-style head can learn from data.

For purely discrete tasks v3's :class:`IterativeDiffCook`
remains the right tool; for purely static (no horizon) tasks
the v2 dual-channel pair head suffices.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "PhysicsStateHead",
    "PhysicsCook",
    "RolloutReport",
    "physics_step_loss",
    "StateLookupHead",
    "distill_physics_cook_to_lookup",
    "PhysicsDistillReport",
]


# ---------------------------------------------------------------------------
# PhysicsStateHead — one-step transition predictor.
# ---------------------------------------------------------------------------


class PhysicsStateHead(nn.Module):
    """Predict the per-timestep state change given the current
    state and (optionally) an external force input.

    Output is interpreted as ``Δstate``, i.e. the head learns
    ``state_{t+1} = state_t + head(state_t, force_t)``. We make
    the residual form explicit because:

    1. It zero-initialises a sensible identity dynamic (Δ=0 →
       static state), letting training start from a stable
       prior.
    2. It mirrors the v3 :class:`pcm.dual_process.SuccessorHead`
       residual interpretation: head outputs a small step,
       cook accumulates them.
    3. The IGNS port-Hamiltonian construction (ICLR 2026
       arxiv 2511.08185) factorises the same way: predict ``Δq``
       and ``Δp`` separately so symplectic structure is preserved
       at the head level. We do not enforce symplectic structure
       at v4 PoC level; that is a follow-up.

    Args:
        state_dim: dimensionality D of the state vector.
        force_dim: dimensionality F of the external force input.
            0 disables force (autonomous dynamics).
        hidden: MLP hidden width.
        dt: simulation timestep, multiplied into the output. Lets
            the same head architecture handle different timesteps
            via a single hyper-parameter.
    """

    def __init__(
        self,
        state_dim: int,
        *,
        force_dim: int = 0,
        hidden: int = 64,
        dt: float = 1.0,
    ) -> None:
        super().__init__()
        if state_dim < 1:
            raise ValueError(f"state_dim must be >= 1, got {state_dim}")
        if force_dim < 0:
            raise ValueError(f"force_dim must be >= 0, got {force_dim}")
        self.state_dim = int(state_dim)
        self.force_dim = int(force_dim)
        self.dt = float(dt)
        in_dim = state_dim + force_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, state_dim)

    def forward(
        self,
        state: torch.Tensor,
        force: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns ``Δstate`` of shape ``(B, state_dim)``.

        Caller is expected to update ``state_{t+1} = state_t +
        Δstate`` and apply any domain-specific post-processing
        (clipping, wall reflection, etc.) before the next call.
        :class:`PhysicsCook` does this automatically.
        """
        if self.force_dim > 0:
            if force is None:
                raise ValueError(
                    "PhysicsStateHead with force_dim>0 requires a "
                    "force tensor; got None"
                )
            x = torch.cat([state, force], dim=-1)
        else:
            x = state
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        delta = self.fc3(h)
        return self.dt * delta


def physics_step_loss(
    pred_delta: torch.Tensor,
    true_next: torch.Tensor,
    state: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Convenience MSE loss for 1-step physics training.

    Args:
        pred_delta: head's predicted Δstate, shape (B, D).
        true_next: ground-truth next state, shape (B, D).
        state: current state, shape (B, D).
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns the MSE between (state + pred_delta) and true_next.
    """
    pred_next = state + pred_delta
    diff = (pred_next - true_next).pow(2).sum(dim=-1)
    if reduction == "sum":
        return diff.sum()
    if reduction == "none":
        return diff
    return diff.mean()


# ---------------------------------------------------------------------------
# PhysicsCook — iterative state-space rollout.
# ---------------------------------------------------------------------------


@dataclass
class RolloutReport:
    """Diagnostics for one :class:`PhysicsCook` rollout."""

    K: int
    wall_seconds: float
    final_state_norm: float
    diverged: bool = False
    step_history: list[float] = field(default_factory=list)


class PhysicsCook:
    """Iterative state-space rollout. Apply PhysicsStateHead K
    times starting from ``initial_state``; optionally clip state
    to a bounding box or reflect off walls.

    Args:
        head: trained :class:`PhysicsStateHead`.
        max_iters: hard upper bound on K (sanity guard).
        state_clip: optional ``(low, high)`` tensors of shape
            ``(state_dim,)`` clamping each dim. Pass ``None`` to
            disable.
        wall_reflect: when ``True`` AND ``state_clip`` is set,
            on any axis where the cursor exceeds the bound, flip
            the sign of the corresponding *velocity* dimension.
            Velocity dimensions are inferred as the second half
            of the state (D/2..D); pass a custom callable via
            ``custom_post_step`` for non-canonical layouts.
        custom_post_step: optional ``(state, K_so_far) → state``
            callable run after each step, before the next head
            call. Lets callers implement domain-specific
            constraints without subclassing.
        diverge_threshold: if any dim of state exceeds this in
            absolute value, mark ``RolloutReport.diverged=True``
            and break.
    """

    def __init__(
        self,
        head: PhysicsStateHead,
        *,
        max_iters: int = 1000,
        state_clip: tuple[torch.Tensor, torch.Tensor] | None = None,
        wall_reflect: bool = False,
        custom_post_step: Callable[
            [torch.Tensor, int], torch.Tensor
        ] | None = None,
        diverge_threshold: float = 1e6,
    ) -> None:
        self.head = head
        self.max_iters = int(max_iters)
        self.state_clip = state_clip
        self.wall_reflect = wall_reflect
        self.custom_post_step = custom_post_step
        self.diverge_threshold = float(diverge_threshold)

    def _post_step(self, state: torch.Tensor) -> torch.Tensor:
        if self.state_clip is not None:
            low, high = self.state_clip
            low = low.to(state.device)
            high = high.to(state.device)
            if self.wall_reflect:
                # For each dim that crossed a bound, clamp position
                # and flip velocity. We assume layout
                # state = [pos_0..pos_{D/2-1}, vel_0..vel_{D/2-1}].
                D = state.shape[-1]
                if D % 2 != 0:
                    raise ValueError(
                        "wall_reflect=True requires even state_dim "
                        "(half pos, half vel)"
                    )
                half = D // 2
                pos = state[..., :half]
                vel = state[..., half:]
                low_p = low[:half]
                high_p = high[:half]
                # Below low: clamp position to low, flip velocity sign
                # if it points away from the interior.
                under = pos < low_p
                pos = torch.where(under, 2 * low_p - pos, pos)
                vel = torch.where(under, vel.abs(), vel)
                over = pos > high_p
                pos = torch.where(over, 2 * high_p - pos, pos)
                vel = torch.where(over, -vel.abs(), vel)
                state = torch.cat([pos, vel], dim=-1)
            else:
                state = torch.clamp(state, low, high)
        return state

    def __call__(
        self,
        initial_state: torch.Tensor,
        force_seq: torch.Tensor | None = None,
        K: int = 1,
    ) -> tuple[torch.Tensor, RolloutReport]:
        """Roll the head out K steps from ``initial_state``.

        Args:
            initial_state: ``(B, state_dim)`` tensor.
            force_seq: optional ``(K, B, force_dim)`` tensor. If
                ``None``, the head is called with no force at
                every step (requires ``head.force_dim == 0``).
            K: rollout horizon.

        Returns ``(trajectory, report)`` where trajectory has
        shape ``(K+1, B, state_dim)`` (index 0 is the initial
        state, index k is the state after k applications).
        """
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if K > self.max_iters:
            raise ValueError(
                f"K={K} exceeds max_iters={self.max_iters}; raise "
                "PhysicsCook.max_iters or split the rollout"
            )
        if force_seq is None and self.head.force_dim > 0:
            raise ValueError(
                "force_seq must be provided when head.force_dim > 0"
            )
        t0 = time.time()
        traj = [initial_state]
        state = initial_state
        diverged = False
        history: list[float] = []
        with torch.no_grad():
            for k in range(K):
                if force_seq is not None:
                    f_k = force_seq[k]
                else:
                    f_k = None
                delta = self.head(state, f_k)
                state = state + delta
                state = self._post_step(state)
                if self.custom_post_step is not None:
                    state = self.custom_post_step(state, k + 1)
                traj.append(state)
                norm = float(state.abs().max().item())
                history.append(norm)
                if norm > self.diverge_threshold:
                    diverged = True
                    break
        full = torch.stack(traj, dim=0)
        return full, RolloutReport(
            K=len(traj) - 1,
            wall_seconds=time.time() - t0,
            final_state_norm=float(state.abs().max().item()),
            diverged=diverged,
            step_history=history,
        )


# ---------------------------------------------------------------------------
# F59 — physics sleep cache: cook → lookup distillation.
# ---------------------------------------------------------------------------


@dataclass
class PhysicsDistillReport:
    """Diagnostics from :func:`distill_physics_cook_to_lookup`."""

    n_steps: int
    n_pairs_distilled: int
    initial_loss: float
    final_loss: float
    cook_oracle_diverge_rate: float = 0.0


class StateLookupHead(nn.Module):
    """Direct ``(state_0, K) → state_K`` lookup, the v4 analogue of
    the F53 RPE classifier. Trained by distillation from a cook
    oracle (see :func:`distill_physics_cook_to_lookup`).

    Once distilled, a single forward pass replaces a K-step cook
    rollout — orders of magnitude faster on neuromorphic
    inference. This is the **mature-physicist** pathway: novice
    runs the cook every time, expert has cached the (s₀, K) →
    s_K mapping.

    Args:
        state_dim: dimensionality D of the state vector.
        max_K: largest horizon the lookup is trained for. The
            head receives ``K`` as a normalised float input
            ``K / max_K``.
        hidden: MLP hidden width.
    """

    def __init__(
        self, state_dim: int, *, max_K: int = 100, hidden: int = 64,
    ) -> None:
        super().__init__()
        if state_dim < 1:
            raise ValueError(f"state_dim must be >= 1, got {state_dim}")
        if max_K < 1:
            raise ValueError(f"max_K must be >= 1, got {max_K}")
        self.state_dim = int(state_dim)
        self.max_K = int(max_K)
        # Input: state_dim (initial state) + 1 (normalised K).
        self.fc1 = nn.Linear(state_dim + 1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, state_dim)

    def forward(
        self, state: torch.Tensor, K: torch.Tensor,
    ) -> torch.Tensor:
        """Predict ``state_K`` given ``(state_0, K)``.

        Args:
            state: ``(B, state_dim)`` initial state.
            K: ``(B,)`` integer horizons (will be normalised).

        Returns ``(B, state_dim)`` predicted state at horizon K.
        """
        K_norm = (K.float() / self.max_K).unsqueeze(-1)
        x = torch.cat([state, K_norm], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


def distill_physics_cook_to_lookup(
    cook: PhysicsCook,
    lookup: StateLookupHead,
    *,
    sample_initial_states: torch.Tensor,
    sample_Ks: list[int],
    optimizer: torch.optim.Optimizer | None = None,
    lr: float = 5e-3,
    n_steps: int = 400,
    batch_size: int = 64,
    rng_seed: int = 0,
) -> PhysicsDistillReport:
    """Sleep cache for the physics cook — F59 v4 analogue of F53
    :func:`pcm.dual_process.distill_cook_to_rpe`.

    For each ``(s0, K)`` pair sampled from the cross-product of
    ``sample_initial_states × sample_Ks``, runs the cook to get
    the K-step rollout terminus and trains
    :class:`StateLookupHead` to predict it directly.

    The function is decoupled from any specific cook architecture
    or domain assumption — pass any cook + lookup pair and it
    will distill them.

    Args:
        cook: trained :class:`PhysicsCook` whose K-step rollouts
            serve as the distillation oracle.
        lookup: :class:`StateLookupHead` to be trained.
        sample_initial_states: ``(N_states, state_dim)`` tensor.
        sample_Ks: list of integer horizons to distill.
        optimizer: optional pre-configured optimiser; if ``None``,
            a fresh AdamW is created.
        lr: learning rate.
        n_steps: distillation gradient steps.
        batch_size: pairs per gradient step.
        rng_seed: deterministic sampling.

    Returns a :class:`PhysicsDistillReport`.
    """
    import random

    if sample_initial_states.numel() == 0:
        raise ValueError("sample_initial_states must be non-empty")
    if not sample_Ks:
        raise ValueError("sample_Ks must be non-empty")

    if optimizer is None:
        optimizer = torch.optim.AdamW(lookup.parameters(), lr=lr)

    rng = random.Random(rng_seed)
    target_device = next(lookup.parameters()).device
    sample_initial_states = sample_initial_states.to(target_device)

    # Pre-compute cook oracle for all (s0, K) pairs.
    n_states = sample_initial_states.shape[0]
    pool: list[tuple[torch.Tensor, int, torch.Tensor]] = []
    n_diverged = 0
    for s_idx in range(n_states):
        s0 = sample_initial_states[s_idx:s_idx + 1]  # (1, D)
        for K in sample_Ks:
            traj, rep = cook(s0, K=K)
            if rep.diverged:
                n_diverged += 1
                continue
            terminal = traj[-1].squeeze(0)  # (D,)
            pool.append((s0.squeeze(0), int(K), terminal))
    if not pool:
        raise RuntimeError(
            "All cook rollouts diverged on the supplied "
            "(s0, K) sample; cannot distill."
        )
    diverge_rate = n_diverged / max(n_states * len(sample_Ks), 1)

    # Initial loss for reporting.
    with torch.no_grad():
        states = torch.stack([p[0] for p in pool])
        Ks = torch.tensor([p[1] for p in pool], device=target_device)
        targets = torch.stack([p[2] for p in pool])
        preds = lookup(states, Ks)
        initial_loss = float(F.mse_loss(preds, targets).item())

    # Distillation loop.
    final_loss = initial_loss
    for _ in range(n_steps):
        batch = [pool[rng.randrange(len(pool))] for _ in range(batch_size)]
        states = torch.stack([b[0] for b in batch])
        Ks = torch.tensor([b[1] for b in batch], device=target_device)
        targets = torch.stack([b[2] for b in batch])
        preds = lookup(states, Ks)
        loss = F.mse_loss(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    return PhysicsDistillReport(
        n_steps=n_steps,
        n_pairs_distilled=len(pool),
        initial_loss=initial_loss,
        final_loss=final_loss,
        cook_oracle_diverge_rate=diverge_rate,
    )
