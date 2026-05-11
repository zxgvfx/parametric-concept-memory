# PCM v4 — Physics-as-Procedural-Cook

**Status**: design proposal, May 2026, motivated by F56's universal
cross-domain cook validation and 2026 literature converging on
"iterative simulation + multi-step training" as the answer to
length-OOD physics rollout (IGNS ICLR 2026; Causal-JEPA arxiv
2602.11389; LeWorldModel arxiv 2603.19312).

This document proposes the third architecture-level extension of
PCM: **physics dynamics modelled as cook rollout**, where
`SuccessorHead` predicts a 1-step state transition under a force
input, and `IterativeDiffCook` accumulates transitions to predict
arbitrary-horizon trajectories. The recipe is mathematically
identical to the F51 number-domain cook, but the domain is
state-space dynamics rather than ordinal arithmetic.

---

## 1. The motivating gap

PCM v3 dual-process is currently a **discrete-displacement
calculator** — cook bridges integer differences in number, hue
displacements in colour, L1 distances in space. The recipe is
universal across PCM domains, but every domain so far has the
same basic primitive: "predict sign of remaining displacement,
iterate until target reached".

Real world dynamics — physics, biology, social systems —
extend the same principle to **continuous state vectors with
external forcing**:

```
   Discrete cook:   target_state = cursor + sum(±1 steps)
   Physics cook:    state_{t+1} = state_t + dt * f(state_t, force_t)
```

This is the natural progression because:

* Cook iteration *is already* a discrete-time simulator —
  applying it to (continuous_state, force) tuples needs almost
  no API change.
* The 2026 IGNS / Causal-JEPA / LeWorldModel literature line
  uses precisely this iterative architecture for graph network
  physics simulators, JEPA world models, and robotic planners.
* Sleep distillation transfers: just as cook procedural →
  RPE retrieval consolidates discrete arithmetic, cook
  trajectories → state-table retrieval consolidates **physical
  intuition**. This is the mature-physicist analogue of
  Year-1-counting → Year-3-retrieval (F53 / F54).

---

## 2. Four-line evidence convergence

### 2.1 Graph Neural Simulators — multi-step training is necessary

> *Improving Long-Range Interactions in Graph Neural
> Simulators via Hamiltonian Dynamics* (IGNS, ICLR 2026,
> arxiv 2511.08185)

* GNSs trained on 1-step transitions accumulate error in
  autoregressive rollouts (the same "1-step accuracy 0.99 →
  K-step accuracy ≈ 0.99^K" failure mode F51 mitigated for
  number arithmetic).
* IGNS adds **multi-step training objective** + port-Hamiltonian
  dynamics to align integrated trajectory with ground truth,
  reducing rollout error.
* PCM correspondence: F51 already has multi-step training
  (cook is trained inside the optimizer) plus the F53 sleep-
  cache distillation. F57 adds the physics-domain version of
  the same IGNS protocol.

### 2.2 JEPA world models — latent dynamics generalise

> *Causal-JEPA* (arxiv 2602.11389)
> *LeWorldModel* (arxiv 2603.19312, May 2026)

* Object-level latent interventions in Causal-JEPA give +20%
  counterfactual reasoning. Cook iteration *is* a latent
  intervention sequence — predicting state_{t+1} = f(state_t)
  under modified force is the same operation.
* LeWorldModel: a 2-loss JEPA on a single GPU, plans 48× faster
  than foundation-model-based world models. PCM cook is even
  cheaper: each rollout step is one MLP forward pass.

### 2.3 Cognitive — System 2 reasoning extends to physics intuition

> *Neural Correlates of Solving Arithmetic Problems in Adults*
> (Springer 2025) — same brain regions (SMA, MTG, cerebellum)
> recruited for hard arithmetic AND mental physics simulation
> (e.g. trajectory prediction tasks).
> *Battaglia 2018 Interaction Networks* — the original
> "physics as graph dynamics" formulation.

* Mental rotation, trajectory prediction, and "imagine the
  ball bounces": all System 2 procedural under fMRI.
* Cook rollout is the literal computational implementation of
  this — apply 1-step transition repeatedly, accumulate state.

### 2.4 Neuromorphic hardware — DMP architectures are deployment-ready

> *Algorithm-hardware co-design of neuromorphic networks
> with dual memory pathways* (arxiv 2512.07602, ICLR 2026)
> *Modern Neuromorphic AI: From Intra-Token to Inter-Token
> Processing* (arxiv 2601.00245, Jan 2026)

* DMP (Dual Memory Pathway) silicon: fast spiking pathway +
  slow memory pathway, **4× throughput + 5× energy efficiency**
  vs equivalent SOTA spiking nets.
* Loihi 2 / Akida already host RSNN-as-FSM mappings; cook is
  literally a finite state machine on (state, force) tuples.
* PCM v4 cook is therefore not just a software algorithm but
  a **hardware-feasible architecture with measurable power
  benefits**.

---

## 3. Core change — cook generalises to state-space dynamics

```
   v3 (May 2026):
     SuccessorHead(slot_a, slot_b) → signed_step ∈ {-1, 0, +1}
     IterativeDiffCook iterates until cursor = target
     State space: discrete ordinal index

   v4 (May 2026 ↪):
     PhysicsStateHead(state_t, force_t) → Δstate
     PhysicsCook iterates state_{t+1} = state_t + Δstate for K steps
     State space: continuous ℝ^D (with optional bounds)
```

The PhysicsStateHead is just a SuccessorHead with continuous
output (regression instead of classification) and a force input
channel. PhysicsCook is IterativeDiffCook with the termination
condition replaced by a fixed step count K (or convergence to
a steady state).

## 4. API additions

```python
# pcm/physics.py — new public module

class PhysicsStateHead(nn.Module):
    """Predict the per-step state change given current state
    and external force.

    Args:
        state_dim: dimensionality of the state vector ℝ^D.
        force_dim: dimensionality of the external force ℝ^F.
                   0 disables the force input (autonomous dynamics).
        hidden: MLP hidden width.
        dt: simulation timestep (default 1.0 in arbitrary units).
    """
    def forward(self, state: Tensor, force: Tensor | None = None) -> Tensor:
        """Returns Δstate of shape (B, state_dim)."""


class PhysicsCook:
    """Iterative state-space rollout. Apply PhysicsStateHead K
    times starting from `initial_state` under a sequence of
    forces."""
    def __init__(
        self, head: PhysicsStateHead, *,
        max_iters: int = 200,
        state_clip: tuple[Tensor, Tensor] | None = None,
        wall_reflect: bool = False,
    ): ...

    def __call__(
        self,
        initial_state: Tensor,
        force_seq: Tensor | None = None,
        K: int = 1,
    ) -> tuple[Tensor, RolloutReport]:
        """Returns (trajectory shape (K+1, B, state_dim), report)."""


def distill_cook_to_lookup(
    cook: PhysicsCook,
    state_table: nn.Embedding,
    sample_initial_states: list[Tensor],
    K: int,
    *,
    optimizer, n_steps, batch_size,
) -> DistillReport:
    """Sleep-cache equivalent: distill K-step rollout outcomes
    into a learned (initial_state) → final_state lookup table.
    Mirrors F53 distill_cook_to_rpe but for continuous state."""
```

## 5. Falsifiability — three new invariants

| ID | property | predicted value |
| --- | --- | --- |
| **P1 1-step accuracy** | `||head(state_t, f_t) - dt·ground_truth_f(state_t, f_t)||` mean over test set | ≤ 1e-2 (relative to state range) |
| **P2 K-step rollout error** | `||cook(s₀)[K] - ground_truth_rollout[K]||` follows polynomial in K, not exponential | linear or quadratic in K, R² ≥ 0.95 |
| **P3 long-horizon recognisability** | At K=100, `cook(s₀)[100]` is still within 10× the natural state-space scale (i.e. has not exploded) | predicted by P2 polynomial |

P1 is the easy precondition — a 1-step regressor that fails
this is broken. P2 is the cook-level claim: error grows with K
but bounded (the *literal* IGNS rollout-error reduction
observation). P3 is the headline: even at K=100 (the natural
v4 OOD test), trajectories should not blow up.

## 6. PoC scope (MVP)

**Domain**: 1-D bouncing ball under gravity, walls at x=0 and
x=L (elastic reflection).

**State**: (position, velocity) ∈ ℝ². L=10, g=−1, dt=0.1.

**Training**: random 1-step transitions; ground-truth physics
applies discrete dt-Euler integration plus wall reflection.
Held-out test: K-step rollouts up to K=100.

**Code budget**: ~200 LoC for `pcm/physics.py` + ~250 LoC for
`experiments/bouncing_ball_poc.py` + ~80 LoC tests.

**Stop conditions:**

* P1 > 1e-1 → head training broken.
* P2 deviates exponentially from polynomial fit → cook
  diverges; abort v4 / preserve v3.
* P3 trajectory norm > 10× state-space scale at K=100 →
  cook is unstable; need IGNS-style port-Hamiltonian
  regularisation.

## 7. Open follow-ups (after F57)

1. **Multi-body physics** — N=3 bodies with pairwise gravity;
   compare cook rollout vs ground-truth orbit precession.
2. **Force-controlled trajectories** — given a target end state,
   solve for the force sequence (control problem). Maps to
   PCM's existing routing dispatcher: cook plays the role of a
   forward simulator, an external optimiser searches over force.
3. **Sleep distillation for physics** — analogous to F53,
   distill frequent (initial_state, K) → final_state outcomes
   into a continuous lookup table, freeing the cook for novel
   trajectories. This is the mature-physicist analogue of
   Year-1-counting → Year-3-retrieval.
4. **Neuromorphic deployment** — see F58 / `PCM_NEUROMORPHIC_PROFILE.md`.

## 8. What this is *not*

* Not a Newtonian solver. The head learns from data and is
  bounded by training distribution; physics laws are not
  hand-coded.
* Not a replacement for symbolic physics (SciPy, MuJoCo).
  The point is *cognitive plausibility* of cook-based rollout,
  not numerical precision.
* Not multi-body 3D fluids. PoC is single body, 1-D, elastic
  walls; v4 milestone is the architectural contract, not the
  benchmark crown.
