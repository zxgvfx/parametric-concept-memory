# PCM v6 — Agent base design

**Status**: F64 first-PoC ships in this commit.
**Date**: May 2026.
**One-line**: extend the F62 universal-operator architecture from
``(slot, Δ) → next_slot`` (passive concept transformation) to
``(slot_state, action) → next_slot_state`` (active state-action
transition), then wrap it in an episode loop with goal
conditioning and a policy head.

This document specifies the v6 agent-base architecture and lists
the falsifiable invariants the first PoC (F64) is designed to
test. The companion implementation lives in:

* `pcm/agent/heads.py`        — TransitionHead, PolicyHead, ValueHead
* `pcm/agent/orchestrator.py` — episode rollout, BC training step
* `pcm/agent/envs/cyclic_nav.py` — toy environment (Z_N navigation)
* `experiments/agent_cyclic_nav_poc.py` — F64 PoC + U1–U5 tests
* `tests/test_agent.py` — unit tests

## 1. Why agent base, why now

F40–F63g built the architecture-vs-content duality and showed
that PCM's universal operator survives:

* discrete cyclic / non-abelian groups (F62, F62b)
* group-axiom checks (F62e)
* number-theoretic graceful degradation (F62f)
* continuous Lie groups (F62c)
* real physics force laws (F62d, force-law-agnostic)
* small-corpus → 5M-token cross-modality (F63d)
* synthetic Markov → real GRCh38 chr22 (F63e)
* 5-modality V3 sweep (F63f)
* per-component mechanistic interpretability (F63g)

The natural next layer is **active interaction**. An MDP transition
``(s, a) → s'`` is *structurally identical* to F62's group action
``(slot, Δ) → slot'``. The action ``a`` plays the role of the
displacement ``Δ``; the transition function plays the role of the
``UniversalCombiner``; the policy plays the role of an inverse-
operator search. F64 tests whether the universal-operator
architecture extends to this active-interaction setting *with no
new architectural primitives*.

## 2. Architecture

### 2.1 Three heads on top of the slot bundle

```
StateEncoder        : observation o → slot_state (D-dim)
ActionEmbedding     : action a (idx) → action_emb (D-dim)
TransitionHead      : UniversalCombiner(slot_state, action_emb)
                       → slot_next_state
PolicyHead          : (slot_state, slot_goal) → action_logits
ValueHead (opt.)    : (slot_state, slot_goal) → expected_return
```

Note that ``TransitionHead`` is *literally* an instance of the F62
``UniversalCombiner`` plus an action embedding. The agent base
does not need to invent new primitives; it reuses the operator
proven across F62 / F62b / F62c / F62d.

### 2.2 Episode loop

```python
def rollout(env, encoder, policy, transition, goal_obs, max_steps):
    obs = env.reset()
    slot_goal = encoder(goal_obs)
    trajectory = []
    for t in range(max_steps):
        slot = encoder(obs)
        action = policy(slot, slot_goal).argmax(-1)
        next_obs, reward, done = env.step(action)
        trajectory.append((obs, action, next_obs, reward, done))
        if done:
            break
        obs = next_obs
    return trajectory
```

At training time we compute two losses:

1. **Behavioural-cloning loss** on ``(state, goal, action)`` tuples
   from oracle trajectories: ``L_pi = CE(policy(s, g), a*)``.
2. **Transition-consistency loss** on ``(state, action,
   next_state)`` tuples: ``L_T = CE(TransitionHead(s, a) ·
   slot_table, idx(s'))`` — i.e. classify which slot is the next
   state, identical to F62's combiner training.

The transition head is *universal* (goal-independent); the policy
is *goal-conditioned*. This split mirrors F62's "operator vs slot
bundle" split.

## 3. F64 falsifiable invariants

We use the same five-letter U1–U5 schema as F62 / F62b / F62c /
F62d so that the agent results align with the rest of the
project. The toy env is a 1-D cyclic navigation task on ℤ_N
(N=20) with 4 actions (``+1, -1, +5, -5``), both because it is
the smallest possible non-trivial agentic env, and because it
collapses naturally to F62 if the agent is trivial — the test
becomes "does the F62 architecture solve agentic versions of
itself?".

| invariant | criterion | rationale |
|---|---|---|
| **U1** in-domain | held-out (start, goal) success ≥ 0.90 | basic agent works |
| **U2** sharing has no penalty | shared-transition acc ≥ separate − 3pp | universal world model |
| **U3** indep transitions align | Procrustes cos ≥ 0.85 | F62-style structural sharing |
| **U4** frozen-transition transfer | new-goal-distribution success ≥ 0.85 with transition frozen | universality at action level |
| **U5** permuted-action neg ctrl | permuted-action-label success ≤ 0.30 | rules out coincidence |

For F64 we *also* add an explicit **U6 multi-step planning**
invariant unique to the agentic setting:

| invariant | criterion | rationale |
|---|---|---|
| **U6** multi-step horizon | mean-step-budget on hard tasks ≤ optimal × 1.20 | policy is not just argmax local; can plan |

## 4. What v6 is NOT (yet)

* **No continuous actions.** F64 uses 4 discrete actions; v6.2
  will add continuous action spaces (RoPE-style action embedding
  by analogy to F62c).
* **No partial observability.** F64's state is fully observable;
  v6.2 will add belief-state reasoning.
* **No tool use.** F64's actions are state-changing primitives;
  v6.3 will add tool calls (variable-arity actions with typed
  arguments).
* **No learned reward / RL.** F64 trains via behavioural cloning
  on oracle trajectories; v6.4 will close the RL loop with
  trial-based reward learning, building on F54's calibration
  framework.
* **No big-model integration.** F64 runs at < 100K parameters;
  v6.5 will be the 1B-parameter Transformer scale validation.

The v6.1 PoC (this commit) demonstrates only that the F62
architecture transfers to active-interaction tasks at the
small-testbed scale. v6.2–v6.5 are explicitly out of scope; each
is its own falsifiable milestone.

## 5. File pointers

* This document: `docs/PCM_V6_AGENT_BASE_DESIGN.md`
* Public API: `pcm/agent/__init__.py`
* PoC: `experiments/agent_cyclic_nav_poc.py`
* Output: `outputs/f64_full/summary.json`

Reproducible from CLI:

    python -m experiments.agent_cyclic_nav_poc \
        --N 20 --epochs 80 --batches-per-epoch 30 \
        --out outputs/f64_full
