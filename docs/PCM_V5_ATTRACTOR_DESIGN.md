# PCM v5 — Statistical-Attractor Heads (F61)

**Status**: design + reference implementation in `pcm/attractor.py`,
validated on Pythagorean three-body in
`experiments/three_body_attractor_poc.py`. Commit hash recorded
on the F61 commit.

## 1. Why v5

F60 (`docs/SHORT_REPORT_2026_FULL.md` §3.5, commit `750d199`)
established a falsifiable contract for the PCM v4 cook:

> **Cook applies cleanly to systems with λ\* ≤ 1/K_target.**
> Beyond that horizon no per-step head accuracy can save the
> rollout, and F59-style sleep distillation does not rescue
> accuracy: the supervisory signal itself is unreliable past
> K\*.

The Pythagorean three-body result (λ\* = 0.091, K\* ≈ 11)
showed that for chaotic systems the cook's K-step state
prediction explodes exponentially regardless of how good the
1-step head is. Cook is a *trajectory* predictor; once the
trajectory is unpredictable, cook is unpredictable too.

But the *outcome distribution* is still predictable:

* In Pythagorean, body 1 (m=4) escapes ~80 % of the time.
* The energy partition concentrates ~54 % on body 1 and ~39 %
  on body 2, regardless of the exact ejection step.
* Time-to-ejection has a non-trivial distribution that depends
  on initial perturbation magnitude.

A human who's seen many three-body collapses doesn't predict
where each planet will be at t=10, but they can confidently
say "the heaviest two will form a binary and the lightest will
fly off." That's the *attractor* — the system's basin of
typical long-horizon outcomes.

PCM v5 introduces the architectural counterpart of that
intuition: an **AttractorHead** that predicts *statistical
descriptors* of long-horizon outcomes from a single state, in
one shot, at ``O(1)`` cost.

## 2. Architecture

### 2.1 `AttractorHead`

A wide-trunk MLP with five output heads (defined on
``state_dim → n_bodies``-shape outputs):

| head | output shape | semantic | loss |
|---|---|---|---|
| `escape` | `(B, n_bodies)` | which body ejects first | cross-entropy |
| `log_time` | `(B,)` | log(steps until ejection) | MSE |
| `energy` | `(B, n_bodies)` | final kinetic-energy fraction per body | KL on simplex |
| `mean` | `(B, state_dim)` | ensemble mean of long-horizon state | Gaussian NLL |
| `log_std` | `(B, state_dim)` | log-σ of long-horizon state | Gaussian NLL |

The head is **`K`-invariant**: in the chaotic regime past
``K*``, the outcome distribution is itself ``K``-invariant
once the system has settled into its attractor. If you want
``K``-conditioned behaviour use the `HybridPhysicsDispatcher`
to route short horizons to cook.

The Gaussian NLL with predicted log-σ is what gives the head
**calibrated uncertainty**: in chaotic regimes log-σ rises
naturally, expressing "I don't know exactly where, but the
spread is this large." log-σ is clamped to [-3, 5] so the
network can't trivially zero its NLL by predicting absurd
variance.

### 2.2 `attractor_loss`

```python
total = (weight_escape * loss_escape
         + weight_time * loss_time
         + weight_energy * loss_energy
         + weight_state * loss_state)
```

Defaults `(1.0, 0.5, 1.0, 1.0)`. Time gets a smaller weight
because its dynamic range is small (chaotic ejection times
cluster) — over-weighting it harms the more discriminative
escape and energy heads.

### 2.3 `HybridPhysicsDispatcher`

```
used = "cook"      if K_target <= K_star
used = "attractor" if K_target > K_star
```

This is the System 1 / System 2 dispatcher: cook (System 2,
``O(K)``) for short horizons inside the cook's predictability
window, attractor (System 1, ``O(1)``) for long horizons past
the chaos boundary. It mirrors the PCM v3 `route_diff` /
`calibrate_rpe_coverage` structure but with a continuous-state
substrate.

The ``K*`` threshold is not the Lyapunov ``K_lyap = 1/λ*``
directly — that bound is for arbitrary-perturbation
amplification. The relevant threshold here is the **cook
prediction horizon** ``K_cook``: the largest ``K`` at which the
cook's pointwise state error is still smaller than the
attractor's mean-prediction error. For Pythagorean (F61
empirical) ``K_cook ≈ 80–200``, well past the Lyapunov
``K_lyap = 11``. We default to ``K_star = 80`` which is the
conservative end of that range.

## 3. Falsifiable invariants

`experiments/three_body_attractor_poc.py` implements four:

* **A1** — `escape_acc > 1/n_bodies + 0.07`. The 3-class baseline
  is 0.333; we require ≥ 0.40. Hits **0.836** in the F61 full
  run.
* **A2** — `log_time_R² > 0`. The constant-mean predictor
  scores 0.0; we require strictly better. Hits **+0.517**.
* **A3** — Hybrid dispatcher's per-K outcome quality dominates
  cook-only past `K*` and matches it inside `K*`. We track two
  metrics:
    * **state L2** — cook for short K, attractor (ensemble
      mean) for long K. At K=800 cook err 832, attractor 503.
    * **escape classification** — the categorical metric where
      cook is fundamentally crippled (it cannot predict
      eventual outcomes from current state at small K, and its
      diverged trajectory at large K only randomly lines up
      with reality). Cook stays at 0.07–0.83 across K;
      attractor is 0.836 across all K.
* **A4** — `energy_KL < uniform_baseline_KL`. Hits 0.098 vs
  uniform 0.345.

All four pass in the F61 run with `n_train=8000, n_test=1000,
epochs=50, K_max=800`.

## 4. Connection to F40-F60

* F54 (`calibrate_rpe_coverage`) introduced adaptive routing
  between System-2 cook and System-1 RPE retrieval for
  arithmetic. v5 generalises this to physics: the System-2
  cook for state, System-1 attractor for distribution.
* F57 (`PhysicsCook`) is the System-2 component for chaotic
  systems past `K*`; v5 supplies the missing System-1 component
  so the dual-process picture is complete in the continuous-
  state regime.
* F59 (`distill_physics_cook_to_lookup`) consolidates cook
  output into a lookup head. v5's attractor is the analogue
  for outputs that are *categorical* / *distributional* rather
  than pointwise.
* F60 measured the Lyapunov `λ*` that gates v5's hybrid
  routing decision; the empirical cook horizon `K_cook` (where
  attractor begins to win on state L2) is computed inside
  the F61 evaluation script and printed in the A3 sweep.

## 5. Cognitive grounding

The dual structure mirrors the human dual-process picture more
directly than v3 did, because the underlying domain (continuous
chaotic dynamics) is the one where humans most obviously rely
on statistical pattern matching:

| capability | human equivalent | PCM v5 |
|---|---|---|
| short-horizon mental simulation | "imagine the next two billiard collisions" | `PhysicsCook` for `K ≤ K*` |
| long-horizon outcome category | "this configuration will eject the smallest body" | `AttractorHead.escape` |
| time estimate | "this should take about 30 seconds" | `AttractorHead.log_time` |
| energy estimate | "the binary will keep most of the energy" | `AttractorHead.energy` |
| calibrated uncertainty | "I don't know exactly where, but somewhere in this region" | `AttractorHead.log_std` |
| dispatching | "should I plan it out or just go with my gut?" | `HybridPhysicsDispatcher` |

This is also the cleanest correspondence we have so far for
Kahneman's System 1 / System 2: a domain where neither system
alone is sufficient, both are architecturally distinct, and
the dispatcher is governed by a measurable physical invariant
(λ\*).

## 6. Open follow-ups

1. **Distributional output for non-physics domains.** Apply v5
   to phoneme V/M/P (where outcomes are syllable-class
   distributions) and to colour holdout (where outcomes are
   percept categories). Tests whether the attractor head is
   genuinely domain-agnostic or contains physics-specific
   inductive bias.
2. **Learned routing.** Replace the hard `K*` threshold with a
   small classifier over `(state, K_target)` predicting whether
   cook will outperform attractor. Mirrors the F45/F46 finding
   that the model only closes its own gate when explicit
   simplification pressure is supplied.
3. **N=4 / N=5 attractor.** The ejection-body categorical
   collapses at N=4: outcomes become "binary + binary" or
   "triple + single" or "all dispersed." An ensemble decoder
   that predicts the number of surviving sub-systems would be
   a richer test.
4. **Sleep distillation v5.** Do an F59-style distillation
   from a slow ensemble simulator into the AttractorHead, so
   the head improves without explicit category labels (which
   are expensive for non-physics domains).
5. **PAPER v3 §17.** Add the v5 dual-process story as the
   closing argument for why PCM is a substrate for both
   System-1 and System-2 cognition — the domain-agnostic claim
   that the project has been building towards.

## 7. File pointers

* Module: `pcm/attractor.py`
* Tests: `tests/test_attractor.py` (12 cases, all passing)
* Experiment: `experiments/three_body_attractor_poc.py`
* Output (full run): `outputs/f61_full/summary.json`
* Reproduction:

  ```bash
  python -m experiments.three_body_attractor_poc \
      --n-train 8000 --n-test 1000 --K-max 800 \
      --K-eval-state 200 --epochs 50 --cook-epochs 15 \
      --cook-steps-per-epoch 200 --out outputs/f61_full
  ```

* Walltime: ~80 s on a single GPU.
