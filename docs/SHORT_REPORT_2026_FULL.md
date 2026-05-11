# PCM 2026 Short Report — From v1 Baselines to Dual-Process Physics

**Status**: project-level synthesis, May 2026. Covers F40 → F59,
the full arc from "literature-driven causal experiments" to
"physics-as-procedural-cook + neuromorphic deployment profile".

This report is the cohesive narrative version of:

* `docs/SHORT_REPORT_2026_S1_S6.md` (F40–F48)
* `docs/PCM_V2_DUAL_CHANNEL_DESIGN.md`
* `docs/PCM_V3_DUAL_PROCESS_DESIGN.md`
* `docs/PCM_V4_PHYSICS_COOK_DESIGN.md`
* `docs/PCM_NEUROMORPHIC_PROFILE.md`

reorganised as a single argument suitable for a workshop /
short-paper venue. Numbers are reproducible from the cited
output JSONs and commit hashes; the suite passes 172 / 172
unit tests at F59.

---

## TL;DR

Across 20 commits (F40 → F59) we used 2025–2026 cognitive-
neuroscience and ML literature to drive nine falsifiable
architectural experiments inside PCM. The results give the
project three new public APIs and one falsifiable contract:

1. **`pcm.dual_channel`** (v2 freeze) — concept = (slot, attr)
   pair + `RelativePositionEmbedding`. Breaks four PCM domains'
   pair-input ceilings to ≥ 0.96 OOD accuracy with the same
   API on number / colour / phoneme / space.
2. **`pcm.dual_process`** (v3 freeze) — `SuccessorHead` learns
   1-step transitions, `IterativeDiffCook` accumulates them
   for arbitrary OOD horizons, `distill_cook_to_rpe` consolidates
   procedural knowledge into retrieval, `calibrate_rpe_coverage`
   surfaces the new capability to routing.
3. **`pcm.physics`** (v4 PoC) — `PhysicsStateHead` +
   `PhysicsCook` + `StateLookupHead` + `distill_physics_cook_
   to_lookup` extend the same recipe to continuous state-space
   dynamics, validated on 1-D bouncing ball at K=100.
4. **Contract**: "**inductive bias must be imposed**" —
   five-config × five-seed evidence (F45 / F46) shows
   architecturally-cleaner solutions need explicit
   simplification pressure (L1 on gates, schedules), not just
   reward signals. Models do not spontaneously prefer simpler
   hypotheses under SGD.

---

## 1. Findings table

Every row maps to a falsifiable invariant tested with 3–8 seeds
and an output JSON.

### 1.1 v1 baseline causal ablations (F40, S1–S6)

| ID | Domain | Finding | Verdict |
| --- | --- | --- | --- |
| S1 | Tier-G | NREM-style two-phase sleep API + G8a/b/c invariants (3/3 pass) | architectural pass |
| S2 | colour | Hierarchical anchor warm/cool NMI 0.26 ± 0.22 (1/8 hits 1.0) | weak partial |
| S2 | number | Dead-codebook collapse: all 30 numbers map to one prototype | falsified |
| S3 | space | mixed_OOD = 0.000 unbreakable by any single-cell D head (4 variants tried) | architectural ceiling diagnosed |
| S4 | colour | IB-frontier: PCM BCD codebook efficiency 0.628 vs Hering4 (vs 0.420 vs RGB) — surprising — most efficient prediction is NOT the supervised RGB target | strong + surprising |
| S5 | colour | Cone-opponent prior emerges k=4 anchors at CMYK (90° equidistant) not Hering4 unique hues | partial |
| S6 | number | Vector analogy top1 = 0.079 < chance (0.10), BCD = A baseline | falsified — bundle geometry ≠ linear compositionality |
| F42 fix | infrastructure | `ConceptGraph._ensure_facet` device-equality bug — `torch.device('cuda') != torch.device('cuda', 0)` silently rebuilt pool every collapse, invalidating optimiser refs (latent throughout v1) | bug fix + 3 regression tests |

### 1.2 PCM v2 — dual-channel + RPE (F42 → F50)

| ID | Domain | Invariant | Result |
| --- | --- | --- | --- |
| V1 | number | Slot purity NMI ≥ 0.85 | **1.000 ± 0.000** ✓ |
| V2 | number | Vector analogy top1 ≥ 0.30 | **1.000 ± 0.000** ✓ |
| V3 | space | mixed_OOD ≥ 0.30 (vs v1's 0.000) | 0.240 ± 0.102 (partial); broken to **1.000 by RPE** |
| F44 D4 | space | Oracle-attr ablation: bottleneck is in head, not bundle | confirmed |
| F44 RPE | space | RPE-only `RelativePositionEmbedding` lookup head saturates mixed_OOD | **1.000 ± 0.000** ✓ |
| F45/F46 | space | A1 schedule (manual decay) → 1.000; A2 learned gate stays at λ ≈ 0.98 across all reward strengths; β=0.1 L1 alone closes gate to 0.002 | "inductive bias must be imposed" verified |
| F48 | cross-domain | RPE vs concat baseline on 4 domains (5 seeds × 20 epochs each): all four go from 0.0–0.013 → 0.764–1.000 | universal architectural lever |

### 1.3 PCM v3 — dual-process number cook (F51 → F56)

| ID | Domain | Invariant | Result |
| --- | --- | --- | --- |
| E1 | number | SuccessorHead 1-step accuracy ≥ 0.99 | 0.966 ± 0.033 (close; min 0.923, max 1.000) |
| E2 | number | Cook K=99 follows polynomial in K, not exponential | log-log R² ≈ 1, **error ≪ 0.99^99 = 0.37** |
| E3 | number | Cook K=99 vs RPE-only K=99 | **1.000 ± 0.000 vs 0.000** (+100 pp) |
| F53 E4 | number | Sleep distill cook → RPE: phase2 K=99 RPE acc | **0.000 → 1.000 ± 0.000** |
| F54 | number | Adaptive routing post-distill: cook_route_fraction | **0.557 → 0.000** (3 seeds, 0 variance) |
| F55 | number | Functional RPE (sinusoidal / ALiBi) on length-OOD | **0.000 at K=25** for both — falsifies "functional RPE solves length extrapolation" claim on precise arithmetic |
| F56 | colour | Cyclic SuccessorHead K=1..5 = 1.000; K=6 = 0.500 (half-ring ambiguity, structurally correct) | universal |
| F56 | space | Cardinal SuccessorHead K=12 (max L1 on 7×7) = 0.833 ± 0.144 | universal |

### 1.4 PCM v4 — physics cook + sleep distillation (F57 → F59)

| ID | Domain | Invariant | Result |
| --- | --- | --- | --- |
| P1 | 1-D bouncing ball | Head 1-step normalised MSE ≤ 0.01 | **0.00159 ± 0.00060** ✓ |
| P2 | 1-D bouncing ball | log-log R² of rollout error ≥ 0.85 (polynomial not exponential) | **0.947** ✓ |
| P3 | 1-D bouncing ball | K=100 final state norm ≤ 100 (trajectory bounded) | **9.77 ± 0.10** ✓ |
| F59 sleep | 1-D bouncing ball | StateLookupHead phase2 MSE drops > 50% | drop 24.48 → 9.43 (–62%) ✓ |
| F59 sleep | 1-D bouncing ball | Lookup at K=50 vs cook at K=50 | **lookup 9.43 < cook 16.97** (lookup wins by 1.8×) |

### 1.5 Neuromorphic profile (F58)

| operation | params | FLOPs/q | resKB |
| --- | ---: | ---: | ---: |
| RPE lookup + classifier | 17 207 | 27 584 | 67.2 |
| **Cook K=99** | **7 491** | **1.47 M** | **29.3** |
| PhysicsCook K=99 | 4 482 | 875 K | 17.5 |
| Sinusoidal RPE | 29 895 | 59 328 | 116.8 |

Cook is 53× more FLOPs but 2.3× less memory than RPE, so on
memory-constrained neuromorphic silicon (Loihi 2: 1 MB / chip;
Akida ~256 KB / reservoir block) cook is the deployment-friendly
choice. After F53/F54, all queries route to RPE — **steady-state
inference is RPE-only at 27.6 K FLOPs**, cook only invoked on
genuinely novel input.

---

## 2. Three structural lessons

### 2.1 Grounding ≠ compositionality (F40–F44 + F55)

The S6 number-domain finding (vector analogy < chance) and the
F44 space-domain D4 oracle-attr ablation (perfect attr +
ReLU classifier still gives mixed_OOD = 0.000) jointly
demonstrate the same lesson the 2026 neuro-symbolic literature
states explicitly:

> **Grounding (PCM bundle geometry) does not automatically
> produce compositional generalisation. Composition is a
> separate capability that must be explicitly supervised.**

PCM v2 RPE (F44/F48) is the explicit compositional supervision
inside-train-range. PCM v3 cook (F51) is the explicit
compositional supervision beyond-train-range. Neither falls out
of v1 supervision alone.

### 2.2 Inductive bias must be imposed (F45 / F46)

Five learned-gate configurations × five seeds, on the
v3 V3 architecture:

| α (reward) | β (L1) | mixed_OOD | λ_final |
| --- | --- | --- | --- |
| 0.0 | 0.0 | 0.720 ± 0.249 | 0.982 |
| 0.5 | 0.0 | 0.900 ± 0.141 | 0.978 |
| 2.0 | 0.0 | 0.760 ± 0.222 | 0.982 |
| **0.0** | **0.1** | **1.000 ± 0.000** | **0.002** |
| 0.5 | 0.1 | 1.000 ± 0.000 | 0.002 |

Reward signals make non-gate components train harder but never
close the gate (α has zero effect on λ_final across all
strengths). Only an explicit L1 penalty on the gate variable
closes it. **Models do not spontaneously discover their own
minimum sufficient statistic via gradient descent**; that
property must be imposed through explicit simplification
pressure. Connects to lottery-ticket / weight-pruning
literature; the user's framing was "学习需要奖励 + 自律，但
只要有自律就够了" — i.e. rewards motivate effort, only
discipline / regularisation simplifies.

### 2.3 Dual-process is the universal pattern (F51–F59)

Across number / colour / space / 1-D physics — four
structurally distinct PCM domains — the same recipe works:

```
       SuccessorHead (1-step)            ← System 2 procedural unit
         ↓ iterate K times
       IterativeDiffCook / PhysicsCook   ← System 2 sequence
         ↓ sleep distill (F53/F59)
       RPE / StateLookupHead             ← System 1 retrieval cache
         ↓ adaptive routing (F54)
       Steady-state inference            ← cook usage falls to 0
```

Cognitive neuroscience analogue (Springer 2025 / Nat Commun
2024 7T): SMA + MTG + cerebellum (System 2 procedural) gives
way to angular gyrus + IPL (System 1 retrieval) as
arithmetic competence matures. Year-1 procedural performance
predicts Year-3 conceptual fact (JNC 2025 longitudinal) — and
PCM v3/v4's distill+route pipeline is the literal computational
implementation of that trajectory.

The functional RPE comparison (F55) shows this is **not
substitutable** by sinusoidal / ALiBi tricks on precise
arithmetic — those work for next-token language tasks where
local distributional regularities carry the signal but fail
when the answer requires exact integer differences outside
the trained range.

---

## 3. Architecture summary

```
PCM v1 (F40 baseline):
  ConceptGraph + ParamBundle + Tier-A/B/C/D/G heads
    └─ pair-input head archetype: Linear(2D, hidden) + MLP
       (the v1 ceiling: pair-fingerprint memorisation,
        no compositionality, S6/F42 mixed_OOD = 0.000)

PCM v2 (F42–F50):
  + pcm/dual_channel.py
    ├─ register_dual_channel_facet  (slot + attr facet pair)
    ├─ collapse_dual_channel  (drop-in for cg.collapse_batch)
    ├─ RelativePositionEmbedding  (V3-RPE lever, F44)
    ├─ {Sinusoidal,ALiBi}RelativePositionEmbedding (F55 baselines)
    └─ pair_attention_logits + four loss primitives
  + pcm/heads/v2_dual_channel.py
    ├─ DualChannelPairHead  (gate_mode in {fixed,schedule,learned})
    └─ SlotIdentityAuxHead

PCM v3 (F51–F56):
  + pcm/dual_process.py
    ├─ SuccessorHead              (1-step head)
    ├─ IterativeDiffCook          (System 2 procedural)
    ├─ route_diff                 (S1/S2 dispatcher)
    ├─ distill_cook_to_rpe        (F53 sleep cache)
    └─ calibrate_rpe_coverage     (F54 adaptive routing)

PCM v4 (F57–F59):
  + pcm/physics.py
    ├─ PhysicsStateHead              (continuous-state 1-step)
    ├─ PhysicsCook                   (state-space rollout)
    ├─ StateLookupHead               (1-shot K-step lookup)
    └─ distill_physics_cook_to_lookup (F59 physics sleep cache)
  + scripts/neuromorphic_profile.py  (F58 hardware spec sheet)
```

Public API surface: 20 new symbols, all in opt-in modules. v1
callers continue working unchanged.

---

## 4. Open follow-ups

These are the natural next steps. None blocks publication of
F40 → F59 as a short report; all are concrete enough that any
of them could be the next milestone if pursued.

1. **Multi-body physics** — N=3 pairwise gravity; tests cook
   on chaotic dynamics (vs the integrable F57 1-D ball).
2. **Force-controlled trajectories** — give cook a target end
   state, search over force sequences (PCM ↔ planning).
3. **Phoneme V/M/P 3-axis successor** — the F56 cross-domain
   test domain we deferred; per-iteration axis selection is the
   open design problem.
4. **Loihi 2 / Akida actual silicon deployment** — F58 N1/N2/N3
   predictions await empirical hardware characterisation.
5. **Routing learned end-to-end** — replace
   `calibrate_rpe_coverage`'s explicit threshold sweep with a
   small classifier predicting whether RPE will succeed,
   tested against the F45/F46 finding (will the model close
   its own gate without explicit pressure?).
6. **PAPER v3 integration** — fold F40–F59 into the main paper
   §8–§16 as the "post-baseline mid-cycle" section.

---

## 5. File pointers (full reproducibility)

### Modules
* `pcm/dual_channel.py` — v2 (F42 + F44 + F55 RPE family)
* `pcm/heads/v2_dual_channel.py` — public v2 heads
* `pcm/dual_process.py` — v3 (F51 + F53 + F54)
* `pcm/physics.py` — v4 (F57 + F59)

### Experiments (each PoC reproducible from CLI)
* `experiments/sleep_*` — S1/S2/S4/S5/S6 from F40
* `experiments/sleep_space_mask_infer.py` — S3 ceiling
* `experiments/space_dual_channel_poc.py` — V3 partial
* `experiments/space_rpe_poc.py` — V3-RPE saturation + A1/A2
* `experiments/rpe_cross_domain.py` — F48 cross-domain RPE
* `experiments/number_dual_process_poc.py` — F51 E1/E2/E3
* `experiments/number_dual_process_sleep_poc.py` — F53 + F54
* `experiments/cross_domain_successor_poc.py` — F56 colour + space
* `experiments/functional_rpe_vs_cook.py` — F55
* `experiments/bouncing_ball_poc.py` — F57 P1/P2/P3
* `experiments/bouncing_ball_sleep_distill.py` — F59 v4 cache

### Tests (172 / 172 passing)
* `tests/test_dual_channel.py` (24 cases — DC1–DC6)
* `tests/test_dual_process.py` (25 cases — DP1–DP5)
* `tests/test_physics.py` (21 cases — PH1–PH4)
* `tests/test_concept_graph_device.py` (3 cases — F42 regression)
* `tests/test_tier_g_sleep.py` (10 cases — G1–G8)
* `tests/test_diagnostics.py` (12 cases — pcm.diagnostics)
* plus the existing v1 suites (test_smoke, test_grow_invariants,
  test_tier_b_gate, test_tier_c_peer, test_cook_subgraph,
  test_cook_all_heads, test_sleep_four_domain — total 77 cases)

### Documentation
* `docs/2026_LITERATURE_AND_PLANS.md` — 23-paper survey
* `docs/PCM_V2_DUAL_CHANNEL_DESIGN.md` — v2 design
* `docs/PCM_V2_MIGRATION_GUIDE.md` — v1 → v2 recipe
* `docs/PCM_V3_DUAL_PROCESS_DESIGN.md` — v3 design + §10 follow-ups
* `docs/PCM_V4_PHYSICS_COOK_DESIGN.md` — v4 design
* `docs/PCM_NEUROMORPHIC_PROFILE.md` — F58 hardware spec sheet
* `docs/SHORT_REPORT_2026_S1_S6.md` — F40–F48 short report
* `docs/SHORT_REPORT_2026_FULL.md` — this document

### Commit log highlights
* `F42` ConceptGraph device bug fix
* `F44` RPE saturates V3 mixed_OOD ceiling
* `F46` Reward vs L1 gate decomposition (inductive bias must be imposed)
* `F48` RPE cross-domain universal lever
* `F51` v3 cook K=99 = 1.000
* `F53` E4 sleep cache RPE OOD = 1.000
* `F54` adaptive routing cook fraction 0.557 → 0.000
* `F55` functional RPE falsified on precise arithmetic
* `F56` cross-domain cook universality (number/colour/space)
* `F57` v4 physics cook P1/P2/P3 all PASS
* `F58` neuromorphic profile + 3 hardware predictions
* `F59` v4 sleep cache lookup beats cook on K=50 by 1.8×

---

*Maintained as the canonical project-level summary. Update when
each new milestone (F60+) ships. Numbers are reproducible from
the cited output JSONs; CLI commands are copy-pasteable from the
docstrings of each experiment module.*
