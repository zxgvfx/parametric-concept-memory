# Counterfactual Bundle Swap — Causal Evidence for "Bundle = Concept Identity"

**Date**: 2026-04-22
**Code**: `mind/experiments/language/counterfactual_swap_study.py`
**Output**: `outputs/counterfactual_swap/summary.json`
**Domains**: numbers (ANS 1–7) + colors (12 hues on the wheel)

## 1. Motivation

Every prior PCM study in the series provides *correlational / necessity*
evidence that the bundle parameters carry concept identity:

- `robustness_study.py` shows Spearman ρ between bundle geometry and
  concept order is stable across seeds;
- `E2_shuffled` shows ρ collapses when concept_id → bundle mapping is
  randomised during training (a necessity condition);
- `E4_permutation` shows cross-facet alignment is statistically
  significant (p = 0.003 / 0.016).

None of these are **interventional**. The strongest possible claim for
PCM — *the bundle is itself the concept's semantic identity, not merely
a learned correlate of it* — requires a causal intervention: **swap two
trained bundles post-hoc and see whether semantic roles follow**. This
is the neurological double-dissociation logic (Shallice 1988) applied
at the parameter level.

## 2. Design

For each domain, after training a dual-muscle model to convergence:

1. Pick two concepts (not adjacent, not antipodal): numbers 3 ↔ 5;
   hues 2 ↔ 5.
2. Record baseline accuracy on **all** evaluation pairs, grouped by
   "involves swapped concept" vs "does not".
3. Apply three interventions in sequence, each followed by the same
   eval and then undone (in-place swap on `bundle.params[facet].data`,
   leaving optimizer state and head weights untouched):
   - **Facet A only**: swap the facet consumed by muscle A.
   - **Facet B only**: swap the facet consumed by muscle B.
   - **Both facets**: swap every facet the two concepts share.
4. Repeat for 3 random seeds.

The predictions encode a **textbook double dissociation**:

| Intervention | Muscle A, involving | Muscle A, not inv. | Muscle B, involving | Muscle B, not inv. |
|---|---|---|---|---|
| Baseline | ≈ ceiling | ≈ ceiling | ≈ ceiling | ≈ ceiling |
| Swap facet A | **collapse** | unchanged | unchanged | unchanged |
| Swap facet B | unchanged | unchanged | **collapse** | unchanged |
| Swap both | collapse | unchanged | collapse | unchanged |

Any deviation from this pattern falsifies the identity claim (e.g.
"Swap facet A" degrading muscle B would imply facet A carries Muscle B's
information — a cross-contamination that PCM's design forbids).

## 3. Results

### 3.1 Number domain (swap concept:ans:3 ↔ concept:ans:5)

- Dual muscles: `ArithmeticHeadV2` (facet `arithmetic_bias`, 64 dim) +
  `ComparisonHead` (facet `ordinal_offset`, 8 dim).
- 7 concepts (1–7), 12 epochs × 120 steps, AdamW lr 1e-3, random
  orthogonal centroids per seed.

| Condition          | Add-inv | Add-not | Cmp-inv | Cmp-not |
|--------------------|---------|---------|---------|---------|
| Baseline           | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| Swap `arith_bias`  | **18.2 %** | 100.0 % | 100.0 % | 100.0 % |
| Swap `ord_offset`  | 100.0 % | 100.0 % | **75.0 %** | 100.0 % |
| Swap both          | 18.2 %  | 100.0 % | 75.0 %  | 100.0 % |

All numbers are **identical** across 3 seeds (σ = 0). Tight determinism
is expected: the eval enumerates the complete pair set, so once both
heads reach ceiling on baseline the downstream effect of the swap is
fully determined by the fixed pair-arithmetic.

Reading the "swap `arith_bias`" row: AddHead accuracy on problems
involving 3 or 5 collapses from 100 % to **18.2 %**. That 18.2 % is not
random (chance = 1/7 ≈ 14.3 %); it is precisely the residual correct
rate expected when every "3" bundle is replaced by the "5" bundle and
vice versa — only pairs whose answer is unchanged under this swap
remain correct. The AddHead is consistently computing the **swapped
semantics**, not random noise.

Simultaneously, CmpHead holds 100 % / 100 % on the very same concepts,
because its consumed facet (`ordinal_offset`) is untouched.

The mirrored effect obtains for "swap `ord_offset`": CmpHead drops to
75 % on involving pairs (the 8 out of 32 pairs where 3↔5 swap flips the
<,=,> label), AddHead unchanged.

### 3.2 Color domain (swap concept:color:2 ↔ concept:color:5)

- Dual muscles: `ColorMixingHead` (facet `mixing_bias`, 64 dim) +
  `ColorAdjacencyHead` (facet `adjacency_offset`, 8 dim).
- 12 hues, 30 epochs × 200 steps, hue 2 and hue 5 are at circular
  distance 3 (neither adjacent nor antipodal).

| Condition           | Mix-inv | Mix-not | Adj-inv | Adj-not |
|---------------------|---------|---------|---------|---------|
| Baseline            | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| Swap `mixing_bias`  | **5.3 %** | 100.0 % | 100.0 % | 100.0 % |
| Swap `adj_offset`   | 100.0 % | 100.0 % | **23.8 %** | 100.0 % |
| Swap both           | 5.3 %   | 100.0 % | 23.8 %  | 100.0 % |

Same double-dissociation pattern as numbers, again σ = 0 across 3
seeds. Crucially the circular domain reproduces the effect exactly —
the mechanism does not depend on linear ordinality.

## 4. What the result rules out

The observed pattern jointly refutes three alternative hypotheses:

- **H-null-swap**: "Bundles are just initialisation noise; swapping is
  no-op." — Falsified: swap causes targeted and large drops.
- **H-all-in-backbone**: "Concept identity lives in shared head / MLP
  weights; bundles are vestigial." — Falsified: swap leaves head
  weights untouched yet collapses exactly the expected pairs.
- **H-cross-contamination**: "Facet A and Facet B carry overlapping
  concept info." — Falsified: facet-A swap leaves muscle B at 100 %
  (double dissociation), so each facet cleanly localises its share of
  concept identity.

## 5. Connection to the H5/H5' line

This experiment directly strengthens **H5'** (cross-facet *identity*
coherence) reported in `SINGLE_VS_DUAL_MUSCLE_FINDING.md`. Those
findings established that when two muscles share compatible task
algebras, their bundle geometries align under the concept identity
map. The swap experiment now shows the converse: when we *break* the
identity map post-hoc (by relabelling which bundle belongs to which
concept), **both** muscles' task performance breaks **on the same
concepts**, without either muscle propagating the error to unrelated
concepts. Identity is localized *and* shared, which is what H5'
predicted.

## 6. Limitations

- Determinism across seeds (σ = 0) is partly a product of full-pair
  enumeration at ceiling accuracy. Running at intermediate training
  epochs would expose variance; we leave that sweep for future work
  since it is not needed to establish the qualitative double
  dissociation.
- `cmp-inv = 75 %` (not 0 %) is a ceiling artefact: on ordinal pairs
  that do not include 3 vs 5 directly, the label (<,=,>) is invariant
  under a 3 ↔ 5 swap for many pairs involving only one of the two.
  The drop from 100 % is still **exclusively on the swapped concepts**.

## 7. Reproducibility

```bash
# ~1 min, GPU. 3 seeds, full config.
python -m experiments.counterfactual_swap_study --n-seeds 3

# ~15s, 1 seed smoke check.
python -m experiments.counterfactual_swap_study --smoke
```

All raw numbers written to `outputs/counterfactual_swap/summary.json`.
