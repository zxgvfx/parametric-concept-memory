# F62 — Universal Operator Hypothesis (Cross-Discipline Transfer)

**Status**: experiment + design note, May 2026. Validated in
`experiments/cross_discipline_operator.py`. All five falsifiable
invariants pass on the F62 full run.

## 1. Hypothesis

The user's framing (paraphrased):

> Different scientific disciplines — math, physics, chemistry —
> are different *muscles* consuming the same underlying *concept*
> structure. If you discover a transformation in one discipline,
> it should apply in another, because the algebraic backbone is
> shared. Cross-discipline application is itself a form of
> reasoning.

In PCM language this is exactly the layered architecture we have
been building:

* **ConceptGraph** + **ParamBundle** ≈ the abstract concept layer
  — knows nothing about disciplines.
* **Muscle heads** ≈ discipline-specific decoders that consume the
  concept layer.
* The user's claim is that the *transformation* (an operator
  acting on concepts) lives in the concept layer; only the
  decoding lives in the muscle layer.

## 2. 2026 literature backing

The hypothesis is consistent with several 2025–2026 results:

* **Maruyama, *Categorical Equivariant Neural Networks* (arXiv
  2511.18417, 2026).** Unifies group / groupoid / poset / lattice
  / graph / sheaf-equivariant networks under naturality in a
  topological category, with a **universal approximation theorem
  for category-equivariant maps**. Proves the abstract operator
  exists and is realisable.
* **Mašulović, *Coalgebraic Foundation for Equivariant
  Representation* (arXiv 2603.03227, 2026).** Set→Vect functorial
  embeddings, lifted endofunctors compatible with embeddings,
  universal approximation of equivariant continuous functions
  across broad symmetry classes.
* **Geometric Alignment + Functor Decomposition (arXiv
  2602.01992, 2026).** Empirically shows Transformers decompose
  analogical reasoning into (1) geometric alignment of relational
  structure and (2) a categorical functor — the same two-piece
  structure as our (slot bundle, RPE + combiner).
* **Feature Resemblance Theory of Analogical Reasoning (arXiv
  2603.05143, 2026).** Transformers encode similar entities into
  similar representations; analogical reasoning works through
  feature alignment with a similarity-then-attribute curriculum.
  Matches the F62 finding that frozen-operator transfer
  succeeds when the new discipline gets to align its slot bundle
  to the operator's expectations.
* **OmniMol (arXiv 2601.10791, 2026).** Empirical proof of
  concept: a particle-physics foundation model transfers to
  molecular dynamics via shared point-cloud structure.
  Demonstrates the universal-operator hypothesis at scale, but
  needs a 100M+ parameter foundation model. F62 demonstrates the
  same phenomenon with a 17K-parameter testbed.
* **Intern-S1-Pro (arXiv 2603.25040, 2026), SciAgent (arXiv
  2511.08151, 2026), SciReasoner (arXiv 2509.21320, 2025), FuXi-
  Uni (arXiv 2601.01363, 2026).** Trillion-scale unified
  scientific foundation models, empirically working but treating
  cross-discipline transfer as a *scaling* phenomenon. F62 frames
  it as an *architectural* one.

The contribution of F62 is a **falsifiable** small-testbed proof
that the architectural piece — a single ``UniversalCombiner +
RPE`` learned in one discipline — transfers to another with
performance loss < 1pp, and that the transfer specifically relies
on shared structure (a permuted-identity negative control fails
to chance).

## 3. Experimental setup

Three structurally-isomorphic tasks under the cyclic group ℤ_N:

| discipline | state | operation |
|---|---|---|
| Math (M) | digit position 0..N−1 | ``a + Δ ≡ b (mod N)`` |
| Physics (P) | 1-D ring lattice cell 0..N−1 | ``x + v·dt ≡ x' (mod N)`` |
| Chemistry (C) | reaction-extent bin 0..N−1 | ``ξ_in + Δ ≡ ξ_out (mod N)`` |

All three implement the same group action; their *semantics*
(what a state means) are entirely different.

The model has three modular pieces:

* **Slot bundle** ``slot_d: ℤ_N → ℝ^D`` — discipline-specific
  embedding of states (one ``nn.Embedding`` per discipline ``d``).
* **RPE table** ``rpe: ℤ_{2K+1} → ℝ^D`` — embedding of
  displacement Δ ∈ {-K, …, +K}.
* **UniversalCombiner** ``T: (ℝ^D × ℝ^D) → ℝ^D`` — a 2-layer
  residual MLP that maps ``(slot_a, rpe_Δ) → slot_b_pred``.

The naive ``slot_a + rpe_Δ`` head was tried first and shown not
to learn (it cannot represent the cyclic-group action because
``sin(a + Δ) ≠ sin(a) + sin(Δ)``). The MLP combiner adds the
expressivity needed; this echoes Maruyama 2026's observation
that group-equivariant architectures need to operate
*multiplicatively*, not additively, on representations.

The classifier is a dot product of ``slot_b_pred`` against all
slot embeddings, trained by cross-entropy on ``b = (a + Δ) mod N``.

## 4. Five conditions, five invariants

| condition | structure | Δ-table | combiner |
|---|---|---|---|
| **A** joint-shared | M ∪ P ∪ C trained jointly | shared | shared |
| **B** joint-separate | M ∪ P ∪ C trained jointly | per-discipline | shared |
| **C** independent (×3 seeds) | each discipline alone | own | own |
| **U4** frozen transfer | math then physics | math-trained, **frozen** | math-trained, **frozen** |
| **U5** permuted-slot control | same as U4, but physics slot ids shuffled at eval | math-trained, frozen | math-trained, frozen |

| invariant | criterion | F62 result | status |
|---|---|---|---|
| **U1** joint-shared works | min per-discipline test acc ≥ 0.95 | **1.000** | PASS |
| **U2** sharing is free | A.acc ≥ B.acc - 0.03 | 1.000 = 1.000 | PASS |
| **U3** indep. RPEs align | Procrustes cos ≥ 0.85 *and* ≥ random + 0.20 | trained **1.000**, random baseline **0.631** | PASS |
| **U4** frozen transfer | new-discipline acc ≥ 0.90 with op frozen | **0.996** | PASS |
| **U5** permuted negative | acc ≤ 0.20 (≈ chance) | **0.051** (random = 1/N = 0.020) | PASS |

## 5. Interpretation

The five invariants together carve out a sharp empirical claim:

* **(U1, U2)** A single ``RPE + UniversalCombiner`` representation
  is *sufficient* to solve all three disciplines simultaneously.
  No accuracy is left on the table by sharing the operator.
* **(U3)** Even when independently trained — disciplines never
  see each other's data — the operators converge to the same
  algebraic structure (Procrustes cos = 1.0 vs 0.63 random).
  This is the strongest possible alignment: the cyclic-group
  representation is *unique* up to orthogonal basis change.
* **(U4)** Trained operators directly transfer. Physics with
  frozen math-operator and only its own slot bundle to learn
  reaches 0.996 — within 0.4 pp of from-scratch.
* **(U5)** When you destroy the structural correspondence
  (shuffle physics's slot identities so they no longer align
  with the math-trained operator), accuracy collapses to chance.
  This rules out "the operator works on anything" and confirms
  the transfer is *structural*.

In PCM terms: **the operator is the concept; the slot bundle is
the muscle skin**. Different disciplines are different *muscle
fittings* of the same conceptual machine. Once the operator is
discovered (in any discipline), bringing a new discipline online
is as cheap as training one ``Embedding`` table — half the cost
of the joint training, with no accuracy loss.

## 6. Cognitive grounding

This matches three established cognitive findings:

* **Schema theory** (Bartlett, Piaget, Rumelhart): humans
  abstract over concrete experience to produce *schemas* —
  reusable structural templates. F62's ``UniversalCombiner`` is a
  schema; the slot bundles are concrete fillings.
* **Analogical transfer** (Gentner & Markman, structure mapping
  theory): analogies are *structure-preserving correspondences*,
  not feature-level resemblances. F62's U5 directly tests this:
  shuffle the structure-preserving correspondence and transfer
  fails.
* **Mathematical universality / Wigner's *unreasonable
  effectiveness***: the same algebraic objects (groups, vector
  spaces, manifolds) describe phenomena in distant disciplines.
  F62's U3 is a tiny demonstration of why: under SGD with
  enough data, *any* faithful representation of a finite group
  ℤ_N converges to the same orthogonal class. The "unreasonable
  effectiveness" is, perhaps, just basis equivalence.

## 7. Limits and follow-ups

The F62 testbed is deliberately tiny — ℤ_N with N = 50 — so the
invariants are learnable in seconds. Real disciplines have
richer structure (continuous Lie groups, non-abelian, infinite-
dimensional). The U-suite would have to be re-tested for each:

1. **F62b — non-cyclic groups.** Same protocol but on dihedral
   ``D_n`` (with reflections), or on ``S_n`` permutation group.
   Tests whether the universal-operator picture survives
   non-abelian structure.
2. **F62c — continuous Lie group.** Replace discrete RPE with
   a sinusoidal / RoPE-style continuous embedding. Tests
   whether U4's frozen-transfer transfers to a Lie-group
   structure.
3. **F62d — empirically natural disciplines.** Replace toy
   ℤ_N with three actual structurally-shared scientific tasks
   (e.g. Hooke's law, electric force, gravitational force —
   all ``F = k · q1 · q2 / r²`` with different ``k`` and units).
   Tests whether the universal-operator effect survives when
   the shared structure is *mathematical* rather than algebraic.
4. **F62e — operator distillation.** Combine F62 with F59
   sleep-distillation: distil the trained universal operator
   into a fixed lookup that any new discipline can plug into,
   without re-running gradient descent.
5. **F62f — partial isomorphism.** What if disciplines share
   only some structure? E.g., ℤ_50 and ℤ_47 (different modulus).
   Quantify graceful degradation as the structural mismatch
   grows.
6. **PAPER §18.** Fold this into the PAPER as the closing
   argument that PCM's layered architecture is a substrate for
   "Wigner's unreasonable effectiveness" *as an architectural
   property*, not as a coincidence.

## 8. File pointers

* Module: experiment-only — reuses
  ``pcm.dual_channel.RelativePositionEmbedding``.
* Experiment: `experiments/cross_discipline_operator.py`
* Output (full run): `outputs/f62_full2/summary.json`
* Reproduction:

  ```bash
  python -m experiments.cross_discipline_operator \
      --N 50 --slot-dim 32 --delta-max 24 \
      --epochs 60 --batches-per-epoch 30 \
      --batch-size 128 --lr 5e-3 --n-seeds-u3 3 \
      --out outputs/f62_full2
  ```

* Walltime: ~95 s on a single GPU.
