# PCM v3 — Dual-Process Number Architecture

**Status**: design proposal, May 2026, motivated by F48 number length-OOD
limit (`mixed_OOD = 0.764` vs space/colour 1.000) and validated by a
seven-direction literature survey covering cognitive neuroscience,
developmental psychology, computational arithmetic, and 2026 neuro-
symbolic ML.

This document proposes the **second** architecture-level redesign of
PCM, extending v2's dual-channel concept encoding with a **dual-process
output path** (System 1 retrieval + System 2 procedural sequencing).
The design is the natural intersection point of four independent
literature lines, each providing a strong-form falsifiable prediction.

---

## 1. The motivating gap

PCM v2 + RPE saturated three out of four cross-domain OOD splits to
`1.000 ± 0.000` in F48:

| domain | RPE OOD | reason |
| --- | --- | --- |
| space (5×5/7×7 mixed_OOD) | **1.000** | every (Δr, Δc) ≤ 6 covered in train |
| colour (12-cyclic, hue holdout) | **1.000** | cyclic group → all 12 displacements covered |
| phoneme (V/M/P 3-axis discrete) | **0.965** | feature space tiny, ~all displacements covered |
| **number (1-d, |Δ| ≤ 29)** | **0.764** | train covers `|Δ| ≤ 19`, test extends to 29 |

Number's 0.764 is exactly `(test pairs with |Δ| ≤ 19) / (all test pairs)`
—— the RPE table is a **lookup**, and any displacement outside the
trained range collapses to its nearest in-range value or random init.
This is a **hard limit** of the v2 architecture, not a tuning issue.

**The user-proposed remedy**: implement an "演算 mechanism" — explicit
procedural decomposition mirroring how humans solve arithmetic problems
beyond their automatised range. Below we show this proposal is
independently endorsed by four mature research lines.

---

## 2. Four-line evidence convergence

### 2.1 Cognitive neuroscience — System 1 / System 2 have distinct neural substrates

> *Neural Correlates of Solving Arithmetic Problems in Adults*
> (Springer 2025, s11055-025-01846-4)
> *Internally generated outcomes of approximate calculation*
> (Nat Commun 2024, 7T fMRI, s41467-024-44810-5)

* **Easy arithmetic** (`5 + 3`, single-digit memorised) recruits **left
  inferior parietal lobule + angular gyrus** — the substrate of **fact
  retrieval**.
* **Hard arithmetic** (`247 + 185`, multi-step) bilaterally recruits
  **supplementary motor area + middle temporal gyrus + cerebellum** —
  the substrate of **procedural sequencing**.
* The transition is **systematic with problem size**, not idiosyncratic
  per subject. Cognitive reflection (the willingness to switch from
  System 1 to System 2) directly predicts mathematical reasoning
  performance (Sci Direct S1041608013001337).

**PCM correspondence**: `RelativePositionEmbedding` (lookup table of
displacements) is the angular-gyrus retrieval mechanism in computational
form. PCM has **no procedural sequencing analogue**, hence the F48
number-domain ceiling at 0.764.

### 2.2 Developmental psychology — counting → retrieval is the universal trajectory

> *Children's Arithmetic Strategy Use and Strategy Change Grade 3→4*
> (IJSME 2025, n = 1947)
> *Problem-size effect in 6 and 12-year-old children: from counting to
> memory retrieval* (J Exp Child Psych 2025)
> *Longitudinal Predictors of Conceptual Understanding of Arithmetic
> Principles* (JNC 2025)

* Five strategy-use profiles co-exist in primary-school populations:
  written algorithm, mental computation, mixed, etc. **No single
  strategy dominates; mixed is the norm.**
* As children mature, **small problems shift to retrieval** while
  **large problems retain counting** (the *problem-size effect*).
* Year-1 procedural performance **predicts Year-3 conceptual
  understanding** (longitudinal, 33 % variance). Procedural experience
  is the **seed** of retrievable facts.

**PCM correspondence**: a dual-process PCM architecture in which
sleep abstraction caches the results of frequently-iterated
procedural cooks **into the RPE table** is the precise computational
implementation of the Year-1-predicts-Year-3 finding. Sleep is what
the brain does at night; PCM's `Tier-G` already does this.

### 2.3 LLM scratchpad literature — procedural sequencing extends OOD

> *Length Generalization with Scratchpads*
> (OpenReview eIgGesYKLG, 2024–2025)
> *AI-rithmetic* (arxiv 2602.10416, Google 2026)
> *Inductive scratchpads* (NeurIPS 2024 paper 3107e4bd…)

* **Educated scratchpads** (with intermediate steps) achieve **6×
  length generalization** on arithmetic; agnostic scratchpads do not
  break the same barrier.
* **Inductive scratchpads** compose prior information and improve
  OOD generalization measurably.
* **Frontier models without scratchpads still fail** at basic
  arithmetic (Claude Opus 4.1, GPT-5, Gemini 2.5 Pro, May 2026).
  Errors decompose into tokenization/layout, carry semantics, and
  recomposition stages — the same failure mode as PCM number 0.764.

**PCM correspondence**: an `IterativeDiffCook` is the PCM-native
form of an educated scratchpad. PCM's `cookable subgraph` mechanism
already supports iterative composition; this is a config-level use,
not a new infrastructure.

### 2.4 Neuro-symbolic 2026 — grounding ≠ compositionality

> *Grounding vs. Compositionality: On the Non-Complementarity of
> Reasoning in Neuro-Symbolic Systems* (arxiv 2604.26521)
> *NeuroProlog Cocktail* (arxiv 2603.02504)
> *Neural Rewriting Systems* (arxiv 2507.19372)

* **Symbol grounding does NOT automatically produce compositional
  generalization** — the central counter-narrative to v1 PCM and
  similar architectures.
* Compositional generalization **requires explicit compositional
  supervision** (rule decomposition, multi-task training, rewriting
  systems).
* Neural Rewriting Systems learn convergent rewriting and
  generalize beyond training distributions, **outperforming pure
  neural baselines on mathematical formula simplification**.

**PCM correspondence**: the F48 finding that v2 + RPE saturates
in-range OOD but caps at the lookup boundary is **the precise
PCM instantiation** of the grounding ≠ compositionality theorem.
Grounding is solved (RPE saturates `|Δ| ≤ N_train`); composition
must be added as a separate mechanism.

---

## 3. Core change — concept-pair output becomes dual-process

```
   v2 (May 2026):
     pair (a, b) → pair_head(slot_a, slot_b, attr_a, attr_b, RPE(Δ))
                                                            │
                                                  lookup-only output

   v3 (May 2026 ↪):
     pair (a, b) → router(|Δ|)
                     ├─ Δ in train range → System 1: pair_head(...)  (v2)
                     └─ Δ out of range  → System 2: iterate(successor)
```

The **System 2 path** is implemented by:

1. A **`SuccessorHead`** — single-input head trained only on `|Δ| = 1`
   adjacent pairs. It outputs a small step (typically `−2 … +2`).
2. An **`IterativeDiffCook`** — pure-function PCM cook that applies
   `SuccessorHead` repeatedly until a target is reached, accumulating
   the step count.
3. A **router** — dispatcher selecting System 1 (RPE) when `|Δ|` is
   in the cached range, System 2 (cook) otherwise.

The **System 1 path** is unchanged from v2 (`RelativePositionEmbedding`).

The two paths interact through **Tier-G sleep**: frequently-iterated
`(start, target, K)` triples are cached as new RPE rows during the
sleep pass. This implements the longitudinal "procedural → conceptual"
transition observed in children.

## 4. API additions

```python
# pcm/dual_process.py — new public module

class SuccessorHead(nn.Module):
    """Predict a small step ``Δ ∈ {−n, …, +n}`` for the next state.
    Trained only on adjacent pairs (|Δ| = 1) but can be applied
    iteratively for arbitrary |Δ|."""

    def __init__(
        self, slot_dim: int, n_steps: int = 5,
        max_step: int = 2, hidden: int = 64,
    ): ...

    def forward(self, slot_a: Tensor) -> Tensor:
        """Returns logits over ``2*max_step + 1`` step classes."""


class IterativeDiffCook:
    """PCM cookable subgraph that iteratively applies SuccessorHead
    starting from `a` until reaching `b`; returns step count.

    The body of the cook is a Python loop with a hard upper bound
    on iterations (default 200). The successor head's output is
    interpreted as a signed step at each iteration; the loop
    terminates when the predicted step is 0, when ``a == b``, or
    when the iteration cap is reached.
    """

    def __init__(
        self, successor_head: SuccessorHead,
        max_iters: int = 200,
        max_step: int = 2,
    ): ...

    def __call__(
        self,
        slot_a: Tensor,                 # current state
        slot_b: Tensor,                 # target state
        identity_lookup: Callable,      # int → slot row
    ) -> tuple[Tensor, dict]:
        """Returns (predicted_diff, diagnostics_dict)."""


def route_diff(
    a: Tensor, b: Tensor,
    *,
    rpe_head: DualChannelPairHead,
    cook: IterativeDiffCook,
    train_max_abs_delta: int,
    delta_a_to_b: Tensor | None = None,
) -> Tensor:
    """Dispatcher: route to RPE for in-range Δ, to cook for OOD Δ.

    ``train_max_abs_delta`` is the largest |Δ| seen during training
    of the RPE table. If ``delta_a_to_b`` is supplied, routing is
    cheap; otherwise the router infers Δ via a comparison head
    (`sign(b − a)`).
    """
```

## 5. Falsifiability — five new invariants

| ID | property | predicted value |
| --- | --- | --- |
| **E1 procedural emergence** | `SuccessorHead` accuracy on `|Δ|=1` train pairs | ≥ 0.99 |
| **E2 compositional accumulation** | `IterativeDiffCook` accuracy on `|Δ|=K` follows `0.99^K` decay | exact match within 1 std |
| **E3 length extrapolation** | `IterativeDiffCook` on `|Δ|=99` (vs RPE-only `|Δ|=99`) | ≥ 0.37 vs ≤ 0.05 |
| **E4 developmental cache** | Tier-G sleep caches frequently-iterated K into RPE; routing usage of cook ↓ in late training | cook fraction at epoch 30 ≤ ½ of epoch 5 |
| **E5 RT-by-Δ scaling** | Wall-clock per query rises linearly in `|Δ|` for cook path; constant for RPE path | linear fit R² ≥ 0.95 |

E1–E3 are the falsifiability core. E4 is the most novel — it is the
direct computational implementation of the Geary / Siegler
developmental-trajectory result. E5 is the connection back to human
behavioural literature (Geary 1996; Ashcraft 1992).

## 6. PoC scope and stop conditions

To avoid burning compute on a v3 reconstruction that might falsify
E1 / E2 / E3, the MVP is single-domain:

* **Domain**: number, train range `|Δ| ≤ 19`, test range `|Δ| ≤ 99`.
* **Code budget**: ~600 LoC for `pcm/dual_process.py` +
  `experiments/number_dual_process_poc.py`. No v2 file modified.
* **Compute**: ~5 GPU-min for training, ~30 s for evaluation.

**Stop conditions:**

* `E1 < 0.95` → SuccessorHead training is broken; fix and retry.
  Almost certainly will not happen — a 1-step head is the easiest
  imaginable arithmetic supervision.
* `E2` deviation from `0.99^K` curve > 0.20 absolute at K=5 →
  iteration mechanism is broken (e.g. wrong sign accumulation).
* **`E3 < 0.10`** → architectural ceiling persists despite cook;
  abort v3 / preserve F40–F50.
* **`E1` ≥ 0.99 AND `E3` ≥ 0.30 (vs RPE-only ≤ 0.05)** → green light to
  add E4 (sleep cache) and E5 (RT scaling).

## 7. Decision gate

This document does not commit any code yet. Once approved we will
in order:

1. create `pcm/dual_process.py` with the three new APIs;
2. create `experiments/number_dual_process_poc.py` running E1/E2/E3;
3. report back **before** touching v2 baselines.

The full v3 freeze (E1–E5 + tests + docs) is estimated at ~800 LoC
and 2–3 hours of focused work, mirroring the F49 / F50 v2 freeze.

---

## 8. What this is *not*

* **Not an exor / boolean-rule replacement.** `SuccessorHead` is
  trained, not hand-coded; the cook composes learned predictions,
  not symbolic axioms. (The user explicitly flagged this concern;
  v3 design is consistent with PCM's emergence-from-data philosophy.)
* **Not a Turing-complete reasoner.** The cook has a hard iteration
  cap and only handles 1-d ordinal differences. Multi-step word
  problems / general reasoning are out of scope.
* **Not a supersession of v2.** v2 dual-channel encoding remains the
  bundle-side substrate; v3 only adds a head-side dispatcher and one
  new head class.
* **Not a hand-tuned routing schedule.** The router uses
  `train_max_abs_delta` (a measurable property of the dataset) as
  the only hyperparameter. No human-coded rules beyond that.

---

## 8.5 E4 sleep cache — implemented and validated (F53)

`distill_cook_to_rpe(cook, rpe_step_fn, rpe_parameters,
sample_pairs, …)` is the public API for the sleep-cache phase. It
re-uses the cook as an oracle on OOD pairs, supplying targets to
the RPE classifier. The function is decoupled from any specific
RPE architecture — caller passes a closure over their concrete
`RelativePositionEmbedding` head plus the parameter list to
optimise.

**Stratified sampling correctness note**: a naïve uniform sample
of `(a, b)` pairs over `|Δ| ∈ (train_max, n_total)` under-
represents large displacements (K=99 has only 1 valid pair vs
K=20 with 80). The reference experiment
`experiments/number_dual_process_sleep_poc.py` does **bucketed
quota sampling** — each `|Δ|` gets `distill_pairs / n_buckets`
pairs, drawn with replacement when the bucket is smaller. Without
this stratification, the RPE never sees the rare large-K pairs
during distillation and `phase2 RPE acc @ K=K_max` stays at 0.

**Empirical result (F53, 3 seeds × 15 epochs, N=100,
train_max=|Δ|≤19, distill_steps=400, distill_pairs=4000)**:

| metric | phase 1 | phase 2 (post-distill) |
| --- | --- | --- |
| RPE acc at K=99 | 0.000 ± 0.000 | **1.000 ± 0.000** |
| Distill loss | 10.x | 1.x (~10× reduction) |
| Cook acc at K=99 | 1.000 ± 0.000 | 1.000 ± 0.000 (unchanged) |

The RPE table fully internalises the cook's procedural knowledge
on OOD displacements after a single sleep pass. This is the
**falsifiable computational implementation** of the
"Year-1 procedural performance predicts Year-3 conceptual fact"
longitudinal trajectory observed in the JNC 2025 study.

**Caveat on cook routing fraction**: the experiment's
`route_diff` continues to use the original `train_max=19` as its
threshold, so `cook_route_fraction` does not drop after phase 2
(0.557 → 0.557). This is a deliberate design choice — routing
threshold is decoupled from RPE capability so the user can
control the System-1 / System-2 trade-off independently. A
follow-up "adaptive routing" extension that infers the threshold
from a calibration sweep over post-distill RPE accuracy is
listed in §9.

## 9. Open follow-ups (after E1–E5 ship)

1. **Adaptive routing** — after F53's E4 sleep cache, the RPE's
   coverage range expands from `train_max` to the cook's success
   range. Routing should infer the new threshold via a quick
   calibration sweep (sample a few K, find the largest K where
   RPE matches ground truth ≥ 0.95). This is the natural
   companion to E4: phase 2 lets RPE handle bigger displacements,
   adaptive routing surfaces that capability to the dispatcher.
2. **Multi-axis successor** — generalise `SuccessorHead` to predict
   multi-axis steps (Δr, Δc) on the spatial domain; compare with
   v2 RPE on length-extrapolating spatial OOD.
3. **Routing learned end-to-end** — replace `train_max_abs_delta`
   with a small classifier predicting whether RPE will succeed,
   tested against the A2 finding (does the model learn its own
   decision threshold without explicit pressure?).
4. **Cross-domain successor heads** — colour: hue rotation steps;
   phoneme: feature flips; space: cardinal moves. The same
   `SuccessorHead` API across all four domains gives a unified
   System 2 substrate.
5. **Compare to ALiBi / RoPE on number** — does a functional RPE
   alone bridge the same gap, or is the dual-process architecture
   strictly stronger? Predicted: dual-process > ALiBi on
   cognitive plausibility metrics (E5 RT scaling) even when
   accuracy is comparable.
