# PCM v8 — How to close the F79 gap to generalist Transformers

**Date**: 2026-05-18
**Context**: F79 measured a 2.07× perplexity gap between PCM-TopK and a
parameter-matched GPT-mini on TinyStories. The user asked: *if we want
PCM to be general, what do the latest (≤ 2026-05) language papers say
about how to do this?*

This document is the literature synthesis + concrete next-experiment
roadmap. It does **not** abandon the "specialist child learner" framing
of F74/F79 — instead, it asks: *what is the minimum architectural
patch that closes the bulk of the gap while keeping PCM's slot-as-
concept interpretability?*

---

## 1. What the 2026 LM landscape converged on

After reading ~10 papers and the Jan–Feb 2026 architecture survey
([Raschka, Feb 2026](https://amyris-cs.ch/10-architectures-from-jan-feb-2026/);
[Hybrid Architectures, 2510.04800, 2025](https://arxiv.org/html/2510.04800v1);
[BabyLM 2025 findings](https://arxiv.org/html/2504.08165v1);
[Titans, NeurIPS 2025](https://arxiv.org/abs/2501.00663);
[Forgetting Transformer, 2503.02130](https://huggingface.co/papers/2503.02130);
[Sessa, 2604.18580, Apr 2026](https://huggingface.co/papers/2604.18580);
[Gated DeltaNet, 2412.06464](https://arxiv.org/pdf/2412.06464);
[MesaNet, 2506.05233, Jun 2025](https://arxiv.org/html/2506.05233);
[MLP Memory, 2508.01832](https://www.arxiv.org/abs/2508.01832v1);
[TransMamba, AAAI 2026](https://arxiv.org/pdf/2503.24067))
**every top model in Jan-Feb 2026 uses some flavour of the same
recipe**:

> **Linear-recurrent backbone with a learned gate, *plus* a few
> sparse "anchor" attention layers for precise retrieval.**

Concretely:

| Model | Year | Long-range layer | Anchor layer | Ratio |
|---|---|---|---|---|
| **Qwen3-Next** / **Qwen3.5** / **Qwen3-Coder-Next** | Feb 2026 | Gated DeltaNet | Gated Attention | 3:1 |
| **Ling 2.5** / **Ring 2.5** | Feb 2026 | Lightning Attention (gated linear) | MLA attention | hybrid |
| **Trinity Large** (Arcee) | Jan 2026 | Sliding-Window Attention + gating | Global Attention | 3:1 |
| **GLM-5** | Feb 2026 | DeepSeek Sparse Attention | MLA | hybrid |
| **Step 3.5 Flash** | Feb 2026 | Gated Attention | — (gated everywhere) | — |
| **Jamba**, **Samba**, **Zamba**, **Hunyuan-TurboS**, **Nemotron nano 2**, **IBM Granite 4.0** | 2024–25 | Mamba (gated SSM) | softmax attention | 1:3 to 1:7 |
| **Titans** (Google) | NeurIPS 2025 | Neural memory module (test-time learned) | Sliding-Window attention | hierarchical |
| **Sessa** | Apr 2026 | Selective SSM | Attention inside feedback path | hybrid |

The Hybrid Architectures systematic analysis ([2510.04800](https://arxiv.org/html/2510.04800v1)) is the definitive
empirical study. Its three findings:

1. **Hybrids beat homogeneous architectures by up to +2.9 % accuracy**
   on language-model quality benchmarks.
2. **Hybrids dominate on in-context retrieval and length generalisation**
   — pure Mamba and pure SWA both fail this test, hybrids pass it.
3. **Sequential (inter-layer) better for short context; parallel
   (intra-layer head-wise split) gives the best Pareto frontier for
   long context.**

This is not aesthetic preference — it is the architecture that wins on
every metric in 2026: speed, scaling, retrieval, and small-model
quality.

### What this means for PCM

PCM-mean is essentially a **degenerate linear-recurrent layer** with
no gate (always-keep-everything cumulative mean). PCM-TopK adds a
sparse-attention path, but the *underlying recurrent backbone* is
still ungated. That is precisely the design point the field has
moved past.

The F79 result — PCM 2.07× behind GPT — is *exactly the size of the
gap predicted by 2510.04800 between a homogeneous linear model and a
hybrid*. We are seeing the textbook problem.

---

## 2. The four architectural levers, in order of "minimum surgery"

These four directions correspond to the four sub-patterns above.
Each one is a concrete F80+ experiment that we can run independently
and combine.

### Lever A — Gated cumulative mean (Mamba-/GLA-/GRU-style)

**The single smallest change** that addresses the F79 gap.

Current PCM-mean:
```
ctx_t = (slot_0 + slot_1 + ... + slot_t) / (t+1)
```
Gated PCM:
```
g_t  = sigmoid(W_g · slot_t + b_g)         # per-channel forget gate
state_t = g_t · state_{t-1} + (1 - g_t) · slot_t
ctx_t = state_t
```

That is **literally the Mamba / GLA / GRU update rule** applied to PCM
slots. Cost: one linear projection per layer (D × D ≈ +16K params at
d_model=128). Each dimension of the state can independently decide how
fast to forget — exactly the "what should I remember about this slot"
semantics that PCM's interpretability story already commits to.

* **Forgetting Transformer** (2503.02130) showed the same gate idea works
  on attention scores.
* **Gated DeltaNet** (2412.06464) showed it works on delta-rule linear
  attention.
* Every Feb-2026 model has it.

**Expected gap closure**: 1.6× → ~1.2× on TinyStories (most of the gap,
based on the 2510.04800 Mamba-vs-Transformer measurements). Preserves
all of PCM's interpretability — the gate values are inspectable per
position, per dim.

**Implementation effort**: ~80 lines in `pcm/lm.py`.

---

### Lever B — Hybrid PCM + Gated Attention layers (Qwen3-Next pattern)

**The 2026-standard play.** Mix layer types in a 3:1 ratio:

```
Layer 1: GatedPCMLayer  (or PCMTopKLayer)
Layer 2: GatedPCMLayer
Layer 3: GatedPCMLayer
Layer 4: GatedAttentionLayer   ← precise retrieval, sparse
Layer 5: GatedPCMLayer
...
```

The Gated Attention layers handle the few positions where the model
really *must* read off a specific past token (named-entity recall,
center embedding, etc.). The PCM layers handle the rest at O(n) cost.

* Qwen3-Next, Qwen3.5, Qwen3-Coder-Next all do exactly this.
* Hybrid Architectures (2510.04800) confirms 1:3 attention:linear is
  near-optimal at 1B scale.

**Expected gap closure**: 1.6× → ~1.05× on TinyStories. Likely matches
GPT.

**Implementation effort**: ~150 lines (a new `GatedAttentionLayer` +
layer interleaving in `PCMMiniLM` / `PCMTopKMiniLM`).

**Cost**: gives up a small amount of interpretability at the attention
layers (which are now standard QKV) but only 1/4 of layers — the slot-
as-concept identity is preserved through 3/4 of the network.

---

### Lever C — Hierarchical memory (Titans-style)

**The architecturally most ambitious play, but already 75 % built.**

PCM v8 = three memory tiers:

```
┌──────────────────────────────────────────────────────────────┐
│  short-term  (~64 tokens window): Gated Attention              │
│  mid-term    (within doc): Gated PCM-TopK                      │
│  long-term   (across docs): EpisodicBuffer + sleep             │
│                                consolidation → ConceptGraph    │
└──────────────────────────────────────────────────────────────┘
```

The mid- and long-term tiers **already exist** as F75 (`EpisodicBuffer`,
`LongTermEpisodicTrace`, `consolidate_to_concept_graph`) and F78
(`EpisodicAgent`). The "new" thing for the LM setting is just to wire
them through the token stream:

* At training: tokens write to short-term; episodic events (sentence /
  paragraph boundaries) get imprinted to long-term via salience.
* At decoding: hierarchical query — first short-term, then mid, then
  long-term retrieval through the existing `recall_by_similarity` API.

* Titans (NeurIPS 2025) showed this **outperforms Transformers at 2 M
  token contexts** on needle-in-haystack.
* MesaNet (2506.05233) generalised it via local conjugate-gradient
  test-time training.
* MLP Memory (2508.01832v1) showed +17–24 % on WikiText.

**Expected impact**: gap closure on bulk LM PPL would be modest
(maybe matches Lever B). But: PCM v8 would inherit Titans-style
**long-context dominance** (the regime where pure Transformers
actually struggle), and the "test-time learning" property — the
model literally writes new episodes into memory as it processes new
text, **without retraining**. This is the killer feature the
specialist-child-learner story needs: a 4-year-old reading a new
book doesn't update its synapses, it *adds episodes to memory*.

**Implementation effort**: ~300 lines (a hierarchical-readout layer
that fans out into the three memories and combines).

---

### Lever D — DeltaNet-style associative update

**A more surgical fix that targets *associative recall* specifically.**

The reason PCM-mean loses on TinyStories is that "Tim found a ball.
Tom was sad. Tim ate the ball." needs **content-addressable recall**
of "Tim" — and cumulative mean blurs identity. The delta rule fixes
exactly this:

```
state_t = state_{t-1} − β_t (state_{t-1} − v_t) k_t k_t^T
        = state_{t-1} (I − β_t k_t k_t^T) + β_t v_t k_t^T
```

This is an *online linear-regression update* in the slot space —
identical to one step of Widrow–Hoff / LMS / delta learning. The
state becomes a content-addressable associative memory where reads
via `state_t · q_t` retrieve the most recently-updated value at a
matching key.

* DeltaNet (NeurIPS 2024) 1.3 B model with this rule beat Mamba and
  GLA at matched params.
* Gated DeltaNet (2412.06464) is what Qwen3-Next actually uses.

**Expected impact**: closes about as much of the gap as Lever A, but
*specifically* for associative-recall failures (which dominate
TinyStories errors). Composes nicely with Lever A's gate.

**Implementation effort**: ~120 lines (a new `PCMDeltaLayer`).

---

## 3. Crossing the BabyLM-Strict-Small benchmark

Independent of architecture, **the right yardstick for "specialist
child learner" is the BabyLM-Strict-Small track** — 10 M words of
developmentally-plausible English (mostly child-directed speech +
children's books + Simple Wikipedia).

[BabyLM 2025 Findings](https://arxiv.org/html/2504.08165v1) reports:

* LTG-BERT on 100 M words **beats** Llama-2 on grammatical tasks.
* The 10 M-word track is *the* small-data sandbox for cognitive-
  plausibility comparisons.
* Curriculum learning largely **failed**; shorter sequences + KD
  succeeded.

PCM v8 should be benchmarked here, not just on TinyStories. This
gives a falsifiable specialist claim: *PCM v8 beats Llama-2 on
BLiMP at 10 M training words*, the way LTG-BERT does.

---

## 4. Recommended roadmap

Each is a self-contained PoC (~1 day of work). Listed in dependency
order; can stop after any one and ship.

| Tag | Title | Effort | Expected closure | Risk |
|---|---|---|---|---|
| **F80** | Gated PCM (Lever A) | 1 day | 1.6× → ~1.2× | low |
| **F81** | Hybrid PCM + Gated Attention (Lever B) | 1 day | → ~1.05× | low |
| **F82** | DeltaNet PCM (Lever D) — alternative to F80 | 1 day | similar to F80, better on assoc-recall | medium |
| **F83** | Hierarchical Titans-style PCM (Lever C) | 2 days | parity + long-context dominance | high |
| **F84** | BabyLM-Strict-Small benchmark of F80–F83 | 1 day | yardstick vs LTG-BERT, Llama-2 | low |

### Suggested execution order

1. **F80 first** (Gated PCM). It is the smallest controlled
   experiment that directly tests the user's "Route B option 1
   (gating)" intuition. If the gap closes to ≤ 1.3×, the question
   "is gating sufficient?" is answered.
2. **F81 next** (Hybrid). This is the 2026-industry-standard play.
   If F80 leaves a gap, F81 will close it.
3. **F83 if we want to make a new claim** (hierarchical memory). This
   is where PCM goes from "matches GPT" to "different from GPT and
   wins on a specific axis" (long-context). Builds directly on F75 +
   F78 already shipped.
4. **F84 at the end** to establish the absolute "specialist child"
   benchmark.

### What does *not* need to be added

* Multi-head attention complexity. Gated single-head suffices in 2026.
* RoPE positional embedding. Several Feb-2026 models drop it (NoPE).
* Full Mamba state-space discretisation. The simpler Gated Linear
  recurrence in Lever A is sufficient at PCM's scale.
* Mixture of Experts. The user's "specialist" framing is the
  opposite goal: one specialist learner, not a router over many.

---

## 5. The "兼并" framing

The user's question was literally how to *兼并* (merge/encompass)
generalist capability into the specialist architecture. The 2026
literature gives a precise answer:

> **You don't replace the specialist with a generalist. You add a
> learned gate + a few sparse attention layers, and you get both.**

That is what every 2026 frontier model does. The PCM equivalent is
F80 + F81: a gated PCM backbone with 25 % of layers replaced by
gated attention. PCM's specialty (slot-as-concept,
universal-combiner, episodic memory, sleep consolidation) survives;
the gap to GPT closes; and the "specialist child" framing remains
the scientific commitment — we just admit that even a 4-year-old has
some attention.

---

## 6. Concrete invariants for F80 (Gated PCM)

To keep the experiment falsifiable, the next PoC should test:

* **G1 init-loss matches uniform**: same as F79 N0 regression test
  (the GPT-2 init fix carries over). Threshold: |loss − ln V| < 1.5.
* **G2 gap closure on TinyStories**: ``ppl_gated / ppl_gpt ≤ 1.30``
  at matched params, vs F79 baseline of 2.07. Threshold: 1.30 ×.
* **G3 PCM-mean recoverable**: when the gate is forced open
  (``g_t = 1/(t+1)`` to recover cumulative mean), final PPL matches
  the F79 PCM-mean baseline within 5 %. Sanity check that gating
  is the *only* change.
* **G4 F77 long-anaphora still passes**: gated PCM at n_sentences=6
  retains ≥ 0.93 accuracy (vs F77 PCM-TopK 0.978). The gate must
  not *break* selective recall.
* **G5 interpretability**: gate values per (position, channel) are
  inspectable and human-readable; visualised gates for a sample
  TinyStory show non-trivial structure (e.g. high retain at named
  entities, low retain at common words).

If F80 passes G1–G5 we have a publishable result: **PCM-Gated is
within 1.3× of GPT on real text, retains long-range coreference,
and exposes its memory dynamics as visualisable gates.**

---

## 7. Next action

Recommend starting **F80 (Gated PCM, Lever A)** immediately. It is
the minimum-surgery option that directly tests the user's gating
intuition, builds on the existing `pcm/lm.py` stack, and produces a
falsifiable answer in one experiment cycle.
