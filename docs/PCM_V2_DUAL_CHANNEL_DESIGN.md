# PCM v2 — Dual-Channel Concept Architecture

**Status**: design proposal → **MVP validated, freeze part 1 shipped** (May 2026).
Motivated by S1–S6 ceilings documented in `docs/SHORT_REPORT_2026_S1_S6.md`;
experimentally validated by F42–F49.

> **Update May 2026:** the original two-channel proposal (slot + attr)
> validated V1 / V2 invariants but only achieved partial pass on V3
> (`mixed_OOD = 0.240 ± 0.102`, vs target ≥ 0.30). Diagnostic D4
> showed the bottleneck was head-side displacement coverage, not
> bundle-side attr learning. We added a **third channel — relative
> position embedding (RPE)** as a first-class option for pair-input
> heads; this version saturates V3 to 1.000 across 5 seeds and
> generalises to colour / phoneme / number with the same pattern.
> See §10 below for the V3-RPE update and §11 for A1/A2 gate
> findings.

This document proposes the first **architecture-level** redesign of PCM
since the original Tier-A/B/C/D split. It does **not** discard the
`ConceptGraph + ParamBundle + Tier-G` core; it changes one well-defined
abstraction — what a "concept" *is* — and lets the rest of the stack
absorb the change through three localised refactors.

---

## 1. Why a v2 at all

Three independent S1–S6 falsification signals converge on the same
structural diagnosis:

| Failure | Source | Symptom | Common cause |
| --- | --- | --- | --- |
| S3 `mixed_OOD = 0.000` for any single-cell D head | F36/F37/S3 | `MoveHead.fc1` cannot generalise (inner, outer) joint pair | head input distribution coverage of single-vector concat |
| S6 vector analogy `top1 < chance` for number, +2.6 pp for colour | S6 | `bundle_d ≠ bundle_a + bundle_c − bundle_b` | bundle rows are cluster-aligned, not direction-additive |
| S2 number `NMI = 0` (dead-codebook) | S2 | All 30 numbers map to a single Tier-G prototype | k-means on 1-d ordinal data + tightly co-clustered bundle rows |

**All three are downstream consequences of one design choice**: a concept
is encoded as **a single vector** in a single facet. That vector has to
simultaneously serve as:

1. a **positional address** for k-means clustering (Tier-G expects
   discrete clusters);
2. a **categorical identity** for downstream MLP heads (`fc1` learns a
   joint-distribution function over pair concats);
3. a **continuous attribute axis** for vector arithmetic.

These three jobs have **different geometric requirements** that a single
vector cannot satisfy at once:

* (1) wants tight intra-cluster + large inter-cluster distance.
* (2) wants the joint distribution of pairs to cover the test-time
  manifold.
* (3) wants attribute differences to be linearly aligned across concepts
  (e.g. `bundle(7) − bundle(5) ≈ bundle(8) − bundle(6)`).

(1) and (3) are mutually antagonistic when D is small; (2) is
head-architecture-bounded but tightens proportionally to
intra-cluster slack from (1).

## 2. Core change — concept becomes a pair

```
   v1:  concept_a  →  bundle_a ∈ ℝ^D
   v2:  concept_a  →  (slot_a, attribute_a)
                       ↑          ↑
                  ℤ (cluster id)  ℝ^{D_attr}  (continuous, contrastive)
```

The two channels live on **different facets** of the same `ConceptGraph`
and are trained by **different loss families**:

* **Slot channel** (`facet_slot`): supervised by Tier-A/B/C/D heads as
  before, plus the existing Tier-G k-means clustering. Its row holds
  the cluster-aligned anchor + small residual that k-means can recover.
* **Attribute channel** (`facet_attr`): trained by a contrastive
  InfoNCE loss whose positives are concepts in the same Tier-G cluster
  and whose negatives are everything else, with an additional
  arithmetic-consistency term that explicitly encourages
  `attr_a + attr_c − attr_b ≈ attr_d` whenever `(a, b, c, d)` is a
  valid analogy quadruple in the domain. Tier-G **does not cluster**
  this channel.

This is why we call v2 *dual-channel*: we keep PCM's existing facet
machinery, but commit to two facets per concept *by design*, not by
accident-of-experiment, and we commit to which kind of training they
each tolerate.

## 3. API diff

```python
# pcm/concept_graph/_core.py
class ConceptGraph:
    # NEW: paired facet declarations
    slot_facets: dict[str, str] = {}
    """Mapping ``base_facet -> slot_facet``. Tier-G clusters slot facets only."""

    attr_facets: dict[str, str] = {}
    """Mapping ``base_facet -> attr_facet``. Trained by contrastive
    + arithmetic-consistency only. Excluded from sleep clustering."""

    def register_dual_channel_facet(
        self, base: str, *, slot_dim: int, attr_dim: int,
    ) -> tuple[str, str]:
        """Allocate two facets ``base + '_slot'`` and ``base + '_attr'``,
        record the pairing, and return the pair."""
```

Heads accept a **dual-channel collapse** helper:

```python
# pcm/sleep.py (renamed pcm/dual_channel.py for v2)
def collapse_dual_channel(
    cg, *, caller, base_facet, concept_ids, slot_shape, attr_shape,
    tick=0, device=None,
) -> tuple[Tensor, Tensor]:
    """Returns ``(slot_rows, attr_rows)`` of shape
    ``(B, slot_dim) and (B, attr_dim)``. Drop-in replacement for
    ``cg.collapse_batch`` whenever a head wants both channels."""
```

Heads built on top:

```python
class DualChannelMoveHead(nn.Module):
    """v2 spatial head — replaces MoveHead.

    * Slot path: pairwise attention over (slot_a, slot_b) → relational
      logit. Fixes S3 mixed_OOD by virtue of attention's
      input-distribution invariance: query (cell_a) attending over
      (cell_b) does not couple the joint pair to the train-time
      distribution of fc1 inputs.
    * Attr path: linear difference (attr_a − attr_b) → direction
      logit. Fixes S6 vector analogy by virtue of explicit linear
      composition.
    """
```

## 4. Training schedule

```
Phase 1 (warm-up):  slot facet only, supervised + Tier-G k-means
Phase 2 (anchor):   freeze slot facet, train attr facet under
                    contrastive InfoNCE (positives=same slot)
Phase 3 (compose):  unfreeze both, joint train with arithmetic-
                    consistency loss on attribute channel
Phase 4 (sleep):    Tier-G + (optional) S1 dual-phase on slot only;
                    attr facet excluded from sleep
```

## 5. What this is *not*

* **Not** a Transformer rewrite. The slot path's attention is a small
  pairwise attention over two embeddings, not a full sequence model.
* **Not** a hyperdimensional computing (HDC) substitution. Bundle rows
  remain plain tensors; we add a second tensor per concept, not a new
  algebra.
* **Not** an abandonment of `ConceptGraph`. Every existing public API
  still works on the slot facet; v1 callers can opt out of v2 by
  declining to register the attr facet.

## 6. Falsifiability — three new invariants for v2

| ID | Property | Test |
| --- | --- | --- |
| **V1** (positional) | Tier-G k-means on the slot facet recovers exactly the supervised cluster identity at NMI ≥ 0.95 on the §7.4 BCD condition | `tests/test_v2_slot_purity.py` |
| **V2** (compositional) | `attr_a + attr_c − attr_b` nearest neighbour matches `d` at top-1 ≥ 0.30 (vs S6 baseline 0.079) on number N=10 BCD | `tests/test_v2_vector_analogy.py` |
| **V3** (architectural) | `DualChannelMoveHead.mixed_OOD` ≥ 0.30 on the §7.5-space 5×5/7×7 grid (vs current 0.000) without augmentation | `tests/test_v2_mixed_ood.py` |

If any of V1–V3 falsifies, v2 is provably *not* the right
generalisation; either the proposal needs revision or one of the three
ceilings is in fact orthogonal to dual-channel encoding.

## 7. PoC scope (MVP)

To avoid burning 200+ GPU-hours on a v2 reconstruction that might fail
falsification, we propose a **single-domain MVP** before any v1 → v2
data migration:

* **Domain**: number N=10 (cheapest, also where S6 falsified hardest).
* **Code**: ~600 LoC for `pcm/dual_channel.py` + `experiments/
  number_dual_channel_poc.py`, no v1 file modified.
* **Eval**: V1, V2, V3 invariants above + the existing §7.4 BCD
  metric panel (spike-10, units-gap, last-digit purity).
* **Stop conditions**:
  * V1 < 0.85 → bug in slot training, fix and retry.
  * V2 < 0.20 → contrastive + arithmetic loss insufficient, escalate.
  * V3 < 0.10 → attention head not the cure, **abort v2 / preserve v1**.
  * **All three ≥ target** → green light to migrate one more domain
    (recommend §6.8 colour for the IB-frontier crosscheck).

## 8. Cost & risk table

| Phase | Code | Compute | Risk |
| --- | --- | --- | --- |
| MVP (number N=10) | ~600 LoC | ~3 GPU-h | low — single domain, can be reverted |
| Crosscheck (colour 12) | ~400 LoC | ~5 GPU-h | low |
| Migration (space + phoneme) | ~800 LoC | ~30 GPU-h | medium — touches §6.9 cross-language transfer |
| Full v1 baseline rerun | 0 LoC | ~80 GPU-h | high — only justified if V1–V3 all green |
| Documentation + paper §8 | n/a | n/a | low |

**Worst case** (V3 falsifies): we throw away ~10 GPU-h of MVP work and
keep v1 + the SHORT_REPORT_2026_S1_S6 findings. No v1 baseline data is
disturbed.

**Best case** (V1–V3 all green at MVP): we have a falsifiable
architectural lever that promises to unlock S3 mixed_OOD, S6 vector
analogy, and S2 dead-codebook simultaneously, and we commit to the
~115 GPU-hour crosscheck + migration to lock it down across domains.

## 9. Decision gate (for the user)

The proposal asks for explicit consent on three independent axes:

1. **Build the MVP?** ~600 LoC + 3 GPU-h. Tests V1/V2/V3 invariants.
2. **If MVP passes, migrate which domains next?** colour / space /
   phoneme can each be picked independently.
3. **If MVP fails, abort v2 and lock in SHORT_REPORT_2026_S1_S6 as
   final?** Or pivot to a different architecture (HDC / Transformer
   backbone)?

This document does not commit any code yet. Once the user approves
axis 1, we will:

* create `pcm/dual_channel.py` with the three new APIs above;
* create `experiments/number_dual_channel_poc.py`;
* create `tests/test_v2_invariants.py`;
* run V1/V2/V3 in a single 3-GPU-h MVP cycle;
* report back **before** touching any §6.x or §7.x baseline.

---

### Appendix — what changed compared to the existing roadmap

The original `docs/2026_LITERATURE_AND_PLANS.md` listed eight schemes
(S1–S8) all of which kept v1 architecture. The S1–S6 findings showed
three structural ceilings cannot be broken from inside v1. PCM v2
supersedes the still-pending S7 (topological grid) and S8 (word-context
phoneme) by addressing the same generalisation goals at a deeper layer:

* S7's "ordered-experience graph" is subsumed by the slot facet's
  attention path (the slot facet *is* the discrete adjacency / order
  structure; attention over slots replaces hand-crafted graph builders).
* S8's "context-conditional phoneme clusters" is subsumed by the attr
  facet's contrastive training (different word contexts become positive
  pairs in InfoNCE; same phoneme → same slot, different attr).

If v2 MVP succeeds, S7 and S8 become **two-line config experiments** on
top of the dual-channel substrate rather than separate research threads.

---

## 10. V3 update — RPE as the third channel

After F42 validated V1 + V2 at 1.000, the V3 (space `mixed_OOD`)
gate cleared `0.240 ± 0.102` (F43): well above v1 baseline 0.000
but below the strict 0.30 target. A four-step diagnostic chain
(F44, see also `docs/SHORT_REPORT_2026_S1_S6.md §V3-RPE`) showed:

| step | hypothesis | result |
| --- | --- | --- |
| v2 V3 trained attr | dual-channel partially fixes coverage | 0.240 ± 0.102 |
| **D4 oracle attr** | bottleneck is in attr learning | **0.100 ± 0.087** (worse!) |
| D4 + ReLU head | maybe linear `attr_diff` was too weak | 0.000 ± 0.000 |
| **RPE-only head** | bottleneck is head-side displacement coverage | **1.000 ± 0.000** ✓ |

**Update**: the v2 architecture now exposes **three** complementary
channels per concept-pair, not two:

```
   v2 (May 2026):
     concept_a → (slot_a, attr_a)
     concept_b → (slot_b, attr_b)
     pair (a, b) → also access RPE(displacement(a, b))
                              ↑
                        learnable embedding of (Δ₁, …, Δₖ)
```

* **slot facet** — discrete cluster identity (same as in §2);
* **attr facet** — continuous, contrastive + arithmetic (§2);
* **RPE table** — head-side embedding of integer displacements,
  keyed on caller-supplied delta tuples
  (`pcm.dual_channel.RelativePositionEmbedding`).

The RPE channel is **head-side, not bundle-side**: it lives in the
:class:`pcm.heads.DualChannelPairHead` rather than as a third
facet on `ConceptGraph`. This is deliberate — RPE encodes
*displacement* between two concepts, not a property *of* a single
concept, so it does not belong on the bundle.

Cross-domain validation (F48) on `pcm.dual_channel.RelativePositionEmbedding`:

| domain | task | concat (v1-style) | RPE (v2.1) | gap |
| --- | --- | --- | --- | --- |
| space (5×5/7×7 mixed_OOD) | direction | 0.000 | **1.000 ± 0.000** | +100 pp |
| colour (12-cyclic, holdout hue) | (b−a) mod 12 | 0.000 | **1.000 ± 0.000** | +100 pp |
| phoneme (V/M/P 3-axis) | joint Δ class | 0.003 | **0.965 ± 0.020** | +96 pp |
| number (1-d, \|Δ\| ≤ 29) | b−a class | 0.013 | 0.764 ± 0.009 | +75 pp |

The number domain's 0.764 (vs others' 0.96+) is a clean coverage
limit, not a failure: train pairs cover only `|Δ| ≤ 19`, but test
contains `|Δ| ≤ 29`; RPE-as-lookup cannot extrapolate beyond seen
displacements. Length-OOD beyond train range needs a **functional**
RPE (sinusoidal / RoPE / ALiBi), which is a clean follow-up
direction explicitly enumerated in §11 below.

## 11. A1 / A2 gate findings — when slot path is auxiliary

When a `DualChannelPairHead` enables both the RPE path and a slot
attention path simultaneously, the resulting fixed-λ mix at λ=1.0
on the V3 grid gives `mixed_OOD = 0.833 ± 0.115`, lower than
RPE-only at 1.000. F45–F46 decomposed why:

* **A1 (linear schedule, λ : 1.0 → 0.0)** — `mixed_OOD = 1.000`.
  Manual prior recovers the architectural lever; the slot path is
  active early and faded out before it can contaminate.
* **A2 (learned gate, sigmoid scalar init at 0.98)** — `mixed_OOD
  = 0.720`, `λ_final = 0.982`. The model **does not** spontaneously
  close the gate; train_acc=1.000 saturates train_loss and gives
  the gate no gradient pressure.
* **A2+ reward (α-weighted holdout CE)** — α ∈ {0.5, 2.0} both
  leave `λ_final ≈ 0.98`. Reward signal makes the model "study
  harder" but does not simplify the architecture.
* **A2+ self-discipline (β-weighted L1 on gate)** — β=0.1 alone
  gives `λ_final = 0.002` and `mixed_OOD = 1.000`. **Explicit
  simplification pressure is what closes the gate; reward is
  neither necessary nor sufficient.**

PCM v2 default policy (encoded in `DualChannelPairHead`):

| context | recommendation |
| --- | --- |
| pair-input head with both slot and RPE active | `gate_mode="schedule"` (cheapest) |
| pair-input head needing data-driven gate | `gate_mode="learned"` + L1 on `head.gate_l1()` |
| pair-input head with slot active, no RPE | `gate_mode="fixed"` (legacy v2 default) |
| pair-input head with RPE only | gate has no effect; pick any mode |

This is the falsifiable form of "inductive bias must be imposed"
within PCM: the model never finds its own minimum sufficient
statistic without an explicit simplification pressure.

## 12. Open follow-ups

The v2 freeze (F49) gives a stable surface for the next research
arc. Concrete follow-ups, in order of expected payoff:

1. **Functional RPE** — replace `RelativePositionEmbedding` lookup
   with sinusoidal / RoPE / ALiBi heads. Test on number length-OOD
   to see whether |Δ| > train_max becomes generalisable.
2. **Cyclic RPE** — current API treats every axis as linear; add
   `cyclic` flag for hue / phase tasks where the displacement
   space is `(0, n-1)` modular.
3. **D4-style ablations cross-domain** — repeat the oracle-attr
   ablation on number / colour / phoneme to verify whether the
   "bottleneck in head, not bundle" diagnosis is universal.
4. **A2+ on other domains** — test whether the "rewards motivate,
   only L1 simplifies" pattern holds outside the V3 grid; if so,
   it is a publishable position paper finding.
5. **Migration of v1 baselines** — port §6.6/6.7/6.8/6.9/7.4/7.5*
   to v2 dual-channel + RPE and re-run; expected: most ceilings
   collapse, several baselines tighten by 5 – 50 pp.
