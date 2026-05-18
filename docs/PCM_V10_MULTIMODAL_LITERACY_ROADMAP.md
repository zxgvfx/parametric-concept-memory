# PCM v10 — Multimodal Literacy Roadmap

**Date**: 2026-05-18
**Predecessor**: F85 Online Teacher Loop (PCM v9.0). 397/397
tests pass.
**Scientific question**: Can PCM cleanly progress through the
child-developmental curriculum: **listen ⇒ speak ⇒ see colors
& shapes ⇒ read printed characters ⇒ write**?

---

## 1. Status of the language substrate (as of F85)

The user asked whether the current language substrate is
sufficient before adding multimodal layers. Empirical answer
from ``scripts/assess_language_substrate.py``:

```
PCM LANGUAGE SUBSTRATE ASSESSMENT
(TinyStories valid, vocab=4096; assessing 'preschool' coverage)

  OK  COLOR        12/12 (100.0%)   red(5110), blue(2834), green(1728), ...
  OK  SHAPE        10/10 (100.0%)   round(746), star(644), heart(446), circle(296), ...
  OK  NUMBER       15/15 (100.0%)   one(29062), two(1793), three(880), ...
  OK  SPATIAL      16/16 (100.0%)   in(40743), up(11874), inside(4860), ...
  OK  EMOTION      13/13 (100.0%)   happy(25946), sad(11140), scared(6372), ...
  OK  FAMILY       12/12 (100.0%)   mom(26495), friend(9841), grandma(890), ...
  OK  BODY         12/12 (100.0%)   face(1468), hand(1456), head(1312), ...
  OK  ACTION       24/24 (100.0%)   play(24439), make(7888), see(7640), ...
  OK  SIZE         10/10 (100.0%)
  OK  TIME         11/11 (100.0%)
  OK  QUANTIFIER    9/9  (100.0%)
  OK  QUESTION      7/7  (100.0%)
  OK  NEGATION      5/5  (100.0%)
  OK  PRONOUN_ADV  25/25 (100.0%)

Summary: 14/14 concept classes complete; 181/181 items present
```

**The language substrate is sufficient.** TinyStories covers
*all* preschool cognitive concept classes with substantial
frequency (lowest item ``nobody`` at 103 occurrences). F81
PPL = 10.86 on this vocab. The model has the right *words*.

What we have NOT yet verified is whether the model has the
right *meanings* — i.e., whether its hidden representations
actually cluster these words by their cognitive class. That
is the cheap F86 probe (~30 min of work). Then the major
new work is vision (F87) and literacy (F88).

---

## 2. The roadmap — F86 / F87 / F88

### F86 — Cognitive-concept probe (light, 1 hour)

**Goal**: verify F81/F85's hidden representations already
encode the 14 cognitive classes from §1.

**Method** (mirrors F74 ``linear_probe_acc``):

1. Sample ~200 sentences from TinyStories containing each
   cognitive class word.
2. Extract the model's final-layer hidden state at the
   class-word position.
3. Train a linear classifier (k-fold) to predict the class
   from the hidden state.
4. Report accuracy. Threshold: ≥ 0.70 per class (well above
   chance 1/14 ≈ 0.07).

**Falsifiable invariants** (5):

* **C1** all-class probe ≥ 0.70
* **C2** COLOR sub-probe (12-way classification) ≥ 0.70
* **C3** SHAPE sub-probe ≥ 0.60 (lower freq → harder)
* **C4** EMOTION sub-probe ≥ 0.70
* **C5** PCM-Gated layers' gates correlate with concept-class
  identity (E1 extension from F80)

If C1-C5 pass, the language substrate has the right *semantic*
structure and we can proceed to grounding it in vision (F87).

### F87 — Visual perception of colors, shapes, symbols (medium, half day)

**Goal**: a 0.5 M-parameter ConvNet that takes 32×32 RGB
images of preschool stimuli and emits **slots in the same
space as the F81 language model's token embeddings**, so that
"red circle" the image and "red circle" the phrase produce
*aligned* slot representations.

This is the **cross-modal grounding** step. By the end of F87
the model knows what "red" looks like, not just what the word
distributes near.

**Architecture**:

```
image (3, 32, 32)
   ↓ Conv-BN-SiLU × 3
   ↓ AdaptiveAvgPool → (d_model,)
   ↓ Linear → slot
PCMUniversalCombiner ← language slot path
   ↓ shared concept-slot space
```

The CNN's final projection lands in PCM's slot space. We then
do **contrastive alignment** (CLIP-style or simpler MSE):

```
loss = MSE(cnn_slot(image), language_slot("the red circle"))
     + contrastive_loss(positive vs negative pairs)
```

**Dataset** — synthetic, generated procedurally:

* 8 colors × 8 shapes × 4 sizes × 4 positions = 1,024
  unique configurations
* For each: render a 32×32 RGB image + the canonical word
  pair ``"the {color} {shape}"``
* Total: ~10,000 (image, caption) pairs with augmentation

**Falsifiable invariants** (6):

| ID | Name | Criterion |
|---|---|---|
| V1 | shape classifier accuracy | ≥ 0.95 (8-way) |
| V2 | color classifier accuracy | ≥ 0.95 (8-way) |
| V3 | image↔caption cosine alignment | ≥ 0.70 on held-out |
| V4 | image-prompted LM | given image, model generates correct color+shape ≥ 60 % |
| V5 | language-prompted retrieval | given "red circle", correct image is top-1 in nearest-neighbour search ≥ 80 % |
| V6 | F62 universal-combiner preserved | combiner still satisfies F62 group-action invariants when slots come from CNN |

V6 is the key structural test: the F62 UniversalCombiner —
which has been the constant across F62→F85 — should treat
visual-origin slots the same as language-origin slots. If it
does, PCM has just *demonstrated cross-modal universality of
its core operator*.

### F88 — Literacy: cross-modal binding of printed glyphs (large, ~1 day)

**Goal**: the model recognises printed words and digits as
images and binds them to the *same* token IDs they already
know from language. This is the "reading" step in the child
curriculum.

**Method**:

1. For each token in the TinyStories vocab (~4096 words),
   render it as a small grayscale image (16×64 pixels at 12pt
   font, e.g. DejaVu Sans).
2. Build a second CNN that maps glyph-image → token-embedding.
3. Train with **contrastive cross-modal loss**: aligned
   (glyph, token) pairs pulled together; misaligned pushed
   apart.
4. At inference, the model can:
   * see a sequence of printed glyphs as images,
   * encode each to its bound token-embedding,
   * run the F81 hybrid LM on the *predicted* tokens,
   * continue generating (read-aloud / continuation /
     question-answering on a printed passage).

**Falsifiable invariants** (6):

| ID | Name | Criterion |
|---|---|---|
| L1 | single-glyph accuracy | ≥ 0.90 — glyph → correct token id |
| L2 | OOV glyph behaviour | rare-train-freq tokens still hit ≥ 0.50 |
| L3 | printed-sentence PPL | model PPL on glyph-encoded sentences ≤ 1.5 × text PPL |
| L4 | online glyph learning | F85 teacher loop with printed *new* concepts (zorgon as image of "zorgon" rendered) — same K = 3 acquisition curve |
| L5 | F62 universal-combiner preserved | combiner works on glyph-encoded slots |
| L6 | Episodic + reading | F75 episodic buffer + F78 dispatcher work with glyph-input contexts |

If L1-L6 pass, PCM v10 has the **full child-curriculum
multimodal stack**: hear → speak (F85) → see colours/shapes
(F87) → read printed text (F88).

### Sleep, dreams, and "write" — F89 and beyond

* **F89 writing**: invert F88. Given a token-embedding, the
  model generates the corresponding glyph-image. The decoder
  is a small ConvT or pixel-level generator. Verify that the
  model can render arbitrary tokens it knows, including
  online-learned ones.
* **F90 multimodal sleep**: extend F75 episodic memory to
  store (image, text) pairs jointly. K-means consolidation
  produces cross-modal concept slots that work for both
  recognition and generation.

These are sketches, not commitments. F86–F88 are the priority.

---

## 3. What we keep, what we add

### Reuse without modification (already shipped)

* :class:`pcm.lm.HybridPCMMiniLM` and
  :class:`pcm.lm.TitansPCMMiniLM` — the language LM.
* :class:`pcm.lm.PCMUniversalCombiner` — the F62 operator that
  must remain unchanged across all modalities (the *structural
  invariant* PCM has been verifying since F62).
* :mod:`pcm.episodic` — F75 episodic + sleep consolidation.
* :mod:`pcm.online` — F85 teacher loop.
* :mod:`pcm.agent.heads` — slot encoder + perception heads
  from F64-F72 (already has an ``ImagePerceptionHead`` from
  F72 that we can extend).

### New modules (to be written)

* ``pcm/vision.py`` — F87 ``VisualEncoder`` (CNN → slot) +
  ``CrossModalAligner`` (contrastive / MSE loss).
* ``pcm/literacy.py`` — F88 ``GlyphEncoder`` (CNN → token
  embedding) + ``LiteracySession`` (the F85 teacher loop
  applied to glyph stimuli).
* ``experiments/cognitive_probe_f86.py`` — the cheap probe.
* ``experiments/visual_grounding_f87.py`` — the main vision
  PoC.
* ``experiments/literacy_f88.py`` — the literacy PoC.

### File-system cleanup (already done in this session)

* Removed ~90 ``_smoke``, ``_dbg``, ``_full[2-9]`` output
  directories from F40-F85 era.
* Removed one-off scripts:
  ``smoke_f74/75/76/77/85.py``, ``patch_f80/81_verdict.py``,
  ``verify_f85_concepts.py``,
  ``debug_f64/75_e3.py``, ``check_*``, ``inspect_*``,
  ``recompute_*``.
* Repository is now lean: 12 canonical _full output
  directories from F60-F85, all design docs in ``docs/``,
  all reproducible experiments in ``experiments/``.

---

## 4. Why this is the right next experiment

The user's intuition is *exactly* the developmental
progression observed in human children:

* **0–2 yr**: passive listening + first words (PCM ≈ F79–F81
  pretraining)
* **2–4 yr**: rapid vocabulary expansion via teacher
  interaction (PCM ≈ F85)
* **3–5 yr**: visual concept formation — colors, shapes, faces
  (PCM ≈ F87)
* **4–6 yr**: letter recognition begins (PCM ≈ F88)
* **5–7 yr**: reading fluency, simple writing (PCM ≈ F89+)

By committing to this curriculum, PCM is not racing toward
LLM-scale general competence. It is mapping out *what cognitive
prerequisite each architectural component answers to*. Every
milestone is a piece of falsifiable evidence that PCM's
slot-as-concept design composes across modalities without
needing to retrain the whole stack.

That cross-modal universality — operator structure (F62
combiner) invariant; content (slot embeddings) modality-
specific — is exactly the "structure as substrate" claim the
user articulated four turns ago. F87 and F88 are the next
falsifiable tests.

---

## 5. Recommended order

1. **F86 cognitive probe** (1 hour) — cheap verification that
   the language substrate's *meanings* are already in place.
   Expected: all C1-C5 PASS, confirming we can proceed.
2. **F87 visual grounding** (half day) — the main new
   contribution. The cross-modal universal-operator test
   (V6) is the most novel part.
3. **F88 literacy** (~1 day) — the integrative milestone.
   PCM reads.

If F86 fails any invariant, we know to *fix the meaning*
before grounding vision. If F87 fails V6, we know the F62
combiner is not as universal as claimed and the architecture
needs revision. F88 is gated on F87 passing.
