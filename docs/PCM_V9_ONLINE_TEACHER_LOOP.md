# PCM v9.0 — Online Teacher Loop (F85)

**Date**: 2026-05-18
**Status**: design fixed; implementation begins immediately.
**Predecessor**: F81 Hybrid PCM (closed the F79 perplexity gap;
matched GPT on TinyStories).
**Scientific claim being tested**: PCM can learn *new concepts*
through online teacher interaction in K = O(10–20) corrections,
*without* catastrophic forgetting and *without* offline retraining
— the way human children acquire new vocabulary.

---

## 1. The cognitive paradigm we are emulating

Real child language acquisition (Tomasello, Saffran, Frank 2013;
Bergelson & Aslin 2017) is **multi-timescale online learning**:

| Phase | Timescale | Mechanism (in humans) | PCM substrate |
|---|---|---|---|
| **Ambient exposure** | weeks–years | Hippocampal indexing of statistics; long-range patterns | F81 hybrid backbone (pretraining) |
| **Production attempt** | seconds | Language production system | LM generation |
| **Teacher correction** | seconds | Caregiver provides target | *new in F85* |
| **Episodic recall ("I was told")** | minutes–hours | Hippocampal episodic memory | F75 ``EpisodicBuffer`` |
| **Salience-gated retention** | days | Hippocampal → cortical consolidation | F75 ``LongTermEpisodicTrace`` |
| **Sleep consolidation** | overnight | NREM replay → semantic memory | F75 ``consolidate_to_concept_graph`` |
| **Slow grammatical reweighting** | months | Synaptic re-tuning across cortex | online micro-gradient (*new in F85*) |

PCM already has six of these eight substrates. F85 adds the
remaining two: the **teacher-correction protocol** and the
**online update policy**. Everything else is reuse.

---

## 2. Architecture overview

```
                  teacher correction event:
                  (context, model_output, target_output, surprise)
                                  │
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
    ╔═══════════════╗   ╔══════════════════╗   ╔════════════════════╗
    ║   M1: Fast    ║   ║  M2: Mid-term    ║   ║   M3: Slow         ║
    ║   episodic    ║   ║  sleep consol.   ║   ║   micro-gradient   ║
    ║   write       ║   ║                  ║   ║                    ║
    ╠═══════════════╣   ╠══════════════════╣   ╠════════════════════╣
    ║ trigger:      ║   ║ trigger:         ║   ║ trigger:           ║
    ║ every event   ║   ║ every N events   ║   ║ same concept       ║
    ║               ║   ║                  ║   ║ corrected ≥ K times║
    ║ writes to:    ║   ║ K-means clusters ║   ║ AND/OR salience    ║
    ║ EpisodicBuffer║   ║ episodes → adds  ║   ║ above τ            ║
    ║ + memory      ║   ║ concept slots to ║   ║                    ║
    ║ readout cache ║   ║ ConceptGraph     ║   ║ writes to:         ║
    ║               ║   ║                  ║   ║ token embedding    ║
    ║ persistence:  ║   ║ persistence:     ║   ║ (slow LR=1e-4) +   ║
    ║ FIFO ~ 200    ║   ║ permanent        ║   ║ optionally last LN ║
    ║ corrections   ║   ║                  ║   ║                    ║
    ║               ║   ║                  ║   ║ replay: sample N   ║
    ║ retrieval:    ║   ║ retrieval:       ║   ║ pretrain examples  ║
    ║ F83 cosine    ║   ║ via concept slot ║   ║ to prevent         ║
    ║ readout layer ║   ║ similarity       ║   ║ catastrophic       ║
    ║               ║   ║                  ║   ║ forgetting         ║
    ╚═══════════════╝   ╚══════════════════╝   ╚════════════════════╝
```

The pretrained F81 backbone stays **frozen by default**. M3
selectively unfreezes:
* the **embedding** of the novel-concept token (always)
* the **final LayerNorm** (always)
* the **F62 combiner** of the *last* PCM layer (optional, off
  by default — too much capacity for the K = 10–20 regime)

This is *parameter-efficient online learning* — analogous to
LoRA / adapter / prefix-tuning but with explicit cognitive
mapping.

---

## 3. Test set design — eight fictional concepts

Use vocabulary expansion at build time:

```python
RESERVED_CONCEPTS = [
    # 4 "animal-like" — selectional class ANIMAL, allows EAT/SLEEP/RUN
    ("zorgon",  "ANIMAL", "animate-singular"),
    ("floob",   "ANIMAL", "animate-singular"),
    ("snerf",   "ANIMAL", "animate-singular"),
    ("vooz",    "ANIMAL", "animate-singular"),
    # 4 "food-like" — selectional class FOOD, allows EATEN/COOKED/TASTED
    ("glimber", "FOOD",   "inanimate-singular"),
    ("quark",   "FOOD",   "inanimate-singular"),
    ("mibble",  "FOOD",   "inanimate-singular"),
    ("prag",    "FOOD",   "inanimate-singular"),
]
```

These 8 strings **never occur in the TinyStories pretraining
corpus** (verified via grep over the corpus). We assign them to
vocab IDs 4088–4095 (last 8 of vocab=4096) at vocab-build time.

For each concept, build:
* **Teaching set (30 sentences)**: simple SVO with the concept,
  matching its selectional class. E.g., for ``zorgon`` (ANIMAL):
  * "the zorgon ate the apple ."
  * "a zorgon ran in the forest ."
  * "the small zorgon slept in the box ."
  * (5 templates × 6 distractor variations = 30)
* **Test set (30 sentences)**: held-out, **different templates**,
  **different distractor vocabulary**:
  * "yesterday the zorgon found a ball ."
  * "the zorgon was happy to see lily ."

Held-out test sentences use:
* Different verbs than teaching (to test compositional
  generalisation of selectional class)
* Different surrounding adjectives, places, objects
* Sometimes embed the concept mid-sentence rather than at SVO
  positions

---

## 4. Six falsifiable invariants

| ID | Name | Criterion | What it falsifies |
|---|---|---|---|
| **O1** | Acquisition | After K=15 corrections per concept, PPL on held-out test sentences with that concept drops by ≥ 50 % vs pre-teaching baseline | "Online learning works at all" |
| **O2** | No catastrophic forgetting | Pretrain-val PPL (TinyStories) stays within 1.10 × of pre-online baseline | "Online updates wreck pretrained knowledge" |
| **O3** | Selectional generalisation | After teaching "the zorgon ATE the apple", the model assigns higher likelihood to "the zorgon SLEPT" than to "the apple SLEPT" (selectional class is learnt, not just lexical co-occurrence) | "Only memorisation, no class abstraction" |
| **O4** | Sample efficiency | Number of corrections needed to halve concept PPL is ≤ 20 (matching child acquisition rates) | "PCM needs LLM-scale data even for one concept" |
| **O5** | Mechanism routing emergence | RECALL fraction increases monotonically across the K corrections (the F78 dispatcher pattern emerges); micro-gradient fires only on uncommon-novel cases | "Updates are random; no learned policy over update modes" |
| **O6** | Sleep consolidation purity | After ``consolidate_to_concept_graph`` over the accumulated episodes, the 8 concept clusters are ≥ 0.65 pure against the {ANIMAL, FOOD} ground-truth split | "Sleep doesn't extract class structure" |

A **PASS** on O1 + O2 alone is already a publishable result:
*PCM learns new concepts online in ~15 corrections, without
catastrophic forgetting*. O3–O6 are stronger structural claims.

---

## 5. Update policy in detail

For each teacher correction event:

### M1 — Episodic write (always fires)

```python
hidden = model.hidden_states(context_tokens)
# Compute the surprise at the concept-token position
surprise = -log(P(target | context))
# Write to F83 readout's kNN cache
model.memory_readout.imprint(hidden, n_per_batch=1)
# Also write to F75 long-term trace if surprise is high
if surprise > SALIENCE_THRESHOLD:
    long_term_trace.maybe_imprint(
        slot=hidden[pos], timestamp=t,
        salience=surprise,
        metadata={"concept_id": novel_id},
    )
```

### M3 — Micro-gradient (fires when …)

* Same concept has been corrected ``≥ K_GRAD`` times (default
  3), **or**
* Surprise > 2 × baseline_surprise (very surprising correction)

```python
optimizer = AdamW([
    model.tok_emb.weight,        # always
    model.ln_final.parameters(), # always
], lr=1e-4, weight_decay=0)

# Replay batch: 1 correction + 7 pretraining examples
replay = random.sample(pretrain_replay_buffer, k=7)
batch = [(context, target)] + replay
loss = ce_loss(model(batch.x), batch.y)
loss.backward()
opt.step()
```

The replay buffer is populated at the end of pretraining (the
last 256 training examples are kept). Catastrophic forgetting is
prevented by ensuring every gradient step sees 7 pretraining
examples for each 1 correction example.

### M2 — Sleep consolidation (periodic)

After every ``N_SLEEP`` (default 50) corrections:

```python
# Collect (hidden, target_token_id) pairs from the episodic
# buffer
slots, labels = buffer.dump()
# K-means with one cluster per novel-token seen so far
cgraph = consolidate_to_concept_graph(
    EpisodicBuffer.from_slots(slots),
    n_clusters=n_novel_concepts_seen,
    n_restarts=5,
)
# Replace token embeddings of novel concepts with cluster
# centroids (averaging with current embedding for stability)
for c, cid in zip(cgraph["centroids"], novel_concept_ids):
    new = 0.5 * c + 0.5 * model.tok_emb.weight[cid]
    model.tok_emb.weight.data[cid] = new
```

This is the "overnight" step: episodic memories get distilled
into semantic embeddings.

---

## 6. The five-step F85 experiment

1. **Pretrain** F81 Hybrid PCM on TinyStories with 4088 vocab
   (8 reserved IDs unused) — ~10 min on RTX 3070.
2. **Snapshot** model weights + pretrain-val PPL for the O2
   baseline.
3. **Online phase**: for each of 8 concepts, present 15
   teaching examples one at a time via
   :class:`OnlineTeacherSession`, applying M1/M3 updates. Check
   held-out test PPL every 5 corrections.
4. **Sleep**: invoke M2 once after all 120 = 8 × 15 corrections.
5. **Evaluate**: O1 (concept PPL), O2 (pretrain-val PPL), O3
   (selectional generalisation), O4 (K-curve), O5 (mechanism
   counts), O6 (consolidation purity).

Expected wall-clock: ~25 min total. Output:
``outputs/f85_full/summary.json`` with full results.

---

## 7. What this commits PCM to

If F85 passes O1 + O2 + O3:

* PCM v9.0 is the first compact LM that can **learn novel
  concepts online through teacher interaction, without
  catastrophic forgetting**.
* The architecture composes all previous PCM components
  (F62 combiner, F75 episodic memory, F78 dispatcher, F81
  hybrid attention, F83 memory readout) into a single
  cognitive substrate.
* The "specialist child learner" framing is upgraded one
  more notch: not just *learns once*, but *keeps learning*.

If F85 fails on any of O1–O6, the failure mode itself is
informative — it tells us exactly which substrate (fast / mid /
slow) is the bottleneck.

---

## 8. Module layout

* ``pcm/lm_synthetic.py`` — add ``RESERVED_CONCEPTS`` +
  ``generate_concept_teaching_sentence`` +
  ``generate_concept_test_sentence``.
* ``pcm/online.py`` (new) —
  * :class:`PretrainReplayBuffer`
  * :class:`OnlineTeacherSession` with ``present_prompt``,
    ``receive_correction``, ``update``, ``sleep``
  * :class:`MechanismCounters` for O5
* ``experiments/online_teacher_f85.py`` — full pipeline.
* ``tests/test_online_teacher.py`` — unit tests for
  ``OnlineTeacherSession``.
