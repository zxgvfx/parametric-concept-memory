# PCM v1 → v2 Migration Guide

**Status: applies to F49 freeze (May 2026).**

This guide explains how to port a v1-style pair-input head — the
archetype that produced the §7.5-space `mixed_OOD = 0.000` ceiling
and the S6 number vector-analogy collapse — to PCM v2 dual-channel
+ RPE. The v1 head is preserved on the slot facet; the v2 head
adds the attribute and RPE channels alongside it.

The guide is **opt-in**: every existing v1 head keeps working
without modification. v2 is what you reach for when:

* the task answer depends on a **displacement / difference**
  between two concepts (direction, size delta, hue rotation,
  phonetic feature delta, semantic offset), AND
* you observe one of the canonical v1 failure modes:
  * `mixed_OOD ≈ 0.000` even after Tier-G runs;
  * vector analogy `top1 < chance`;
  * Tier-G dead-codebook (every concept maps to one prototype).

---

## 1. Five-line minimum-impact migration

### 1.1 Register a dual-channel facet pair

```python
from pcm import ConceptGraph
from pcm.dual_channel import register_dual_channel_facet

cg = ConceptGraph(feat_dim=32)
# ... register concepts as usual ...
register_dual_channel_facet(
    cg, base="motion", slot_dim=32, attr_dim=16,
)
# This creates two paired facets:
#   "motion_slot" — clustered by Tier-G (legacy behaviour)
#   "motion_attr" — excluded from Tier-G, trained by InfoNCE /
#                   arithmetic / successor losses
```

### 1.2 Replace the head

| v1 (concat MLP) | v2 (DualChannelPairHead) |
| --- | --- |
| `class MoveHead(nn.Module)` | `from pcm.heads import DualChannelPairHead` |
| `self.fc1 = nn.Linear(2*facet_dim, hidden)` | `head = DualChannelPairHead(n_classes=5, slot_dim=32, attr_dim=16, rpe_ranges=[(-6, 6), (-6, 6)])` |
| `head(ids_a, ids_b, cg, tick)` | `pair_collapse_and_forward(head, cg, base_facet="motion", ids_a=ids_a, ids_b=ids_b, slot_shape=(32,), attr_shape=(16,), deltas=(dr, dc), tick=t)` |

### 1.3 Compute (Δr, Δc) at the call site

The caller is responsible for computing the integer displacement
to feed the RPE table. For PCM domains where concept ids encode
their coordinates (e.g. `concept:space:r_c`), this is a one-line
parse:

```python
def _displacement(triples, *, n_cols):
    # triples: list of ((r_a, c_a), (r_b, c_b), label)
    dr = torch.tensor([t[1][0] - t[0][0] for t in triples])
    dc = torch.tensor([t[1][1] - t[0][1] for t in triples])
    return dr, dc
```

For domains where concepts have no obvious axis (e.g. arbitrary
phonemes), pre-compute a `cid → (v, m, p)` mapping at concept
registration and look up the diffs from there.

### 1.4 Add the auxiliary identity head

```python
from pcm.heads import SlotIdentityAuxHead

aux = SlotIdentityAuxHead(slot_dim=32, n_concepts=49)

# Train aux side-by-side with the main head; targets are the
# concept's flat index (cell, digit, phoneme id, hue, ...).
aux_logits = aux(slot_rows)
loss_aux = F.cross_entropy(aux_logits, concept_ids_flat)
```

This drives the slot facet to learn a clean cluster-aligned
representation, exactly as `RowIndexHead` / `LastDigitHead` did
in v1.

### 1.5 Add the auxiliary attribute losses

Three loss primitives from `pcm.dual_channel` give the attr facet
its `attr_a + attr_c − attr_b ≈ attr_d` structure:

```python
from pcm.dual_channel import (
    arithmetic_consistency_loss, successor_consistency_loss,
)

# After collapsing the full attr table:
_, attr_table = collapse_dual_channel(
    cg, caller="attr-loss", base_facet="motion",
    concept_ids=all_cids,
    slot_shape=(32,), attr_shape=(16,), tick=t,
)
loss_arith = arithmetic_consistency_loss(
    attr_table, ia, ib, ic, id_,
)
loss_succ = successor_consistency_loss(attr_table)
total = loss_main + 1.0 * loss_arith + 5.0 * loss_succ
```

Default weight schedule (validated on number V2 = 1.000):

| weight | value |
| --- | --- |
| `loss_main` | 1.0 |
| `loss_arith` | 1.0 |
| `loss_succ` | 5.0 (norm penalty β default 1.0 inside `successor_consistency_loss`) |
| `info_nce` | 0.0 (only useful when concepts share slots) |
| `spread` | 0.0 (only when arith fails to spread) |

---

## 2. Choosing the gate mode (v2 V3 + A1/A2 findings)

When you enable both the RPE path and the slot attention path
inside `DualChannelPairHead`, the slot logits compete with the
RPE logits. **A1 (manual schedule)** is the recommended default:

```python
head = DualChannelPairHead(
    n_classes=5, slot_dim=32, attr_dim=16,
    rpe_ranges=[(-6, 6), (-6, 6)],
    gate_mode="schedule",
)

# In the training loop:
total_steps = epochs * steps_per_epoch
for global_step, batch in enumerate(loader):
    head.set_progress(global_step / max(total_steps - 1, 1))
    # ... forward / backward as usual
```

`set_progress` decays the slot path's contribution from 1.0 →
0.0 over the first half of training. Validated on V3 to give
`mixed_OOD = 1.000 ± 0.000`.

If you must use a learned gate (e.g. when training cycles too
short for a meaningful schedule), pair it with an L1 penalty:

```python
head = DualChannelPairHead(..., gate_mode="learned")

# In training loop, β ≈ 0.1 was sufficient on V3:
loss = loss_main + 0.1 * head.gate_l1()
```

**Do not** use `gate_mode="learned"` without an L1 penalty —
F46 showed the gate stays at ≈0.98 across all reward strengths,
and `mixed_OOD` falls to 0.72.

---

## 3. Mapping v1 heads to v2 idioms

| v1 head | task | v2 replacement | rpe_ranges |
| --- | --- | --- | --- |
| `MoveHead` (5×5 grid, direction) | (a, b) → 5-class | `DualChannelPairHead(n_classes=5, …, rpe_ranges=[(-N+1, N-1), (-M+1, M-1)])` | 2-D |
| `DistanceHead` (5×5, L1 dist) | (a, b) → 9-class | `DualChannelPairHead(n_classes=9, …, rpe_ranges=[(-N+1, N-1), (-M+1, M-1)])` | 2-D |
| `QuadArithHead` (number arith) | (a, b, op) → c | not directly RPE-able (3-input); keep v1 |  |
| `LastDigitHead` (number identity) | a → digit | `SlotIdentityAuxHead` | n/a (single input) |
| `ColorMixingHead` (hue mix) | (a, b) → c | `DualChannelPairHead(n_classes=N, …, rpe_ranges=[(0, N-1)])` | 1-D cyclic |
| `ColorAdjacencyHead` | (a, b) → adjacent | `DualChannelPairHead(n_classes=2, …, rpe_ranges=[(0, N-1)])` | 1-D cyclic |
| `RipeFruitHead` | a → ripe | `SlotIdentityAuxHead` (binary) | n/a |
| `RowIndexHead` (cell → row) | a → row | `SlotIdentityAuxHead` | n/a |
| `MaskInferHead` (4-neighbour) | (n1..n4) → centre | not directly RPE-able (k-input, k>2); keep v1 |  |
| `InverseMoveHead` (a, dir → b) | (a, dir) → b | `DualChannelPairHead(n_classes=N_cells, slot_dim=…, rpe_ranges=[(0, n_dirs-1)])` (with dir as the displacement axis) | 1-D |
| `MinimalPairHead` (V/M/P feature) | a → V/M/P bit | `SlotIdentityAuxHead` (multi-bit) | n/a |

**General rule of thumb:**

* **Single input** (`a → label`) → `SlotIdentityAuxHead`.
* **Pair input where answer depends on displacement** → `DualChannelPairHead`
  with `rpe_ranges` matching the displacement axes.
* **Pair input where answer depends on absolute identities** (rare in PCM
  domains; e.g. "are these two cells both landmarks?") → `DualChannelPairHead`
  with `rpe_ranges=None` and `use_slot_path=True` only.

---

## 4. When NOT to migrate

* The v1 head already passes its OOD invariants and you have no S6/S3 ceiling.
* The task is not pair-input.
* The displacement space is unbounded and discrete (e.g. arbitrary text
  edit distance) — RPE-as-lookup cannot scale; wait for functional RPE
  in v3.

---

## 5. Reproducibility checks before migration

Before migrating a v1 baseline, run:

```bash
# Confirm v2 internals work correctly on your platform.
python -m pytest tests/test_dual_channel.py tests/test_concept_graph_device.py -q

# Reproduce the v3 saturation (~3 min, 5 seeds).
python -m experiments.space_rpe_poc --variant rpe_only \
    --n-seeds 3 --epochs 20 --steps-per-epoch 240 \
    --out outputs/v3_rpe_only_smoke

# Reproduce the v2 V1 + V2 invariants (~5 min, 3 seeds).
python -m experiments.number_dual_channel_poc \
    --n-seeds 3 --epochs 20 --steps-per-epoch 200 \
    --out outputs/v2_number_smoke
```

Expected outputs:

* `tests/test_dual_channel.py`: 19 passed
* `tests/test_concept_graph_device.py`: 2 passed (3 if CUDA available)
* `space_rpe_poc rpe_only`: `mixed_OOD = 1.000 ± 0.000`
* `number_dual_channel_poc`: `V1_NMI = 1.000 ± 0.000`, `V2_top1 = 1.000 ± 0.000`

If any of these falsifies, do not migrate — file an issue with the
output of `pytest -v` and the smoke summary.

---

## 6. Reference implementations

* `experiments/number_dual_channel_poc.py` — V1 + V2 reference
  (number domain, 1-D ordinal). Demonstrates `successor_consistency_loss`
  with norm penalty as the way to break the trivial `attr=0` minimum.
* `experiments/space_dual_channel_poc.py` — V3 partial-pass reference
  (space domain, 2-D lattice). Shows `DualChannelMoveHead` with
  translation-invariant slot path; pre-RPE.
* `experiments/space_rpe_poc.py` — V3 saturation reference. Three
  variants (rpe_only / rpe_plus_attn / gate-mode sweep) with the
  full `attn_logit_scale`, `gate_mode`, `reward_alpha`, `gate_l1_beta`
  ablation hooks.
* `experiments/rpe_cross_domain.py` — cross-domain reference
  (number/colour/phoneme). Validates the F48 pattern that RPE
  saturates 4/4 domains.

All four scripts run end-to-end on CPU in under 2 hours total
(GPU recommended); none requires more than 32 GB RAM.
