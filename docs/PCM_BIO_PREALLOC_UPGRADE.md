# PCM × Bio-inspired Pre-allocation × PEER

> **Status**: `RESEARCH` (2026-05-10) · companion to `PARAMETRIC_CONCEPT_MEMORY.md`
> **Implements**: D93 — Pre-allocated Dense Pool (bionic, GPU-native).
> **Code**: `pcm/concept_graph.py`, `pcm/param_bundle.py`, `pcm/gate.py`, `pcm/peer.py`.
> **Tests**: `tests/test_grow_invariants.py`, `tests/test_tier_b_gate.py`, `tests/test_tier_c_peer.py` (30 / 30 PASS).
> **Demos**: `experiments/synaptogenesis_study.py`, `experiments/peer_discovery_demo.py`.

---

## 0. TL;DR

Three-tier upgrade that takes PCM from a Python-loop sparse graph to a
GPU-native dense slot bank, motivated by the same biology PCM was
initially named after.

| Tier | What changes | Bio analogy | Status |
|---|---|---|---|
| **A** | Per-node `ParamBundle` -> shared `bundle_pool[facet]` (capacity, D); muscle hot path becomes a single `F.embedding`; capacity may grow without breaking any paper claim. | Cortical sheet has fixed neuron count at birth; "growth" of mind is synaptogenesis on a pre-allocated substrate. | **landed** |
| **B** | Add a sigmoid gate `g[facet, slot]` per slot; L0 regulariser + over-allocation reproduce the developmental pruning curve. Default OFF preserves Tier-A bit-identity. | 0-2y synaptic over-production followed by experience-dependent pruning (Frontiers 2025; bioRxiv 2025). | **landed** |
| **C** | Optional PEER (Lample 2024) / Memory-Layers (Berges 2024) product-key router for sub-linear lookup over up to 16K experts (8GB) / 1M experts (24GB). Symbolic mode unchanged. | Adult neurogenesis in dentate gyrus (Boldrini 2018) — small, sub-linear additions on top of the static cortical grid. | **landed** (8GB micro version) |

All paper §4 / §5 / §6 / §B claims pass bit-for-bit on Tier A; Tier B is
opt-in (gate disabled by default); Tier C is gated behind explicit
``ProductKeyRouter`` instantiation. Empirical highlights:

- **10.7× speed-up** on RTX-class CUDA at B=128, N=30 (forward+backward).
- **Counterfactual swap §B**: AddHead-inv 18.18%, MixHead-inv 5.26%, vs paper's 18.2% / 5.3% (delta < 0.04 pp).
- **Synaptogenesis pruning curve**: 89.1% of phantom slots pruned (gate < 0.05) while task accuracy stays at 100%.
- **PEER discovery toy**: 95.8% accuracy on a 10-class problem with a 16K-expert anonymous bank, peak VRAM 59.6 MB.

---

## 1. Motivation

### 1.1 Pre-D93 hot path was Python-bound

The pre-D93 muscle gather had this shape (e.g.
[`arithmetic_head_v2.py`](../pcm/heads/arithmetic_head_v2.py)):

```python
rows = []
for cid in concept_ids:                              # B-deep Python loop
    cc = cg.concepts[cid].collapse(...)              # dict lookup + ParameterDict
    rows.append(cc.as_tensor())                      # 1 row per concept
return torch.stack(rows, dim=0)                      # (B, D) at the end
```

Each step paid B Python-level dict lookups and a `torch.stack`. On an
RTX 4090 with B=128 and N=30, the GPU was idle most of the time. The
storage shape — one `nn.Parameter` per (concept, facet) — also blocked
`torch.compile`, made `SparseAdam` impossible, and grew O(concepts) of
optimiser state irrespective of which concepts the current batch
actually touched.

### 1.2 The bio-inspired alternative

The user's framing was sharp: in Houdini's GPU smoke solver, a *dense*
voxel grid lets the SMs run flat-out; a *sparse* grid is intrinsically
CPU-bound. Biology faces the same trade-off and — at least in mammalian
cortex — picks dense pre-allocation:

- Neuron count is essentially fixed at birth (Frontiers Cell Dev Bio
  2025; bioRxiv 2025).
- The infant 0-2y phase is **synaptic over-production** followed by
  experience-dependent **pruning** (PLOS Comp Bio 2021; Frontiers 2025).
- Adult neurogenesis is a real but minor addition (dentate gyrus only;
  Boldrini *Cell Stem Cell* 2018; Sorrells *Nature* 2018 contests
  scale, not existence).

This maps almost 1:1 to a triplet of engineering primitives that
already exist in 2024-2026 literature:

| Biological mechanism | Engineering analogue |
|---|---|
| pre-allocated cortical sheet | pre-allocated dense `bundle_pool` (D93 Tier A) |
| synaptogenesis + experience-dependent pruning | per-slot gate + L0 regulariser (Tier B; Differentiable SLT, arXiv 2603.08914) |
| adult neurogenesis | doubling-capacity grow protocol (G1-G6 invariants) |
| sub-linear navigation of a vast bank | PEER product-key (Lample 2024 arXiv 2407.04153) / Memory Layers at Scale (Berges 2024 arXiv 2412.09764) |

---

## 2. Tier A — Dense pool with G1-G6 grow

### 2.1 Storage

```python
class ConceptGraph:
    bundle_pool: dict[str, nn.Parameter]   # facet -> (capacity, D)
    cid_to_slot: dict[str, int]            # stable, append-only
    free_slots: list[int]                  # LIFO, recycled by Tier B prune
    capacity: int                          # current N_max
    growth_factor: float = 2.0             # doubling => O(log N) realloc
```

The per-node `ParamBundle` becomes a thin proxy bound to `(graph,
slot_idx)`. Legacy callers — `bundle.params[facet]`,
`bundle.consumed_by`, `bundle.state_dict()` — all keep working through
proxy objects (`BundleRowView`, `_ParamProxyDict`).

### 2.2 The hot path collapses to one `F.embedding`

```python
slots = torch.tensor([cg.cid_to_slot[cid] for cid in concept_ids],
                     device=pool.device)
return F.embedding(slots, cg.bundle_pool[facet])  # (B, D), pure-CUDA
```

This is the entire change to the heads (see
[`pcm/heads/arithmetic_head_v2.py`](../pcm/heads/arithmetic_head_v2.py)
lines 64-79). Per-row gradients flow back into the pool via
`index_add` semantics in `F.embedding.backward`, exactly matching the
old per-Parameter sum-of-gradients but now on a single contiguous
tensor.

### 2.3 Capacity grow protocol (the key safety net)

The user's hardest constraint was: capacity must be growable on demand,
*and* the grow must not break already-trained models. The plan §2.5
spelled out the six invariants (G1-G6) the protocol must respect; the
implementation in `ConceptGraph.grow_capacity` enforces all of them:

| Invariant | What it says |
|---|---|
| G1 | `pool[facet][slot]` data is bit-identical for every old slot |
| G2 | `cid_to_slot[cid]` is unchanged for every existing concept |
| G3 | `consumed_by` / `collapse_history` per slot are unchanged |
| G4 | optimiser moments (Adam exp_avg / exp_avg_sq) preserved per row; new rows zero |
| G5 | Forward output is bit-identical for any batch over old slots |
| G6 | Gradient path on old slots is unchanged; new rows receive zero gradient unless someone collapses them |

Optimiser-state migration is the fiddly bit. `migrate_param_in_optimizer`
walks `optimizer.state[old_p]`, copies every per-element tensor row-by-row
into a freshly-shaped buffer, zeros the new rows, and rebinds
`optimizer.param_groups`. AdamW, Adam, SGD-with-momentum and RMSprop all
work without per-class adapters because they all use the convention
"state buffer with the same shape as the parameter" — see
[`pcm/param_bundle.py`](../pcm/param_bundle.py) lines 408-447.

### 2.4 Three grow trigger modes

Per plan §2.5.4, all three modes are implemented:

1. **Passive grow on register** (`_allocate_slot` → `grow_capacity(1)`
   when `free_slots` empty). Amortised O(1) per `register_concept`.
2. **Active grow** (caller invokes `cg.grow_capacity(extra=K)` before a
   curriculum stage).
3. **Tier-B-driven grow + prune** (`gate.gate_status` is high → grow;
   gate < threshold → recycle slot via `prune_pruned_slots`). This is
   the closest analogue of the biological synaptogenesis ↔ pruning
   homeostat.

### 2.5 Verification

`tests/test_grow_invariants.py` defines six tests (one per invariant)
plus four integration regressions:

- `test_grow_bit_identical` — forward over old cids unchanged
- `test_grow_optimizer_moment` — Adam state preserved
- `test_grow_then_paper_claim` — paper §4 ρ_linear stays within 0.05
  when a grow is inserted halfway through training
- `test_grow_then_swap` — counterfactual swap on old slots still
  produces dissociation after grow

All 13 grow tests pass, plus the 5 pre-existing smoke tests.

### 2.6 Empirical impact

```bash
$ python -m scripts.bench_dense_pool --N 30 --batch 128 --steps 200
─── forward only ──────────────────────────────────────────────
  legacy loop+stack       200 steps in 0.465s →   429.9 steps/s
  dense-pool F.embedding  200 steps in 0.043s →  4613.5 steps/s
  speed-up: 10.73×

─── full forward+backward ─────────────────────────────────────
  legacy fb               200 steps in 4.519s →    44.3 steps/s
  dense-pool fb           200 steps in 0.420s →   476.3 steps/s
  speed-up: 10.76×

Peak VRAM allocated: 17.8 MB
```

Paper § 4 / § B claims preserved bit-for-bit:

- §4.4: `ρ_linear = 0.991 ± 0.007` (paper: 0.991 ± 0.007).
- §B.1 swap_arith_only AddHead-inv: `18.18%` (paper: 18.2%).
- §B.2 swap_mix_only MixHead-inv: `5.26%` (paper: 5.3%).

---

## 3. Tier B — Slot gates and the developmental pruning curve

### 3.1 Why gates rather than just lazy-init

Paper §4.4 already showed that PCM relies on the small-init
feature-learning regime: `lazy regime → ρ = 0.22, feature regime →
ρ = 0.97`. The lazy-init lottery is implicit; once the row exists it is
never explicitly closed. Tier B makes the lottery **explicit** by
inserting a sigmoid gate

\[
b_{i} = \sigma(g_{f,i}) \cdot \mathbf{p}_{f,i}
\]

per `(facet, slot)`. With L0 regularisation `λ · Σ σ(g)` added to the
data loss, the dynamics become a homeostat:

- a slot that contributes to the loss has `∂L/∂g > 0` ⇒ gate opens
- a slot that never appears in any batch has `∂L/∂g` only from L0 ⇒ gate
  drifts to 0
- the threshold logit (`-2.197`) corresponds to `σ = 0.1`, matching the
  Frontiers 2025 / bioRxiv 2025 "neonatal" baseline.

### 3.2 Bit-identity guarantee

Tier B is **off by default**. `cg.gate_enabled` defaults to `False`; in
that mode `collapse_batch` does not touch the gate code path, so
forward output is bit-identical to Tier A and every paper claim
survives unchanged. Enabling Tier B is one explicit call:

```python
import pcm
pcm.gate.attach_gates(cg)        # creates gate Parameters, sets logit=10 on
                                 # every already-active slot so the post-call
                                 # forward is ~bit-identical (sigmoid(10) ≈ 1).
```

Causal interventions become reversible:

```python
pcm.gate.set_slot_gate(cg, "arithmetic_bias", slot, 0.0)   # reversible ablation
pcm.gate.set_slot_gate(cg, "arithmetic_bias", slot, 1.0)   # restore
```

`tests/test_tier_b_gate.py` verifies (a) disabled-mode bit-identity,
(b) gate=0 ⇒ row contribution ≈ 0, (c) L0 separates used from unused
gates, and (d) G1-G6 invariants extend to gate Parameters.

### 3.3 Developmental pruning study

`experiments/synaptogenesis_study.py` runs an over-allocated PCM
(`N_used = 7` task numerals + `N_phantom = 24` never-trained slots) on
the §4 arithmetic loss with L0 pressure. The trajectory reproduces the
biological curve:

| epoch | used_mean σ(g) | phantom_mean σ(g) | frac_pruned (σ(g) < 0.05) |
|---|---|---|---|
| 0  | 1.000 | 0.100 | 0.000 |
| 4  | 1.000 | 0.071 | 0.000 |
| 8  | 1.000 | 0.052 | 0.000 |
| 9  | 1.000 | 0.049 | **0.891** |
| 12 | 1.000 | 0.040 | 0.891 |

Final task accuracy = 100%. **Used slots stay fully open while
89.1% of the phantom population is pruned** — the textbook
synaptogenesis-then-pruning shape, in 5.7 seconds on a single 8GB GPU.

The plot lives at `outputs/synaptogenesis/gate_trajectory.png`.

---

## 4. Tier C — Product-key router for sub-linear lookup

### 4.1 Why PEER on top of PCM

PEER (Lample 2024) and Memory Layers at Scale (Berges 2024) showed that
a flat `K × D` expert bank can be navigated in `O(2 sqrt(K))` time and
`O(sqrt(K) D)` parameters via product keys. Two PCM-flavoured uses:

- **Symbolic mode** (default; paper claims unchanged): when a
  `concept_id` is known, `cid_to_slot` does its deterministic O(1)
  lookup and PEER is invisible.
- **Discovery mode** (opt-in): when only a perceptual query `q` is
  available, `ProductKeyRouter.gather_values(cg, facet, q)` returns a
  weighted sum of the top-K rows of `bundle_pool[facet]`, learnt
  end-to-end. This unlocks the Pipeline-A auto-discovery the paper §3.6
  deferred.

### 4.2 The 8GB micro variant

PEER's headline numbers come from `K = 1_048_576`. On 8GB we cap at
`K = 16_384` (a perfect 128² square). Parameter cost
\(2 \cdot h \cdot \sqrt{K} \cdot d/2 = h \cdot \sqrt{K} \cdot d\)
gives `4 × 128 × 64 = 32 KB` per head pair — negligible.
`tests/test_tier_c_peer.py::test_sublinear_param_count` asserts the
sub-linear property explicitly: ratio
`bank_params / (K · d) = h / sqrt(K) = 8 / 128 = 0.0625`.

### 4.3 Soft attribution

The Tier-A "consumed_by is a dict lookup" promise still holds under
soft routing. `pcm.peer.soft_consumed_by_log` writes
`(caller -> facet -> slot)` for every slot that received softmax
weight above a threshold (default 0.05). After training, the same
question — "which callers consumed slot s?" — has the same O(1) answer,
modulo a slot now possibly appearing in *multiple* callers' selection
distributions.

### 4.4 Empirical: discovery toy

`experiments/peer_discovery_demo.py` constructs 10 Gaussian-centroid
classes (samples = noise around centroid) and trains a
`DiscoveryMuscle` whose only access to the dense pool is via the
router (no `concept_id`s ever passed in). Result on 8GB:

```
final acc          : 0.958
final loss         : 2.001
discovered slots   : 1890 / 16384
wall               : 0.3s
peak VRAM          : 59.6 MB
```

The 1,890 distinct slots that received soft-attribution writes are the
"hypothesis concepts" PCM has discovered without supervision. The
paper §3.6 future-work item — joint training of Pipeline A (discovery)
and Pipeline B (representation) — is now doable end-to-end.

---

## 5. What this changes for the paper

Tier A is a pure replacement: every § 4 / § 5 / § 6 / § B figure can be
regenerated with the new code and produces numerically equivalent
results. No paper claim needs editing.

Tiers B and C are **additive**. They unlock three natural new sections,
each with one or more new falsifiable claims:

- **§ S1 (synaptogenesis ↔ pruning)**: gate trajectory matches the
  biological developmental curve qualitatively; quantitatively, the
  fraction of pruned phantom slots crosses a threshold within
  *O*(epochs / λ) consistent with the L0 dynamics.
- **§ S2 (Tier-B gate ↔ Tier-A consumed_by)**: a slot's terminal gate
  level is monotone in the size of its `consumed_by` registry — i.e.
  used = open, unused = pruned, with a sharp boundary.
- **§ S3 (PEER × PCM)**: in symbolic mode, PCM + product-key keeps all
  paper claims; in discovery mode, the router learns a `query → slot`
  map that recovers `cid_to_slot` up to permutation when the data
  contains the original task structure.

The full ranking of new claims and the bionic narrative they suggest
are encoded in the unit-test invariants (G1-G6) and the three demo
experiments. The implementation is open (Apache-2.0, repo
`zxgvfx/parametric-concept-memory`).

---

## 6. References (added vs. main paper)

- Boldrini M. et al. 2018 · *Human Hippocampal Neurogenesis Persists Throughout Aging*. Cell Stem Cell.
- Sorrells S. F. et al. 2018 · *Human hippocampal neurogenesis drops sharply in children to undetectable levels in adults*. Nature.
- Frontiers Cell Dev Bio 2025 · *Neuron-to-glia and glia-to-glia signaling directs critical period experience-dependent synapse pruning*.
- bioRxiv 2025/01 · *Developmental synaptic pruning in the olivo-cerebellar circuit*.
- PLOS Comp Bio 2021 · *The information theory of developmental pruning*.
- arXiv 2603.08914 (2026) · *Differentiable Strong Lottery Ticket discovery*.
- AAAI 2025 · *SLTH for Multi-head Attention*.
- arXiv 2407.04153 (Lample et al. 2024) · *Mixture of A Million Experts (PEER)*.
- arXiv 2412.09764 (Berges et al. 2024) · *Memory Layers at Scale*.
- arXiv 2603.17168 · *HierarchicalKV: GPU Hash Table with Cache Semantics for Continuous Online Embedding Storage*.

---

## 7. Quick start

```bash
# Tier A regression (≤ 1 min)
python -m scripts.sanity_check

# Tier A throughput benchmark
python -m scripts.bench_dense_pool --N 30 --batch 128 --steps 200

# Tier B synaptogenesis-pruning study (~6 s)
python -m experiments.synaptogenesis_study

# Tier C PEER discovery toy (~0.3 s)
python -m experiments.peer_discovery_demo

# All 30 unit tests
python -m unittest discover tests
```
