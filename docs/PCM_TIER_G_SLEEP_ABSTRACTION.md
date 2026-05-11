# PCM Tier-G · Sleep Abstraction Pass

> **Status**: `RESEARCH` (2026-05-10) · companion to
> [`PARAMETRIC_CONCEPT_MEMORY.md`](PARAMETRIC_CONCEPT_MEMORY.md),
> [`PCM_BIO_PREALLOC_UPGRADE.md`](PCM_BIO_PREALLOC_UPGRADE.md), and
> [`PCM_NODE_AS_FUNCTION_DESIGN.md`](PCM_NODE_AS_FUNCTION_DESIGN.md).
> **Implements**: D95 — Sleep-cycle structure abstraction (NREM-style
> codebook compression + interleaved replay).
> **Code**: [`pcm/sleep.py`](../pcm/sleep.py) (NEW),
> [`pcm/dna_ops.py`](../pcm/dna_ops.py) (+1 op).
> **Tests**: [`tests/test_tier_g_sleep.py`](../tests/test_tier_g_sleep.py)
> (NEW; G1–G6 invariants).
> **Demo**: [`experiments/sleep_pass_demo.py`](../experiments/sleep_pass_demo.py)
> (NEW; before/after ρ on §4 numbers).
> **Authority**: 2024–2026 cognitive-neuroscience literature on sleep-
> dependent abstraction (Aton et al. 2024 *Nat Commun*; Liu, Sibille &
> Dragoi 2024 *Nat Neurosci*; bioRxiv 2026.04.10.717748;
> Saxena, Shobe & McNaughton 2022 *PNAS*).

---

## 0. TL;DR

User asked: can PCM be upgraded so that

1. **Cross-instance abstraction** (e.g. "various crows → bird → fowl")
   happens *automatically* during a periodic offline pass, mirroring the
   NREM/REM division of labour now established in mouse and human
   electrophysiology.
2. **Old-task geometry** is protected during this pass via interleaved
   replay, preventing the catastrophic-forgetting failure mode that
   would otherwise wreck the ρ_linear / ρ_circular / ρ_L1 / Procrustes
   metrics established in PAPER §4–§6.
3. The new abstract concepts are **first-class graph nodes**, cookable
   via the existing Tier-D evaluator, with mechanical attribution
   (`bundle.consumed_by["sleep"]`), so PAPER §3.4 命题 1 still holds.

**Answer: STRONGLY YES.** The Tier-D `parametric_muscle_subgraph`
machinery already gives us everything we need to express an abstract
concept as "anchor codebook entry + per-member residual function";
Tier-G adds (a) the offline K-means pass that finds the codebook,
(b) the interleaved-replay loss that protects old muscles, and (c) two
small DNA ops (`concept.codebook_lookup`, `concept.relation_apply`) to
make the abstract subgraphs cookable.

This document specifies:

- **§1** the Tier-G architecture (offline pass, op extensions, new node
  kinds)
- **§2** the algorithm (snapshot → K-means → register → interleaved
  replay → idempotency check)
- **§3** the formal invariants G1–G6 (the falsifiable contract)
- **§4** the public API (`pcm/sleep.py`)
- **§5** integration with the five existing training loops (number /
  color / space / phoneme / purity)
- **§6** validation plan (unit tests + per-domain ρ thresholds)

---

## 1. Tier-G architecture

### 1.1 One new `ConceptNode.kind` + one reused-with-hint

To preserve Tier-D's strict cookability check
(`graph_eval.py:122-126`: `kind != PARAMETRIC_KIND` raises
`SubgraphEvalError`), Tier-G does **not** introduce a new cookable
`kind`. Instead it reuses the existing `parametric_muscle_subgraph`
kind for its cookable nodes and adds a single new non-cookable kind
for the prototype data slots.

| `kind` | Storage | Cookable? | Created by | Notes |
|---|---|---|---|---|
| `data` | `bundle_pool` row | ✗ | `register_concept` (Tier-A) | unchanged |
| `parametric_muscle_subgraph` | `metadata.nodes` op DAG | ✓ | `register_muscle_subgraph` (Tier-D) | now also used by Tier-G abstract relations; `metadata.constants["kind_hint"]` distinguishes the sub-type |
| **`abstract_prototype`** | `bundle_pool` row | ✗ | `pcm.sleep.register_prototype` (NEW) | one row per cluster centre; **non-cookable** by design (it *is* the value, not a function) |

Disambiguation between Tier-D heads and Tier-G abstract relations
is via the metadata hint:

```python
node.kind == "parametric_muscle_subgraph"
node.metadata["constants"]["kind_hint"] in {"tier_d_head", "abstract_relation"}
```

`pcm.sleep.iter_abstract_relations(cg)` returns only the latter.

A `data` node like `concept:bird:crow:42` (the 42nd crow exemplar) and
its sibling `concept:bird:raven:7` will, after a sleep pass, gain two
upstream nodes in the graph:

- `concept:cluster:bird_outline:0` (`abstract_prototype`) holding the
  K-means centroid on facet `shape_outline`.
- `concept:rel:bird_outline:0:crow:42` (`abstract_relation`) — a
  cookable subgraph that reconstructs `crow:42`'s shape_outline as
  `cluster:bird_outline:0 ⊕ residual`, where `⊕` is a learnable
  `relation_apply` op.

The `abstract_prototype` node is just a normal slot in the same
`bundle_pool[facet]` — no new pool, no new optimiser group, no new
gate. This means it inherits Tier-B gates (if enabled), Tier-C peer
routing (if enabled), grow-capacity migration (G1–G6 of
`tests/test_grow_invariants.py`), and Tier-D cookability for free.

### 1.2 Two new ops in `pcm/dna_ops.py`

```python
"concept.codebook_lookup": kind="collapse"
    fn(facet, codebook_id, *, caller, tick, concept_graph) -> Tensor (1, D)
    # F.embedding lookup of one specific abstract slot, with attribution.

"concept.relation_apply": kind="pure"
    fn(anchor: Tensor, residual: Tensor, mode: str = "add") -> Tensor
    # Currently: mode in {"add", "mul", "concat"}. Future Tier-E: "rotate"
    # / "shift" with learnable params will plug in here.
```

`concept.codebook_lookup` is an alias of `muscle.collapse_facet` with
`concept_ids=[codebook_id]` and `caller="sleep"`. We expose it as a
separate op for two reasons: (a) clarity in the cook DAG (`@anchor` vs
`@residual` reads explicit), (b) it lets the sleep pass stamp a
distinct attribution marker so PAPER §3.4 命题 1's static enumeration
still works.

`concept.relation_apply` is `pure` — no graph touch, no gradient
boundary surprises. The default `"add"` mode reduces the cook subgraph
exactly to `centroid + residual`, which is *bit-identical* to the
original Tier-A `bundle_pool[facet][slot]` row when the residual was
initialised as `(row − centroid).detach().clone()` (G4).

### 1.3 The cook subgraph for an abstract relation

```python
ConceptNode(
    node_id="concept:rel:shape_outline:0:crow:42",
    kind="parametric_muscle_subgraph",   # Tier-D cookable kind, reused
    metadata={
        "inputs": [],  # bindings come from constants
        "constants": {
            "kind_hint":      "abstract_relation",   # Tier-G sub-type marker
            "anchor_facet":   "shape_outline",
            "anchor_id":      "concept:cluster:shape_outline:0",
            "residual_facet": "shape_outline_residual",
            "member_id":      "concept:bird:crow:42",
            "mode":           "add",
        },
        "facet_specs": {},
        "nodes": [
            {"id": "anchor",   "op": "concept.codebook_lookup",
             "args": ["shape_outline", "concept:cluster:shape_outline:0"]},
            {"id": "residual", "op": "muscle.collapse_facet",
             "args": ["shape_outline_residual", ["concept:bird:crow:42"]]},
            {"id": "out",      "op": "concept.relation_apply",
             "args": ["@anchor", "@residual", "add"]},
        ],
        "output": "out",
    },
)
```

Cooking this returns a `(1, D)` tensor identical (within float
tolerance) to what `bundle_pool["shape_outline"][slot_of_crow42]` was
holding *before* the sleep pass. **This is the equivalence that lets
G3 hold** — see §3.

### 1.4 Pool layout after one sleep pass

Before sleep pass, on facet `f` with `K` member rows in capacity `C`:

```
bundle_pool[f] : nn.Parameter (C, D)
  rows 0..K-1 : trained data slots
  rows K..C-1 : free
```

After sleep pass with `k_clusters=3`:

```
bundle_pool[f]                      : nn.Parameter (C, D)
  rows 0..K-1                       : ORIGINAL trained data (untouched, G2)
  rows K..K+2                       : abstract_prototype centroids (3 new slots)
  rows K+3..C-1                     : free

bundle_pool[f + "_residual"]        : nn.Parameter (C, D)   # NEW facet
  rows 0..K-1                       : (original_row − centroid_of_assigned_cluster)
  rows K..K+2                       : zero (centroids have no residual)
  rows K+3..C-1                     : free
```

**G2 is the load-bearing invariant**: the sleep pass does *not* mutate
`bundle_pool[f].data[slot]` for any pre-existing slot. The codebook
information lives in (i) **new slots K..K+2 on the same pool** and
(ii) **a new pool `f + "_residual"`**. Old training-time forward
calls that read `bundle_pool[f][0..K-1]` get the same bytes back.

### 1.5 What the sleep pass writes to attribution

For every new abstract slot $s_k$ on facet $f$:

```python
cg._consumed_by_by_slot[s_k][f] = {"sleep"}
cg._collapse_history_by_slot[s_k][f] = [("sleep", tick)]
cg._active_facets_by_slot[s_k] = {f}
```

For every original slot $s$ that gets a residual on facet `f_residual`:

```python
cg._consumed_by_by_slot[s][f_residual] = {"sleep"}
cg._collapse_history_by_slot[s][f_residual] = [("sleep", tick)]
cg._active_facets_by_slot[s].add(f_residual)
# Note: cg._consumed_by_by_slot[s][f] is UNCHANGED (still the old muscles).
```

This satisfies PAPER §3.4 命题 1's *static enumerability* requirement:
any gradient flowing into `bundle_pool[f][s_k]` after the sleep pass
must come from a cook subgraph whose metadata names
`concept:cluster:f:k`. Since the sleep pass exhaustively enumerates
those subgraphs, we keep the O(1) attribution lookup property.

---

## 2. Algorithm

The sleep pass on a single facet `f` with `k_clusters=K` proceeds in
five phases, all running in `torch.no_grad()` except phase D.

### 2.1 Phase A — Snapshot

```python
pool = cg.bundle_pool[f].detach()                                  # (C, D)
active_slots = sorted({s for s in cg._active_facets_by_slot
                       if f in cg._active_facets_by_slot[s]})       # list[int]
rows = pool[active_slots].clone()                                   # (N, D)
```

Active slots are exactly those the user has actually trained against
on facet `f` — i.e. the rows whose `ρ_linear` / `ρ_circular` etc.
contribute to the PAPER metrics. Inactive slots (allocated for other
concepts but never consumed on this facet) are skipped.

### 2.2 Phase B — Cluster

K-means++ init + Lloyd's iterations, all on PyTorch. Default config:

```python
SleepConfig(
    k_clusters="auto",       # "auto" → max(2, round(sqrt(N) / 2))
    kmeans_iters=32,
    kmeans_tol=1e-5,
    init="kmeans++",
    distance="cosine",       # critical: arithmetic_bias geometry is angular
)
```

Output:

```python
centroids   : Tensor (K, D)
assignments : Tensor (N,)        # int64, in [0, K)
```

**Cosine distance, not L2**: PAPER §4–§6 measures geometry through
**cosine** matrices (`_cos_matrix`, `_rho_circular`, `_rho_L1`). If we
cluster by L2 we'd silently optimise a different objective than the
geometry under test. Cosine clustering with L2-normalised codebook
matches `_l2_normalize`/`F.normalize` reach-throughs already in the
PAPER pipeline.

### 2.3 Phase C — Register prototypes + residuals

```python
for k in range(K):
    proto_id = f"concept:cluster:{f}:{k}"
    cg.register_concept(node_id=proto_id, label=f"PROTO_{f}_{k}",
                        scope="ABSTRACT", provenance=f"sleep:{tick}")
    proto_slot = cg.cid_to_slot[proto_id]
    with torch.no_grad():
        cg.bundle_pool[f].data[proto_slot] = centroids[k]
        cg._record_attribution(proto_slot, f, "sleep", tick)

# Residuals on a NEW facet "<f>_residual":
res_facet = f + "_residual"
cg._ensure_facet(res_facet, shape=tuple(rows.shape[1:]),
                 device=rows.device, init="zero")
for i, slot in enumerate(active_slots):
    with torch.no_grad():
        cg.bundle_pool[res_facet].data[slot] = rows[i] - centroids[assignments[i]]
        cg._record_attribution(slot, res_facet, "sleep", tick)
```

### 2.4 Phase D — Register cookable abstract-relation subgraphs

For every `(member_slot, cluster_k)` pair we wire one cook subgraph
(see §1.3). All subgraphs share `evaluator.module_registry` (no
registered modules — `concept.relation_apply` is `pure`).

### 2.5 Phase E — Interleaved replay (Saxena-style protection)

This is the **only** phase that touches gradients. It exists to
protect old muscles from drift caused by the new facets being seen
during downstream cooking.

```python
for replay_step in range(cfg.replay_steps):       # default 32
    batch = replay_buffer.sample(batch_size=cfg.replay_batch_size)
    losses = []
    for muscle_caller, forward_fn, target in batch:
        out = forward_fn()                         # uses cg as before
        losses.append(F.mse_loss(out, target) if target.dtype.is_floating_point
                      else F.cross_entropy(out, target))
    loss = torch.stack(losses).mean()

    # Hard-mask: zero grad on abstract slots and residual facets so
    # replay protection only updates the ORIGINAL trained rows.
    optimizer.zero_grad()
    loss.backward()
    _mask_abstract_grads_(cg, abstract_slots, residual_facets)
    optimizer.step()
```

The replay buffer is built by the host training loop and passed in as
a `ReplaySource` callable returning `(caller, forward_fn, target)`
triples. We do **not** maintain a buffer inside `pcm/sleep.py` to keep
that file independent of any specific muscle / dataset.

If `cfg.replay_steps == 0`, phase E is skipped (useful for unit tests
and for the first sleep pass on synthetic toy graphs where no muscles
exist yet).

### 2.6 Idempotency

A second `run_sleep_pass(cg, ...)` immediately after the first, with
the same data, **must be a no-op** (G6). We achieve this by:

1. Detecting that `bundle_pool[f + "_residual"]` already exists and the
   `_consumed_by_by_slot[s][f + "_residual"]` contains `"sleep"` for
   every active slot;
2. If so, skipping phases A–D and only running phase E (replay
   protection is allowed to repeat).

To force re-clustering after new training, callers pass
`force_recluster=True`.

---

## 3. Invariants — the falsifiable contract

All six are unit-tested by `tests/test_tier_g_sleep.py`.

### G1 — Bit-identity when sleep is not attached

> If `pcm.sleep` is never imported / `attach_sleep(cg)` is never called,
> all PAPER §4–§7 forward outputs are byte-identical to Tier-D.

**Verification**: `tests/test_tier_g_sleep.py::test_g1_no_attach_bit_identical`
constructs the same `ConceptGraph + ArithmeticHeadV2` under two
processes — one that imports `pcm.sleep`, one that does not — and
asserts `torch.equal(out_no_sleep, out_with_sleep_unused)` over 8
random batches.

### G2 — Pool memory safety

> For every `(facet f, slot s)` such that `s ∈ active_slots(f)` *before*
> the sleep pass, `bundle_pool[f].data[s]` is byte-identical *after*
> the sleep pass (modulo phase E updates, which the test disables by
> setting `replay_steps=0`).

**Verification**: snapshot `pool[f][:K_active].data.clone()`, run
`run_sleep_pass(replay_steps=0)`, assert `torch.equal`. Since K-means
runs under `no_grad` and writes only to *new* rows, this holds
trivially — the test is a regression net for future refactors.

### G3 — ρ regression bound under interleaved replay

> After a sleep pass with `replay_steps ≥ 32`, the geometry metrics on
> the *original* facet drop by at most 1% (absolute):
>
> | domain | metric | baseline | post-sleep lower bound |
> |---|---|---|---|
> | numbers N=7 | ρ_linear | 0.973 | ≥ 0.965 |
> | numbers N=30 | ρ_linear | 0.991 | ≥ 0.981 |
> | colors 12 hues | ρ_circular | 0.977 | ≥ 0.965 |
> | space 5×5 | ρ_L1 | 0.860 | ≥ 0.850 |
> | space 5×5 | Procrustes disparity | 0.071 | ≤ 0.150 |
> | phoneme voicing | intra-inter cos gap | +1.97 | ≥ +1.85 |

Numerical thresholds match the pre-registered hypotheses in PAPER
§4.2, §5.2, §6.2, §6.3.

**Verification**: `tests/test_tier_g_sleep.py::test_g3_rho_no_regression`
uses the §4 N=7 dual-muscle pipeline as the smallest representative
case, asserts ρ_linear ≥ 0.965 and ρ_arith_ord cross-facet align ≥
0.85 (vs. paper baseline 0.907).

The full per-domain G3 verification lives in
`experiments/sleep_pass_demo.py` (unit-test scope is too small for
ρ_circular and ρ_L1 baselines).

### G4 — Centroid reconstruction

> For each cluster $k$ on facet $f$:
> $\| \text{centroid}_k − \text{mean}(\{r_i : a_i = k\}) \|_2 < 10^{-5}$.

(Trivially true at K-means convergence; the test guards against future
drop-in replacements that might silently break the assumption.)

### G5 — Cookability of abstract subgraphs

> For every `abstract_relation` node registered by the sleep pass:
> `evaluator.eval(node_id, bindings={})` returns a tensor whose shape
> equals `cg.bundle_pool[f].shape[1:]` and whose values reconstruct the
> original member row to within `1e-6` (when `mode="add"` and
> `replay_steps=0`).

**Verification**: `tests/test_tier_g_sleep.py::test_g5_abstract_cook_reconstructs`.

### G6 — Idempotency

> `run_sleep_pass(cg, optimizer, replay_steps=0)` followed immediately
> by another `run_sleep_pass(cg, optimizer, replay_steps=0)` produces
> a `ConceptGraph` byte-identical to after the first call (no extra
> nodes, no extra prototypes, attribution unchanged).

**Verification**: snapshot `len(cg.concepts)`, `cg.bundle_pool` keys,
and `cg._consumed_by_by_slot` deep-cloned dict; run sleep pass twice;
assert deep equality.

### Relation to existing invariants

- **PAPER §3.4 命题 1 (architectural attribution)** — survives because
  every gradient into a new abstract slot has caller `"sleep"` (or a
  cook subgraph that statically references the slot via
  `concept.codebook_lookup`). Static enumerability is preserved.
- **G1–G6 of `tests/test_grow_invariants.py`** (Tier-A grow capacity)
  — survive because the sleep pass uses public `register_concept` /
  `_ensure_facet` APIs; if these grow under load the same migration
  path runs.
- **D1 (Tier-D bit-identity)** — survives unchanged for any pre-sleep
  cook subgraph.
- **Tier-B gates / Tier-C peer routing** — orthogonal: gates apply
  per-`(slot, facet)` so abstract slots get their own gate (default
  open); peer routing operates on `bundle_pool` rows by index so
  abstract slots are valid routing targets if the peer router is
  rebuilt after the sleep pass (caller's responsibility).

---

## 4. Public API — `pcm/sleep.py`

The whole module is opt-in (mirrors `pcm/gate.py` / `pcm/peer.py`):

```python
# pcm/sleep.py — public surface
from dataclasses import dataclass
from typing import Callable, Iterable

@dataclass
class SleepConfig:
    k_clusters: int | str = "auto"          # "auto" -> max(2, round(sqrt(N)/2))
    kmeans_iters: int = 32
    kmeans_tol: float = 1e-5
    distance: str = "cosine"                 # "cosine" | "l2"
    init: str = "kmeans++"                   # "kmeans++" | "uniform"
    replay_steps: int = 32
    replay_batch_size: int = 64
    seed: int = 0
    abstract_scope: str = "ABSTRACT"
    abstract_kind_proto: str = "abstract_prototype"
    abstract_kind_relation: str = "abstract_relation"

def attach_sleep(cg, *, facets: Iterable[str] | None = None) -> None:
    """Mark sleep machinery as enabled on ``cg``. Idempotent.

    Sets ``cg.sleep_enabled = True`` and stamps the per-facet allow-
    list. Must be called before any forward to guarantee G1.
    """

def run_sleep_pass(
    cg,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    facets: Iterable[str] | None = None,
    config: SleepConfig | None = None,
    replay_source: Callable[[], list[tuple[str, Callable[[], torch.Tensor],
                                           torch.Tensor]]] | None = None,
    tick: int = 0,
    force_recluster: bool = False,
) -> "SleepReport":
    """Run one offline sleep pass.

    Returns a ``SleepReport`` dataclass with per-facet ``(K, N,
    silhouette, centroid_norms, residual_norms, replay_loss_curve)``.
    ``optimizer`` is required iff ``config.replay_steps > 0``.
    """

def sleep_status(cg) -> dict:
    """Return a JSON-serialisable summary of sleep state on ``cg``.

    Keys: ``enabled``, ``facets_clustered``, ``n_prototypes``,
    ``n_relations``, ``last_tick``.
    """

# Internal helpers exposed for tests:
def register_prototype(cg, *, facet: str, k: int, centroid, tick: int) -> str: ...
def register_relation(cg, *, anchor_id: str, member_id: str, facet: str,
                      mode: str = "add", tick: int = 0) -> str: ...
```

The interleaved-replay loop is wholly implemented inside
`run_sleep_pass`; users only need to provide a `replay_source` lambda
that returns a list of `(caller, forward_fn, target)` triples each
time it is called.

For host training loops (the five `train_one`s in `experiments/`), a
canonical adapter is provided:

```python
def make_replay_source_from_buffer(buffer: list[tuple[Callable[[], torch.Tensor],
                                                      torch.Tensor]],
                                   muscle_callers: list[str]) -> Callable: ...
```

---

## 5. Integration with the five existing training loops

We do **not** modify `pcm/concept_graph` to call the sleep pass — that
would violate the opt-in principle behind `pcm/gate.py` and
`pcm/peer.py`. Instead each `train_one` opts in with 4–6 lines.

### 5.1 The canonical opt-in pattern

```python
from pcm.sleep import SleepConfig, attach_sleep, run_sleep_pass

def train_one(seed: int, *, sleep_every: int | None = None, **kw):
    cg = ConceptGraph(...)
    ...
    if sleep_every is not None:
        attach_sleep(cg, facets=["arithmetic_bias", "ordinal_offset"])

    opt = torch.optim.AdamW(
        list(head.parameters()) + list(cg.iter_bundle_parameters()),
        lr=LR, weight_decay=1e-4,
    )

    replay_buffer = []                       # filled during training
    for epoch in range(1, epochs + 1):
        for step_i in range(steps_per_epoch):
            ...
            replay_buffer.append((forward_fn_for_this_batch, target))
            opt.zero_grad(); loss.backward(); opt.step()

        if sleep_every and epoch % sleep_every == 0:
            run_sleep_pass(
                cg, optimizer=opt,
                facets=["arithmetic_bias", "ordinal_offset"],
                replay_source=lambda: random.sample(replay_buffer, 64),
                tick=epoch * 10000 + 9999,
            )
```

### 5.2 Per-domain attachment points

Documented exact line numbers in the existing code (read-only — these
are the lines that *will* receive the new branch):

| Domain | File | Insertion line | Facets to cluster |
|---|---|---|---|
| Numbers (§4) | `experiments/quad_study.py:201` (AdamW) and `:233` (epoch end) | `arithmetic_bias`, `ordinal_offset`, `identity_prototype` |
| Numbers (purity audit) | `experiments/purity_audit/train.py:80` and `:110` | `arithmetic_bias`, `ordinal_offset` |
| Colors (§5) | `experiments/color_concept_study/train.py:67` and `:95` | `mixing_bias`, `adjacency_offset` |
| Space (§6.2) | `experiments/space_concept_study/train.py:107` and `:128` | `motion_bias`, `distance_offset` |
| Phoneme (§6.3) | `experiments/phoneme_concept_study/train.py:81` and `:111` | `voicing_proto`, `manner_proto`, `place_proto` |

For **migration**, each `train_one` gains:
- One new kwarg `sleep_every: int | None = None` (default disabled).
- Three lines in the body: import, `attach_sleep`, the per-epoch
  `run_sleep_pass` call.
- One line in the corresponding `_config.py` (or `_main_.py`) to
  pass `sleep_every` from the CLI / experiment runner.

This is the *whole* code surface the user has to review per domain.

### 5.3 What to do when the host loop has no replay buffer

For `quad_study.py` (the only loop without a clean `replay_buffer`
abstraction), we append the bare minimum:

```python
recent_batches: collections.deque = collections.deque(maxlen=512)
...
recent_batches.append((triples_batch, op_onehot_batch, target_batch))
```

`make_replay_source_from_buffer` then walks `recent_batches` to build
the `(caller, forward_fn, target)` triples on demand.

---

## 6. Validation plan

### 6.1 Unit tests — `tests/test_tier_g_sleep.py`

Six methods on `class TestTierGSleep(unittest.TestCase)`, each
mapping to one G:

| test | invariant | scope |
|---|---|---|
| `test_g1_no_attach_bit_identical` | G1 | one ArithmeticHeadV2 forward, B=8 random batches |
| `test_g2_pool_memory_unchanged` | G2 | snapshot `pool[f][:N].clone()` before/after, `torch.equal` |
| `test_g3_rho_no_regression_n7` | G3 | full §4 N=7 dual-muscle, ρ_linear ≥ 0.965 |
| `test_g4_centroid_is_mean` | G4 | direct numerical check |
| `test_g5_abstract_cook_reconstructs` | G5 | `evaluator.eval(rel_id)` ≈ original row |
| `test_g6_idempotent` | G6 | run_sleep_pass twice, deep dict equality |

Test runtime budget: ≤ 60 seconds total on CPU. The `test_g3_*` test
uses N=7 (not the full N=30) precisely to keep this budget; full-scale
verification belongs to the demo (§6.3).

### 6.2 Demo — `experiments/sleep_pass_demo.py`

Reproduces the §4 N=7 dual-muscle pipeline twice (sleep_every=None vs
sleep_every=10) over 5 seeds and prints:

```
domain   metric              baseline       with sleep     gap
numbers  ρ_linear (single)   0.973 ± 0.007  0.97x ± 0.0xx  +/-Δ
numbers  ρ_linear (dual)     0.973 ± 0.004  0.97x ± 0.0xx  +/-Δ
numbers  cross-facet align   0.907 ± 0.037  0.9xx ± 0.0xx  +/-Δ
```

Acceptance: every `gap` ≥ −0.01 (i.e. sleep does not regress by more
than 1 abs-percentage-point); every `with sleep` mean ≥ G3 lower
bound.

If accepted, we then run the equivalent demos in
`color_concept_study`, `space_concept_study`, `phoneme_concept_study`
and only after **all four** pass do we proceed to the full migration
of `experiments/render_paper_figures/`.

### 6.3 Per-domain integration tests (after migration)

A new test file `tests/test_sleep_four_domain.py` runs the smallest
config of each domain (`N=7` numbers, 4 hues color, 3×3 space, 8
phoneme) with `sleep_every=5` over 1 seed and asserts the geometry
indicator is within ε of the no-sleep reference. Budget: ≤ 5 minutes
on CPU.

### 6.4 Expected migration outcome

If §6.1, §6.2, and §6.3 all pass:

1. The five `train_one`s gain the `sleep_every` kwarg permanently.
2. `experiments/render_paper_figures/_bundles.py:131-136` gains a
   companion `cos_matrix_from_abstract` so F4/F8 can render the
   prototype geometry alongside the data geometry.
3. `experiments/sleep_pass_study/` (NEW) holds the per-domain demo
   results as `outputs/sleep_pass_<domain>/summary.json` for F9
   (sleep-pass alignment-by-cluster) figure rendering.

If any phase fails: roll back the migration, leave `pcm/sleep.py`
under `RESEARCH` status, and document the failure mode in
`docs/TIER_G_DECISION.md` (template aligned with
`docs/TIER_D_DECISION.md`).

---

## 7. Equivalence to literature mechanisms

This section maps each Tier-G design choice to a 2024–2026 finding so
that any future change can re-justify itself by the same map.

| Tier-G phase | Cognitive-neuroscience analogue | Citation |
|---|---|---|
| **A snapshot** (active slots only) | Hippocampal cell ensembles flagged during awake encoding for later replay (recent-memory contracted-pupil sub-state) | Sleep microstructure organises memory replay (2024) *Nature* |
| **B K-means in cosine space** (NREM compression) | Nested compressed co-representations of multiple sequential experiences during sleep | Liu, Sibille & Dragoi (2024) *Nat Neurosci* |
| **C prototypes + residual facet** (codebook + per-member offset) | NREM organises scattered learned knowledge into a hierarchical structure | Aton et al. (2024) *Nat Commun* |
| **D abstract-relation cook subgraphs** (compositional retrieval of member from prototype + residual) | Compositional memory in hippocampus through replay; landmark cells = conjunction of object-vector primitives + spatial code | Bakermans, Warren, Whittington & Behrens (2025) *Nat Neurosci* |
| **E interleaved replay** (protect old muscles by sampling old batches) | Similarity-weighted interleaved learning prevents catastrophic forgetting; SWS interleaves novel and familiar traces | Saxena, Shobe & McNaughton (2022) *PNAS*; bioRxiv 2025.06.25.661579 v2 |
| **G3 falsifiable bound** (sleep must not regress old geometry) | Sleep-dependent abstraction enables *transfer*, not at the cost of source-task fidelity | bioRxiv 2026.04.10.717748 |
| **`abstract_prototype` node kind** (a real graph node owning real bundle rows) | "Naming guides 12-month-olds": labels rewrite the underlying memory trace, not just attach a tag | Bergen et al. PNAS 2020 / Weaver et al. 2024 *Child Dev* |

Each of these can be **swapped** independently. For example, if a
follow-up wants to test a hierarchical-DPMM-style cluster generation
instead of K-means (Aton's "hierarchy" finding), only phase B
changes; A/C/D/E and all G1–G6 stay valid.

---

## 8. Out of scope (deferred)

These were considered but explicitly excluded from Tier-G to keep the
landing surgical:

- **REM-style inferential pass**: making *new* relations between
  abstract prototypes (e.g. `bird → fowl → animal`) would require a
  second offline pass that reads the prototype facet rather than the
  data facet. The hooks are all in place (`abstract_prototype` is just
  a slot, so a second sleep pass on the prototype facet would build
  the next level), but we want to land NREM-only first.
- **Awake fast-mapping** (Tier-H in earlier scoping): registering
  speculative concept nodes from perceptual clusters during the
  *awake* loop. Orthogonal to Tier-G — both can coexist, and the
  awake-side `concept:hypothesis:*` nodes would flow through Tier-G
  on the next sleep boundary.
- **Tier-E (spatial coordinates)** and **Tier-F (vmap cooking)** —
  pre-existing deferrals from `docs/TIER_D_DECISION.md`. Tier-G is
  designed to compose with both: a `spatial_pos` field on a prototype
  is just another facet, and `evaluator.eval_batch` would batch over
  abstract relations the same way as over Tier-D heads.
- **Cluster pruning / merging across sleep passes**: this is the
  offline analogue of Tier-A's `grow_capacity` shrink. Doable but
  needs a separate decision document.

---

## 9. References

- Bakermans, J. J. W., Warren, J., Whittington, J. C. R., & Behrens,
  T. E. J. (2025). Constructing future behavior in the hippocampal
  formation through composition and replay. *Nature Neuroscience*.
- Liu, K., Sibille, J., & Dragoi, G. (2024). Nested compressed
  co-representations of multiple sequential experiences during sleep.
  *Nature Neuroscience*, 27, 1816–1828.
- Aton, S. J. et al. (2024). Prefrontal coding of learned and inferred
  knowledge during REM and NREM sleep. *Nature Communications*.
- *Sleep microstructure organises memory replay* (2024/2025).
  *Nature*, s41586-024-08340-w.
- Saxena, R., Shobe, J. L., & McNaughton, B. L. (2022). Learning in
  deep neural networks and brains with similarity-weighted interleaved
  learning. *PNAS*, 119, e2115229119.
- *Memory reactivation during sleep promotes structure abstraction*.
  bioRxiv 2026.04.10.717748 (2026).
- *Interleaved Replay of Novel and Familiar Memory Traces During
  Slow-Wave Sleep Prevents Catastrophic Forgetting*. bioRxiv
  2025.06.25.661579 v2 (2025).
- Whitehead, J. C. et al. (2024). Barcode activity in a recurrent
  network model of the hippocampus enables efficient memory binding.
  *eLife*.
- [`docs/PCM_NODE_AS_FUNCTION_DESIGN.md`](PCM_NODE_AS_FUNCTION_DESIGN.md)
  — Tier-D companion (concept-as-function, evaluator).
- [`docs/PCM_BIO_PREALLOC_UPGRADE.md`](PCM_BIO_PREALLOC_UPGRADE.md) —
  Tier-A/B/C bio-preallocation, gates, peer routing.
- [`docs/TIER_D_DECISION.md`](TIER_D_DECISION.md) — formatting
  template for the eventual `TIER_G_DECISION.md`.
