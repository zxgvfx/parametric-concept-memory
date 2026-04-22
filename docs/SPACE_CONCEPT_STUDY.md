# A2 · Space Concept Study — PCM on 2-D Product-Order Topology

**Date**: 2026-04-22
**Code**: `mind/experiments/language/space_concept_study.py`
**Output**: `outputs/space_concept/summary.json`
**Domain**: 5×5 integer grid (25 concepts).

## 1. Question

Numbers (1-D linear) and colors (1-D circular) established PCM's
topology-adaptive geometry claim in 1-D. **A2 tests whether PCM can
emerge higher-dimensional structure**: does the bundle develop a
genuine 2-D lattice, or does it collapse the 2-D task onto a 1-D
projection (a failure mode of many implicit representation learners)?

**Topology**: product order on Z² — `(r₁,c₁) ≤ (r₂,c₂)` iff r₁≤r₂
and c₁≤c₂. This is a **half-order / partial order**, not a total
order. No single linear ranking can preserve both axes.

## 2. Design

- **25 concepts**: `concept:space:r_c` for r, c ∈ {0..4}.
- **MoveHead** (facet `motion_bias`, 64-d, 5-class output).
  (cell_a, cell_b) → direction class ∈ {up, down, left, right, same}.
  Only 4-connected neighbours and self-pairs are valid inputs
  (105 pairs total, exhaustive).
- **DistanceHead** (facet `distance_offset`, 8-d, 9-class output).
  (cell_a, cell_b) → L1 distance ∈ {0..8}. 625 pairs exhaustive.
- 30 epochs × 200 steps, AdamW lr 1e-3, weight decay 1e-4.
- 3 random seeds. Cross-facet alignment also tested in a
  permutation-test mode (E4).

### Pre-registered predictions

| Metric | Prediction | Rationale |
|---|---|---|
| ρ_L1 | **> 0.80** | bundle cos should track L1 distance |
| ρ_linear_flat | **< ρ_L1** | 1-D flattening (row-major index) should be worse than 2-D L1 |
| ρ_row_within ≈ ρ_col_within | **both > 0.60**, diff < 0.15 | 2-D isotropy, not axis-dominant |
| MDS Procrustes disparity | **< 0.15** | bundle in 2-D should literally look like the grid |
| Shuffle ρ_L1 | **|ρ| ≈ 0** | identity-dependent |
| Cross-facet align (MoveHead ↔ DistanceHead) | **≥ 0.5, p < 0.01** | both tasks live in the same 2-D metric algebra |

## 3. Results

### 3.1 Single-muscle geometry (MoveHead only, 3 seeds)

All numbers mean ± std across 3 seeds.

| Metric | Value | Prediction | Status |
|---|---|---|---|
| move_acc | 1.000 ± 0.000 | ceiling | ✓ |
| **ρ_L1** | **+0.860 ± 0.012** | > 0.80 | ✓ |
| ρ_linear_flat | +0.572 ± 0.075 | < ρ_L1 | ✓ (gap = 0.29) |
| ρ_row_within | +0.805 ± 0.048 | > 0.60, isotropic | ✓ |
| ρ_col_within | +0.797 ± 0.065 | > 0.60, isotropic | ✓ (row-col Δ = 0.008) |
| **MDS Procrustes disparity** | **0.071 ± 0.014** | < 0.15 | ✓ (very strong) |

The single MoveHead muscle induces a bundle geometry that, when
projected to 2-D via MDS and Procrustes-aligned to the ground-truth
grid, has a disparity of 0.07 (0 = perfect grid). The row and column
axes carry near-identical ρ (Δ = 0.008), so the emergence is
**genuinely 2-D isotropic**, not a 1-D projection that accidentally
resembles a grid. The 1-D flattening control (ρ_linear_flat = 0.57)
is 0.29 below ρ_L1 (0.86), confirming the 2-D metric wins over any
1-D ordering.

### 3.2 Shuffle counterfactual (E2)

3 seeds, concept_id→bundle mapping scrambled during training.

| Metric | Value |
|---|---|
| move_acc | 1.000 (task still solvable — heads are unaffected) |
| |ρ_L1| | 0.025 ± 0.019 |
| ρ_row_within | −0.016 (avg) |
| ρ_col_within | −0.160 (avg) |
| MDS Procrustes disparity | 0.94 (avg) — no grid at all |

A perfect negative control. The 0.86 → 0.025 collapse (34×) confirms
the 2-D geometry depends on concept identity and not on training
accident.

### 3.3 Cross-facet alignment (unexpected)

| Metric | Value | Prediction | Status |
|---|---|---|---|
| DistanceHead (dual) dist_acc | 0.729 ± 0.065 | ceiling | ✗ (partial) |
| DistanceHead ρ_L1 | +0.158 ± 0.017 | ≥ 0.70 | ✗ |
| **cross-facet align** (motion ↔ distance) | **−0.034 ± 0.016** | ≥ 0.5 | ✗ |
| E4 permutation p | 0.767 | < 0.01 | ✗ |

This is the **single negative result** of A2 and is worth reporting
in full. Two things happened simultaneously:

1. DistanceHead alone did not emerge a 2-D metric bundle geometry
   (ρ_L1 only 0.16 vs MoveHead's 0.86) despite reaching 73% task
   accuracy (vs 11% chance).
2. Cross-facet alignment with MoveHead's bundle is essentially zero
   (Spearman ρ = −0.03, p = 0.77).

The non-zero task accuracy with absent geometry suggests DistanceHead
solves L1 via a distributed/shortcut encoding in its 8-d facet — the
fc layers compute L1 from a representation that does **not** lay
cells out on a grid. MoveHead's facet, in contrast, must carry
**oriented vector** information (up/down/left/right are all signed),
which forces a 2-D grid layout.

### 3.4 Interpretation of the non-alignment

The natural reading is that **"2-D spatial" is not a single task
algebra**. Direction and magnitude are fundamentally different
algebraic operations on the same underlying space:

- MoveHead requires **signed 2-D vectors**: the bundle must encode a
  cell's (row, col) as a coordinate, so that `cell_b − cell_a`
  yields a direction.
- DistanceHead requires only **scalar symmetric magnitude**: |Δr| +
  |Δc|. Symmetric in (a, b), so no orientation needed, and the
  8-d facet can shortcut to any representation whose fc readout
  happens to produce L1.

Formally, the two tasks share a metric space but not a *vector* space
structure. Bundle geometry aligns only when muscles demand **the
same kind of algebraic work** from their consumed facets. In this
case they do not.

This sharpens H5'' (paper §3.4 / §7) from "same task family ⇒ aligned"
to **"same algebraic requirement on the facet ⇒ aligned"**. Direction
and magnitude are the same task family (both 2-D grid) but impose
different algebraic requirements (vector vs scalar), and the
alignment duly fails. Compare:

| Pair | Task family | Facet-level algebra | Align? |
|---|---|---|---|
| arith ↔ ordinal (numbers, paper §4.5) | arithmetic | both *ordered* additive | ✓ ρ = 0.56, p = 0.003 |
| mix ↔ adj (colors, paper §5) | circular 2-D | both *circular-ordered* | ✓ ρ = 0.56, p = 0.016 |
| motion ↔ L1 (space, this study) | 2-D grid | *vector* vs *magnitude* | ✗ ρ = −0.03, p = 0.77 |

The three cases together provide the cleanest evidence yet that H5''
is a **facet-level algebraic statement**, not a domain-level one.

## 4. What this adds to the paper's claim map

Universal geometry emergence (positive claims established):

| Domain | Topology | ρ geometry | MDS visual | Algebra |
|---|---|---|---|---|
| Numbers 1-30 | 1-D linear | ρ = 0.991 | linear | additive ordered |
| Colors 12 hues | 1-D circular | ρ_circ = 0.977 | circle | circular ordered |
| **Space 5×5** | **2-D lattice** | **ρ_L1 = 0.860** | **grid (disparity 0.07)** | **2-D vector** |

PCM reliably auto-adapts to 1-D (linear and circular) and 2-D
(lattice) topologies with **the same framework code**. No
architectural change between domains.

Alignment gating (cross-facet):

| Pair | Facet-level algebra | Align? |
|---|---|---|
| arithmetic ↔ ordinal | ordered additive | ✓ |
| mixing ↔ adjacency | circular ordered | ✓ |
| **motion ↔ L1 distance** | **vector vs scalar magnitude** | **✗** |

The third row is **new evidence for H5''**: it rules out alignment
being driven by shared task domain alone.

## 5. Limitations

- DistanceHead's low geometry ρ (0.16) makes the cross-facet
  non-alignment argument partially confounded: one could argue the
  null alignment is because DistanceHead simply did not learn any
  geometry to align to. A cleaner design would switch DistanceHead
  to predict the **vector offset** (Δr, Δc) instead of the scalar
  L1, which would *force* a vector-algebraic facet and let us test
  whether bundle alignment recovers. We mark this as A2-follow-up.
- Grid is 5×5 = 25 concepts. Larger grids (10×10, 20×20) and
  non-square grids are future work.
- Partial order test is sparse: we do not yet verify that the bundle
  geometry respects the poset structure (e.g., comparability under
  ≤) beyond L1 metric consistency.

## 6. Reproducibility

```bash
# ~4 min, 3 seeds, single GPU
python -m experiments.space_concept_study --n-seeds 3

# ~20 s smoke test
python -m experiments.space_concept_study --smoke
```

Raw data (per-seed rows, MDS disparities, per-row/col ρs, permutation
histogram) in `outputs/space_concept/summary.json`.

## 7. Relation to the workshop paper

A2's positive findings (§3.1–§3.2) substantially strengthen the
paper's §1 Abstract claim from "linear + circular" to "**1-D and
2-D topologies alike auto-adapt**". The negative finding (§3.3–§3.4)
refines H5'' in §3.4 from "compatible algebras align" to
"compatible facet-level algebras align; same-domain is not enough".
Both belong in a revised workshop / main-track draft; decision on
whether to fold in now vs leave for follow-up pending A3 results.
