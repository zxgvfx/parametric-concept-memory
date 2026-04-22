# A3 · Phoneme Concept Study — PCM on a Discrete Categorical Domain

**Date**: 2026-04-22
**Code**: `mind/experiments/language/phoneme_concept_study.py`
**Output**: `outputs/phoneme_concept/summary.json`
**Domain**: 20 phonemes × 3 orthogonal categorical attributes.

## 1. Question

Numbers (1-D linear), colors (1-D circular), and spatial grid (2-D
lattice) all share a **metric topology**: cosine similarity can track a
ground-truth distance function. A3 stress-tests PCM on a domain where
no natural metric exists and concept identity is defined by a sparse
categorical feature matrix.

**Sharper claim being tested (H5'' necessity)**: when a fixed set of
concepts is trained by **three orthogonal** muscles, each facet should
develop geometry on its own axis but the three facets should be
**mutually non-aligned**. This probes the "not-if" direction of H5''
to complement numbers / colors (which satisfied the "if" direction).

## 2. Design

20 phonemes selected to approximately orthogonalize three attributes
drawn from simplified SPE features:

| attribute | values | distribution |
|---|---|---|
| voicing | 2 (-, +) | 8 voiceless / 12 voiced |
| manner | 4 (STOP, FRIC, NAS, APR) | 7 / 6 / 3 / 4 |
| place | 4 (LAB, COR, DOR, GLT) | 6 / 7 / 5 / 2 |

Three single-input classifier muscles (each consuming its own facet,
jointly trained):

| muscle | facet | dim | output |
|---|---|---|---|
| VoicingHead | `voice_bias` | 16 | 2-class (±voice) |
| MannerHead | `manner_bias` | 16 | 4-class manner |
| PlaceHead | `place_bias` | 16 | 4-class place |

60 epochs × 120 steps × BS 32, AdamW lr 1e-3, weight_decay 1e-4.
3 seeds (E1), 3 shuffle-concept runs (E2), 1 permutation-test run (E4).

### Pre-registered predictions

| Metric | Prediction | Rationale |
|---|---|---|
| Per-facet ρ_same_axis | **≥ 0.6** | each facet should group its axis classes |
| Per-facet intra-inter cos gap | **> 0.5** | same-class pairs closer than different-class |
| Cross-facet align (v-m, v-p, m-p) | **all \|ρ\| < 0.2** | axes are orthogonal |
| Shuffle |ρ_same_axis| | **< 0.1** | identity-dependent geometry |
| Permutation p-values | **> 0.01 for all 3 pairs** | null alignment (contra the 0.003 / 0.016 seen in numbers / colors) |

## 3. Results

All runs reached 100 % accuracy on all three axes by epoch ~20.

### 3.1 Per-facet geometry (E1, 3 seeds)

Values are **structural maxima** for the Spearman-vs-binary-indicator
correlation given each axis's class partition; hitting them exactly
(± 0.000 across seeds) indicates perfectly separated same-class vs
different-class cosines, i.e. saturated class geometry.

| facet | accs | ρ_same_axis | intra-inter cos gap |
|---|---|---|---|
| VoicingHead (axis 0) | 1.000 | **+0.866 ± 0.000** | **+1.970 ± 0.012** |
| MannerHead (axis 1) | 1.000 | **+0.736 ± 0.000** | **+1.232 ± 0.018** |
| PlaceHead (axis 2) | 1.000 | **+0.747 ± 0.000** | **+1.229 ± 0.027** |

Each facet locks onto its assigned axis: same-class pairs converge to
cos ≈ +1 and different-class pairs to cos ≈ −1, producing an
intra-inter cos gap near 2.0 for voicing (binary) and ~1.2 for manner
/ place (4-way). The cos-gap saturation matches what we expect from
prototype-style class encoding.

Critically, the bundle for a given facet only separates **on that
facet's axis**; it carries near-zero information about other axes
(measured but omitted here — see `rho_leakage` in `summary.json`).

### 3.2 Cross-facet alignment (the critical test)

| pair | align ρ (mean ± std) | perm-test p | status |
|---|---|---|---|
| voice ↔ manner | **+0.108 ± 0.022** | p = 0.052 | not significant |
| voice ↔ place | **−0.011 ± 0.014** | p = 0.991 | null |
| manner ↔ place | **−0.124 ± 0.018** | p = 0.037 | marginally significant |

Compare to the metric domains:

| pair | domain | \|align ρ\| | p |
|---|---|---|---|
| arithmetic ↔ ordinal | numbers | 0.56 | 0.003 |
| mixing ↔ adjacency | colors | 0.56 | 0.016 |
| **voicing ↔ manner** | **phonemes** | **0.11** | 0.052 |
| **voicing ↔ place** | **phonemes** | **0.01** | 0.991 |
| **manner ↔ place** | **phonemes** | **0.12** | 0.037 |

Phoneme cross-facet alignments are **5× smaller** than metric-domain
alignments and none pass a p < 0.01 threshold. The marginal
significance on v-m and m-p is explained by **phonotactic feature
correlations** in the natural phoneme inventory, not by a PCM
alignment mechanism:

- **v-m**: approximants (4 phonemes) and nasals (3) are **all voiced**
  in our inventory. Knowing manner partially reveals voicing. This
  data-level dependence is reflected faithfully by the bundles.
- **m-p**: glottal place contains only STOPs and FRICs (no nasals or
  approximants). Knowing place partially constrains manner.
- **v-p**: voicing and place are truly independent in our set → the
  corresponding alignment is the smallest (−0.01, p = 0.99).

PCM therefore carries **the real statistical structure of the
input** (a correctness property, not a bug), but the magnitudes are
0.01–0.13 — an order of magnitude below the algebraic alignments
seen in metric domains.

### 3.3 Shuffle counterfactual (E2, 3 seeds)

| facet | |ρ_same_axis| (shuffled) |
|---|---|
| voicing | 0.030 ± 0.028 |
| manner | 0.037 ± 0.022 |
| place | 0.022 ± 0.015 |

All three axes' own-axis geometry collapses to noise when the
concept_id→bundle mapping is scrambled, mirroring the earlier
numerical / color / space shuffle controls. Task accuracy remains
100 % (heads are identity-invariant), confirming the geometry lives
*only* in the bundle.

## 4. Interpretation — H5'' is both necessary and sufficient

| Domain | Task pair | Facet-level algebra | Observed align | Predicted by H5'' |
|---|---|---|---|---|
| numbers | arith ↔ ordinal | ordered additive (both) | +0.56 (p = 0.003) | ✓ |
| colors | mix ↔ adj | circular ordered (both) | +0.56 (p = 0.016) | ✓ |
| space | motion ↔ L1 | vector vs scalar magnitude | −0.03 (p = 0.77) | ✓ (incompatible) |
| **phonemes** | **voice ↔ manner / place** | **orthogonal categorical** | **~0.1 or less (p > 0.03)** | ✓ (orthogonal ⇒ no align) |

The four domains together span:
- **1-D linear ordered** (numbers) ✓ metric geometry + algebraic align
- **1-D circular** (colors) ✓ metric geometry + algebraic align
- **2-D lattice** (space) ✓ metric geometry on one task; no align because one task is vector, the other is scalar magnitude
- **discrete categorical** (phonemes) ✓ class-prototype geometry per facet; no align because axes are orthogonal

This gives H5'' a **four-domain claim schema**:

> **H5''**: two facets of the same concept align iff their consuming
> muscles impose the same algebraic requirement on the facet.

- Same algebra ⇒ align (number, color)
- Different algebra (vector vs scalar) ⇒ no align (space)
- Orthogonal partitions ⇒ no align (phonemes)

A3 cleanly fills the "orthogonal tasks ⇒ no algebraic alignment"
quadrant of the H5'' prediction grid.

## 5. Positive universality claim

Even though the domain has no natural metric, **PCM still emerges
structured bundle geometry**:

- Each facet's cosine matrix separates same-class from different-class
  pairs by a gap of +1.2 to +2.0, i.e. same-class pairs have cos ≈ 1
  and different-class pairs have cos ≈ 0 or negative.
- Class prototypes organize the 20 phonemes into clean clusters along
  each axis, without any architectural changes to the PCM framework.

Combined with A1 (color) and A2 (space), A3 establishes that PCM's
topology-adaptive geometry is **not restricted to metric domains**.
Whatever structure the task demands — linear, circular, lattice, or
partition — the bundle develops the matching geometry.

## 6. Limitations

- **Single-input classification**: the task design is trivial to
  solve (20 phonemes, 2/4/4 classes). A harder task would force the
  bundles to encode *within-class* sub-structure (e.g. distinguishing
  labial from coronal voiced stops while retaining voicing info).
  We did not test whether intra-class gradient structure emerges.
- **Phoneme inventory is a sample, not a theory of phonology**: the
  SPE features are simplified and our 20-phoneme set is deliberately
  near-orthogonal to make the prediction clean. Natural inventories
  have more complex dependencies.
- **No pair-based task**: we did not test a "minimal pair" muscle of
  the form `(p₁, p₂) → hamming distance`, which would directly test
  whether the bundle supports compositional feature distance reading.
  Marked as A3-follow-up.

## 7. Reproducibility

```bash
# ~4.5 min, 3 seeds, single GPU
python -m experiments.phoneme_concept_study --n-seeds 3

# ~25 s smoke
python -m experiments.phoneme_concept_study --smoke
```

Raw data (per-seed cross-facet ρs, leakage ρs, intra-inter gap
breakdowns, permutation histograms) in
`outputs/phoneme_concept/summary.json`.

## 8. Relation to the workshop paper

A3 contributes:

1. **Fourth domain** for PCM's universality claim
   (`1-D linear`, `1-D circular`, `2-D lattice`, `categorical`
   — all with emergent structured geometry, zero architectural
   changes).
2. **H5'' necessity test**: three orthogonal muscles produce three
   mutually non-aligned facets, filling the quadrant left blank by
   numbers / colors / space. Combined, the four domains show the
   alignment predictor is fine-grained (facet-level algebraic
   compatibility), not coarse (domain or task family).

Recommendation: merge A2 + A3 findings into a single revision of
`WORKSHOP_PAPER.md` under a new §6 ("Cross-domain universality") or
retain the current paper scope (numbers + colors) and ship A2 + A3
as a follow-up note. Decision deferred to the user.
