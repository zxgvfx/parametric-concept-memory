# Parametric Concept Memory (PCM)

[![CI](https://github.com/zxgvfx/parametric-concept-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/zxgvfx/parametric-concept-memory/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Paper](https://img.shields.io/badge/paper-PAPER.md-brightgreen.svg)](./PAPER.md)

**Language**: English | [简体中文](./README.zh-CN.md)

> *Concepts Collapse into Muscles — Domain-Topology-Adaptive Parametric
> Concept Memory.*
> Code, full paper, and a falsifiable cognitive-science testbed.

Instead of asking "which neurons represent concept *X*?", PCM makes
the question a **dict lookup**: every `ConceptNode` in a symbolic
graph owns a multi-facet `ParamBundle`, and every task-specific
"muscle" head consumes those bundle tensors on demand via a
**contextual collapse** operation. Attribution
(`bundle.consumed_by[facet]`) becomes a first-class property of the
graph, not a downstream inference problem.

Across **four domains with disjoint topologies** — numbers (1-D
linear), colors (1-D circular), a 5 × 5 spatial grid (2-D lattice),
and 20-phoneme categorical features — the *same* framework code
induces geometry that matches each domain's topology, under
geometry-free (random-orthogonal or one-hot) supervision. A
**post-hoc bundle swap** then establishes causal ownership: swapping
the trained bundle of two concepts on one facet collapses exactly the
muscle consuming that facet on exactly the involved pairs (100 % →
18.2 % for numbers, 5.3 % for colors) while every other muscle and
every other concept stay at 100 %. Textbook double dissociation.

On top of the empirical core, PCM provides a **Tier-G sleep
abstraction subsystem** with seven falsifiable invariants
(G1–G7), and a **five-condition `A / B / C / D / B+C+D` causal
ablation protocol** that lets researchers ask, of any candidate
representational primitive, whether it can emerge unconditionally
or whether it requires a biological prior, an ecological
statistic, or a task-asymmetric pressure.

Full paper in [`PAPER.md`](./PAPER.md) (English) and
[`PAPER.zh-CN.md`](./PAPER.zh-CN.md) (Chinese). A 2,700-word
short-report distillation of §6.7 / §6.8 / §7 / §7.4 / §7.5 /
§7.5-color / §6.9 lives in
[`docs/SHORT_REPORT_EN.md`](./docs/SHORT_REPORT_EN.md). Twelve
publication-quality figures in [`docs/figures/`](./docs/figures/).

![Four-domain universality of bundle geometry](./docs/figures/F4_four_domain_universality.png)

## What's new — PCM v2 milestone (May 2026)

A nine-finding mid-cycle published as F40–F49 (commits in
`git log --oneline | head -30`). Three highlights:

* **RPE breaks the §7.5-space `mixed_OOD = 0.000` ceiling** that
  resisted every v1 D-head variant. `pcm.dual_channel.RelativePositionEmbedding`
  saturates 4/4 PCM domains (space / colour / phoneme / number)
  to 0.96 – 1.00 mean OOD accuracy across 5 seeds — the same
  API, three tweaks of `ranges`. See
  [`docs/SHORT_REPORT_2026_S1_S6.md`](./docs/SHORT_REPORT_2026_S1_S6.md)
  for the diagnostic chain (concat → trained attr → oracle attr → RPE).

* **Inductive bias must be imposed (falsifiable)**. F45 / F46
  show that a learned attention gate stays at λ ≈ 0.98 across
  all reward strengths; only an explicit L1 penalty (β = 0.1)
  closes the gate to λ_final ≈ 0.002. Five-config × five-seed
  evidence in [`docs/SHORT_REPORT_2026_S1_S6.md §V3-RPE`](./docs/SHORT_REPORT_2026_S1_S6.md).

* **PCM v2 dual-channel encoding**. Concepts now decompose into
  `(slot, attr)` pairs supporting Tier-G clustering on the
  former and contrastive + arithmetic + successor losses on the
  latter (V1, V2 invariants both at 1.000 ± 0.000 on number).
  See
  [`docs/PCM_V2_DUAL_CHANNEL_DESIGN.md`](./docs/PCM_V2_DUAL_CHANNEL_DESIGN.md)
  for the full design and
  [`docs/PCM_V2_MIGRATION_GUIDE.md`](./docs/PCM_V2_MIGRATION_GUIDE.md)
  for the five-line v1 → v2 migration recipe.

Public API additions (107 / 107 tests passing):
- `pcm.dual_channel.{register_dual_channel_facet, collapse_dual_channel,
  RelativePositionEmbedding, info_nce_loss, arithmetic_consistency_loss,
  successor_consistency_loss, spread_regularizer, pair_attention_logits}`
- `pcm.heads.{DualChannelPairHead, SlotIdentityAuxHead, pair_collapse_and_forward}`
- `pcm.sleep.run_dual_phase_sleep` (S1 NREM-style two-phase) +
  G8a/b/c invariants

## TL;DR of the claims

| # | Claim | Evidence | Section |
|---|---|---|---|
| 1 | Attribution is a dict lookup, not an inference | H1–H4 all pass at 100 % on a 7-numerosity toy | §4.2 |
| 2 | Bundle geometry tracks task topology across four qualitatively different domains | F4: ρ = 0.991 (numbers) · ρ_circ = 0.977 (colors) · ρ_L1 = 0.860 / Procrustes disp 0.07 (space) · intra-inter cos gap +1.2 to +2.0 (phonemes) | §4–§6.5 |
| 3 | Cross-muscle alignment is gated by **facet-level algebraic compatibility** (H5″), not by shared domain or task family | F7: 2×2 schema fully populated — same-algebra align (p = 0.003, 0.016); same-domain incompatible-algebra null (p = 0.77); orthogonal categorical null (p ≥ 0.04) | §6.4 |
| 4 | Bundle = concept semantic identity (causal, not correlational) | F6: post-hoc swap causes targeted double dissociation with zero seed variance | §8 + Appendix B |
| 5 | PCM gives *geometric* emergence, not *algorithmic* | F5: spike₁₀ ≈ 0.001, p = 0.44; hand-coded positional priors recover 100 % extrapolation with ~10³ fewer samples | §7 |
| 6 | **Tier-G sleep abstraction is safe across all 4 domains** (no task-acc regression) and Pareto-better in task-difficulty-matched non-saturated regimes | F9: ΔOOD = +0.018 on number domain (5 seeds); ρ-std collapses to 38 % of baseline on colour | §6.6 |
| 7 | **PCM does not invent perceptual primitives** under cyclic mixing alone (sleep abstraction with k ∈ {3, 4, 6} returns equidistant rotations of arbitrary orientation; RYB primaries hit 0/24) | F11 §6.7: 24-seed colour-ring rotation analysis | §6.7 |
| 8 | **Three-causal-layer recipe drives RGB-aligned anchors** (B = LMS centroid, C = green-peak sampling, D = ripe-fruit head); D dominates under cyclic task symmetry | F11 §6.8: red-wedge anchor 0.62 → 1.00 (8/8 seeds) | §6.8 |
| 9 | **Three-causal-layer recipe reverses the §7 base-10 negative**; D drives spike₁₀ from +0.290 to +0.667 (× 2.3), units_gap sign-flips, last-digit cluster purity 0.876 | F12 §7.4: 5 cond × 8 seeds | §7.4 |
| 10 | **Length-extrapolation has a clean architectural ceiling** under D91/D92: A/B/C strictly at chance, D/BCD lift OOD by only +1.1 pp; colour hue-holdout is even cleaner (25/25 strictly 0.000) | F13/F14 §7.5 / §7.5-color | §7.5 |
| 11 | **Phoneme cross-language transfer is B-dominant** (articulator centroid alone gives V/M/P transfer = 1.000 / 0.943 / 0.771); reveals a **task-symmetry × dominant-layer principle**: cyclic / translational tasks need D, orthogonal-categorical tasks let B transfer cleanly | F15 §6.9: 5 cond × 5 seeds | §6.9 |

## Repo layout

```
pcm/                       core framework
├── concept_graph/         ConceptGraph + ConceptNode (packagified)
├── param_bundle/          ParamBundle + ContextualizedConcept
├── heads/                 task-specific muscles + Tier-D cook layer
├── sleep.py               Tier-G sleep abstraction (G1-G7 invariants)
├── gate.py                Tier-B slot gates
├── peer.py                Tier-C peer discovery
├── graph_eval.py          GraphEvaluator (cookable subgraph executor)
└── dna_ops.py             DNA op registry (concept.codebook_lookup, etc.)

experiments/               paper replications + ablations (`python -m experiments.<name>`)
├── color_concept_study/   §5 colour study (packagified)
├── space_concept_study/   §6.2 space study
├── phoneme_concept_study/ §6.3 phoneme study
├── counterfactual_swap_study/  Appendix B causal swap
├── purity_audit/          §4.4 attribution audit
├── render_paper_figures/  F2-F8 + F9 + F11-F15 renderers
├── quad_study.py          §4.5 four-operation arithmetic + Tier-G plumbing
├── number_decimal_priors.py   §7.4: decimal cones + LastDigitHead + samplers
├── phoneme_transfer_priors.py §6.9: articulator cones + MinimalPairHead
├── sleep_ablation.py             §6.6 V2/V3 ablation (quad domain)
├── sleep_ablation_four_domain.py §6.6 4-domain × 5-seed safety
├── sleep_color_primaries.py      §6.8 colour 5-condition × 8-seed
├── sleep_inspect_color_anchors.py §6.7 24-seed rotation analysis
├── sleep_number_decimal.py       §7.4 number 5-condition × 8-seed
├── sleep_number_extrapolate.py   §7.5 length OOD ceiling
├── sleep_color_holdout.py        §7.5-color hue holdout ceiling
└── sleep_phoneme_transfer.py     §6.9 cross-language transfer

tests/                     63 unit tests (Tier-A grow, Tier-B gate, Tier-C peer,
                           Tier-D cook, Tier-G G1-G7, 4-domain integration)

docs/
├── figures/               F2-F8 + F9, F11, F12, F13, F14, F15 figures
├── PCM_TIER_G_SLEEP_ABSTRACTION.md  full Tier-G design doc
└── SHORT_REPORT_EN.md     2,700-word TICS Forum / Cog-Sci short report draft

outputs/
└── ans_encoder/final.pt   shipped pre-trained ANS encoder (≈ 108 KB)

PAPER.md / PAPER.zh-CN.md  full paper (English / Chinese)
CHANGELOG.md               D91-D96 changelog with literature mapping
```

## Installation

Python 3.10+ and a working PyTorch install (CPU or CUDA) is all that
is needed.

```bash
git clone https://github.com/zxgvfx/parametric-concept-memory.git
cd parametric-concept-memory
pip install -r requirements.txt
# or, to install pcm as an importable package in dev mode:
pip install -e .
```

Dependencies: `torch`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`.
No ML-specific deps beyond PyTorch.

## Quick start — inspect the framework in ≤ 15 lines

```python
import torch
from pcm import ConceptGraph

cg = ConceptGraph(feat_dim=128)
for n in range(1, 8):
    cg.register_concept(node_id=f"concept:ans:{n}",
                         label=f"ANS_{n}", scope="BASE",
                         provenance=f"smoke:n={n}")

c = cg.concepts["concept:ans:3"]
cc = c.collapse(caller="AddHead", facet="arithmetic_bias",
                shape=(64,), tick=0, init="normal_small")
print(cc.as_tensor().shape)                        # torch.Size([64])
print(cg.concepts["concept:ans:3"].bundle.consumed_by)
# → {'arithmetic_bias': {'AddHead'}}      # attribution is a dict lookup
```

## Reproducing the paper's four domains

Total compute: ≈ 25 min on a single RTX 4090 (≪ 1 hour on CPU for
everything except the N = 100 numerical study).

```bash
# numbers — linear domain, H1-H4 attribution, H5 refutation, H5′ support
python -m experiments.robustness_study \
    --encoder-ckpt outputs/ans_encoder/final.pt --n-seeds 10

# numbers — scale study (N ∈ {7, 15, 30})
python -m experiments.scale_study --n-seeds 3

# numbers — purity audit (A1-A4)
python -m experiments.purity_audit \
    --encoder-ckpt outputs/ans_encoder/final.pt --n-seeds 5

# numbers — 4-op (± × ÷), N = 100
python -m experiments.quad_study --N 100 --n-seeds 3

# numbers — D93a positional prior, 100 % digit-length extrapolation
python -m experiments.compositional_number_study --head-type slot_equivariant

# COLORS — circular domain, H5″ same-algebra "if" direction
python -m experiments.color_concept_study --n-seeds 5

# SPACE — 2-D lattice + H5″ vector-vs-scalar null
python -m experiments.space_concept_study --n-seeds 3

# PHONEMES — discrete categorical + H5″ orthogonal-algebra null
python -m experiments.phoneme_concept_study --n-seeds 3

# NEGATIVE — pure base-10 emergence fails
python -m experiments.emergent_base10_study --scan 50 100 --n-seeds 3

# CAUSAL — post-hoc bundle swap (Appendix B)
python -m experiments.counterfactual_swap_study --n-seeds 3
```

### Reproducing the Tier-G + three-causal-layer ablations

Six additional experiments support paper sections §6.6 / §6.7 /
§6.8 / §6.9 / §7.4 / §7.5 / §7.5-color (≈ 1 hour total on a
single GPU):

```bash
# §6.6 — 4-domain Tier-G safety (5 seeds × 4 domains × A/C ablation)
python -m experiments.sleep_ablation_four_domain --n-seeds 5 \
    --out outputs/sleep_ablation_4domain

# §6.7 — colour-ring rotation analysis (Sleep does NOT invent
#         perceptual primitives; 24 seeds × k ∈ {3, 4, 6})
python -m experiments.sleep_inspect_color_anchors --k 3 --n-seeds 8
python -m experiments.sleep_inspect_color_anchors --k 4 --n-seeds 8
python -m experiments.sleep_inspect_color_anchors --k 6 --n-seeds 8

# §6.8 — colour 5-condition × 8-seed three-causal-layer ablation
python -m experiments.sleep_color_primaries --n-seeds 8 \
    --out outputs/primaries_5cond_8seed

# §6.9 — phoneme cross-language transfer (B-dominant)
python -m experiments.sleep_phoneme_transfer --n-seeds 5 \
    --n-target 7 --out outputs/phoneme_transfer_5seed

# §7.4 — number base-10 reversal (D dominates: spike_10 +0.29 → +0.67)
python -m experiments.sleep_number_decimal --n-seeds 8 \
    --out outputs/decimal_5cond_8seed

# §7.5 — length extrapolation ceiling (input-side)
python -m experiments.sleep_number_extrapolate --N-train 30 --N-total 100 \
    --n-seeds 5 --out outputs/extrap_30_100_5seed

# §7.5-color — hue holdout ceiling (output-side; 25/25 strictly 0.000)
python -m experiments.sleep_color_holdout --holdout-hue 5 --n-seeds 5 \
    --out outputs/color_holdout_h5_5seed
```

### Regenerating the figures

Figures F2, F4, F5, F6, F7, F8 are auto-generated by a single
entry point. Figures F9 (Tier-G safety), F11 (§6.8), F12 (§7.4),
F13 (§7.5), F14 (§7.5-color), F15 (§6.9) are produced by
dedicated renderers that read the corresponding `summary.json`:

```bash
# Original 4-domain figures (≈ 3 min)
python -m experiments.render_paper_figures
# or a subset:
python -m experiments.render_paper_figures --only F4 F7

# Tier-G + three-causal-layer figures (each < 10 s, no retraining)
python -m experiments.render_paper_figures.F9_sleep_four_domain
python -m experiments.render_paper_figures.F11_color_primaries
python -m experiments.render_paper_figures.F12_number_decimal
python -m experiments.render_paper_figures.F13_number_extrapolate
python -m experiments.render_paper_figures.F14_color_holdout
python -m experiments.render_paper_figures.F15_phoneme_transfer
```

### Regenerating the shipped ANS encoder

The ≈ 108 KB `outputs/ans_encoder/final.pt` is shipped with the
repo for turn-key reproduction. If you want to retrain from
scratch:

```bash
python -m experiments.train_ans --epochs 30 --out outputs/ans_encoder
```

## PCM as a falsifiable cognitive-science testbed

Beyond the empirical four-domain study, PCM provides a
methodologically explicit ablation protocol for asking "does
representational primitive *Y* require external priors, or
does it fall out of generic learning?" — see
[`docs/SHORT_REPORT_EN.md`](./docs/SHORT_REPORT_EN.md) for the
2,700-word write-up targeting *Trends in Cognitive Sciences*
Forum / *Cognitive Science* short reports.

The protocol is the **A / B / C / D / B+C+D** five-condition
template, mapped onto three causal layers from the human
trichromacy literature (Stockman & Sharpe 2000; Jacobs 2009;
Conway et al. 2007):

- **A** baseline: random centroids, uniform sampling, no auxiliary
  head. Symmetric task only.
- **B** biological prior: e.g. an LMS-cone-like centroid layout
  for colour, decimal-cone centroids for numbers, articulator
  cones for phonemes.
- **C** ecological statistics: non-uniform sampling boosting
  task-relevant input distributions.
- **D** task-driven asymmetry: an auxiliary head that singles out
  a small subset of inputs as behaviourally relevant
  (ripe-fruit head for colour, last-digit head for numbers,
  minimal-pair head for phonemes).
- **B+C+D**: all three layers stacked.

Across colour (§6.7 / §6.8), number (§7 / §7.4 / §7.5), and
phoneme (§6.9), the protocol consistently:

1. **Refutes** spontaneous emergence of human-perceptual
   primitives under symmetric tasks (RGB / RYB hits exactly
   equal the strict-equidistant rate; 0/24 RYB hits in 24 colour
   seeds; spike₁₀ ≈ 0.001 in 5 number seeds).
2. **Supports** prior-driven emergence: BCD reaches red-wedge
   1.00 on colour, spike₁₀ +0.67 on number, V/M/P transfer
   0.97 / 0.77 / 0.71 on phoneme.
3. **Predicts** which layer dominates by **task symmetry**:
   cyclic / translational tasks need D (colour, number);
   orthogonal-categorical tasks let B alone do the job
   (phoneme).
4. **Maps** clean architectural ceilings: input-side
   (length-OOD-100 = chance + 1.1 pp) and output-side
   (hue-5 holdout = 25/25 strictly 0.000).

## What makes PCM different from prior work

PCM sits at the intersection of several research lines but is
specifically **not** any of them:

| Prior approach | What they do | How PCM differs |
|---|---|---|
| **Concept bottleneck models** (Koh 2020, Zarlenga 2022) | Concepts are labels on *activations* in a layer | Concepts are first-class graph nodes that *own* parameters |
| **Hypernetworks / fast weights** (Ha 2017, Schmidhuber 1992) | Generate weights from a context signal at use-time | Bundles *store* weights at the node; no generator MLP |
| **External / episodic memory** (DNC, NEC, kNN-LM) | Retrieve vectors or key-value pairs | Memory *is* parameters (`nn.Parameter`), with consumer registry |
| **Mech-interp & SAEs** (Anthropic monosemanticity, SAE circuits) | Discover features post hoc in residual streams | PCM *decrees* features by graph construction, then empirically tests emergence |
| **Multi-task representation learning** (Caruana, Maurer) | Treat shared reprs as a bandwidth trade-off | H5″ identifies a *structural* precondition — facet-level algebraic compatibility |

See `PAPER.md` §2 for full contrasts.

## Citation

If you use PCM or these results, please cite:

```bibtex
@misc{zhang2026pcm,
  title        = {Concepts Collapse into Muscles: Domain-Topology-Adaptive
                  Parametric Concept Memory},
  author       = {Zhang, Xugang},
  year         = {2026},
  howpublished = {\url{https://github.com/zxgvfx/parametric-concept-memory}},
  note         = {Framework + 4-domain empirical study; MIT licensed},
}
```

## License

- **Code** (`pcm/`, `experiments/`, `tests/`, `outputs/ans_encoder/`)
  is released under **MIT** — see [`LICENSE`](./LICENSE).
- **Paper and figures** (`PAPER.md`, `submission/`, `docs/`) are
  released under **CC BY 4.0** — attribution required, derivatives
  and commercial use allowed. Matches the license attached to the
  arXiv preprint.
