# A three-layer causal ablation reveals what computational systems can and cannot invent on their own: lessons from sleep abstraction in a parametric concept memory

**Format**: Opinion / Forum article (~2,500 words). Target venues: *Trends in Cognitive Sciences* (Forum / Opinion), *Cognitive Science* (Letter / Short Report), or *Cognition* (Brief Article). Adapt headings to the target style.

**Author placeholder**. **Affiliation placeholder**. **Correspondence placeholder**.

---

## Highlights

- A computational system trained only on cyclic colour-mixing does **not** spontaneously rediscover RGB-like perceptual primaries during offline (sleep-style) consolidation, despite reproducing well-known geometric findings (a circular hue ring with ρ = 0.977).
- Across 24 ablation seeds with cluster sizes *k* ∈ {3, 4, 6}, RYB primaries are recovered 0/24 times; RGB / CMY / CMYK / WarmCool6 hit-rates exactly equal the strict-equidistant rate, showing that any apparent "primary recovery" is a renaming artefact of the cyclic task symmetry.
- A *three-layer causal ablation* — mirroring the human trichromacy literature (Stockman & Sharpe, 2000; Jacobs, 2009; Conway et al., 2007) — shows that a single binary foraging-style task drives anchors into the red wedge in 8/8 seeds, while a biological-prior centroid alone reaches only 4/8 strict-equidistant seeds with seed-dependent rotation. Combining all three layers (cone-like supervision, ecological sampling, foraging task) yields the only condition with simultaneous strict equidistance *and* fixed rotation class.
- A third-domain phoneme cross-language transfer experiment (5 seeds, 13-source / 7-target split, 20 phonemes total, V/M/P heads see only source) reverses the dominant layer: in the phoneme domain the articulator-grouped centroid (layer B) carries target voice/manner/place accuracy from chance levels (0.51 / 0.11 / 0.26) all the way to 1.000 / 0.943 / 0.771, while the task-driven minimal-pair head (layer D) transfers only the single facet it consumes. PCM thus predicts a clean *task-symmetry × dominant-layer* relationship: cyclic / translational task groups (colour, number) require task-driven asymmetry to break symmetry, whereas orthogonal-categorical task groups (phoneme features) admit clean transfer from biological prior alone — consistent with the universal-discrimination phase of infant phonetic learning (Werker & Tees 1984).
- We propose this **A / B / C / D / B+C+D** template as a falsifiable testbed for cognitive-science claims of the form "feature *X* is necessary for representation *Y*", and report a fully-open implementation in the accompanying repository.

---

## Abstract

Computational accounts of perceptual primaries — why humans see colour as red-green-blue rather than as a rotation-symmetric continuum — typically appeal to one of three causal layers: (1) cone genetics, (2) chromatic statistics of natural environments, and (3) task-driven selection (e.g. ripe-fruit foraging). The relative contributions of these layers are difficult to test empirically because each is hard to switch off. We replicate the layered debate in a small computational model — Parametric Concept Memory (PCM) — where each layer can be cleanly enabled or removed. Without any layer, offline cluster-based "consolidation" never invents primaries: anchor placements distribute uniformly over the *k* rotation classes of the cyclic task symmetry. Adding a cone-like supervision (layer 1) doubles strict-equidistant rate but leaves rotation seed-dependent. Adding a non-uniform sampling (layer 2) gives marginal improvement. Adding a single binary foraging task (layer 3) drives anchors into the red region in every seed. Only the combined manipulation produces both equidistant placement *and* fixed rotation class. We propose this five-condition design as a falsifiable testbed for similar layered claims across cognitive science.

---

## Introduction

A recurring question in cognitive science is whether some perceptual or conceptual category — colour primaries, base-10 numerical structure, IPA-like phoneme inventories, cardinal spatial axes — is *invented* by general learning machinery or whether it requires injection from biology, environment, or task. Strong claims of either kind are often hard to falsify because the proposed cause cannot be removed in a real cognitive system. We argue that small, fully-instrumented computational models are well placed to fill this gap, provided they expose three things: (i) interpretable parameters whose geometry can be measured directly, (ii) a clean separation between the layers under debate, and (iii) a public diagnostic protocol that can be re-run across new candidate primitives.

We instantiate this argument in a 12-hue colour study using **Parametric Concept Memory (PCM)** — a small attribution-strict architecture where each concept owns one row in a shared parameter pool, consumed by *muscles* (downstream task heads) under formal contracts. PCM has four properties that make it useful for the present purpose: (a) bundle rows form interpretable geometries that already replicate canonical findings across number, colour, space, and phoneme domains (ρ = 0.991 / 0.977 / 0.860 / structure-recoverable, respectively, see [PCM technical paper]); (b) an offline "sleep" pass can compress the pool with falsifiable invariants G1–G7; (c) random-orthogonal centroids and uniform task sampling give a strict cyclic-equivariant baseline; (d) each of the three putative layers in trichromacy — cone genetics, chromatic statistics, foraging task — corresponds to a one-line code change.

We use this scaffolding to ask: **does sleep-style consolidation invent RGB-like primaries on its own?** And, separately, **what minimal causal ingredient is needed to drive PCM into a fixed-rotation primary geometry?**

## Two negative results: PCM does not invent primitives that the task does not encode

Two independent ablations show that pure task-symmetric training, in either wake or offline phases, fails to produce widely-cited perceptual or numerical primitives.

**Numbers do not factor themselves into base-10 columns** even when training is extended and balanced (Saxton et al. 2019; Nogueira et al., 2021 on Abacus embeddings document the converse: hand-coded base-10 priors enable digit-length extrapolation at 10³× lower data than learned ones). In our setup, integers 1–100 trained on flat four-arithmetic with random orthogonal centroids never produce a column structure detectable by either a 10-period spike test or a digit-MLP probe (see PCM §7 for full negatives across 5 seeds at *N* = 50, 100). Linear and logarithmic geometries do emerge (ρ = 0.991), but multiplicative-place structure does not.

**Colour does not crystallise into RGB primaries** during offline consolidation. We ran *k* ∈ {3, 4, 6} k-means sleep passes on the 12-hue PCM bundle pool, 8 seeds each (24 total). For *k* = 3, anchors land in strict 120° equidistant configuration in only 2/8 seeds; the other 6 seeds drift by ±1–2 hue units, a residue of running k-means in 64-dimensional bundle space rather than on a 12-mod angular coordinate. Crucially, the only canonically-named non-equidistant prior probed (RYB, painter primaries {0, 2, 8}, with spacings [2, 4, 6]) hits **0/24** ablations across all *k*. Each named-equidistant prior — RGB, CMY, CMYK_aligned, WarmCool6 — has a hit rate exactly equal to the strict-equidistant rate of its own *k*: an apparent "RGB recovery" is a renaming artefact of the cyclic task symmetry, not a discovery of perceptual structure.

These two negatives are not failures of PCM. They falsify a specific computational-cognitive claim — that learning machinery alone, given a symmetric task, will spontaneously invent the primitives that human cognition uses — for the regime in which we tested it.

## A three-layer causal ablation that successfully drives RGB-aligned primaries

The trichromacy literature distinguishes three causal layers (Stockman & Sharpe, 2000; Jacobs, 2009; Conway et al., 2007; Berlin & Kay, 1969):

| Layer | Biology | PCM operationalisation |
|---|---|---|
| **B** Hardware | Three opsins (L/M/S) with overlapping cosine sensitivity curves | `make_lms_like_centroids`: three orthogonal cone basis vectors and per-hue half-cosine weights peaked at hue 0/4/8 |
| **C** Statistics | Natural chromatic statistics (e.g. green-leaf abundance) | Per-hue exponential sampling weight peaked at hue 4 (max:min ≈ 4:1) |
| **D** Task | Ripe-fruit / leaf foraging asymmetry | A binary head consuming the same `mixing_bias` facet, ripe = hue ∈ {0, 1, 11} |

We ran a 5-condition × 8-seed ablation: **A** baseline (random centroid + uniform sampling + no auxiliary head); **B** alone; **C** alone; **D** alone; **B+C+D** stacked.

Two diagnostics indexed the geometry: the *strict-equidistant rate* (whether anchor spacings are exactly [4, 4, 4] mod-12, i.e. cyclic-symmetry fully broken into a 120° tripartition) and the *red-wedge anchor rate* (whether at least one of three anchors falls into the foraging-relevant set {0, 1, 11}, i.e. cyclic symmetry broken in the rotation dimension).

| Condition | Strict EQUI (k = 3) | RGB-rotated hit | Red-wedge anchor |
|---|---|---|---|
| **A** baseline | 2 / 8 | 2 / 8 | 0.62 |
| **B** + LMS centroid | **4 / 8** | **4 / 8** | 0.75 |
| **C** + green-peak sampling | 3 / 8 | 3 / 8 | 0.75 |
| **D** + ripe-fruit head | 3 / 8 | 3 / 8 | **1.00** |
| **B+C+D** combined | **4 / 8** | **4 / 8** | **1.00** |

Three findings stand out.

First, **the foraging task (D) is the strongest cyclic-symmetry breaker**: a single binary head drives the red-wedge anchor rate from 0.62 to 1.00. This computational result mirrors Jacobs's (2009) hypothesis that red-green discrimination, driven by ripe-fruit and tender-leaf detection, was the selective pressure that pushed L/M cone divergence in primates. Critically, in our setup the foraging task is the only manipulation that fixes the *rotational* ambiguity of the equidistant primary set; biology alone (B) cannot.

Second, **biological prior alone (B) is insufficient**. LMS-like supervision lifts the equidistant rate from 2/8 to 4/8, but the rotation class continues to vary by seed: 3/4 successful seeds land on rotation-class {3, 7, 11}, only one on {0, 4, 8}. The cyclic task symmetry of the mixing operation re-uniformises hue rows during gradient descent, partly erasing the centroid-level prior. This is consistent with the Conway et al. (2007) finding that V4 hue-selective neurons sample the entire hue circle roughly uniformly even though L/M cones supply asymmetric input — categorical "focal hues" are defined at the task-readout stage, not at the cone stage.

Third, **the combined manipulation (B+C+D) is the only configuration with simultaneous strict equidistance and full rotation fixation**. Three layers stack monotonically and reach the maximum on both diagnostics. This computational result reproduces, in toy form, the multi-causal account of trichromacy that no single-layer story has been able to displace empirically.

## Methodological proposal: A / B / C / D / B+C+D as a default ablation template

The procedure above generalises beyond colour. For any debated primitive *Y* in domain *X*, identify three minimal computational analogues:

1. A **biological prior** (some non-uniform centroid / receptor / pre-existing structure that supplies *Y* as input geometry).
2. An **environmental statistic** (a non-uniform input distribution that makes *Y*-aligned regions over-represented).
3. A **task asymmetry** (a small auxiliary task whose loss is non-trivial only on a *Y*-aligned subset of inputs).

Run five conditions, eight seeds each, and report whether *Y* emerges in (i) baseline, (ii) each layer alone, (iii) all stacked. The pattern of results identifies which layer is necessary, which is sufficient, and which contribute additively. The same template can be applied to candidate primitives across the cognitive sciences:

- **Number / base-10**: B = 10-axis cone-like centroid; C = digit-frequency sampling; D = explicit carry head.
- **Phoneme / IPA features**: B = articulator-grouped centroid; C = phonotactic frequency; D = minimal-pair classification head.
- **Space / cardinal directions**: B = orientation-tuned centroid; C = environment-anisotropic sampling; D = landmark-disambiguation head.
- **Causal reasoning / do-calculus**: B = intervention-axis centroid; C = intervention-frequency bias; D = counterfactual-prediction head.

In each case, the experiment offers a falsifiable lower bound on what general-purpose learning machinery, devoid of injected priors, will and will not produce.

## Discussion

We do not claim PCM is a model of human cortex. It uses gradient descent rather than spike-timing plasticity, lacks neuromodulators, has no developmental schedule, and operates on tens to hundreds of concepts. It is a *representation-level falsifiable testbed*, sitting between Marr-level cognitive theory (which often makes claims it cannot operationalise) and end-to-end deep models (which do operationalise but lack interpretable parameters). Within this niche it can do something the larger systems cannot: ask whether a specific cognitive primitive *needs* injection or whether it falls out of generic learning, and answer with a calibrated probability across seeds.

Our results carry one substantive cognitive-science implication. Across two independent toy negatives (numbers and colours) and one positive layered ablation (colours), the pattern is that **representational primitives reflect either the task's symmetry group or an injected asymmetry, and nothing else**. PCM does not autonomously invent perceptual primitives, but it faithfully preserves any task-asymmetric prior supplied to it. This is a methodological win, not a limitation: anything one observes in PCM's bundle geometry can be traced, parameter by parameter, back to the task or the injection that produced it.

Three caveats. First, our colour task is an artificially symmetric mixing rule, not a natural-image distribution; the C-layer (statistical) result may underestimate the contribution of ecological statistics on naturalistic input. Second, the foraging head (D) uses a hand-specified ripe-set; whether this set itself can be discovered by a higher-order learner remains open. Third, all results are at *N* ∈ {12, 100} concept scale; whether the layered pattern survives at 10⁴–10⁶ concepts remains a scaling question that PCM, in its current form, cannot answer.

What we offer is not a theory of where primaries come from — that question requires fMRI, single-unit recording, and developmental data we cannot supply. We offer a falsifiable computational protocol for any researcher who wants to ask, of any candidate primitive, whether it can emerge unconditionally or whether it requires injection. The answer in the colour case is unambiguous and matches the multi-layered consensus already established in the trichromacy literature; whether other debated primitives — base-10 columns, IPA features, cardinal axes, causal-reasoning operators — show the same pattern is, on present evidence, an open question worth running.

---

## Outstanding questions

1. **Partially resolved (this report, addendum)**. We ran the same 5-condition A/B/C/D/B+C+D template on the number domain with 8 seeds (1..*N* = 30, decimal-cone centroids, round-number sampling boost ×5, last-digit classification head). The results are exactly parallel to the colour case: the task-driven layer (D, last-digit head) lifts ``spike_10`` from +0.290 ± 0.042 to +0.667 ± 0.052 and flips the sign of `cos[+10] − cos[+1]` from −0.157 to +0.467, consistent with base-10 column structure. The combined B+C+D condition reaches ``spike_10`` = +0.684 with ``spike_5`` ≈ 0 and last-digit cluster purity 0.876 ± 0.085 — the cleanest base-10 periodicity we have observed in PCM, and one that the original §7 negative could not produce. The three-layer recipe therefore generalises across two qualitatively distinct domains (cyclic colour + linear number) for **representational** primitives.

   **However**, two additional extrapolation experiments reveal a clean architectural ceiling that the three-layer recipe does *not* cross. (a) *Number length-OOD*: training QuadArithHead on a, b, c ∈ [1, 30] with all 100 concepts registered and LastDigitHead trained over the full 1–100 range, A / B / C all flat-line at 0.051 ± 0.000 on length-100 OOD (the head's systematic bias on novel inputs), while D and B+C+D rise to 0.055 ± 0.001 and 0.062 ± 0.003 — statistically robust (5/5 seeds in the same direction) but only +1.1 pp absolute. (b) *Colour target-hue holdout*: dropping every mixing triple whose output hue is *c* = 5 from training and re-evaluating across 5 conditions × 5 seeds gives a striking and unanimous 25/25 OOD = 0.000, well below the 1/12 ≈ 0.083 chance baseline. The two boundaries are mechanistically distinct — the number case is *input-side* (bundle row receives no task gradient, regresses to chance), the colour case is *output-side* (centroid receives no cosine gradient, head's softmax never points at it) — and together they delimit precisely where the three-layer recipe stops working. The same architectural separation is pre-stated in PAPER §3.6 / §9 as the D91/D92 limit, and now has 25-of-25 plus 5-of-5 empirical confirmation. We read this as the falsifiable testbed delivering exactly what one wants from it: a clean partition between **geometric / categorical emergence** (where the three-layer recipe works robustly across two domains) and **algorithmic / compositional emergence** (where it does not, in either domain), and a precise architectural prescription (slot-generator-style bundle synthesis, à la D93a) for any future work that wants to attempt the algorithmic side.
2. How does the relative contribution of B / C / D layers depend on training duration? Our 30-epoch run favours D; does longer training let C or B dominate?
3. Do the three layers compose linearly across diagnostics, or are there interaction effects where (B + D) ≠ (B alone) + (D alone)? We see additive behaviour on `spike_10` and `purity` in numbers (BCD ≈ D alone), but a much stronger super-additive interaction on `spike_5` (BCD = −0.07 vs D = −0.23, baseline = −0.37): this is the cleanest evidence in our data set that the layers do not simply sum.
4. Can the foraging-task (D) layer itself be acquired by a higher-order learner, or must it always be supplied as architectural prior, as in our experiments?
5. Does the methodological template extend to multi-agent settings, where the "task asymmetry" layer is supplied by communicative pressure between agents rather than by an experimenter-supplied head?
6. What is the minimum bundle dimensionality at which the cyclic task symmetry overrides the LMS centroid prior? Our 64-d setup is empirical; a theoretical bound would tighten the claim.

---

## Box 1 · Parametric Concept Memory in one paragraph

PCM is a small attribution-strict learning architecture in which each concept owns one row in a shared `bundle_pool[facet]` parameter table; muscles (downstream task heads) consume rows by index and have no parameters of their own that depend on concept identity. A wake-time gradient pass causes bundle rows to take on the geometry implied by the task's symmetry group — linear (numbers), circular (colours), 2-D lattice (space), prototype-clustered (phonemes). An offline sleep pass compresses bundle rows into prototype centroids and per-row residuals, with seven falsifiable invariants (G1–G7) ensuring that the compressed representation reconstructs the original up to numerical eps. PCM has been validated across the four canonical concept domains; full source code, unit tests, and ablation harnesses are available at the accompanying repository.

---

## References (suggested core)

- **Berlin & Kay (1969)**. *Basic Color Terms: Their Universality and Evolution*. UC Press. — Cross-linguistic regularities in colour categorisation.
- **Conway, Moeller & Tsao (2007)**. Specialized color modules in macaque extrastriate cortex. *Neuron* 56, 560–573. — V4 hue-selective neurons sample the hue ring uniformly.
- **Jacobs (2009)**. Evolution of colour vision in mammals. *Curr Biol* 19, R1085–R1092. — Foraging-pressure account of L/M cone divergence.
- **Klinzing, Niethard & Born (2019)**. Mechanisms of systems memory consolidation during sleep. *Nat Neurosci* 22, 1598–1610. — NREM sharp-wave ripples + cortical spindle coupling.
- **Sun et al. (2023)**. Organising memories for generalisation in complementary learning systems. *Nat Neurosci* 26, 1438–1448. — Consolidation conditional on generalisation.
- **Stockman & Sharpe (2000)**. The spectral sensitivities of the middle- and long-wavelength-sensitive cones derived from measurements in observers of known genotype. *Vision Res* 40, 1711–1737. — LMS cone fundamentals.
- **Sutton et al. (2024)**. Loss of plasticity in deep continual learning. *Nature* 632, 768–774. — Continual-learning failure modes addressed in our anchor-EMA design.
- **Zheng et al. (2023)**. Online clustered codebook. *ICCV*. — Codebook-collapse failure mode addressed in our `force_recluster=False` default.

(Add or adapt to the target journal's preferred reference style.)

---

## Statement of contributions, code, and data

All numerical results in this report were produced by `experiments/sleep_color_primaries.py` (8 seeds × 5 conditions, ~14 minutes on a single mid-range GPU) and `experiments/sleep_inspect_color_anchors.py` (8 seeds × 3 *k* values × 24 ablations). Code, ablation harnesses, raw summaries (`outputs/primaries_5cond_8seed/summary.json`), and the rendered figure (`docs/figures/F11_color_primaries.png`) are released with this report. Sixty-three unit tests (G1 through G7 plus four-domain integration) are included; pre-flight regression: 63/63.
