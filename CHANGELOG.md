# Changelog

All notable changes to **Parametric Concept Memory (PCM)** are recorded
here. This project follows [Semantic Versioning](https://semver.org/)
and the [Keep a Changelog](https://keepachangelog.com/) conventions.

## [Unreleased]

### Added — PCM v2 Dual-Channel + RPE Architecture (F40 – F49)

The first architecture-level redesign of PCM since the original
Tier-A/B/C/D split. Motivated by three independent S1–S6
falsifications (S3 `mixed_OOD = 0.000`, S6 vector analogy < chance,
S2 number dead-codebook), all of which trace to one design flaw:
encoding a concept as **a single vector** that has to serve as
positional address, categorical identity, and continuous attribute
axis simultaneously.

**Public API (frozen at F49):**
- **`pcm/dual_channel.py`** — opt-in dual-channel concept encoding:
  - `register_dual_channel_facet(cg, base, slot_dim, attr_dim)` →
    paired `<base>_slot` (Tier-G clustered) + `<base>_attr`
    (excluded from sleep, contrastive-trained) facets.
  - `collapse_dual_channel(cg, base_facet, ids, ...)` → drop-in
    dual-channel replacement for `cg.collapse_batch`.
  - `info_nce_loss`, `arithmetic_consistency_loss`,
    `successor_consistency_loss` (with norm penalty),
    `spread_regularizer` — attribute-channel loss primitives.
  - **`RelativePositionEmbedding(ranges, embed_dim)`** — k-axis
    learned embedding of integer displacements; the V3 lever.
  - `pair_attention_logits` — minimal pair-attention primitive.
- **`pcm/heads/v2_dual_channel.py`** — public PCM v2 muscles:
  - `DualChannelPairHead` — pair-input head with three optional
    paths (slot attention / attribute MLP / RPE lookup) and three
    gate modes (`fixed` / `schedule` / `learned`).
  - `SlotIdentityAuxHead` — single-input identity aux head
    (LastDigit/RowIndex analogue for v2).
  - `pair_collapse_and_forward` — v1-style ergonomics.
- **`pcm.sleep.run_dual_phase_sleep`** — S1 NREM-style two-phase
  sleep (small-pupil "fresh" + large-pupil "old"); new
  `PROTO_CID_PHASE_TEMPLATE` + `RELATION_CID_PHASE_TEMPLATE`
  cid templates; G8a / G8b / G8c invariants in
  `tests/test_tier_g_sleep.py`.

**Bug fix:**
- `ConceptGraph._ensure_facet` device-equality regression: the
  pre-fix `pool.device != torch.device(device)` comparison
  silently rebuilt the bundle pool `nn.Parameter` whenever the
  caller passed `"cuda"` while the pool was on `cuda:0`,
  invalidating optimiser references and freezing v2 V1/V2
  training at its random init. Fixed in F42 with regression
  tests `tests/test_concept_graph_device.py` (3 cases).

**Empirical results (5-seed mean ± std, see `docs/SHORT_REPORT_2026_S1_S6.md`):**
- **V1 (slot purity)** — number N=10: NMI = 1.000 ± 0.000 ✓
- **V2 (vector analogy)** — number N=10: top1 = 1.000 ± 0.000 ✓
- **V3 (mixed_OOD)** — space 5×5/7×7 grid: 1.000 ± 0.000 ✓
  (RPE-only; A1 schedule reproduces; A2 learned alone fails to
  0.720; A2 + L1 β=0.1 recovers to 1.000).
- **Cross-domain RPE (F48)** — concat baseline → RPE (5 seeds):
  - space (5×5 mixed_OOD): 0.000 → **1.000 ± 0.000** (+100 pp)
  - colour (12-cyclic, hue holdout): 0.000 → **1.000 ± 0.000**
  - phoneme (V/M/P 3-axis): 0.003 → **0.965 ± 0.020** (+96 pp)
  - number (1-d, |Δ| ≤ 29): 0.013 → 0.764 ± 0.009 (+75 pp;
    bounded by lookup range, motivates functional RPE in v3)

**Design observation, falsifiable form:**
"Models do not spontaneously discover their own minimum
sufficient statistic via gradient descent" — A2 learned gates
stay at λ ≈ 0.98 across all reward strengths; only an explicit
L1 penalty (β=0.1) closes the gate. See F46 commit message and
`docs/SHORT_REPORT_2026_S1_S6.md §V3-RPE` for the five-config
× five-seed evidence.

**Documentation:**
- `docs/PCM_V2_DUAL_CHANNEL_DESIGN.md` — design proposal +
  V3-RPE update (§10) + A1/A2 gate findings (§11) + open
  follow-ups (§12).
- `docs/PCM_V2_MIGRATION_GUIDE.md` — five-line v1 → v2
  migration recipe + per-head mapping table + reproducibility
  smoke commands.
- `docs/SHORT_REPORT_2026_S1_S6.md` — full F40–F48 spin-off
  short-report draft.
- `docs/2026_LITERATURE_AND_PLANS.md` — 23-paper literature
  survey across cognitive neuroscience / developmental psych /
  anthropology / philosophy / 2026 ML.

**Tests (107 / 107 passing):**
- 19 new in `tests/test_dual_channel.py` (DC1 facet pairing,
  DC2 collapse, DC3 losses, DC4 RPE 1-D/2-D/3-D, DC5 pair
  attention, DC6 public heads).
- 3 new in `tests/test_concept_graph_device.py` (F42 regression).
- 3 new in `tests/test_tier_g_sleep.py::TestG8DualPhaseSleep`
  (G8a/b/c S1 invariants).

### Added — Tier-G Sleep Abstraction Pass (D95)
- **`pcm/sleep.py`** — opt-in offline pass that performs NREM-style
  codebook compression on `bundle_pool[facet]` rows, registers
  cluster centroids as `abstract_prototype` nodes, member residuals
  on a sibling `<facet>_residual` pool, and cookable
  `abstract_relation` subgraphs (anchor + residual). Public API:
  `attach_sleep`, `run_sleep_pass`, `sleep_status`,
  `register_prototype`, `register_relation`,
  `iter_abstract_relations`, `make_replay_source_from_buffer`.
- **Two new DNA ops** in `pcm/dna_ops.py`:
  - `concept.codebook_lookup` (collapse) — single-row read of an
    abstract prototype slot, with sleep-flagged attribution.
  - `concept.relation_apply` (pure) — combine anchor + residual via
    `add` / `mul` / `concat`.
- **Falsifiable contract G1–G6** documented in
  `docs/PCM_TIER_G_SLEEP_ABSTRACTION.md`:
  G1 bit-identity off, G2 pool memory safety, G3 ρ regression bound
  (numbers ρ_linear ≥ 0.965 / colors ρ_circular ≥ 0.965 / space
  ρ_L1 ≥ 0.85 / Procrustes ≤ 0.15 / phoneme intra-inter gap ≥ 1.85),
  G4 centroid = mean(members), G5 abstract cook reconstructs row,
  G6 idempotency.
- **Unit tests** (`tests/test_tier_g_sleep.py`, 8 cases): one method
  per invariant + sleep_status sanity. Runs in ~1.5 s.
- **Per-domain integration** (`tests/test_sleep_four_domain.py`,
  11 cases): backward-compat (`sleep_every=None`) and live
  (`sleep_every=2`) for color / space / phoneme `train_one`s, plus
  signature smoke for `quad_study.train_quad` and
  `purity_audit.purity_train_one`.
- **Five training loops migrated** with backward-compatible
  `sleep_every: int | None = None` kwarg (default `None` =
  byte-identical to v0.1.0):
  `experiments/color_concept_study/train.py`,
  `experiments/space_concept_study/train.py`,
  `experiments/phoneme_concept_study/train.py`,
  `experiments/purity_audit/train.py`,
  `experiments/quad_study.py`.
  Each loop returns a new `sleep_reports: list[dict]` field
  (empty when sleep is off) for downstream analysis.

### Added — Tier-G failure-mode diagnosis & functional sleep (D96)
- **G7 invariant** (post hoc) in `pcm/sleep.py` and
  `tests/test_tier_g_sleep.py`: the new
  `collapse_via_abstract` returns rows numerically equal to
  `cg.collapse_batch` immediately after a sleep pass; differences
  measured later are *strictly* due to gradient-flow rerouting. Two
  unit tests cover registered-member equality and unregistered-member
  fallback.
- **Functional consumption of abstract relations**
  (`pcm/sleep.py::collapse_via_abstract`,
  `pcm.sleep::collapse_with_optional_abstract`,
  `pcm.sleep::materialize_effective_bundle_state`): a head set with
  `use_abstract=True` reads each member as `anchor + residual`,
  routing gradient simultaneously through the shared prototype slot
  and the per-member residual row. This turns Tier-G from a read-only
  archive into a functional schema while preserving G1–G7.
- **Three SleepConfig knobs** mapping directly to known VQ-VAE / MoE
  failure modes documented in 2024 literature:
  - `anchor_ema ∈ (0, 1]` — Online-Codebook / VectorQuantizeEMA blend
    for re-cluster passes (cures the *topology hop* failure mode of
    `force_recluster=True` reported by Zheng et al. ICCV 2023).
  - `assignment ∈ {"hard", "soft"}` + `soft_tau` — softmax routing
    over all anchors in a facet (Default-MoE / SparseMixer style)
    that distributes gradient to every anchor and cures *expert
    starvation*.
  - `sleep_warmup` plumbed through every host
    `train_one`: skip sleep until the wake-time geometry has had time
    to form, mirroring CLS slow-cortical timescale.
- **CLI ablation harness** (`experiments/sleep_ablation.py`,
  `experiments/sleep_ablation_four_domain.py`): A/B/C ablation
  (no-sleep / sleep+direct / sleep+abstract) with full sweep over
  `--k-clusters`, `--anchor-ema`, `--assignment`, `--soft-tau`,
  `--sleep-warmup`, `--force-recluster`, `--ood-ratio`. Used to
  produce the F9 figures in the paper.
- **OOD evaluation in color and space `train_one`** via a new
  `ood_ratio: float = 0.0` kwarg (legacy default = 0 keeps
  full-train behavior bit-identical). Used to extend the F4 four-
  domain sleep ablation to F5 OOD-aware ablation.
- **F9 figure renderer**
  (`experiments/render_paper_figures/F9_sleep_four_domain.py`)
  produces F9a (4-domain ρ) and F9b (train + OOD accuracy) bar
  charts from the four-domain ablation summary JSON.
- **PAPER §6.6 Tier-G — Sleep Abstraction** added in
  `PAPER.zh-CN.md`: §6.6.1 four failure modes with literature
  mapping, §6.6.2 quad-domain Pareto-better at OOD=0.30
  (5 seeds: ΔOOD = +0.018, Δρ = +0.004), §6.6.3 four-domain safety
  table, §6.6.4 limitations.
- **PAPER §6.7 Sleep does not invent perceptual primitives**
  (negative result mirroring §7 base-10): on the 12-hue color
  domain, k ∈ {3, 4, 6} sleep anchors land on equidistant 360°/k
  hue rings up to k-means noise (≤ ±1-2 hue), but **the rotational
  offset is uniformly seed-dependent**: 0/24 seeds match RYB
  (the only non-equidistant prior probed), and RGB / CMY /
  CMYK_aligned / WarmCool6 hit-rates equal the strict-equidistant
  rate exactly. PCM does not invent visual-system primaries; it
  only realises one of *k* rotation classes of the cyclic
  task-symmetry group. Falsifies H_perceptual; supports H_taskSym.
  Diagnostic harness:
  `experiments/sleep_inspect_color_anchors.py`.
- **PAPER §6.8 Recovering perceptual primitives requires
  biological priors or ecological pressure** (positive
  counterpart to §6.7): three-layer causal ablation
  (B = LMS-like centroid; C = green-peak hue sampling;
  D = `RipeFruitHead` foraging task) on 12-hue color × k=3 sleep
  × 8 seeds. Findings:
  - **D alone** drives `red-wedge anchor` rate from baseline
    0.62 to **1.00** (8/8 seeds) — task asymmetry is the strongest
    cyclic-symmetry breaker (Jacobs 2009 *Curr Biol* foraging
    pressure for L/M cone divergence).
  - **B alone** lifts strict-equidistant from 2/8 to 4/8, but
    rotation class still varies by seed (Conway et al. 2007
    *Neuron* observation that V4 hue-selective neurons follow
    cone-level sampling but pick categorical hues only with
    ecological pressure).
  - **B+C+D combined** achieves both EQUI = 4/8 and red-wedge =
    1.00 simultaneously — minimal rotation-fixed primary
    geometry. Mirrors the human evolutionary path:
    cone genetics + chromatic statistics + foraging task.
  - New: `make_lms_like_centroids` in
    `experiments/color_concept_study/graph_builder.py`,
    `RipeFruitHead` in `experiments/color_concept_study/heads.py`,
    `mix_sample_weight` + `enable_ripe_head` plumbed through
    `train_one`, ablation harness
    `experiments/sleep_color_primaries.py`, F11 figure renderer
    `experiments/render_paper_figures/F11_color_primaries.py`.
- **PCM corollary**: PCM does not autonomously invent perceptual
  primaries (§6.7) but **faithfully preserves any task-asymmetric
  prior injected** (§6.8). This makes it an interpretability
  win, not a limitation: the structure you see in PCM's bundle
  geometry is exactly the symmetry of the task plus whatever
  external priors you supply, and **nothing else**.
- **PAPER §7.4 Three-layer causal injection reverses the §7
  base-10 negative** (positive counterpart to §7, exact mirror of
  §6.8 on the linear number domain). Mirrors the §6.8 colour
  ablation: B = `make_decimal_cone_centroids` (10 unit + 10 tens
  cones), C = `round_number_weights` (×5 sampling boost on
  multiples of 10), D = `LastDigitHead` (single-input
  10-class classifier consuming `arithmetic_bias`). 8 seeds × 5
  conditions on N=30 four-arithmetic. Findings:
  - **D alone** lifts ``spike_10`` from +0.290 ± 0.042 to
    +0.667 ± 0.052 (×2.3) and flips ``cos[+10] − cos[+1]`` from
    −0.157 to +0.467 (sign flip = same-units numbers more
    similar than adjacent numbers, the direct signature of
    base-10 column structure).
  - **B+C+D combined** reaches ``spike_10`` = +0.684 ± 0.038
    with ``spike_5`` ≈ −0.07 (close to zero), giving the
    cleanest 10-periodicity observed in PCM. Sleep k=10 anchor
    purity against last-digit equivalence classes is 0.876 ±
    0.085 (chance = 0.10).
  - **B alone** and **C alone** are weak (Δspike_10 ≈ +0.07 and
    +0.09 vs baseline), confirming the §6.8 finding that
    biological prior + ecological statistics by themselves do
    not break a strongly task-symmetric domain.
  - **Trade-off**: BCD's OOD acc 0.74 vs baseline 0.82 — the
    same ρ↔OOD trade-off observed in §6.6.2 / §6.8: cleaner
    structural geometry costs some held-out task accuracy, a
    pure structural-vs-utility trade-off intrinsic to
    asymmetric prior injection.
  - New: `experiments/number_decimal_priors.py` (centroid +
    head + sampling helpers), `experiments/sleep_number_decimal.py`
    (5-condition × 8-seed harness), `experiments/render_paper_figures/F12_number_decimal.py`
    (3-panel figure), `centroid_mode / digit_sample_weight /
    enable_last_digit_head` plumbed through `train_quad`. All
    63 unit tests still pass.
- **§6.7 / §6.8 / §7 / §7.4 joint claim**: across two
  qualitatively distinct domains (cyclic colour, linear
  number), the same 5-condition A/B/C/D/B+C+D protocol reverses
  the negative result, with the task-driven layer always the
  dominant contributor. This generalises the §6.7+§6.8 colour
  finding into a **falsifiable methodological proposal** for
  cognitive-science questions about emergent primitives.
- **PAPER §7.5 Length extrapolation hits a clean architectural
  ceiling**. Extends `train_quad` with `n_total: int | None`:
  registers 1..n_total concepts up front while QuadArithHead
  trains only on a, b, c ∈ [1, N] triples; LastDigitHead samples
  the full [1, n_total] range so length-OOD bundle rows still
  receive last-digit gradient. New harness
  `experiments/sleep_number_extrapolate.py` and figure renderer
  `experiments/render_paper_figures/F13_number_extrapolate.py`.
  5-condition × 5-seed result on N_train=30 / N_total=100:
  - A/B/C all flat-line at 0.051 ± 0.000 on length-100 OOD
    (≈ chance level due to the head's systematic OOD bias).
  - D = 0.055 ± 0.001 and BCD = 0.062 ± 0.003 (5/5 seeds in
    the same direction) — statistically robust but only
    +1.1 pp absolute over chance.
  - A baseline in-range OOD = 0.786 ± 0.054, BCD = 0.664 ±
    0.084 — the same trade-off seen in §7.4 (cleaner geometry
    costs random interpolation accuracy).
  - **Interpretation**: the three-layer recipe drives
    *representational/categorical* emergence (§6.7 / §6.8 /
    §7.4) but does NOT drive *algorithmic/compositional*
    emergence (length extrapolation). This is consistent with
    PAPER §3.6 / §9 / §7.3's pre-stated D93/D93a architectural
    boundary: per-concept indexed bundles support geometry
    over a fixed concept inventory but not unbounded
    digit-place composition.
  - This negative result is itself a methodological
    contribution: it gives a clean, falsifiable separation
    between two kinds of "emergence" that cognitive science
    routinely conflates, and tells future PCM-based work
    exactly what kind of architectural extension would be
    required to cross the boundary.
- **PAPER §7.5-color hue holdout** (cleaner mirror of the
  number length-OOD ceiling). New `holdout_target_hues` kwarg
  on color `train_one`: drops every mixing triple whose target
  hue ``c ∈ holdout`` from training and evaluates the held-out
  ones separately. New harness
  `experiments/sleep_color_holdout.py` and figure renderer
  `experiments/render_paper_figures/F14_color_holdout.py`.
  Result on hue-5 holdout, 5 conditions × 5 seeds = 25 runs:
  - **All 25 runs strictly 0.000 on held-out hue 5**, well
    below the 1/12 = 0.083 chance baseline.
  - Train accuracy is 1.000 for A / C / D and ≈ 0.70 for
    B / BCD (LMS centroids overlap, making the closed
    in-domain task slightly harder).
  - **Closed-output-set ceiling**: the head's softmax is
    never trained to point at the held-out target's centroid,
    so even when prior layers (LMS / sampling / ripe head)
    pre-shape the held-out concept's bundle row, the head
    can still never predict it.
  - Together with §7.5 number length-OOD, this gives PCM
    two clean, falsifiable architectural ceilings:
    *input-side* (bundle row never receives task gradient,
    chance-level OOD; numbers) and *output-side*
    (centroid never receives task gradient, strictly-zero
    OOD; colour). Both are pre-stated D91/D92 limits in
    PAPER §3.6 / §9 and now have empirical 25-25 / 5-5
    confirmation.
  - Cognitive-science parallel: human infants have full
    LMS cone responses from birth (sensory representation
    present) but categorical colour naming stabilises at
    4–6 months and depends on the specific language being
    acquired (Berlin & Kay 1969; Skelton et al. 2017
    *PNAS*). PCM's "centroid present, head untrained →
    holdout = 0" mirrors "cone responses present, language
    label absent → categorical access blocked".
- **PAPER §6.9 phoneme cross-language transfer** (third
  domain in the three-causal-layer protocol; reveals a
  qualitative dominant-layer switch). New
  `experiments/phoneme_transfer_priors.py`
  (`make_articulator_centroids`, `zipf_phonotactic_weights`,
  `MinimalPairHead`), upgraded
  `experiments/phoneme_concept_study/train.py` with
  `source_indices`, `centroid_init`,
  `enable_minimal_pair_head` kwargs, harness
  `experiments/sleep_phoneme_transfer.py`, figure
  `experiments/render_paper_figures/F15_phoneme_transfer.py`.
  Setup: 20-phoneme inventory split into 13-source / 7-target
  per seed; V/M/P heads see only source; target accuracy on
  V (chance 0.5) / M (chance 0.25) / P (chance 0.25) is the
  transfer indicator. 5-condition × 5-seed result:
  - **A baseline** target acc = 0.514 / 0.114 / 0.257
    (chance or below — V/M/P heads' systematic OOD bias).
  - **B articulator centroid alone** target acc = **1.000 /
    0.943 / 0.771** — the strongest single-layer transfer
    signal observed in any PCM domain so far.
  - **D minimal-pair head alone** transfers only the facet
    it consumes (default voice_bias): tgt_V = 1.000, M/P
    stay at chance.
  - **Phoneme is B-dominant**, in contrast with colour and
    number which are D-dominant. PCM thus reveals a
    *task-symmetry × dominant-layer* prediction principle:
    cyclic / translational task groups need D to break
    symmetry; orthogonal-categorical task groups let B
    transfer cleanly on its own. This matches Werker & Tees
    1984 *Infant Behav Dev* on infant universal phonetic
    discrimination — articulator anatomy supplies axis
    geometry from birth, no foraging-style task pressure
    required.
- **Cross-domain dominant-layer table** now spans three
  qualitatively different domains:
  - colour mixing (Z₁₂ cyclic): D dominant
  - number arithmetic (ℤ translational): D dominant
  - phoneme V/M/P (orthogonal categorical): B dominant
  Together with the §7.5 / §7.5-color extrapolation ceilings,
  this gives the falsifiable testbed claim its strongest
  triple-domain support.
- **PAPER §7.5-space — spatial 2-D length extrapolation reveals
  a new input-distribution-interaction ceiling type**.
  New `experiments/space_cardinal_priors.py` (cardinal-axis
  cones, center-bias weights, RowIndexHead),
  `experiments/sleep_space_extrapolate.py` (5-condition × 5-seed
  harness on a 7×7 grid, training MoveHead only on the inner 5×5
  sub-grid, three test splits), and
  `experiments/render_paper_figures/F16_space_extrapolate.py`.
  5-condition × 5-seed = 25 runs at chance ≈ 0.200:
  - **A baseline** outer-OOD = 0.263 ± 0.005 (≈ chance).
  - **B cardinal centroid alone** outer-OOD = 0.570 ± 0.020
    (≈ 2.9 × chance). **B is the dominant single layer for
    space**, mirroring the §6.9 phoneme finding.
  - **D row-index head alone** outer-OOD = 0.498 ± 0.103.
  - **B+C+D combined** outer-OOD = 0.572 ± 0.038 — small lift
    over B alone, suggesting B already captures most of the
    transferable cardinal-axis information.
  - **mixed-OOD strictly 0.000 in 25 / 25 runs**, well below
    the 0.200 chance baseline. This is a new ceiling type
    distinct from §7.5 input-side and §7.5-color output-side:
    MoveHead's fc1 receives `concat(bundle_a, bundle_b)`,
    trained only on the inner × inner joint distribution; the
    (inner, outer) joint distribution is OOD even when each
    individual cell's bundle is prior-injected. Term:
    **input-distribution interaction ceiling**, unique to
    two-input muscles with asymmetric input roles.
  - Three-domain unified ceiling taxonomy (input-side / 
    output-side / symmetric-OOD / asymmetric-OOD) added to
    PAPER §7.5-space.
- **PAPER §7.5-space dominant-layer placement**: B ≈ D > C,
  intermediate between phoneme (B-dominant, orthogonal
  categorical) and colour / number (D-dominant, cyclic /
  translational). Consistent with the task-symmetry × 
  dominant-layer principle: 5-class direction has a partial
  cyclic group + categorical "same" + row × col factorisation.

- **`pcm.diagnostics` — formal causal-ablation protocol API**.
  New module `pcm/diagnostics.py` packages the §6.6 / §6.7 /
  §6.8 / §6.9 / §7.4 / §7.5 / §7.5-color / §7.5-space
  five-condition × N-seed pattern as a reusable abstraction.
  Public surface:
  - `CAUSAL_LAYERS = ("B", "C", "D")` — canonical layer order.
  - `AblationLayers` — frozen dataclass with B/C/D flags and
    `is_active(layer)` helper, accepts `None / False / True / 
    str` activation values.
  - `AblationCondition` — named (B, C, D) condition with
    `to_dict()` for JSON serialisation.
  - `DEFAULT_CONDITIONS` — the five canonical conditions
    (A_baseline, B_prior, C_statistics, D_task, BCD_combined).
  - `CausalAblationProtocol` / `run_causal_ablation` — driver
    that orchestrates 5 × N runs and produces a summary dict
    in the same `config / by_condition / per_seed / mean / std`
    layout already used by all bundled experiment scripts and
    F11–F16 figure renderers.
  - `summarise_per_seed` — NaN-safe stats helper.
  Five new unit tests (D1–D5) in `tests/test_diagnostics.py`
  cover the canonical condition shape, layer activation
  semantics, driver-invocation contract, summary-dict layout,
  and statistical helper behaviour. The module is **purely
  orchestration**: it does not assume any specific PCM
  architecture, head shape, or evaluation metric, and the
  bundled domain-specific experiment scripts continue to work
  unchanged. Future work that wants to add a new domain to
  the falsifiable causal-ablation protocol can now do so in
  ~50 lines, by writing a domain `run_one(seed, layers, **kw)`
  callback and passing it to `run_causal_ablation`.

Tests: 75 / 75 pass (63 prior + 12 new diagnostics smoke). No
regressions in Tier-A grow / Tier-B gate / Tier-C peer / Tier-D
cook bit-identity; G1–G7 invariants still hold.

- **F35 — PAPER §7.5 scale-up to N=50 / N_total=200 confirms the
  ceiling tightens with scale**. Re-ran the §7.5 length-OOD
  protocol on a 5×-larger training range (50 numbers) and a
  4×-larger output space (200 numbers registered) using the F34
  dogfooded `experiments/sleep_number_extrapolate.py` with no
  code changes. 3 seeds × 5 conditions × 20 epochs × 120 steps,
  proportionally scaled from the N=30/100 baseline.
  - A baseline length-100 OOD: 0.048 ± 0.000 (matches the
    N=30/100 chance level of 0.051 ± 0.000).
  - **D last-digit head signal disappears**: from +0.055 ± 0.001
    at N=30 (+1.1 pp over chance) to 0.049 ± 0.001 at N=50/200
    (≈ 0.1 pp, within noise of A baseline).
  - **BCD shows catastrophic-seed behaviour**: 1 of 3 seeds
    drops to length-100 OOD = 0.004 (well below chance = 0.005);
    aggregate 0.035 ± 0.027 vs A baseline 0.048 ± 0.000.
  - **Conclusion**: the +1.1 pp signal observed at N=30/100 is
    a finite-sample effect specific to the small-N regime, not
    a structurally robust transfer mechanism. The input-side
    ceiling **tightens with scale rather than relaxing**, which
    *strengthens* §7.5's main claim: D91/D92 static-bundle
    architecture cannot cross length-OOD without D93a slot-
    generator upgrade.
  - This experiment also serves as a **F34 dogfood validation**:
    the dogfooded `sleep_number_extrapolate.py` ran cleanly at
    a 4-5× larger problem scale with no modifications, producing
    a self-consistent and informative result. Schema produced by
    `pcm.diagnostics.run_causal_ablation` is robust across scale
    changes.
  - PAPER §7.5 Chinese and English versions both updated with
    the scale-up sub-section.

Tests: 75 / 75 pass (no test changes required for F35).

- **F37 — PAPER §7.5-space addendum: mixed-OOD ceiling is fc1
  distribution-coverage, broken by 5 % training-pair
  augmentation**. New experiment script
  `experiments/sleep_space_mixed_augment.py` and figure renderer
  `experiments/render_paper_figures/F17_space_mixed_aug.py`.
  Builds on the §7.5-space BCD_combined condition, splits all
  mixed pairs (one inner + one outer cell) 50 / 50 per seed
  into mixed_train_pool / mixed_test_pool, and varies the
  fraction of training batches drawn from mixed_train_pool.
  4 rates × 5 seeds = 20 runs:
  - rate=0.00 (no aug): mixed_test = 0.000 ± 0.000
    (replicates F32 §7.5-space ceiling), outer_OOD =
    0.596 ± 0.047 (replicates F32).
  - **rate=0.05: mixed_test = 0.600 ± 0.245** (+60 pp jump
    from rate 0.00; 5 / 5 seeds give mixed_test ≥ 0.40),
    outer_OOD = 0.372 ± 0.040 (−22 pp trade-off cost).
  - rate=0.15 / 0.30: mixed_test rises monotonically to
    0.640 / 0.680, outer_OOD drops to 0.337 / 0.315.
  - **5 % augmentation completely breaks the mixed-OOD
    ceiling**, providing direct evidence that the §7.5-space
    asymmetric-OOD ceiling is not a PCM-fundamental boundary
    but an fc1 input-distribution-coverage issue — a softer
    ceiling than the §7.5 input-side or §7.5-color output-
    side PCM-fundamental boundaries.
  - **Monotonic trade-off**: outer-OOD (cardinal-prior-driven
    transfer) degrades 0.596 → 0.315 (−28 pp) as rate
    increases, indicating tension between cardinal centroid's
    abstract geometry and fc1's input-distribution fitting.
  - Refined PCM ceiling taxonomy in PAPER §7.5-space addendum
    (Chinese + English): input-side (numbers) and output-side
    (colour hue) are D91/D92-fundamental and uncrossable by
    augmentation; symmetric-OOD (space outer) is partial-prior-
    driven and augmentation hurts it; asymmetric-OOD (space
    mixed) is fc1-distribution-coverage and 5 % augmentation
    breaks it.
  - Practical implication for D93a follow-up: joint-
    distribution-aware bundle synthesis is *not* needed to
    cross asymmetric-OOD; mixed-pair augmentation in the
    existing D91/D92 training pipeline suffices. Contrasts
    sharply with §7.5 number length-OOD, where augmentation
    is impossible and a true D93a slot-generator upgrade
    is required.

Tests: 75 / 75 pass (no test changes for F37).

- **F38 — `docs/PCM_CAUSAL_ABLATION_GUIDE.md` author's reference for
  the five-condition protocol API**. Crystallises the F33 (and F34
  dogfood) experience into a 50-line recipe for new domains:
  - When to use the protocol (3-criterion checklist).
  - Three-step recipe (declare conditions, write dispatcher, drive
    with `run_causal_ablation`).
  - Output schema with exact key names and example
    `summary.json` skeleton.
  - Optional cond-level post-aggregation pattern (for §6.7 / §6.8
    style ad-hoc fields like `fraction_equidistant`).
  - "What you do not need to do" anti-pattern list (no nested
    loops, no domain `_stats`, no inventing new condition names).
  - Six worked-example pointers to existing scripts.
  - "When this protocol is not the right tool" exception list
    (long-running continual ablations, augmentation rate sweeps,
    multi-domain joint training).
  Aimed at researchers adopting PCM for new cognitive-science
  questions.

- **F39 — dogfood numerical-drift verification**. Re-ran the
  F34-dogfooded `experiments/sleep_color_holdout.py` (commit
  `d81f00a`) at the original PAPER §7.5-color configuration
  (5 seeds × 5 conditions, holdout-hue=5, 30 epochs × 200 steps).
  Output:
  - A_baseline:    1.000 / 0.000  (matches pre-dogfood)
  - B_lms:         0.689 ± 0.048 / 0.000  (matches pre-dogfood)
  - C_greenpeak:   1.000 / 0.000  (matches pre-dogfood)
  - D_ripehead:    1.000 / 0.000  (matches pre-dogfood)
  - BCD_combined:  0.711 ± 0.057 / 0.000  (matches pre-dogfood)
  All 25 / 25 runs reproduce the original §7.5-color
  closed-output-set ceiling (mixed_test_OOD strictly 0.000).
  Confirms PyTorch deterministic-seeding holds across the F34
  refactor: `pcm.diagnostics.run_causal_ablation` introduces zero
  numerical drift versus the hand-rolled per-script orchestration
  it replaced. The same seed produces the same per-seed metrics
  bit-for-bit. F34 is therefore a pure clarity / abstraction win.

Tests: 75 / 75 pass (no test changes for F38 / F39).

### Authority — extended
Above plus VQ-VAE / continual-learning literature mapped to the
four observed failure modes:
Zheng et al. *ICCV* 2023 (online clustered codebook); ECVQ-VAE
*Multimedia Systems* 2024 (control-chart codebook regulation);
VQGAN-LC *NeurIPS* 2024 (large codebook utilization); Zhang
*NeurIPS* 2025 (dimensional collapse in VQ-VAE);
Sutton et al. *Nature* 2024 (loss of plasticity); SparseMixer
*ICLR* 2024 (sparse-to-dense backprop); Default-MoE 2025 (EMA
expert outputs); Sun et al. *Nat Neurosci* 2023 (consolidation
conditional on generalization).

### Test summary
63/63 unit tests pass in 7.13 s on CPU (61 prior + G7 ×2). No
regressions in Tier-A grow / Tier-B gate / Tier-C peer / Tier-D
cook bit-identity.

## [0.1.0] — 2026-04-22

Initial public release: PCM framework + the four-domain empirical
paper (numbers · colors · space · phonemes) + causal bundle-swap
experiment + full pre-trained ANS encoder.

### Added
- **Core framework** (`pcm/`):
  - `ConceptGraph` + `ConceptNode` — symbolic graph container with
    attribution-closure guarantees (Proposition 1 in `PAPER.md` §3.4).
  - `ParamBundle` (`nn.ParameterDict` wrapper) — per-node multi-facet
    parameter storage with lazy initialisation and a consumer
    registry (`bundle.consumed_by`).
  - `ContextualizedConcept` — ephemeral handle returned by
    `node.collapse(caller, facet, shape, tick)`.
- **Muscle heads** (`pcm/heads/`): `ArithmeticHeadV2`, `ComparisonHead`,
  `NumerosityClassifier`, `NumerosityEncoder` (+ `DatasetConfig`,
  `generate_dot_canvas`, `encode_numerosity`).
- **Paper experiments** (`experiments/`):
  - Numbers: `robustness_study`, `purity_audit`, `scale_study`,
    `quad_study`, `emergent_base10_study`, `compositional_number_study`.
  - Colors: `color_concept_study` (1-D circular domain, §5).
  - Space: `space_concept_study` (2-D lattice, §6.2).
  - Phonemes: `phoneme_concept_study` (discrete categorical, §6.3).
  - Causal: `counterfactual_swap_study` (Appendix B).
  - Reproduction: `train_ans` (regenerate encoder), `_graph_builder`
    (shared helper).
- **Figure renderer** (`experiments/render_paper_figures.py`):
  one-shot script regenerating F2, F4, F5, F6, F7, F8 as PDF + PNG.
- **Paper** (`PAPER.md`, 1 060 lines): abstract, §1 intro, §2 related
  work, §3 method (incl. formalisation), §4–§6 experiments (four
  domains), §7 base-10 null, §8 discussion (four-domain H5″ schema),
  §9 limitations, §10 conclusion, §11 reproducibility, §12 figure
  list, Appendix A raw data, Appendix B causal bundle swap, 27
  references.
- **Study writeups** (`docs/`, 13 markdown files): architectural
  design docs (`PARAMETRIC_CONCEPT_MEMORY.md`,
  `CONTEXTUAL_CONCEPT_COLLAPSE.md`) and per-study writeups for all
  experiments.
- **Figures** (`docs/figures/`): F2, F4, F5, F6, F7, F8 as both PDF
  and PNG at 300 dpi.
- **Pre-trained artefacts** (`outputs/ans_encoder/final.pt`, ≈ 108
  KB): ANS `NumerosityEncoder` used by all numerical experiments.
- **Unit tests** (`tests/test_smoke.py`): 5 smoke tests covering
  public-API import, collapse attribution, consumer registry,
  bundle-leaf parameter status, and gradient flow into bundles.
- **GitHub Actions CI** (`.github/workflows/ci.yml`): matrix over
  Python 3.10 / 3.11 / 3.12 running unit tests + phoneme-domain
  smoke on every push / PR.
- **Project metadata**: `pyproject.toml`, `requirements.txt`,
  `LICENSE` (MIT), `CITATION.cff`, `README.md` with badges.

### Headline empirical findings
- **Geometry emergence universality** across four qualitatively
  different topologies (linear · circular · 2-D lattice · discrete
  categorical) with zero domain-specific architectural change —
  see F4.
- **Facet-algebraic compatibility (H5″) governs cross-muscle
  alignment** across all four predicted quadrants of a 2 × 2 schema
  (same-algebra align, incompatible-algebra / orthogonal null) — see
  F7.
- **Causal bundle identity** demonstrated by a post-hoc swap: single-
  facet swap produces textbook double dissociation (numbers
  100 → 18.2 %, colors 100 → 5.3 %) with zero seed variance — see
  F6.
- **Algorithmic emergence boundary**: pure base-10 factorisation does
  not emerge from arithmetic signal on a flat bundle — see F5.

### How to cite
See `CITATION.cff`. BibTeX block lives in `README.md`.

### Planned for v0.2 (non-binding)
- Close the loop between Pipeline A (concept discovery from
  `grounding.py`) and Pipeline B (concept representation, this
  release) — end-to-end joint training.
- Loss-plateau-driven facet capacity growth ("bundle regrowth"
  protocol — §9 in `PAPER.md`).
- A4 "different-domain, same-algebra" experiment filling the last
  quadrant of the H5″ schema.
- Zenodo DOI once a tagged release is published.
