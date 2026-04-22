# Changelog

All notable changes to **Parametric Concept Memory (PCM)** are recorded
here. This project follows [Semantic Versioning](https://semver.org/)
and the [Keep a Changelog](https://keepachangelog.com/) conventions.

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
  `generate_dot_canvas`, `encode_numerosity`), plus legacy
  `ArithmeticHead` for older experiments.
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
