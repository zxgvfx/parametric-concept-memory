# Parametric Concept Memory (PCM)

> *Concepts Collapse into Muscles — Domain-Topology-Adaptive Parametric
> Concept Memory.*
> Code and full paper for the four-domain PCM study.

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

Full paper in [`PAPER.md`](./PAPER.md). Six publication-quality
figures in [`docs/figures/`](./docs/figures/).

![Four-domain universality of bundle geometry](./docs/figures/F4_four_domain_universality.png)

## TL;DR of the claims

| # | Claim | Evidence |
|---|---|---|
| 1 | Attribution is a dict lookup, not an inference | H1–H4 all pass at 100 % on a 7-numerosity toy (`experiments/purity_audit.py`). |
| 2 | Bundle geometry tracks task topology across four qualitatively different domains | F4: ρ = 0.991 (numbers) · ρ_circ = 0.977 (colors) · ρ_L1 = 0.860 / Procrustes disp 0.07 (space) · intra-inter cos gap +1.2 to +2.0 (phonemes). |
| 3 | Cross-muscle alignment is gated by **facet-level algebraic compatibility** (H5″), not by shared domain or task family | F7: 2×2 schema fully populated — same-algebra pairs align (p = 0.003, 0.016); same-domain incompatible-algebra pair is null (p = 0.77); orthogonal categorical axes are null (p ≥ 0.04). |
| 4 | Bundle = concept semantic identity (causal, not correlational) | F6: post-hoc swap of two trained bundles on one facet causes targeted double dissociation with zero seed variance. |
| 5 | PCM gives *geometric* emergence, not *algorithmic* — pure base-10 factorisation does not emerge | F5: spike₁₀ ≈ 0.001, p = 0.44; hand-coded positional priors recover 100 % extrapolation with ~10³ fewer samples than Abacus. |

## Repo layout

```
pcm/                      core framework
├── concept_graph.py      ConceptGraph + ConceptNode
├── param_bundle.py       ParamBundle (nn.ParameterDict) + ContextualizedConcept
└── heads/                task-specific muscles
    ├── arithmetic_head_v2.py
    ├── comparison_head.py
    ├── numerosity_encoder.py      (+ DatasetConfig, generate_dot_canvas)
    ├── numerosity_classifier.py
    └── arithmetic_head.py         (legacy, for older experiments)

experiments/              paper replications (run with `python -m experiments.<name>`)
├── _graph_builder.py              shared ANS graph builder
├── train_ans.py                   regenerate NumerosityEncoder checkpoint
├── robustness_study.py            numbers §4
├── purity_audit.py                numbers §4.4
├── scale_study.py                 numbers §4.4
├── quad_study.py                  numbers §4.5 (four-operation)
├── emergent_base10_study.py       numbers §7 (negative)
├── compositional_number_study.py  numbers §4.5 (slot + carry)
├── color_concept_study.py         colors §5
├── space_concept_study.py         space §6.2
├── phoneme_concept_study.py       phonemes §6.3
├── counterfactual_swap_study.py   causal swap, Appendix B
└── render_paper_figures.py        regenerate F2-F8 (PDF + PNG)

docs/                     per-study writeups + architectural design docs
└── figures/              F2-F8 publication-quality figures

outputs/
└── ans_encoder/final.pt  shipped pre-trained ANS encoder (≈ 108 KB)

PAPER.md                  full paper (abstract, method, 4 experiments, appendices)
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

### Regenerating the figures

Figures F2, F4, F5, F6, F7, F8 are all auto-generated by a single
entry point (≈ 3 min — retrains one representative seed per domain
for bundle access, reads summary JSONs for the rest):

```bash
python -m experiments.render_paper_figures
# or a subset:
python -m experiments.render_paper_figures --only F4 F7
```

### Regenerating the shipped ANS encoder

The ≈ 108 KB `outputs/ans_encoder/final.pt` is shipped with the
repo for turn-key reproduction. If you want to retrain from
scratch:

```bash
python -m experiments.train_ans --epochs 30 --out outputs/ans_encoder
```

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
@misc{pcm2026,
  title        = {Concepts Collapse into Muscles: Domain-Topology-Adaptive
                  Parametric Concept Memory},
  author       = {{Percept PCM authors}},
  year         = {2026},
  howpublished = {\url{https://github.com/zxgvfx/parametric-concept-memory}},
  note         = {Framework + 4-domain empirical study; MIT licensed},
}
```

## License

MIT — see [`LICENSE`](./LICENSE).
