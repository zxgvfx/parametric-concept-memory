# Tier-D Decision Record · 节点即可执行函数 (2026-05-10)

> **Decision**: Tier-D (Houdini-style cook) **LANDED**. Tier-E (spatial
> coordinates) and Tier-F (vmap cooking) are **deferred to future work**.

This document closes out the
["PCM 节点即可执行函数 · 仿生空间排列升级" plan](../../../c%3A/Users/zxgvfx/.cursor/plans/pcm_%E8%8A%82%E7%82%B9%E5%8D%B3%E5%8F%AF%E6%89%A7%E8%A1%8C%E5%87%BD%E6%95%B0%E5%8D%87%E7%BA%A7_761f2f68.plan.md)
by recording the empirical evidence and the decision criterion the
plan's §6 step 3 asked for.

---

## 1. Tier-D landed — empirical evidence

| Check | Result | Source |
|---|---|---|
| Unit tests (all tiers) | **39/39 PASS** | `python -m unittest discover tests` |
| New cook unit tests | **9/9 PASS** | [tests/test_cook_subgraph.py](../tests/test_cook_subgraph.py) |
| Tier-A regression (smoke + grow + scale + swap) | **4/4 PASS** | [scripts/sanity_check.py](../scripts/sanity_check.py) |
| Bit-identity D1 (forward, B=3) | `torch.equal == True` | `test_d1_bit_identical_forward` |
| Cook gradient flows to pool + backbone | non-zero | `test_cook_grad_flows_to_pool_and_backbone` |
| Subgraph error surface (5 paths) | all raise `SubgraphEvalError` | `TestEvaluatorErrorSurface` |
| §4 N=7 paper run (cook vs direct) | `delta_acc=0.0000`, `delta_rho=0.0000` | [experiments/cook_demo.py](../experiments/cook_demo.py) |
| §B counterfactual swap | `add_inv=18.18%`, `mix_inv=5.26%` (paper: 18.2% / 5.3%) | sanity_check |

The D1 bit-identity claim is met **exactly** (delta = 0.0 at 12 epochs ×
120 steps), confirming that Tier-D is a bit-for-bit monotonic extension
of Tier-A.

## 2. Why Tier-E / Tier-F are deferred

The original plan §4 sketched two new tiers on top of Tier-D:

- **Tier-E (spatial coordinates)** — every ConceptNode gets a learnable
  `spatial_pos: Tensor(D_pos,)` plus a `graph.spatial_query` op so cook
  subgraphs can pull weighted neighbour bundles.
- **Tier-F (vmap over cooking)** — `eval_batch(node, bindings_batch:
  Tensor(B, ...))` so the same cook node runs B different parameter
  sets in parallel.

Both are sound. We defer for these specific reasons:

1. **They change paper claims, not just code**. Tier-D ships as a
   monotonic extension (D1 bit-identity), so all existing §4 / §5 / §6
   / §B numbers stay valid. Tier-E introduces a *new* dynamics (spatial
   loss term, neighbour weighting) that requires re-running every
   four-domain experiment — and either matching or improving on the
   current ρ values to be a publishable claim. Tier-F similarly opens a
   new optimisation regime (vmap over `(B, params)`) whose convergence
   behaviour is not characterised in any current §4-§7 result.
2. **They warrant a separate paper**. The Tier-A/B/C bionic tier already
   gives the v2 paper 5 new claims (G1-G6 invariants, gate trajectory
   ↔ developmental pruning, PEER discovery). Adding Tier-E + F would
   crowd the narrative. We suggest a sequel paper *PCM-Cook* which can
   cite this Tier-D landing as foundation and develop Tier-E + F
   end-to-end with their own §4-§6.
3. **The pcm-agent project is the natural sandbox**. The sister project
   already runs the full Houdini paradigm (`muscle_subgraph` +
   `GraphEvaluator` + `(defprimitive)` runtime DSL extension) on the
   ARC environment. New tiers should be prototyped there first; if a
   prototype yields a clean falsifiable claim, it can be ported back
   into PCM as Tier-E / F under a paper-quality regression.

## 3. Reopening criteria

Tier-E / Tier-F implementation should be reopened only when **at least
one** of the following is true:

- A pcm-agent experiment produces a falsifiable claim that requires
  PCM's bundle-pool plumbing (e.g. spatial-pos-only muscle achieves
  ρ_L1 ≥ 0.86 on the §6.2 5×5 lattice without any explicit edges).
- A user explicitly opens scope for a sequel PCM-Cook paper.
- A downstream consumer of PCM (`mind/`, `pcm-agent`) requests cook
  semantics with multi-parameter parallelism in a way that direct
  forward (Tier A) cannot satisfy.

## 4. Code map for Tier-D

| File | Purpose | LoC |
|---|---|---|
| [docs/PCM_NODE_AS_FUNCTION_DESIGN.md](PCM_NODE_AS_FUNCTION_DESIGN.md) | architecture spec, equivalence proof, 命题 1 reformulation | — |
| [pcm/dna_ops.py](../pcm/dna_ops.py) | minimal op vocabulary (8 ops, 3 op_kinds) | 195 |
| [pcm/graph_eval.py](../pcm/graph_eval.py) | cook interpreter (one-pass walker, cycle guard) | 257 |
| [pcm/concept_graph.py](../pcm/concept_graph.py) | `kind` / `metadata` fields + `register_muscle_subgraph` helper | +56 |
| [pcm/heads/cook_factory.py](../pcm/heads/cook_factory.py) | generic `MLPBackbone` / `HeadAsBackbone` / `make_cook_subgraph` (used by every head) | 204 |
| [pcm/heads/cook_wrappers.py](../pcm/heads/cook_wrappers.py) | per-head `build_*_cook` (number domain) | 162 |
| [experiments/{color,space,phoneme}_concept_study/cook.py](../experiments/) | per-head `build_*_cook` (color · space · phoneme domains) | ~60 each |
| [tests/test_cook_subgraph.py](../tests/test_cook_subgraph.py) | evaluator error-surface regressions | 109 |
| [tests/test_cook_all_heads.py](../tests/test_cook_all_heads.py) | D1 bit-identity for all 8 muscle heads (number · color · space · phoneme) | 259 |
| [experiments/cook_demo.py](../experiments/cook_demo.py) | §4 N=7 cook-vs-direct regression (single-head path) | 254 |
| [experiments/cook_four_domain/](../experiments/cook_four_domain/) | end-to-end direct-vs-cook training across all 4 paper domains | (package) |

## 5. One-line summary

> Tier-D is in. Tier-E / F are sound but stay out of this PR until the
> sequel paper opens.
