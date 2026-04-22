# Workshop Paper Outline — *Concepts Collapse into Muscles: A Parametric Memory that Forces Cause-and-Effect Attribution*

**Target venue (candidates)**
- NeurIPS 2026 Workshop — **UniReps** (Unifying Representations in Neural Models) · **主攻**
- NeurIPS 2026 Workshop — **Symmetry & Geometry in Neural Representations** · 备选
- ICML 2026 Workshop — **Mechanistic Interpretability** · 备选
- Cognitive Science Society 2026 — *Conceptual Knowledge & Representation* · 扩刊版

**字数预算**: 4 页主文 + 无限 appendix (workshop 标准)
**作者定位**: 1 篇实验扎实、但命题野心克制的 workshop paper; 主 narrative 用**负结果 → 精修命题**的三层证据链

---

## 0 · Hook (abstract 第一句 + intro 第一段要传达的 one-liner)

> When a neural network's parameters are attached to symbolic concepts rather than to muscle modules, a concept stops being "a vector" and becomes **what it is used for** — a Wittgensteinian claim we can now falsify experimentally. We show that this reparameterisation (i) yields 100% attribution soundness, (ii) makes concept-level ablation causal rather than correlational, and (iii) produces cross-facet geometric alignment **if and only if** the consuming muscles carry compatible algebraic structure — a condition we identify empirically by introducing a non-algebraic classifier as a negative control.

主卖点两条, 必须在 abstract、intro 末、conclusion 都重复出现:
1. **架构上的归因可证 (H1-H4)**: 参数位置决定了干预语义, 把概念消费图变成 testable 对象
2. **对齐条件的发现 (H5'')**: 跨肌肉 facet 几何对齐不是自动的, 它需要**任务代数结构兼容**

---

## 1 · Title & Abstract

**候选标题**:
- ✨ **"Concepts Collapse into Muscles: Parametric Concept Memory with Task-Conditional Alignment"**
- **"From Correlation to Causation in Concept Representations via Muscle-Consumed Parameters"**
- **"What is a Concept? A Parametric Bundle Consumed by Muscles, If and Only If the Task Has Algebraic Structure"**

首选: **"Concepts Collapse into Muscles"** — 短、具象、与 Wittgenstein / quantum cognition 呼应.

**Abstract (150 词模板)**:

> Interpretability work routinely asks "which neurons represent concept X?". We argue this question is ill-posed until parameters are explicitly *attached* to concepts rather than to layers. We introduce **Parametric Concept Memory (PCM)**: every ConceptNode in a symbolic graph owns a multi-facet parameter bundle, consumed on demand by task-specific "muscle" modules through an operation we call **contextual collapse**. In a 7-numerosity toy (concepts 1-7, three muscles: arithmetic, ordinal comparison, classification), PCM delivers 100% attribution soundness — *ablating concept $c$'s bias destroys only tasks involving $c$, leaving every other input untouched* — and 100% attribution completeness. More surprisingly, we find that **cross-facet geometric alignment is not automatic**: the two algebraically-structured muscles (arithmetic, ordinal) produce facets whose cosine matrices align at Spearman $\rho = 0.803$, whereas the classifier's facet is **random with respect to numerical order** ($\rho = 0.088$) and un-aligned with the other two ($\rho \in [-0.07, 0.05]$). A single-muscle control further rules out "multi-muscle causes coherence", establishing that it is the **algebraic signature of the task**, not the plurality of muscles, that shapes the concept.

---

## 2 · Introduction (≈ ⅓ 页)

**Three-beat structure**:

1. **The interpretability gap**: Why "representation X lives in layer Y" is not causal. Ablation studies conflate layer-level and concept-level hypotheses; probing gives correlations; circuit work is heroic but not reproducible. What's missing is a **data structure** that makes *which parameter belongs to which concept* a first-class fact.
2. **Our claim**: If a concept's parameters are bundled in its symbolic node and consumed by muscles, two things follow for free: (a) attribution is mechanical (read the consumption registry), (b) concepts have no resting state, only "use-states" — making concepts empirically more like verbs than like vectors (Wittgenstein §43, Zuhandenheit, quantum cognition's observable-dependent collapse).
3. **What we actually show** (tight contract):
   - **C1 (architecture)**: PCM + collapse implements (a,b) in standard PyTorch; no privileged ops.
   - **C2 (positive)**: H1-H4 hold on 7-numerosity — attribution is sound *and* complete; void concepts are behaviourally nil.
   - **C3 (negative → precise)**: Original H5 ("coherence from plurality") is falsified by single-muscle control; the correct claim H5'' ("coherence from task-algebra, aligned across muscles iff their algebras are compatible") is supported by a **triple-muscle experiment with a non-algebraic classifier as negative control**.

---

## 3 · Related Work (≈ ¼ 页, 6 bullets max)

Structure: we sit at the intersection of four threads, none of which hybridises symbolic + parametric the way we do.

1. **Neuro-symbolic & concept bottleneck**: CBM (Koh et al. 2020), CEMs (Zarlenga et al. 2022) — predefined supervised concepts, no bundle ownership.
2. **Hypernetworks & fast weights**: Ha et al. 2017; Schmidhuber 1992 — generate params from embeddings; we *store* params on nodes (inverted dataflow).
3. **Differentiable Neural Dictionaries / External Memory**: Pritzel et al. 2017, Graves et al. 2016 — key-value memory but content is feature vectors not task-ready parameters; no consumer registry.
4. **Mechanistic interpretability**: Anthropic 2024 "Scaling monosemanticity", Elhage et al. 2022 — features are *discovered* in residual streams; we *decree* them and test falsifiability.
5. **Cognitive theories of concept-as-use**: Wittgenstein 1953, Barsalou 1999 situated cognition, Quantum Cognition (Busemeyer & Bruza 2012) — formalising the "concept has no resting state" claim.
6. **Multi-task shared representations**: Caruana 1997, Maurer 2016 MTL bounds — closest empirical sibling, but their shared backbone lacks a notion of concept-level attribution.

---

## 4 · Method (≈ ½ 页)

### 4.1 ParamBundle
Each `ConceptNode` owns a `ParamBundle` = `nn.ParameterDict[facet → Parameter]`, initialised lazily on first request with a specified shape.

### 4.2 Contextual Collapse
```
cc = node.collapse(caller="AddHead", facet="arithmetic_bias", shape=(64,), tick=t)
# → ContextualizedConcept(concept_id, caller, facet, facet_params)
```
Side effects (invariants):
- `bundle.consumed_by[facet] ← {…, caller}` (D91 attribution)
- `bundle.collapse_history[facet].append((caller, t))` (D92 use-history)

Only the returned handle is differentiable; between collapses a concept has **no state**.

### 4.3 Muscle contract
Every muscle's forward pass calls `collapse()` for the concept ids it needs; its own backbone **must not carry concept-conditional information** (we enforce this by passing zero embeddings in control experiments, see §5.4). Loss flows through `cc.facet_params` back into the bundle — training routes concept-specific gradient exclusively through the graph.

### 4.4 Acceptance tests (preregistered)
| Hyp. | Operationalisation | Threshold |
|---|---|---|
| H1 sound | ablate `facet` at node $c$ ⇒ accuracy drop on tasks involving $c$ | ≥ 50 pp |
| H2 complete | ablate `facet` at node $c$ ⇒ no drop on tasks not involving $c$ | ≤ 2 pp |
| H3 orthogonal | ablate facet $f_A$ ⇒ no drop on muscle that uses only $f_B$ | ≤ 2 pp |
| H4 void-nil | request for `VOID_CONCEPT` ⇒ chance-level accuracy | ≤ 60% (binary) |
| H5'' algebra-conditional | pairwise Spearman ρ of cos-matrices ≥ 0.7 between *algebraic* facets; random between non-algebraic and algebraic | see §6.3 |

---

## 5 · Experimental Setup (≈ ¼ 页)

### 5.1 Domain
7-numerosity toy: concepts `concept:ans:1` … `concept:ans:7`, plus `concept:void` (never consumed, negative control for H4).

### 5.2 Three muscles
| Muscle | Facet | Shape | Task | Label-generating rule |
|---|---|---|---|---|
| **ArithmeticHeadV2** | `arithmetic_bias` | (64,) | $\{+,-\}$ on two numbers | $a \pm b \in [n_\min, n_\max]$ only |
| **ComparisonHead** | `ordinal_offset` | (8,) | $a <,=,> b$ | ternary label |
| **NumerosityClassifier** | `identity_prototype` | (16,) | 1-of-7 classification | $\text{argmax}_k = N_c - n_\min$ |

Critical design choice: **muscle backbones ignore embeddings** (fed with `torch.zeros`), forcing 100% of concept-specific signal through the bundle.

### 5.3 Training
- AdamW, lr $3\!\times\!10^{-3}$ for heads, $10^{-2}$ for bundles; 3000 steps per condition; single GPU, ≈4 min total.
- Three conditions: **single-muscle (ADD only)**, **dual (ADD+CMP)**, **triple (ADD+CMP+CLS)**.

### 5.4 Evaluation artefacts
- `attribution_report.json` — per-concept `consumed_by` + `collapse_history`
- `eval_collapse.py` — runs H1-H4 cross-ablation matrix
- `eval_identity_coherence.py` — computes cos-matrix per facet, pairwise Spearman ρ, and ρ vs. $-|\Delta n|$

---

## 6 · Results

### 6.1 C1 · Attribution is mechanical (Table 1)

| Metric | Value |
|---|---|
| H1 soundness | **100%** (7/7 concepts) |
| H2 completeness | **100%** (all unrelated concepts unaffected) |
| H3 facet-orthogonal | **100%** (ablating `arithmetic_bias` leaves comparison untouched, and vice versa) |
| H4 void-nil | chance on binary cmp; 0/1 on add |

> *Take-away*: Once you read `bundle.consumed_by`, the "which parameters represent concept 3?" question collapses from a research project to a dict lookup.

### 6.2 C2 · The single-muscle falsification (Figure 2)

| Condition | `ρ(arithmetic_bias, -|Δn|)` |
|---|---|
| Single-muscle (ADD only) | **0.915** |
| Dual-muscle (ADD + CMP)  | 0.892 |
| $\Delta\rho$ | **−0.023** (within noise) |

*Claim refuted*: "coherence emerges from multi-muscle coupling".
*Replacement claim*: coherence inside a facet is a side-effect of the task's algebraic demand; it **appears even with a single consumer**.

### 6.3 C3 · Cross-facet alignment is task-algebra-conditional (Figure 3, our money figure)

Triple-muscle condition; Spearman $\rho$ of pairwise-cosine matrix entries:

|  | arith ↔ ordinal | arith ↔ id-proto | ordinal ↔ id-proto |
|---|---|---|---|
| $\rho$ | **0.803** | 0.053 | −0.073 |

And `ρ(facet, -|Δn|)`:

| facet | ρ |
|---|---|
| arithmetic_bias | 0.918 ✓ |
| ordinal_offset  | 0.878 ✓ |
| **identity_prototype** | **0.088** ✗ |

> Two algebraic muscles share the same condensation of identity; the 1-of-K classifier — whose loss has permutation symmetry over labels — does **not** participate in that condensation. **Cross-facet alignment requires compatible algebraic structure among the consumers.** This is H5''.

The classifier is our **negative control**: without it, one might attribute alignment to "any two muscles working on the same 7 numbers". With it, alignment is shown to depend on *what the muscles compute*, not on how many there are.

### 6.4 Bonus · Counterfactual swap (appendix, optional 实验 — 已 marked pending)

Swap the trained `arithmetic_bias` of `concept:ans:3` ↔ `concept:ans:5`, re-run all tests:
- Accuracy on problems involving $3$ or $5$ collapses (expected)
- Cross-facet alignment index drops by ≥ 0.3 (to-be-measured)

This is the cleanest causal test; do it before camera-ready.

---

## 7 · Discussion (≈ ¼ 页)

Three beats:

1. **What kind of object is a concept, according to these experiments?** An **indexed parameter bundle** whose semantic content is indistinguishable from its consumption history. Resting concepts *do not exist* — liveness $L(v) = 0$ is behaviourally nil (H4). This is a mechanistic version of Wittgenstein's §43 and of Heidegger's Zuhandenheit.
2. **Why H5'' matters for multi-task learning**: MTL literature usually treats shared representations as a bandwidth tradeoff. We show a *structural* condition: shared-representation alignment is **not automatic** and not determined by sharing-the-input; it is gated by task-algebra compatibility. This gives an explanation for why some auxiliary tasks help and others hurt.
3. **For interpretability**: attribution soundness/completeness is not a virtue the model achieves; it is a **consequence of parameter location**. Changing where a parameter lives changes the space of interventions that have meaning.

---

## 8 · Limitations / Honest Scope (≈ ⅛ 页 — put before conclusion)

- **Toy domain** (7 concepts, atomic numerosity). We do not yet show PCM scales to visual or language domains; cg + bundles for 10⁶ concepts is future work.
- **Ground-truth concept IDs are given**, not discovered. Coupling PCM with unsupervised concept discovery (D88 prelinguistic grounding in the wider project) is left to follow-ups.
- **H5'' is verified for two specific task algebras** (ring-like for +/−, total-order for comparison); we conjecture but do not prove that the "compatible-algebra" condition is necessary in general.
- **No lesion-recovery**: we ablate but do not retrain to see which concepts are most distillable; the dual of H2 should be tested.

---

## 9 · Conclusion (≈ ⅛ 页)

Two one-liners, both already seeded in abstract:
- Put the parameters on the concepts and attribution is free.
- Alignment among shared representations is *not* free; it is the signature of the shared algebra.

---

## 10 · Reproducibility statement (workshop requires)

All code and artefacts in `mind/experiments/language/` and `mind/core/cognition/language/`; full training of all three conditions < 5 minutes on a single RTX 4090; concrete commands:

```
python -m experiments.train_single_muscle
python -m experiments.train_dual_muscle
python -m experiments.train_triple_muscle
python -m experiments.eval_collapse
python -m experiments.eval_coherence_compare
python -m experiments.eval_identity_coherence
```

Seeds (42), hyperparams, and the full ablation matrix are shipped as JSON artefacts next to the checkpoints. The `ConceptGraph Viewer` (`mind.diagnostics.concept_graph_viewer`) renders `bundle.describe()` and the alignment sidecar without additional dependencies.

---

## 11 · Figures (to be rendered before submission)

| # | What | Source |
|---|---|---|
| **F1** | Architecture diagram: `muscle.forward(cid) → node.collapse(caller, facet) → ParamBundle` with consumer registry + collapse history | new TikZ, hand-drawn OK for workshop |
| **F2** | Bar plot: ρ under single / dual / triple muscle (arithmetic_bias); highlight Δρ₍single−dual₎ = −0.023 | `eval_coherence_compare.json` |
| **F3** | **Money plot**: three 7×7 cos heatmaps side by side (arith / ordinal / id-proto) with Spearman ρ between pairs annotated | `identity_coherence.json`; we already render this in the viewer |
| **F4** | Attribution matrix (concept × muscle heatmap of ρ-drop under ablation, corner saturated) | `eval_collapse.json` |

---

## 12 · Writing plan / timeline

| 周 | 交付物 | 负责 |
|---|---|---|
| W1 | intro + method 草稿 (2 pp), F1 draft | 作者 |
| W1 | 写 counterfactual-swap 实验 + run (§6.4) | 作者 (≈ 4h) |
| W2 | results + figures 全做出, appendix 落盘 | 作者 |
| W2 | discussion + related work; 找 2 个外部人 review | 作者 |
| W3 | 改到 camera-ready quality; submission check | 作者 |

**Hard no-go**: 在 H5'' 之上过度推广 (比如声称在视觉/语言域成立) — 坚决不做, 留给后续 full paper.

---

## 13 · Open decisions (需作者 / 合作者拍板)

- [ ] Workshop final pick: UniReps vs SGGeneRal vs MI — 取决于 CFP 最新 deadline
- [ ] 是否跑 §6.4 counterfactual-swap (强烈建议跑, ≤ 半天)
- [ ] 加不加 language/vision domain 的 toy 扩展 (如果 workshop 愿意收 short paper 可以只做上面的; 如果投 full paper 就必须加)
- [ ] Fig 3 要不要配一版 signed-cosine 而不是 signed-Spearman (补充实验)

---

## 14 · Connection to Percept long-term roadmap

本论文对应 Percept 架构决策 **D91 (Parametric Concept Memory) + D92 (Contextual Concept Collapse)** 的首次完整外部化表达. 对 roadmap 的贡献:

1. 为未来的 **D93 (Reconstructive Invertibility)** 提供 forward-projection 端的完整证据: 如果 ConceptGraph + bundle + muscles 能 forward-compose 出任何当前语义状态, 那么**反向重建 (perceive → reconstruct)** 也是同一图谱 / 同一参数的逆用 — 这是下一篇 paper 的自然延伸.
2. 本工作的归因可证性 (H1-H4) 直接支撑 `docs/language/PHILOSOPHY.md §5 5-step provenance` 的"每一个 concept 都可被解释"的红线.
3. Viewer + 对齐热图是 **AGI-4 mechanistic debugging** 这条长期线的原型.
