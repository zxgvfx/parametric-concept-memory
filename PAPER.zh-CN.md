# 概念坍缩为肌肉：领域拓扑自适应的参数概念记忆

**匿名作者**
*2026-04-22* — 为双盲评审保留匿名。
非匿名公开 artifact 仓库位于 `github.com/zxgvfx/parametric-concept-memory`。

**Artifact**：`pcm/`（框架），`experiments/`（复现实验）。
全部实验可在单张 RTX 4090 上约 25 分钟内复现；除 *N* = 100 的数值实验外，在 CPU 上也可于 1 小时内完成。

![四领域 bundle 几何普适性](./docs/figures/F4_four_domain_universality.png)

*图 1*（核心图；另见 §6 / F4 与 §8 / F7）：同一个 PCM 框架，在没有领域专用架构改动的情况下，诱导出与各领域拓扑一致的 bundle 几何。左上：线性数轴（ρ = 0.991）。右上：圆形色相环（ρ_circular = 0.977）。左下：2-D 空间格点（Procrustes 对齐后的 MDS，disparity = 0.07）。右下：20 个音素的余弦热图，出现按发音方式组织的块状结构。

---

## 摘要

可解释性研究经常问："哪些神经元表示概念 *X*？" 我们认为，在参数被明确地附着到概念而不是附着到层之前，这个问题本身就没有被良好定义。本文提出 **Parametric Concept Memory（PCM，参数概念记忆）**：符号图中的每个 `ConceptNode` 都拥有一个多 facet 参数 bundle，任务专用的 "muscle" 模块通过我们称为 **contextual collapse（上下文坍缩）** 的操作按需消费这些参数。除了由结构本身保证的架构级归因外，我们在四个领域上建立了三个经验发现；这些领域覆盖线性、圆形、2-D 格点和离散类别拓扑：

1. **四领域几何涌现的普适性**。1–30 上的单个算术 muscle 诱导出 **线性数轴**（ρ = 0.991，*N* = 30）；12 个色相上的单个混色 muscle 诱导出 **圆形色相环**（ρ_circular = 0.977，径向残差 ≤ 11%，5/5 seeds 均为循环序）；5×5 网格上的两个 muscle 诱导出 **2-D 格点**（ρ_L1 = 0.860，MDS Procrustes disparity 0.07，3/3 seeds）；20 个音素上的三个 muscle 沿类别轴诱导出 **原型簇**（类内外余弦差 +1.2 到 +2.0，3/3 seeds）。同一个框架和同一套代码会自动适配任务拓扑。
2. **Facet 代数决定跨 muscle 对齐（H5″）**。朴素假设 "多个 muscle 会导致 coherence"（H5）被明确反驳（Welch *p* = 0.93，*N* = 10 seeds）。修正后的命题是：跨 facet 对齐反映 *facet 级代数兼容性*。该命题填满了 2 × 2 预测表的四个格子：相同代数 ⇒ 强对齐（数字 ρ = 0.91 ± 0.04，permutation *p* = 0.003；颜色 ρ = 0.38 ± 0.29，permutation *p* = 0.016）；同一领域中的向量 vs 标量不匹配 ⇒ null（空间 ρ = −0.03，*p* = 0.77）；正交类别轴 ⇒ null 到残余（音素 |ρ| ≤ 0.12，*p* ≥ 0.04）。
3. **纯 base-10 涌现失败**。如果没有 slot / carry 先验，平坦 PCM 不能仅从算术信号中诱导出数字位分解（spike₁₀ ≈ 0.001，*p* = 0.44）。这给出一个清晰边界：PCM 给出的是 *几何* 涌现，不是 *算法* 涌现。

反事实 shuffle 会使所有正向几何指标崩塌（圆形 |ρ| 0.97 → 0.09，线性 0.97 → 0.18），说明几何依赖于概念身份，而不是训练 artifact。训练后的 **bundle swap**（附录 B）进一步给出因果闭环：在两个领域中，只交换两个概念在某个 facet 上训练好的 bundle，只会击中消费该 facet 的 muscle，且只影响涉及这两个概念的 pair（数字：100% → 18.2%；颜色：100% → 5.3%），其他 muscle 和其他概念保持 100%。这是教科书式的 double dissociation，说明 **bundle 是概念语义身份本身，而不只是它的相关物**。

---

## 1 引言

**可解释性缺口。** "表示 *X* 存在于第 *Y* 层" 这类说法并不具备因果性。Ablation 会混淆层级假设和概念级假设；probe 只能给出相关性；circuit 工作虽然艰难而有价值，但难以大规模复现。缺失的是一种 *数据结构*，能让 "哪些参数属于哪个概念" 成为模型的一等事实，而不是事后推理。

**我们的主张。** 如果一个概念的参数被 bundle 在它的符号节点中，并由 muscle 消费，那么两个结论自然成立：

- **(a) 归因是机械的。** 读取 `bundle.consumed_by` 即可；归因问题从一个研究项目退化为一次 dict lookup。
- **(b) 概念没有静止态。** 在 collapse 之间，节点没有 activation；它的语义内容与其消费历史不可区分。这是维特根斯坦 §43 和海德格尔 *Zuhandenheit* 的一种机制化版本。

**四领域概览。** 我们选择四个拓扑彼此不相同的领域来测试框架（图 4）：

| § | 领域 | 拓扑 | muscle | 100% 任务准确率 | 几何指标（seeds） |
|---|---|---|---|---|---|
| 4 | 数字 1–30 | 1-D 线性 | AddHead + CmpHead + IdClassifier | ✓ | ρ = 0.991 (10) |
| 5 | 颜色 12 hues | 1-D 圆形 | MixHead + AdjHead | ✓ | ρ_circular = 0.977，径向残差 ≤ 11% (5) |
| 6.2 | 空间 5×5 | 2-D product-order lattice | MoveHead + DistanceHead | ✓ / 73% | ρ_L1 = 0.860，Procrustes disp 0.07 (3) |
| 6.3 | 音素 20×3 | 离散类别 | VoicingHead + MannerHead + PlaceHead | ✓ | 类内外余弦差 +1.2 到 +2.0 (3) |

同一套框架代码 —— `ConceptGraph + ParamBundle + collapse`，每个领域一个 `train_one` 风格循环 —— 产生了各自的几何。没有领域专用的架构改动。

**贡献。** (1) 我们形式化 **Parametric Concept Memory**（PCM，§3），包括归因契约和 muscle API（§3.3）。(2) 在一个 7-numerosity toy 上，我们经验性证明该契约（H1–H4 全部 100%，§4.2）。(3) 在四个领域中，我们显示 PCM 诱导出的几何 **跟随任务拓扑，而非监督几何**；每个案例中的监督都是 random-orthogonal 或 one-hot（§4–§6）。(4) 我们明确反驳朴素多 muscle coherence 假设（H5，Welch *p* = 0.93），并以 **H5″** 替代：facet 级代数兼容性是跨 facet 对齐的预测因子；该预测在 2 × 2 schema 的所有已测试格子中无一失败（§6.4，F7）。(5) 我们指出 PCM 涌现失败的一条清晰边界：纯 base-10 factorisation 不会从平坦 bundle 上的算术信号中自然出现（§7）。(6) 训练后 bundle swap（附录 B，F6）在数字和颜色领域中产生教科书式 double dissociation，给出因果而非相关证据，说明 bundle **就是** 概念语义身份。

---

## 2 相关工作

我们将 PCM 与六条研究线对比；后面三条与本文发现最直接相关。

**概念瓶颈模型（CBM, CEM）。** Koh et al. (2020) 先预测固定的人类标注概念词表，再通过监督 bottleneck 消费它；Zarlenga et al. (2022) 将其放宽为 *concept embeddings*，但它们仍然绑定到下游层中的逐样本 dense activation。二者都把概念视为前向过程中网络携带的 **activation 空间中的位置**。PCM 在三点上不同：(i) 概念是一等 *图节点*，不是 activation；(ii) 概念 **拥有** 参数（每个节点一个 `ParamBundle`），而不是作为标注输出；(iii) 归因是 `bundle.consumed_by` 上的 dict lookup，不是 probe 或 saliency 分析。CBM/CEM 每次 forward 都需要从 activation 中重新识别 concept 3 的贡献；PCM 从设计上直接暴露它。

**Hypernetworks 与 fast weights。** Hypernet 从上下文 / 条件信号生成目标权重。在 PCM 中，我们反转数据流：bundle *存储* 权重，muscle 在使用时 *读取* 它们，中间没有 generator MLP。这使得归因注册表（`bundle.consumed_by`）成为图的静态属性，而不是模型级推断。最接近的 hypernet 变体是 condition-conditional adapter，但它们仍提供一个 *key* 来选择权重；我们提供的是拥有这些权重的 *concept id*。

**外部 / episodic memory。** DNC、NEC、kNN-LM 等从无参数 memory 中检索内容（向量或 key-value pair）。PCM 的 memory **就是** 参数：bundle 是 `nn.Parameter` leaf，由标准 SGD 训练，每个 entry 由唯一 concept id 拥有。Ablate 一个 DNC cell 的语义并不明确；ablate PCM 中 `concept:ans:3` 的 bundle 是局部化干预，并有可直接证伪的预测（H1–H4，§4.2）。

**机制可解释性与稀疏自编码器（SAE）。** 这些方法在预训练模型 residual stream 内部事后发现并命名特征，通常依赖强稀疏假设。PCM 的关系是正交的：我们通过图构造 *规定* 哪些特征存在，然后经验性检验：(i) 训练是否在其上发现非平凡几何（§4–§6）；(ii) 该几何是否由 bundle 因果拥有（附录 B swap）。在一个 dual-muscle 数字模型上跑 SAE，预计会恢复与 bundle row 对齐的特征；这是本文未测试但可证伪的预测。

**算术泛化。** Abacus、位置编码方法、NAU/NMU、Neural GPU 等关注原始数值外推，常使用专用位置编码或算术 primitive。本文 §4 研究的是另一个问题：一个 **通用概念记忆** 中的算术信号是否会诱导出可解释的线性 bundle 几何；§7 则建立清晰负结果：同一框架不会在没有位置先验时发现 base-10 分解。这支持 Abacus 的收益主要来自位置方案，而不是规模本身。D93a 显式提供位置先验，并以约少 10³ 倍的数据达到 100% digit-length extrapolation，代价是把 base-10 写进架构。

**多任务 / 表征学习。** 多任务理论通常把共享表征看作带宽权衡，并在相似分布假设下证明泛化收益。我们识别出共享表征的另一个结构条件：*facet 级代数兼容性*；它可以用每个任务对一个 permutation test 来证伪（§6.4，§8）。F7 的四领域 schema 据我们所知是对该条件首个正负并列测试。

**认知科学基础。** 三条线索影响了 PCM 的哲学，但不直接约束实现：grounded cognition / perceptual symbols；mental number line 与 ANS 文献；神经心理学中的 double dissociation 方法。维特根斯坦《哲学研究》§43 —— "一个词的意义就是它的使用" —— 在 PCM 中得到机制性影子：collapse 之间的 bundle 在行为上为 nil（H4），其语义内容与消费历史不可区分（§3.2）。

---

## 3 方法

### 3.1 ParamBundle

每个 `ConceptNode` 拥有一个 `ParamBundle`，即 `nn.ParameterDict`，它从 `facet → Parameter` 映射。Bundle 在第一次请求时按指定 shape 懒初始化，之后作为标准 PyTorch 参数参与训练（AdamW、weight decay、gradient clipping 都不需要改动）。

### 3.2 Contextual Collapse

```python
cc = node.collapse(caller="AddHead", facet="arithmetic_bias",
                   shape=(64,), tick=t, init="normal_small")
# → ContextualizedConcept(concept_id, caller, facet, facet_params)
```

每次调用维护以下不变量：

- `bundle.consumed_by[facet] ← bundle.consumed_by[facet] ∪ {caller}`（D91 归因注册表）
- `bundle.collapse_history[facet].append((caller, t))`（D92 使用历史）

只有返回的 handle 可微；在 `collapse` 之外，概念没有状态。

### 3.3 Muscle 契约

每个 muscle 的 `forward` 都会为所需概念 id 调用 `collapse()`。Muscle 的 backbone **不得** 携带 concept-conditional 信息；我们通过将零 embedding 输入到任何感知输入位置来强制这一点（见 `ArithmeticHeadV2`）。因此，所有 concept-specific gradient 都必须只通过图流动。

### 3.4 形式化

令 **V** 为有限 concept id 集合，**F** 为有限 facet 名集合。图状态为

\[
  \mathcal{G} = \bigl(V,\; \{ \mathbf{B}_v \}_{v \in V} \bigr),\qquad
  \mathbf{B}_v : F \hookrightarrow \bigcup_{d\ge 1} \mathbb{R}^{d},
\]

即从 facet 到 tensor 的偏映射，也就是节点 *v* 的 `ParamBundle`。每次调用
`v.collapse(caller = c, facet = f, shape = s)` 的操作语义为

\[
  \mathbf{B}_v[f] \longleftarrow
    \begin{cases}
      \mathbf{B}_v[f] & \text{if defined} \\
      \mathrm{init}(s) & \text{otherwise (lazy init)}
    \end{cases},
  \quad
  \mathrm{cons}(v, f) \leftarrow \mathrm{cons}(v, f) \cup \{c\},
\]

并返回 **ContextualizedConcept** handle
$(v, c, f, \mathbf{B}_v[f])$。懒初始化与 identity 无关：新建 facet 从 concept-blind 分布中 i.i.d. 抽样（`normal_small`），因此不会注入语义先验。*归因闭包* 是下面事实的内容，也是本文动机所在：

> **命题 1（归因闭包）。** 对每个优化步骤，流入 $\mathbf{B}_v[f]$ 的每个梯度都来自某个经过 collapse $(v, c, f, \cdot)$ 的 muscle $c$ 的 loss term。反过来，如果当前步骤中没有 muscle collapse $(v, f)$，则 $\mathbf{B}_v[f]$ 上的梯度严格为零。

这是实现的性质：bundle 是 leaf `nn.Parameter`；collapse 返回的是同一个参数而不是 copy；muscle 契约禁止 collapse 之外的 concept-conditional 路径（§3.3）。它让我们能从契约本身证明 H1–H3。H4 和 H5 / H5′ / H5″ 则是经验命题。

### 3.5 预注册假设

| ID | 操作化 | 阈值 |
|----|----|----|
| **H1** Soundness | Ablate 节点 *c* 的 facet ⇒ *c*-tasks 准确率下降 | ≥ 50 pp |
| **H2** Completeness | Ablate 节点 *c* 的 facet ⇒ *¬c*-tasks 下降 ≤ 2 pp | ≤ 2 pp |
| **H3** Facet-orthogonal | Ablate facet *f₁* ⇒ 使用 *f₂* 的 muscle 不下降 | ≤ 2 pp |
| **H4** Void-nil | 请求 `VOID_CONCEPT` ⇒ chance accuracy | ≤ 60%（binary） |
| **H5**（朴素）Multi-muscle coherence | dual ρ > single ρ | Δρ ≥ 0.05 |
| **H5′** Cross-facet identity | facet 余弦矩阵之间 Spearman ρ ≥ 0.7 | ≥ 0.7 |
| **H5″** Facet-algebra-conditional | 跨 facet 对齐当且仅当两个消费 muscle 对 facet 施加相同代数要求；同一领域既非必要也非充分 | 见 §6.4, F7 |

**H5″ 的形式表述。** 令 $\mu_A, \mu_B$ 为两个 muscle，$f_A, f_B \in F$ 为其消费的 facet。每个 muscle 定义一个 *代数签名* $\sigma(\mu)$，形如
$(\text{arity}, \text{is-symmetric}, \text{structure})$，其中 "structure" 编码该 loss 对 facet 隐含要求的群 / 有序集 / 划分结构。H5″ 声称：

\[
\operatorname{align}(\mathbf{B}_{\cdot}[f_A],\;\mathbf{B}_{\cdot}[f_B])
\;\;\gg\;\; 0
\quad \Longleftrightarrow \quad
\sigma(\mu_A) = \sigma(\mu_B).
\]

我们在 §6.4 和 F7 中填充预测 2 × 2 真值表的三个格子；第四格（不同领域 + 相同代数）留作未来工作。

### 3.6 本文范围：测试什么，推迟什么

Percept 项目包含 **两条互补 pipeline**：

- **Pipeline A — 概念发现**：感知流被聚类；未匹配 cluster 通过 `register_concept` 自动注册为新的 `concept:hypothesis:{modality}:N` 节点。这实现了项目中 "概念从经验规律中涌现" 的承诺。
- **Pipeline B — 概念表示**（本文 §3.1–§3.3）：给定图中已有节点，bundle/collapse 机制将 per-concept 参数路由到任务专用 muscle。

本文故意绕过 Pipeline A，并通过 `_graph_builder.build_ans_graph` 和 `build_color_graph` 直接提供 ground-truth concept ID。原因是方法论上的：如果同时训练发现和表示，任何负结果或混合结果都可能归因于任一阶段。解耦后，§4–§7 与附录 B 的每个发现都能明确归因于 bundle/collapse 机制。

注意，监督信号（来自 `NumerosityEncoder` 的 ANS centroid，颜色中的 random-orthogonal）仍然来自感知或无语义先验构造；我们只提供 **identity tag**。Bundle 本身从未以任务结构 seeded；这由 `VOID_CONCEPT` control（H4）和 shuffle 反事实（`E2_shuffled`，|ρ| → 0.09–0.18）验证。后续论文将联合训练 Pipeline A 和 B；架构支持已经存在。

---

## 4 实验 1 — 数字（线性领域）

### 4.1 设置

概念：`concept:ans:1` … `concept:ans:N`（以及用于 H4 的 `concept:void`）。Muscle：`ArithmeticHeadV2`（facet `arithmetic_bias`，64-d）、`ComparisonHead`（`ordinal_offset`，8-d）、`NumerosityClassifier`（`identity_prototype`，16-d）。监督：random-orthogonal centroids，因此 **标签中没有 ordinal signal**。基线 *N* = 7；规模实验扩展到 *N* ∈ {15, 30}；四运算实验扩展到 *N* = 100。

### 4.2 架构级归因（H1–H4）

| Metric | Result |
|---|---|
| H1 soundness | **100%**（7/7 概念） |
| H2 completeness | **100%**（所有无关概念不受影响） |
| H3 facet-orthogonal | **100%**（ablate `arithmetic_bias` ⇒ ComparisonHead 不受影响） |
| H4 void-nil | binary cmp 为 chance；add 为 0/1 |

只要查看 registry，"哪些参数表示 concept 3？" 就是一次 dict lookup。

### 4.3 H5 被明确反驳，H5′ 被明确支持

10 个独立 seeds，每个 seed 分别训练 single-muscle 和 dual-muscle 条件。

| Condition | ρ(`arithmetic_bias`, −\|Δn\|) mean ± std | *n* |
|---|---|---|
| Single (AddHead only) | **0.973 ± 0.007** | 10 |
| Dual (Add + Cmp) | **0.973 ± 0.004** | 10 |
| Welch *t*-test | *t* = −0.089，**p = 0.930** | — |
| Dual extra `ordinal_offset` ρ | 0.922 ± 0.017 | 10 |
| Dual cross-facet align (arith ↔ ord) | **0.907 ± 0.037** | 10 |

H5 死亡：`arithmetic_bias` 的 coherence 是 **任务内生的**，不是 multi-muscle 效应。封闭加法任务是一个代数约束：`MLP(bias_a ‖ bias_b) ≈ centroid(a+b)` 是 soft homomorphism；低容量解自然是 `bias_n ∝ n·u`。

H5′ 显著（permutation test，*n*_perm = 1000，single seed）：observed ρ = 0.847，null mean ≈ 0，**p = 0.003**。

图 2 展示单个 dual-muscle seed 上两个 facet 的 7×7 余弦热图：两个 facet 都复现线性数轴（ρ_add = +0.973，ρ_ord = +0.918），跨 facet align ρ = +0.847。

![F2 数字 bundle 余弦热图 — arith + ord](./docs/figures/F2_number_cos_heatmaps.png)

### 4.4 稳健性、规模与纯度

- **Shuffle 反事实。** 用随机置换的 `concept_id → bundle` 训练，并在自然顺序上测试。`|ρ|` 从 0.97 降到 0.18；若按 shuffle inverse remap，ρ 恢复到 0.97，说明几何跟随 **任务驱动的身份**，不是标签顺序。
- **规模。** ρ 随 *N* 单调不降：0.966（*N*=7）→ 0.987（*N*=15）→ 0.991（*N*=30）。
- **纯度审计。** Coherence 不依赖 (i) 监督目标几何（A1，random orthogonal centroids），(ii) concept ID 字符串（A4，UUID-as-id）。它依赖 (iii) optimizer implicit bias（A3）：`normal_small` init 给出 ρ = 0.975；`normal` init（std 1，lazy regime）在 99.3% task accuracy 下只有 ρ = 0.22。我们将其报告为 **feature-learning vs lazy-regime** 区分，而不是污染。
- **线性而非 Weber。** 在 *N* ∈ {7, 15, 30} 上，ρ_linear > ρ_log；模型选择 **等距** 数轴，这不同于生物 ANS。

### 4.5 组合算术（四运算，*N* = 100）

在 balanced-op sampling 与四种运算（±, ×, ÷）下，pair-held-out OOD triples 在 +/− 上达到 92% 泛化，在 ×/÷ 上较弱。带 slot-equivariant head 的 positional composer（ripple-carry 风格）可对 ± 达到 **任意位数 100% 外推**，但需要手写 base-10 先验；这也引出 §7。

---

## 5 实验 2 — 颜色（圆形领域）

### 5.1 设置

12 个概念 `concept:color:{0..11}` 均匀分布在 hue wheel 上，每 30° 一个。两个 muscle：

- **ColorMixingHead**（facet `mixing_bias`，64-d）：(*a*, *b*) → 圆形中点 *c*。排除对立色中点歧义后，共 120 个有效 triples。
- **ColorAdjacencyHead**（facet `adjacency_offset`，8-d）：(*a*, *b*) → 圆形距离三分类 {1 / 2–3 / ≥4}。共 132 个 triples。

Centroid 为 random-orthogonal（没有颜色相似性通过标签泄漏）。5 seeds，30 epochs × 200 steps，总计约 5 分钟。

### 5.2 圆形几何涌现

| Metric | Single (Mix only) | Dual (Mix + Adj) |
|---|---|---|
| mix_acc | 1.000 | 1.000 |
| adj_acc | — | 1.000 |
| **ρ(mix, −circ_dist)** | **0.977 ± 0.007** | 0.978 ± 0.003 |
| ρ(mix, −\|Δi\|) (linear) | 0.681 ± 0.016 | — |
| ρ(adj, −circ_dist) | — | 0.378 ± 0.292 |
| Cross-facet align (mix ↔ adj) | — | 0.376 ± 0.292 |

ρ_circular − ρ_linear = 0.30 在 seeds 间稳定，说明 bundle 捕捉的是 **圆形结构**，不是线性近似。

### 5.3 MDS 可视化：Bundle ≈ Circle

基于 bundle 余弦距离的 2-D MDS（5 seeds，mixing-only）：

| seed | radial residual | angular order (atan2-sorted) | cyclic |
|---|---|---|---|
| 1000 | 0.113 | [6,5,4,3,2,1,0,11,10,9,8,7] | rev |
| 1001 | 0.037 | [5,4,3,2,1,0,11,10,9,8,7,6] | rev |
| 1002 | 0.054 | [5,4,3,2,1,0,11,10,9,8,7,6] | rev |
| 1003 | 0.061 | [5,6,7,8,9,10,11,0,1,2,3,4] | fwd |
| 1004 | 0.053 | [5,6,7,8,9,10,11,0,1,2,3,4] | fwd |

全部 5 个 seeds 都把 12 个概念放成严格循环序；chirality（fwd / rev）只是 random init 的任意对称性。径向残差为半径的 3–11%，即 **bundle 在几何上就是一个圆**。

### 5.4 Shuffle 反事实

用 shuffled `concept_id → bundle` 训练：mix_acc 仍为 1.000，但自然顺序 |ρ_circ| = 0.086 ± 0.044。几何下降 11×；它依赖 **概念身份**，不是标注 artifact。

### 5.5 跨 muscle 对齐方差

跨 facet align 均值 = 0.376，但 std = 0.29；单 seed permutation test 给出 *p* = 0.016（显著）。方差集中在 **adjacency** facet：三分类 bucket loss 对同 bucket pair 施加相同压力，允许多个非等价旋转都满足 loss。连续 target（预测圆形距离标量）预计会增强对齐；留作后续工作。

### 5.6 Bundle 采用领域拓扑

同一个 PCM + collapse 框架，在没有领域专用改动的情况下，在 arithmetic-like loss 下产生 **线性** 几何，在 hue-mixing loss 下产生 **圆形** 几何。Bundle 的余弦结构跟随任务拓扑，而不是监督标签拓扑。

---

## 6 跨领域普适性 — 空间与音素

### 6.1 动机

数字和颜色都是 1-D 拓扑（线性和圆形）。如果 PCM 的自适应几何反映真实结构性质，它应当扩展到 1-D metric domain 之外。我们加入两个实验，同时补齐 H5″ 预测 schema 的两个缺失格子：

- **空间（§6.2）**：5×5 整数网格；2-D partial-order / product topology，有两个正交 metric axis。它测试 PCM 能否涌现真正高维几何，也测试 H5″ 的一侧：同一领域内由不兼容 facet algebra 消费的两个 muscle 不应对齐。
- **音素（§6.3）**：20 个辅音，跨三个发音轴（voicing / manner / place）。这是没有自然 metric 的离散类别领域。它测试 categorical geometry emergence，也测试 H5″ 的 orthogonal-algebra 象限：三个独立 muscle 应产生三个互不对齐的 facet。

所有结果都使用与 §4–§5 相同的 PCM 框架和训练循环；没有领域专用架构改动。

### 6.2 空间 — 2-D 格点

**设置。** 25 个节点 `concept:space:r_c`，两个 muscle 消费不同 facet：

- **MoveHead**（`motion_bias`，64-d）→ 5 类方向 {up, down, left, right, same}；105 个穷尽 pair（自身 + 4-neighbour）。
- **DistanceHead**（`distance_offset`，8-d）→ 9 类 L1 距离 {0..8}；625 个穷尽 pair。

30 epochs × 200 steps，AdamW lr 1e-3，3 seeds。

**几何涌现（single-muscle，3 seeds mean ± std）**：

| metric | value | pre-reg | status |
|---|---|---|---|
| move_acc | 1.000 ± 0.000 | ceiling | ✓ |
| **ρ_L1** (cos vs −L1) | **+0.860 ± 0.012** | > 0.80 | ✓ |
| ρ_linear_flat (1-D index control) | +0.572 ± 0.075 | < ρ_L1 | ✓ (gap 0.29) |
| ρ_row_within | +0.805 ± 0.048 | > 0.60, isotropic | ✓ |
| ρ_col_within | +0.797 ± 0.065 | > 0.60, isotropic | ✓ (Δ = 0.008) |
| **MDS Procrustes disparity** | **0.071 ± 0.014** | < 0.15 | ✓ |
| Shuffle \|ρ_L1\| | 0.025 ± 0.019 | ≈ 0 | ✓ (34× collapse) |
| Shuffle MDS disparity | 0.943 ± 0.028 | ≈ 1 | ✓ |

Bundle 余弦矩阵通过 MDS 投影到 2-D 并 Procrustes 对齐到 ground-truth (row, col) 坐标后，disparity = 0.07（0 表示完美网格）。Row 和 column 轴贡献几乎相同的 ρ（Δ = 0.008），因此涌现结构是 **真正 2-D 且各向同性**，不是某个恰好与 L1 排名相关的 1-D 投影。1-D row-major flattening control 比 ρ_L1 低 0.29。

**跨 facet 不对齐（预注册 H5″ 失败案例）**：

| quantity | value | prediction |
|---|---|---|
| DistanceHead dist_acc | 0.729 ± 0.065 | 远高于 11% chance |
| DistanceHead ρ_L1 (own facet) | +0.158 ± 0.017 | low (scalar shortcut suffices) |
| **cross-facet align (motion ↔ L1)** | **−0.034 ± 0.016** | null under H5″ |
| permutation *p* | 0.767 | > 0.01 ✓ |

两个 muscle 都在同一个 5×5 网格中，但它们施加在 facet 上的代数不同。MoveHead 对 (a, b) 是 *反对称* 的："a 在 b 上方" 与 "b 在 a 上方" 是不同类，迫使 bundle 编码有符号 2-D 坐标。DistanceHead 对 (a, b) 是 *对称* 的，只需要标量 magnitude；8-d facet 可以通过分布式 shortcut 编码满足任务（ρ_L1 = 0.16，尽管任务准确率 73%）。H5″ 预测 vector vs scalar-magnitude algebra 不应对齐；观察到 ρ = −0.03（p = 0.77），吻合。

### 6.3 音素 — 离散类别

**设置。** 20 个辅音，三个属性近似正交交叉：

| axis | classes | counts |
|---|---|---|
| voicing | ± | 8 / 12 |
| manner | STOP / FRIC / NAS / APR | 7 / 6 / 3 / 4 |
| place | LAB / COR / DOR / GLT | 6 / 7 / 5 / 2 |

三个单输入 classifier muscle，各消费自己的 facet（均为 16-d）：VoicingHead（2 类）、MannerHead（4 类）、PlaceHead（4 类）。60 epochs × 120 steps，3 seeds。三个 muscle 都达到 100% 任务准确率。

**每个 facet 在自身轴上的几何（3 seeds，σ = 0）**：

| facet | ρ_same_axis (own) | intra-inter cos gap |
|---|---|---|
| VoicingHead (binary) | **+0.866** | **+1.970 ± 0.012** |
| MannerHead (4-way) | **+0.736** | **+1.232 ± 0.018** |
| PlaceHead (4-way) | **+0.747** | **+1.229 ± 0.027** |

观察到的 ρ_same_axis 正好达到 Spearman against binary same-class indicator 在各轴 class partition 下的 **结构上限**；跨 seeds σ = 0 说明 same-class 与 different-class cosines 完全分离。类内 cos 接近 +1，类间 cos 接近 −1（二值 voicing 的 gap 达 1.97），因此即便在无自然 metric 的领域，PCM 也会沿每个轴涌现清晰的 prototype-style cluster。

泄漏接近 null：voicing facet 关于 manner 或 place 几乎不携带信息。

**跨 facet 对齐 — H5″ orthogonal-algebra 测试**：

| pair | ρ align (3-seed mean) | perm-test *p* | predicted |
|---|---|---|---|
| voice ↔ manner | +0.108 ± 0.022 | 0.052 | null (|ρ| < 0.2) ✓ |
| voice ↔ place | −0.011 ± 0.014 | 0.991 | null ✓ |
| manner ↔ place | −0.124 ± 0.018 | 0.037 | null (|ρ| < 0.2) ✓ |

三种对齐均比 metric domains 中的对齐小约 5×。非零残余来自我们 20-phoneme inventory 中真实的 phonotactic correlation，而不是对齐机制：

- v ↔ m：所有 approximants 和 nasals 都 voiced；知道 manner ∈ {nasal, approximant} 就完全决定 voicing。
- m ↔ p：glottal place 只包含 stops 和 fricatives（没有 nasals 或 approximants）。
- v ↔ p：近独立，因此 residual 最小（−0.01）。

Shuffle 反事实：三个 facet 上 |ρ_same_axis| 都降到 0.022–0.037。任务准确率仍为 100%（identity-invariant heads）。

### 6.4 H5″ 的四领域 schema

结合 §4、§5 和 §6，可占据预测象限：

| | same facet-algebra | different facet-algebra |
|---|---|---|
| **same domain** | 数字 arith ↔ ord：**ρ = 0.91 ± 0.04**（10 seeds，permutation *p* = 0.003）；颜色 mix ↔ adj：**ρ = 0.38 ± 0.29**（5 seeds，permutation *p* = 0.016）—— **align** | 空间 motion ↔ L1：**ρ = −0.03 ± 0.02**（3 seeds，permutation *p* = 0.77）—— **null** |
| **different domain** | （本文未测试） | 音素 v ↔ m / v ↔ p / m ↔ p：**\|ρ\| ≤ 0.12 ± 0.02**（3 seeds，permutation *p* ∈ {0.05, 0.99, 0.04}）—— **null-to-residual** |

每个已占据格子都符合 H5″ 预测。对齐由 **facet 级代数兼容性** 控制，而不是由领域或任务家族控制。F7 展示六个 alignment bar、permutation-null band 与三个 algebra regime 的颜色标注。

### 6.5 四领域几何涌现普适性

在同一个未改变框架下：

| domain | topology | indicator | value |
|---|---|---|---|
| numbers 1–30 | 1-D linear | ρ(cos, −\|Δn\|) | **0.991** |
| colors 12 hues | 1-D circular | ρ_circular / MDS radial residual | **0.977** / ≤ 11% |
| space 5×5 | 2-D lattice | ρ_L1 / MDS Procrustes disparity | **0.860** / **0.071** |
| phonemes 20×3 | discrete categorical | intra-inter cos gap | **+1.23 到 +1.97** |

PCM 自动适配四种质性不同的拓扑：线性、圆形、格点和类别。F4 展示所有四种几何的并列可视化；F8 展示空间网格 MDS 从 trained disp 0.07 到 shuffled disp 0.97 的崩塌。

![F4 四领域 bundle 几何普适性](./docs/figures/F4_four_domain_universality.png)

![F7 H5″ 四领域跨 facet 对齐 schema](./docs/figures/F7_h5pp_alignment_schema.png)

![F8 空间网格 MDS：trained vs shuffled-identity 反事实](./docs/figures/F8_space_mds_trained_vs_shuffle.png)

---

## 7 实验 4 — 纯 Base-10 涌现（负结果）

### 7.1 设置

每个整数 1..*N* 一个 `ConceptNode`（无 slot / carry / digit prior）；一个带 balanced ± sampling 的平坦 `QuadArithHead`。Random orthogonal centroids。Base-10 结构的多个指标：

- `spike_k` = avg cos(n, n+k) − ½·(avg cos(n, n+k−1) + avg cos(n, n+k+1))。若 `spike_10` 为正且邻居平坦，则说明 10-periodicity。
- `residual_units_effect` = 移除线性 cos ≈ α·(−|Δ|) + β 趋势后，比较 same-units-digit 与 different-units-digit pair 的残差（permutation-tested *p*）。

### 7.2 结果（5 seeds total: *N* = 50 × 3 + *N* = 100 × 2）

| *N* | seed | train | ρ_linear | spike₁₀ | spike₅ | resΔunits | *p* | resΔtens |
|---|---|---|---|---|---|---|---|---|
| 50 | 80500 | 1.00 | 0.985 | +0.0014 | +0.0028 | −0.028 | 0.175 | +0.092 |
| 50 | 80501 | 1.00 | 0.985 | +0.0015 | +0.0028 | −0.022 | 0.245 | +0.070 |
| 50 | 80502 | 1.00 | 0.984 | +0.0012 | +0.0029 | −0.030 | 0.120 | +0.093 |
| 100 | 81000 | 0.999 | 0.990 | +0.0008 | +0.0011 | −0.006 | 0.445 | +0.057 |
| 100 | mean | 0.999 | 0.990 | +0.0009 | +0.0011 | −0.008 | ~0.44 | +0.057 |

所有 base-10 指标均为 null。残差 `resΔunits` 实际略为负；`resΔtens` 为正，但这只是线性 ordinality 的二阶 artifact（同 decade pair 平均更近）。F5 绘制 N ∈ {50, 100} 的 avg cos(n, n+k) vs shift k：两条曲线都是平滑单调线，没有 k = 10 的峰。

![F5 Base-10 涌现 null — k=10 无峰](./docs/figures/F5_base10_spike_null.png)

### 7.3 解释

PCM 会诱导出解决任务所足够的 **pairwise geometry**（加法为线性，混色为圆形）。它不会诱导 **algorithmic factorisation**（个位、十位、进位），原因是：(i) 对 random orthogonal centroid 的 cross-entropy 除了 unique directions 外不奖励结构，(ii) 64-d 平坦 `ParamBundle` 没有 factorisation prior，(iii) 纯语义监督没有视觉压力（例如 "12" 与 "32" 之间共享像素）。这是 PCM 当前形式的一个 **清晰经验边界**，有助于校准期待并推动后续工作（视觉 glyph 输入、slot priors、curriculum）。

手写 base-10 prior 确实能解锁 100% digit-length extrapolation（见 D93a / `COMPOSITIONAL_NUMBER_STUDY.md`），与 Abacus embeddings 相当，但数据少约 10³×；代价是把 base-10 写进架构，而不是让它被学出来。

---

## 8 讨论

**Bundle = indexed parameter，且没有静止态。** 概念语义内容与消费历史不可区分（H1–H4）。Liveness *L*(*v*) = 0 在行为上为 nil（H4）。这是维特根斯坦 §43 与 Zuhandenheit 的机制化版本。

**因果身份。** 训练后 bundle swap 关闭相关–因果缺口：在某个 facet 上交换两个训练好的 bundle，只会产生教科书式 double dissociation —— 精准击中消费该 facet 的 muscle，并且只影响涉及被交换概念的 pair（数字：100 → 18.2%；颜色：100 → 5.3%），而其他 muscle 和其他概念保持 100%。F6 展示两个领域中 baseline、swap A only、swap B only、swap both × involving/not-involving × muscle A/B 的所有条件。Bundle 不是身份的 *correlate*；在 downstream muscle 可见范围内，它 **就是** 身份。

![F6 训练后 bundle swap — 教科书式 double dissociation](./docs/figures/F6_swap_dissociation.png)

**几何跟随任务拓扑，而非监督几何。** 数字为线性，颜色为圆形，空间 cell 为 2-D lattice，类别音素为 prototype cluster。四个案例的监督标签都是 random orthogonal centroid 或 one-hot，因此 bundle 中可见结构完全由任务代数 / 组合形状诱导，而不是标签。这是 PCM 允许的最强普适性主张：**任务需要什么拓扑，bundle 就在同一框架下长出什么拓扑**。

**Facet algebra 而非任务领域控制对齐。** 同一领域中相同 facet algebra ⇒ 强对齐：数字（arith ↔ ordinal）ρ = 0.91 ± 0.04，颜色（mix ↔ adj）ρ = 0.38 ± 0.29。相同领域中不同 facet algebra ⇒ null：空间 motion（signed vector） vs L1（symmetric scalar magnitude），ρ = −0.03，p = 0.77。正交类别轴 ⇒ null，残余对齐可由 inventory 的 phonotactic statistics 解释。多任务文献通常把共享表征看成带宽权衡；我们识别出一个结构前提：facet 级代数兼容性。

**Facet 信息密度。** 对齐稳定性取决于 loss 保留多少连续结构。三分类 bucket（color adjacency）给出 align std = 0.29；totally ordered set 上的 ternary ordinal `<,=,>` 给出 std = 0.04。这提示一个设计原则：**尽可能使用连续 / 回归 target**。

**PCM 范围。** PCM 给出 *几何* 涌现（只要任务代数提供压力，跨四拓扑成立），但不给出 *算法* 涌现（base-10 factorisation，§7）。算法涌现可能需要 (a) 带共享 glyph 子结构的视觉 grounding，(b) 架构层面的显式组合先验（D93a），或 (c) multi-agent curriculum pressure。我们认为这条边界是优点：PCM 诚实回答了 "仅从任务信号中能诱导出哪类结构？"

---

## 9 局限

- **Toy domains**（7–100 个 numerosity；12 个 hue；25 个 grid cell；20 个 phoneme）。我们不声称 PCM 原样可扩展到视觉 / 语言；10⁶ 概念的 cg + bundle 需要 storage / sharding 工作。
- **给定 ground-truth concept ID，而非发现它们**（见 §3.6）。Percept 项目已包含从 unmatched percept cluster 创建 `concept:hypothesis:…` 节点的 auto-discovery pipeline；本文故意绕过它以隔离 bundle/collapse 机制。发现 + 表示的端到端联合训练是下一篇论文。
- **Facet capacity 静态。** 每个 facet 的 shape（如 64-d `arithmetic_bias`）由 caller 声明，不会在训练压力下增长。Lazy init 处理新 facet（新 muscle），但不处理既有 facet 的容量扩张。基于 loss plateau 的 "bundle regrowth" 协议已设计但未实现。
- **H5″ 测试了四类 algebra**（ordered-additive、circular-ordered、2-D metric vector vs scalar magnitude、orthogonal categorical）。所有已测试类别都符合 facet-algebra compatibility 预测，但系统性 algebra sweep（lattice、tree、group-valued、mixed continuous-discrete）仍是未来工作；我们 conjecture 但未证明一般必要性。
- **Space DistanceHead 的 73% ceiling / 0.16 ρ_L1 caveat**。Motion 与 L1 facet 之间的 null alignment 部分受到 DistanceHead 自身未涌现 metric geometry 的混淆。更干净设计是把 scalar target 替换为 vector offset (Δr, Δc) prediction，强制 vector-algebraic structure；这是 A2 后续工作。
- **Lazy-regime caveat。** 几何涌现依赖 small-init feature-learning dynamics；在 lazy regime 下任务准确率保留但几何不保留。这是 optimizer 性质，不是 PCM 本身，但限定了适用范围。

---

## 10 结论

三句总结：

1. **把参数放到概念上，归因自然成立。**
2. **Bundle 会跨四类质性不同领域采用任务拓扑**：数字为线性、颜色为圆形、空间为 2-D lattice、音素为 prototype-clustered categorical；即使监督不携带几何，且架构不改。
3. **共享表征之间的对齐不是免费的**；它是 *facet 级代数兼容性* 的信号，当 algebra 在同一领域不兼容（空间）或在类别轴之间正交（音素）时，它会干净地断裂。

---

## 11 可复现性

```bash
# numbers — D91/D92 attribution, H5 vs H5'
python -m experiments.robustness_study \
    --encoder-ckpt outputs/ans_encoder/final.pt --n-seeds 10

# numbers — scale study (N in {7,15,30})
python -m experiments.scale_study --n-seeds 3

# numbers — purity audit (A1–A4)
python -m experiments.purity_audit

# numbers — quad arithmetic (N=100, ± × ÷)
python -m experiments.quad_study --N 100 --n-seeds 3

# numbers — D93a hand-coded base-10 (100% extrapolation)
python -m experiments.compositional_number_study \
    --head-type slot_equivariant

# COLORS — circular domain universality
python -m experiments.color_concept_study --n-seeds 5

# SPACE — 2-D lattice universality + incompatible-algebra null
python -m experiments.space_concept_study --n-seeds 3

# PHONEMES — discrete categorical + orthogonal-algebra null
python -m experiments.phoneme_concept_study --n-seeds 3

# NEGATIVE — pure base-10 emergence fails
python -m experiments.emergent_base10_study \
    --scan 50 100 --n-seeds 3

# CAUSAL — post-hoc bundle swap (Appendix B, ~1 min)
python -m experiments.counterfactual_swap_study --n-seeds 3
```

完整四领域复现（数字 + 颜色 + 空间 + 音素 + base-10 null + swap）在单张 RTX 4090 上 < 25 分钟。所有 seeds 和超参数都作为 JSON 与 checkpoint 一起提供。

---

## 12 图表

全部六张自动生成图位于 `mind/docs/research/figures/`（PDF + PNG）。约 3 分钟即可重新生成：

```bash
python -m experiments.render_paper_figures
```

| # | File | What |
|---|---|---|
| F1 | `F1_pcm_architecture.*` | PCM 架构图：`muscle.forward(cid) → node.collapse(caller, facet) → ParamBundle`，含 consumer registry 与 collapse history |
| F2 | `F2_number_cos_heatmaps.*` | 数字 bundle 7×7 cos heatmaps：dual-muscle run 中 arithmetic_bias + ordinal_offset；标注 cross-facet align |
| F4 | `F4_four_domain_universality.*` | 四领域普适性 panel：线性数轴 · 圆形色环 MDS · 2-D 空间网格 MDS · 音素 cos heatmap |
| F5 | `F5_base10_spike_null.*` | Base-10 涌现 null：N ∈ {50, 100} 的 avg cos(n, n+k) vs k，k = 10 无峰 |
| F6 | `F6_swap_dissociation.*` | Counterfactual swap double-dissociation bars |
| F7 | `F7_h5pp_alignment_schema.*` | H5″ 四领域 schema：六个 cross-facet-alignment bars |
| F8 | `F8_space_mds_trained_vs_shuffle.*` | 空间网格 MDS overlay：trained vs shuffle counterfactual |

---

## 引文

- Barsalou L. W. 1999 · *Perceptual Symbol Systems*. BBS.
- Bricken T. et al. 2023 · *Towards Monosemanticity: Decomposing Language Models with Dictionary Learning*. Anthropic tech report.
- Caruana R. 1997 · *Multitask Learning*. Machine Learning 28.
- Chen Y. et al. 2022 · *HyperPrompt: Prompt-based Task-Conditioning of Transformers*. ICML.
- Dehaene S. 2011 · *The Number Sense: How the Mind Creates Mathematics*. Oxford University Press.
- Elhage N. et al. 2022 · *Toy Models of Superposition*. Transformer Circuits.
- Feigenson L., Dehaene S., Spelke E. 2004 · *Core Systems of Number*. Trends in Cognitive Sciences.
- Graves A. et al. 2016 · *Hybrid Computing Using a Neural Network with Dynamic External Memory*. Nature.
- Ha D., Dai A., Le Q. 2017 · *HyperNetworks*. ICLR.
- Jacot A., Gabriel F., Hongler C. 2018 · *Neural Tangent Kernel*. NeurIPS.
- Kaiser Ł., Sutskever I. 2016 · *Neural GPUs Learn Algorithms*. ICLR.
- Kazemnejad A. et al. 2023 · *The Impact of Positional Encoding on Length Generalization in Transformers*. NeurIPS.
- Khandelwal U. et al. 2020 · *Generalization through Memorization: Nearest-Neighbor Language Models*. ICLR.
- Koh P. W. et al. 2020 · *Concept Bottleneck Models*. ICML.
- Madsen A., Johansen A. R. 2020 · *Neural Arithmetic Units*. ICLR.
- Maennel H. et al. 2020 · *What Do Neural Networks Learn When Trained With Random Labels?* NeurIPS.
- Marks L. et al. 2025 · *Sparse Feature Circuits*. ICLR.
- Maurer A. 2016 · *The Benefit of Multitask Representation Learning*. JMLR.
- McLeish S. et al. 2024 · *Transformers Can Do Arithmetic with the Right Embeddings* (Abacus). NeurIPS.
- von Oswald J. et al. 2020 · *Continual Learning with Hypernetworks*. ICLR.
- Pritzel A. et al. 2017 · *Neural Episodic Control*. ICML.
- Raghu M. et al. 2017 · *SVCCA*. NeurIPS.
- Ruder S. 2017 · *An Overview of Multi-Task Learning in Deep Neural Networks*. arXiv.
- Schmidhuber J. 1992 · *Learning to Control Fast-Weight Memories*. Neural Computation.
- Shallice T. 1988 · *From Neuropsychology to Mental Structure*. Cambridge University Press.
- Templeton A. et al. 2024 · *Scaling Monosemanticity*. Anthropic tech report.
- Welch B. L. 1947 · *The Generalisation of "Student's" Problem*. Biometrika.
- Wittgenstein L. 1953 · *Philosophical Investigations* §43.
- Zarlenga M. E. et al. 2022 · *Concept Embedding Models*. NeurIPS.

---

## 附录 A — 各研究完整原始数字

见 `mind/docs/research/SINGLE_VS_DUAL_MUSCLE_FINDING.md`、`SCALE_STUDY.md`、`PURITY_AUDIT.md`、`QUAD_STUDY.md`、`COMPOSITIONAL_NUMBER_STUDY.md`、`COLOR_CONCEPT_STUDY.md`、`SPACE_CONCEPT_STUDY.md`、`PHONEME_CONCEPT_STUDY.md`、`EMERGENT_BASE10_STUDY.md`、`COUNTERFACTUAL_SWAP.md`。所有数据位于 `outputs/**/summary.json`。

## 附录 B — Counterfactual bundle swap（因果测试）

此前所有 PCM 证据要么是结构上成立的（通过 `consumed_by` 归因），要么是相关 / 必要性证据（shuffle 使 ρ 崩塌、permutation test 给出低 *p*）。PCM 最强命题 —— **bundle 是概念的语义身份本身，而不是学到的相关物** —— 需要训练后的因果干预。

**设计（两个领域相同）**：先把 dual-muscle 模型训练到 ceiling，然后在某个指定 facet 上原地交换两个概念的 bundle `.data`，再在完整 pair set 上重新评估。这保持 head weights 与 optimizer state 不变，只改变 identity map（concept_id ↔ bundle tensor）。每个 seed 三种干预：只交换 facet A、只交换 facet B、两个都交换。每个领域 3 个 random seeds。

**B.1 数字（交换 `concept:ans:3` ↔ `concept:ans:5`）**。
Muscles：AddHead（`arithmetic_bias`，64d）和 CmpHead（`ordinal_offset`，8d）。

| Condition | Add-inv | Add-not | Cmp-inv | Cmp-not |
|---|---|---|---|---|
| Baseline | 100.0% | 100.0% | 100.0% | 100.0% |
| Swap `arith_bias` | **18.2%** | 100.0% | 100.0% | 100.0% |
| Swap `ord_offset` | 100.0% | 100.0% | **75.0%** | 100.0% |
| Swap both | 18.2% | 100.0% | 75.0% | 100.0% |

**B.2 颜色（交换 `concept:color:2` ↔ `concept:color:5`）**。
Muscles：MixHead（`mixing_bias`，64d）和 AdjHead（`adjacency_offset`，8d）。

| Condition | Mix-inv | Mix-not | Adj-inv | Adj-not |
|---|---|---|---|---|
| Baseline | 100.0% | 100.0% | 100.0% | 100.0% |
| Swap `mixing_bias` | **5.3%** | 100.0% | 100.0% | 100.0% |
| Swap `adj_offset` | 100.0% | 100.0% | **23.8%** | 100.0% |
| Swap both | 5.3% | 100.0% | 23.8% | 100.0% |

所有数字在两个领域的 3 seeds 中完全一致（σ = 0）；ceiling accuracy 下的 full-pair enumeration 使 post-swap 预测在固定 pair-arithmetic 下确定。

**解释。** 该模式是参数级的教科书式 **double dissociation**：

1. *定向崩塌*。单 facet swap 只会使涉及被交换概念的 pair 准确率下降，而且只作用于消费该 facet 的 muscle。未涉及概念保持 100%；bundle tensor 是可移植的，它携带自己的语义角色。
2. *无污染*。交换 `arithmetic_bias` 不影响 CmpHead；交换 `ordinal_offset` 不影响 AddHead。两个 facet 携带概念身份中可干净分离的部分。
3. *非随机退化*。18.2% 不是 chance（chance ≈ 14.3%）；它是每个 "3" bundle 确定性地表现为 "5"、反之亦然时的残余正确率。AddHead 在计算 **被交换后的语义**，不是噪声。
4. *跨领域一般性*。数字（线性代数）和颜色（圆形代数）给出同样的定性模式，排除线性特有解释。

**排除的解释。** (a) Bundle 是初始化噪声，训练会绕开它 —— 被反驳，swap 导致精确定向下降。(b) 概念身份存在于 head / backbone weights —— 被反驳，head 未动而 targeted pairs 崩塌。(c) Facet 泄漏 —— 被反驳，每个 facet-swap 对另一个 muscle 不可见。

这是 PCM 中心主张最干净的因果证据，也把 §4–§6 的相关性几何闭合到机制身份。完整数字与讨论见 `COUNTERFACTUAL_SWAP.md`。代码：`counterfactual_swap_study.py`（约 1 分钟，3 seeds，单 GPU）。
