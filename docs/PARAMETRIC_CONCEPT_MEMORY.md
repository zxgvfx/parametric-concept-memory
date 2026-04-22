# PARAMETRIC CONCEPT MEMORY — 图谱即参数池 + 架构级归因

> **状态**: `RESEARCH` (2026-04-22) · 支撑 `DESIGN_DECISIONS.md#D91`
> **提出者**: 用户 + 架构师共设计 (2026-04-22 对话)
> **核心命题**: ConceptNode 不再只是符号身份 + 单向 centroid, 而是**按 symbolic key 组织的多面 (multi-facet) 参数池**, 被多个肌肉模块共享消费, 且"谁消费了谁"作为架构一等公民可 introspect, 直接给出**无需事后分析的功能归因**.

---

## 0. TL;DR

| 维度 | 当前 Percept (D85-D89) | 本文提议 (D91) |
|---|---|---|
| ConceptNode 载体 | `centroid` (静态 embedding) + `label` + `scope` | **同上 + `ParamBundle` (多面可学参数池)** |
| 肌肉权重位置 | 全部存肌肉自己 (`.pt`) | **拆成 (backbone 存肌肉) + (concept-specific 存图谱)** |
| 训练中图谱是否更新 | 只 EMA 更新 centroid, 无梯度 | **梯度直接回流进图谱参数** |
| "某肌肉用了哪些 concept" | 运行时用 cosine match, 无持久记录 | **`consumed_by` registry 架构级记录, 训练结束自动就有功能归因** |
| 可解释性 | 事后 probe / saliency | **架构契约内置** |

**一句话**: 把 mechanistic interpretability 从**事后反向工程**变成**架构契约**.

---

## 1. 问题陈述

### 1.1 现状的两个痛点

**痛点 A · 符号身份与参数身份分家**

目前 ConceptGraph 里 `concept:ans:3` 拥有:
- symbolic id (字符串)
- label + scope + provenance (元数据)
- centroid (128-d embedding, EMA 更新, **不含梯度**)
- surface_form 指针列表

而**所有与这个 concept 有关的运算能力** (如 `AddHead` 如何处理数字 3) 完全存在 `arith_head/final.pt` 里, 跟图谱无绑定. 训练时梯度**穿不过 ConceptGraph**, 图谱变成纯 inference-time 的字典.

这有三个直接后果:
1. **无法归因**: 给一个 concept, 不能回答"它跟哪些能力有关?" 除非事后活体 probe
2. **无法剪枝**: 死概念 (无用) 和活概念 (多用) 看起来没差别
3. **无法迁移**: 图谱导出后新实例拿不到"该 concept 对哪些 head 意味着什么", 只能从头训

**痛点 B · Mechanistic Interpretability 是事后考古**

Anthropic Circuits / Sparse AutoEncoder / Activation Patching 这些方法有效, 但都是**反向工程**:
- 训练一个黑盒
- 然后花巨资找 "某神经元对应什么概念"
- 只能近似, 无法证明

我们明明知道 Percept 里有哪些 concept (因为图谱里写着), 却让自己用考古方式再找一遍 — 这是**架构浪费**.

### 1.2 解决方案骨架

把图谱从"身份表"升级为"身份表 + 参数表":

```
ConceptNode.bundle.arithmetic_bias (64,)  ──►  AddHead.forward() 读
ConceptNode.bundle.visual_gain (32,)      ──►  NumerosityEncoder 读
ConceptNode.bundle.comparison_w (8,)      ──►  ComparisonHead 读
ConceptNode.bundle.audio_prior (64,)      ──►  speech_encoder 读
                                              │
每次 forward 读取: bundle.consumed_by[<param>].add(<caller>)
                                              │
训练完: 图谱里自带 "谁消费了我" 的完整记录 ─┘
```

---

## 2. 提议的核心定义

### 2.1 ParamBundle

```python
class ParamBundle(nn.Module):
    """一个 ConceptNode 的"多面可学参数池". 
    
    - 懒创建 (首次 request 时按调用方 shape 初始化)
    - 参数是 nn.Parameter, 梯度可回流
    - consumed_by 记录读取者 (架构级归因)
    - 结构化: 每个 param 有 name, 允许不同肌肉各读各的面
    """
    def __init__(self):
        super().__init__()
        self.params: nn.ParameterDict = nn.ParameterDict()
        self.consumed_by: dict[str, set[str]] = defaultdict(set)
    
    def request(
        self, name: str, shape: tuple[int, ...], caller: str,
        init: str = "normal_small",
    ) -> torch.Tensor:
        """肌肉调用此 API 从 bundle 取参数 (并注册 consumer)."""
        if name not in self.params:
            if init == "normal_small":
                p = nn.Parameter(torch.randn(*shape) * 0.01)
            elif init == "zero":
                p = nn.Parameter(torch.zeros(*shape))
            else:
                raise ValueError(f"unknown init: {init}")
            self.params[name] = p
        self.consumed_by[name].add(caller)
        return self.params[name]
    
    def has(self, name: str) -> bool:
        return name in self.params
    
    def ablate(self, name: str) -> None:
        """归因实证: 把某个参数清零, 观察下游影响."""
        if name in self.params:
            with torch.no_grad():
                self.params[name].zero_()
```

### 2.2 ConceptNode 扩展

```python
@dataclass
class ConceptNode:
    # ... 原有字段 ...
    bundle: ParamBundle = field(default_factory=ParamBundle)
```

### 2.3 肌肉契约 (新的 forward 接口)

约定所有接受 concept context 的肌肉都要:

```python
class ConceptAwareModule(nn.Module):
    """接受 concept_ids, 从图谱 bundle 里读 concept-specific 参数."""
    
    @property
    def caller_name(self) -> str:
        return type(self).__name__
    
    def forward(self, *inputs, concept_ids: list[str] | None = None) -> torch.Tensor:
        raise NotImplementedError
```

**原则**:
- 不是所有肌肉都必须是 concept-aware (底层 NumerosityEncoder 目前就是 concept-agnostic, 它吐出 embedding 后再 match concept)
- **只有做 "concept-specific 计算" 的肌肉** (AddHead / ComparisonHead / CharDecoder / ...) 需要实现这个接口
- Backbone 参数 (不 concept-specific 的) 仍住肌肉里, 保证不膨胀

---

## 3. 文献 Review (按相关度降序)

### 3.1 最接近的前身

**Hypernetworks (Ha, Dai, Le · ICLR 2016)**  
核心: 一个小 "hypernet" 生成另一个大网络的权重.  
与本文关系: 思想一致 (权重外置可生成), **差异**: hypernet 是**单个** global generator, 不是"每 symbol 一个独立参数块"; 也无 consumer registry.

**Differentiable Neural Dictionary (Pritzel et al., DeepMind · ICML 2017)**  
核心: RL agent 带外部 key-value memory, key 做 lookup, value 是可学 tensor, 梯度可回流.  
与本文关系: **最相近**的工程实现. **差异**: DND 的 key 是 neural activation (非 symbolic), memory 是 flat 一 bank, 无概念结构无归因.

**Prototypical Networks (Snell, Swersky, Zemel · NeurIPS 2017)**  
核心: few-shot 学习中每类一个 prototype (一个可学向量), 分类靠距离.  
与本文关系: **每 class 一个参数** 的先驱. **差异**: 单 prototype / 单面, 非 multi-facet; 无 consumer 记录.

### 3.2 参数共享范式

**Mixture of Experts + Routing (Shazeer et al. 2017; Switch Transformer · Fedus 2021)**  
核心: N 个 expert, gating 决定用哪个, sparse activation.  
与本文关系: expert 对应不同子网络. **差异**: expert 不绑 symbolic concept; routing 是学的, 不是声明的.

**LoRA / Adapters (Hu et al. 2021; Houlsby 2019)**  
核心: 冻住 base, 给每个任务/领域加小参数 adapter.  
与本文关系: 参数可按单位 (task/concept) 切分. **差异**: task 级, 不是 concept 级; adapter 跟肌肉一对一, 不共享.

**Memorizing Transformer (Wu et al. ICLR 2022); LongMem (Wang 2023)**  
核心: Transformer 带 external memory bank, attend 到旧 KV 上.  
与本文关系: 外置 key-value store 被主模型消费. **差异**: memory 是 flat, 非 symbolic, 不持久归因.

### 3.3 Symbolic + Neural 混合

**Neuro-Symbolic Concept Learner (Mao et al. · ICLR 2019)**  
核心: 每 concept 是小 MLP, 端到端学习.  
与本文关系: concept 作为函数. **差异**: concept function 静态绑定在代码, 不是"图谱里的可读参数块"; 无 consumer 归因.

**Neural Theorem Provers (Rocktäschel & Riedel · NIPS 2017)**  
核心: 规则是可学向量, unification 可微.  
与本文关系: 符号级参数化. **差异**: 规则层面, 非 concept 层面; 无共享消费机制.

**DeepProbLog (Manhaeve et al. · NeurIPS 2018)**  
核心: Prolog 谓词内嵌神经网络.  
与本文关系: 神经 ↔ 符号双向. **差异**: 谓词 = 静态函数, 无状态化参数, 无归因.

### 3.4 Mechanistic Interpretability

**Mechanistic Interpretability / Circuits (Anthropic, 2022-2024)**  
核心: 手工反向工程, 找每段 circuit 对应的功能.  
与本文关系: **目标一致** (知道哪段权重 做 啥). **方法完全相反**: 本文提议事前声明, MI 事后发现.

**Sparse Autoencoders for Features (Cunningham et al. 2023; Bricken/Anthropic 2023)**  
核心: 用 SAE 在 activation 空间找 monosemantic feature.  
与本文关系: 找"谁负责什么"的工程化尝试. **差异**: 事后 activation level, 不触及权重分配.

**Concept Bottleneck Models (Koh et al. · ICML 2020); Concept Activation Vectors (Kim et al. 2018)**  
核心: 中间层 unit 对应 concept, 可强制对齐.  
与本文关系: concept ↔ internal rep 绑定. **差异**: 单向 (concept 监督 activation), 无参数持久化到图谱.

### 3.5 认知架构

**K-Lines (Minsky · Society of Mind 1986)**  
核心: 概念激活同时调用数据 + 过程.  
与本文关系: **思想祖宗**. **差异**: 语言级提出, 从未工程化.

**ACT-R Chunk + Production; Soar Working Memory**  
核心: chunk 是 concept + attribute-value 组合, production 读 chunk 触发动作.  
与本文关系: attribute-value 类似 multi-facet bundle. **差异**: symbolic only, 无可学连续参数; 无跨 production 的 consumer 自动记录.

### 3.6 汇总: 5 个技术点的覆盖度

| 技术点 | 在哪有过 | 在本文的处理 |
|---|---|---|
| 1. 外部参数池, 梯度回流 | ✅ DND / Hypernet | 沿用, 按图谱拆分 |
| 2. 参数按 symbolic key 索引 | ✅ Proto / 词 embedding | 推广到 ConceptGraph |
| 3. 每 key 有多个命名参数块 | ⚠️ MoE 部分有 (但 expert 不绑 symbol) | **新 (per-concept multi-facet)** |
| 4. 被多个肌肉共享消费 | ✅ shared embedding | 扩到 concept × 面 双维度 |
| 5. 消费者注册表 + 归因 | ❌ 无完整前例 | **本文核心新意** |

---

## 4. 新颖性声明

**N1 (整合)**: 将 (Hypernetwork 参数外置) × (Prototypical 按符号切分) × (MoE 多面) × (Symbolic Graph) 在单一图谱结构中合一, 首次落地.

**N2 (架构归因, 本文最锋利的贡献)**: 将 mechanistic interpretability 从后验分析问题转化为架构契约问题. 通过 `consumed_by` registry, 训练完即得到 "concept X 被哪些能力消费" 的完整图景, 无需 probe / SAE / activation patching.

**N3 (可剪枝可迁移)**: 由归因派生出 dead-concept 自动识别 + concept-level 迁移学习能力 — 导出 bundle 即导出该 concept 的"全部跨肌肉知识".

**N4 (与 D88 / D89 共振)**: ParamBundle 是 D89 ProcedureNode 的自然存储底座 (procedure 的状态就存这里); 也是 D88 hypothesis concept 晋升的客观依据 (有 consumer = 有价值).

---

## 5. 形式化

### 5.1 数学描述

设 ConceptGraph $G = (V, E)$, 每个节点 $v \in V$ 有:
- 静态符号: $\text{id}(v), \text{label}(v), \text{scope}(v)$
- 感知向量: $c_v \in \mathbb{R}^d$ (centroid, 原有)
- **参数束**: $B_v = \{(n_k, \theta_k) : k \in K_v\}$, 每个 $\theta_k \in \mathbb{R}^{s_k}$ 是可学参数
- **消费记录**: $C_v = \{(n_k, M_k) : M_k \subseteq \mathcal{M}\}$, $\mathcal{M}$ 是系统所有肌肉模块集合

一个 concept-aware 肌肉 $m \in \mathcal{M}$ 的 forward 函数:
$$
y = m(x, \{v_i\}_{i=1..n}; \Theta_m, \{\theta^{(m)}_{v_i}\}_{i=1..n})
$$

其中:
- $\Theta_m$ 是 $m$ 自身的 backbone 参数 (跟 concept 无关, 存肌肉里)
- $\theta^{(m)}_{v_i} = B_{v_i}[\text{name}_m]$ 是 concept $v_i$ 在"面 $\text{name}_m$" 上的参数 (存图谱里)
- 读取时自动: $C_{v_i}[\text{name}_m] \leftarrow C_{v_i}[\text{name}_m] \cup \{m\}$

### 5.2 训练动力学

Total loss $\mathcal{L}$ 对肌肉参数和图谱 bundle 参数**同时**回传梯度:

$$
\theta^{(m)}_v \leftarrow \theta^{(m)}_v - \eta \frac{\partial \mathcal{L}}{\partial \theta^{(m)}_v}
$$

$\theta^{(m)}_v$ 只有在 batch 中涉及 concept $v$ 的样本才收到梯度 → 天然稀疏更新 → 建议 SparseAdam 或 LazyAdam.

### 5.3 归因命题 (可验证假设)

**命题 H1 (Attribution Soundness)**:
> 若 $m \notin C_v[\text{任何 name}]$ (即 $m$ 从未读过 $v$ 的任何参数), 则 ablate $B_v$ (全部置零) 不应显著影响 $m$ 在涉及 $v$ 的测试数据上的表现.

**反例命题 H2 (Attribution Completeness)**:
> 若 $m \in C_v[\text{name}]$, 则 ablate $B_v[\text{name}]$ 应显著降低 $m$ 的表现.

H1 和 H2 都应该通过 PoC 中 ablation 实验 empirically 验证 — 这是**本文自带的可证伪 claim**.

---

## 6. 实现草案 (PyTorch)

### 6.1 最小改动

新增文件:
- `mind/core/cognition/language/param_bundle.py`

修改:
- `mind/core/cognition/language/concept_graph.py` 加 `bundle` 字段
- `mind/core/cognition/language/visual/arithmetic_head_v2.py` 新肌肉示范

### 6.2 AddHead V2 示例

```python
class ArithmeticHeadV2(nn.Module):
    """从 ConceptGraph bundle 读 concept-specific bias."""
    
    def __init__(self, cg: ConceptGraph, embed_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.cg = cg
        self.backbone = nn.Sequential(
            nn.Linear(embed_dim * 2 + 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
    
    def forward(
        self,
        ea: torch.Tensor, eb: torch.Tensor, op: torch.Tensor,
        concept_ids_a: list[str], concept_ids_b: list[str],
    ) -> torch.Tensor:
        h = self.backbone(torch.cat([ea, eb, op], -1))
        bias_a = torch.stack([
            self.cg.concepts[cid].bundle.request(
                "arithmetic_bias", (h.shape[-1],), caller="ArithmeticHeadV2"
            )
            for cid in concept_ids_a
        ])
        bias_b = torch.stack([
            self.cg.concepts[cid].bundle.request(
                "arithmetic_bias", (h.shape[-1],), caller="ArithmeticHeadV2"
            )
            for cid in concept_ids_b
        ])
        return F.normalize(h + bias_a + bias_b, dim=-1)
```

### 6.3 训练循环变化

```python
# 把图谱 bundle 里所有参数也收进 optimizer
all_params = list(head.parameters())
for node in cg.concepts.values():
    all_params += list(node.bundle.parameters())
optim = torch.optim.AdamW(all_params, lr=2e-3)
```

### 6.4 Introspection

```python
for cid, node in cg.concepts.items():
    for pname, consumers in node.bundle.consumed_by.items():
        if consumers:
            print(f"{cid}.{pname} ← {consumers}")
```

---

## 7. 风险 / 缓解

| 风险 | 等级 | 缓解 |
|---|---|---|
| 参数爆炸 (10k concept × 5 facet × 64 dim) | ★★ | Lazy init + LRU evict + 稀疏性正则 |
| 稀疏更新下 Adam 不稳 | ★★★ | SparseAdam / LazyAdam; 图谱参数单独 optim, 低 lr |
| Attribution 泄漏 (过拟合导致 false consumer) | ★★★ | Top-k routing + L1 on bundle activations |
| 动态 hypothesis concept 如何初始化 bundle | ★★ | 首次被 request 时懒初始化 (已设计) |
| 违 D19 (看起来像 LLM KV cache) | ★★ | 严格白名单 caller; 每 facet ≤ 64 dim; bundle 总大小 cap |
| Verification 难度 (测单肌肉要 mock graph) | ★ | 提供 `MockConceptGraph` fixture |
| 图谱持久化膨胀 | ★★ | `.pt` 单独存, JSON 只存 metadata + 指针 |

---

## 8. 路线图

### Phase 1 · PoC (本周)
- [x] 写本研究文档 + D91 决策
- [ ] ParamBundle 实现 + 单元测试
- [ ] ArithmeticHeadV2 上线, 训练到 ≥ 95% accuracy
- [ ] **Attribution ablation test**: 验证 H1 / H2

### Phase 2 · 扩展 (1 周)
- ComparisonHead / OrdinalHead 改 V2
- viewer 升级: 图谱节点显示 "谁消费了我"
- 跟 D90 (parametric state) 合并 API

### Phase 3 · 论文准备 (长期)
- 标准数据集 (MNIST / Omniglot / ARC) 上 benchmark
- 比较 vs baseline (flat embedding / MoE / standard finetune)
- Ablation 系统化: H1/H2 在多任务上的普遍性
- 投稿目标: NeurIPS 2026 / ICLR 2027 ("Parametric Concept Memory" or "Architectural Attribution")

---

## 9. 开放问题

1. **Bundle 初始化如何影响收敛?** 零初始化 vs 小随机 vs centroid-derived, 三者收敛速度差别?
2. **Consumer 多的 concept 是否更稳定?** 多个肌肉同时施加梯度是否互相干扰?
3. **跨 scope evict 如何处理 bundle?** BASE scope 的 bundle 能否被 level scope 的肌肉 request?
4. **Routing vs Declaration 的 tradeoff**: 让 caller 自己声明读哪些 concept vs 让图谱自己 gate 选读哪些?
5. **D90 (stateful nodes) 和 D91 (parametric bundle) 在同一 ConceptNode 上冲不冲突?** state vs param 的语义分界?
6. **Attribution 的因果性**: consumer ≠ 因果贡献. ablation 是定义因果的一种方式, 但如何扩展到更精确的 Shapley value 式归因?
7. **跨实例迁移**: 把 bundle 拷到新 Percept 实例, 新实例的肌肉能否立即复用? 如果肌肉 backbone 不同怎么对齐?

---

## 10. 参考文献

### 参数共享 / Hypernet / Memory
- Ha, D., Dai, A., Le, Q.V. (2016). *HyperNetworks*. ICLR 2017.
- Pritzel, A. et al. (2017). *Neural Episodic Control*. ICML 2017.
- Shazeer, N. et al. (2017). *Outrageously Large Neural Networks: Sparsely-Gated Mixture-of-Experts Layer*. ICLR 2017.
- Fedus, W. et al. (2021). *Switch Transformer: Scaling to Trillion Parameter Models...*. JMLR 2022.
- Wu, Y. et al. (2022). *Memorizing Transformers*. ICLR 2022.
- Wang, W. et al. (2023). *Augmenting Language Models with Long-Term Memory*. NeurIPS 2023.
- Hu, E.J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
- Houlsby, N. et al. (2019). *Parameter-Efficient Transfer Learning for NLP*. ICML 2019.

### Prototypical / Symbolic / Neuro-Symbolic
- Snell, J., Swersky, K., Zemel, R. (2017). *Prototypical Networks for Few-shot Learning*. NeurIPS 2017.
- Mao, J. et al. (2019). *The Neuro-Symbolic Concept Learner*. ICLR 2019.
- Rocktäschel, T., Riedel, S. (2017). *End-to-end Differentiable Proving*. NIPS 2017.
- Manhaeve, R. et al. (2018). *DeepProbLog: Neural Probabilistic Logic Programming*. NeurIPS 2018.
- Andreas, J. et al. (2016). *Neural Module Networks*. CVPR 2016.

### 可解释性 / Attribution
- Koh, P.W. et al. (2020). *Concept Bottleneck Models*. ICML 2020.
- Kim, B. et al. (2018). *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors*. ICML 2018.
- Elhage, N. et al. (Anthropic, 2021-2024). *Mechanistic Interpretability / A Mathematical Framework for Transformer Circuits*.
- Cunningham, H. et al. (2023). *Sparse Autoencoders Find Highly Interpretable Features*.
- Bricken, T. et al. (Anthropic, 2023). *Towards Monosemanticity: Decomposing Language Models with Dictionary Learning*.

### 认知架构
- Minsky, M. (1986). *The Society of Mind*. Simon & Schuster.
- Anderson, J.R. (1993). *Rules of the Mind* (ACT-R). Lawrence Erlbaum.
- Laird, J.E. (2012). *The Soar Cognitive Architecture*. MIT Press.

### Percept 内部
- `DESIGN_DECISIONS.md#D85`: 多模态符号接地
- `DESIGN_DECISIONS.md#D87`: Surface Form vs Concept Identity (Hub-Spoke)
- `DESIGN_DECISIONS.md#D88`: Pre-Symbolic Grounding
- `DESIGN_DECISIONS.md#D89`: Procedural Memory Node
- `research/PROCEDURAL_MEMORY.md`: 对应 D89 的研究背景

---

## 11. 变更日志

| 日期 | 变更 |
|---|---|
| 2026-04-22 | 初版. 10 节 + 25 篇文献 + 可证伪命题 H1/H2 + 三阶段路线图. |
