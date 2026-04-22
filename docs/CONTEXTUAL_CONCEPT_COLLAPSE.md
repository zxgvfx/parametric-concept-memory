# CONTEXTUAL CONCEPT COLLAPSE — 概念即 Wittgenstein 量子

> **状态**: `RESEARCH` (2026-04-22) · 支撑 `DESIGN_DECISIONS.md#D92`
> **提出者**: 用户 (2026-04-22 对话) + 架构师共形式化
> **命题**: ConceptNode 本身没有"当前状态". 它只是一个符号身份 + 一包"可能参数". 只有当某个肌肉 (caller) 读取某个命名 facet 的那一刻, concept 才**塌缩**成一个具体的、当下的语义实例. "3"在算术肌肉下的"3"跟"3"在视觉肌肉下的"3"是**同一个 ID, 不同的塌缩**.

> **与 D91 的关系**: D91 说"图谱节点上可以存可学参数, 被肌肉消费". D92 说"被消费这件事本身就是 concept 的语义发生条件 — 没消费, concept 是空的". D91 是工程, D92 是本体论.

---

## 0. TL;DR

**一句话**: Wittgenstein 的"意义即使用 (meaning is use)" 第一次被当作**架构契约**而不是哲学口号; 量子认知 (Busemeyer & Bruza 2012) 的 superposition-collapse 比喻第一次在符号系统中被**字面执行**.

**五个可证伪命题 (H1-H5)**: 参见 §5.

**可直接操作的架构改变**:
1. `ConceptNode.collapse(caller, facet) → ContextualizedConcept`
2. `ContextualizedConcept` 是 ephemeral (不持久化)
3. 肌肉 forward 时被要求显式塌缩, 拿到的是塌缩对象, 不是裸节点
4. 归因 (`consumed_by`) 由"参数被读过"变成"concept 以哪些方式塌缩过"的语义史

---

## 1. 动机

### 1.1 D91 留下的问题

D91 让 ConceptNode 可以存 multi-facet 参数 bundle, 肌肉按 name 取. 但它没回答:

**Q1**: 同一个 ConceptNode `3`, 在 AddHead 调用时和 VisualDecoder 调用时, "是同一个东西"吗?  
**Q2**: 如果 ConceptNode `7` 的 bundle 里从来没被任何肌肉读过, 它存在吗? 跟"不存在的 concept"有区别吗?  
**Q3**: 为什么要用 symbolic ID 而不是 distributed embedding? 如果 bundle 本质是权重, ID 还有什么意义?

D91 工程可以 work, 但语义不清. 用户的这个提议给出清晰的答案:

- **A1**: 是同一个 ID (符号锚点同一), 不是同一个状态 (具体语义不同, 因为读的 facet 不同)
- **A2**: 没被任何肌肉消费过的 concept ≈ 无. 它只是占位 ID. 这是可证伪的 (H4)
- **A3**: ID 提供跨 muscle 的**身份锚**: 多个 muscle 对"3"的不同解读**必须协同学习**, 因为它们共享同一个 symbolic anchor — 这产生涌现的跨任务一致性 (H5)

### 1.2 哲学动机

**Wittgenstein (1953) PI §43**:
> "Die Bedeutung eines Wortes ist sein Gebrauch in der Sprache."
> ("The meaning of a word is its use in the language.")

Wittgenstein 在质疑"意义是词内部的一个东西 (所指 / idea / image)"这种观点, 主张意义由**使用语境**决定. 例如 "game" 没有统一定义, 每次使用激活不同的"家族相似"特征.

**Heidegger (1927) Being and Time, §15-16, Zuhandenheit (上手存在)**:
> 锤子不是"有锤子属性的物体", 而是**在钉钉子时才成为锤子**. 脱离使用, 它只是堆原子.

**Merleau-Ponty (1945) Phenomenology of Perception**:
> 意义随身体的当下实践而涌现, 不是脑内预先存在的表征.

**Radical Embodied Cognition (Chemero 2009)**:
> 认知不需要内部表征. 概念即 affordance, 即"与当前目标的关系结构".

这些都在说同一件事: **意义不在物体里, 在物体-使用者-情境的关系网中**. 但哲学家从不设计架构. 我们可以.

### 1.3 科学动机

**Quantum Cognition (Busemeyer & Bruza 2012, Cambridge Univ Press)**:

真的有认知科学家用**量子概率论**建模人类认知决策. 核心论断:
- 判断前概念处于 superposition (多个兼容态并存)
- 问问题 (提供 context) 的行为本身改变了概念状态
- 不同问法顺序给出不同答案 (**order effects**, 经典概率论无法解释)

他们的**数学结构**:
- concept = ray in Hilbert space
- context = projection operator
- "当前状态" = projected ray

**你的提议与之同构**:
- concept = ConceptNode ID + full ParamBundle (superposition 空间)
- context = (caller, facet) 对
- "当前状态" = `collapse(caller, facet)` 返回的 ContextualizedConcept

注意: 这里"量子"是**结构同构**, 不是物理量子. 我们不声称 concept 在物理意义上服从量子力学. 但数学模式 (superposition → collapse via context) 是直接搬用的.

**Elman (1990) Simple Recurrent Networks + Rogers & McClelland (2004) Semantic Cognition**:
- 同一个 word (e.g., "bank") 的内部表征随 sentential context 变化
- 分布式表征中, context-dependent representation 早有研究
- **但这是隐式的** — 全在神经网络内部, 无架构约束
- 本提议: 显式化为架构契约

**Frame Semantics (Fillmore 1982, 1985)**:
- 词激活 frame, frame 承载意义
- 但 Fillmore 没说词本身是空的 — 他说词**指向** frame
- 本提议更激进: **词本身是空的, 意义纯粹在 frame-word 的激活事件中**

**Schema Theory (Bartlett 1932; Piaget 1952; Rumelhart 1980)**:
- 概念是 schema — 可被实例化的模板
- schema 必须被 slot-filling 激活
- 跟本提议的 facet 塌缩同源, 但 schema theory 是描述性, 无架构性

### 1.4 AI 动机

**Polysemantic Neurons (Elhage et al., Anthropic 2022)**:
- 单个 neuron 编码多个 concept → 反直觉的 feature superposition
- 本提议**反转**: 单个 concept 分布到多个 facet/muscle, **monosymbol, polyfacet**. 更清晰, 更可解释.

**CLIP (Radford et al. 2021) + VQ-VAE (van den Oord 2017)**:
- Token/codebook index 没有内在语义, 语义由下游 head 赋予
- 本提议相似, 但加上**符号化 + multi-facet + 明示 collapse 事件**

**Concept Bottleneck Models (Koh et al. 2020)**:
- 中间层每个 unit 被强制对齐一个 concept
- 单向 (concept → output), 无双向 collapse, 无 multi-facet

**Meta-Learning / Task-Conditioned Representations (MAML, LEO, ProMP...)**:
- 表征 随 task context 变化
- 但 context 是 flat (task id 或 support set), 无"每 concept × 每 facet"的细粒度 collapse

**总结: 现有工作要么是隐式 (Elman), 要么是单向 (CBM), 要么无符号 (CLIP). 都不是"空 symbol + 多面 bundle + 架构级 collapse 事件".**

---

## 2. 正式形式化

### 2.1 对象定义

**定义 2.1 (空概念)**:
$$
\text{ConceptNode}(v) = \left( \text{id}_v, \; \text{meta}_v, \; c_v, \; B_v \right)
$$

其中:
- $\text{id}_v$: 符号 ID (e.g., `"concept:ans:3"`)
- $\text{meta}_v$: 元数据 (label, scope, provenance, 等; D85-D88 已有)
- $c_v$: centroid (D85 已有, EMA 更新的 perception-space 位置)
- $B_v$: ParamBundle (D91 提议, multi-facet 可学参数池)

**关键断言**: $\text{ConceptNode}$ 无 "当前状态". 它只有"可能状态空间" — 即 $B_v$ 所有 facet 的组合.

### 2.2 塌缩算子

**定义 2.2 (Contextualized Concept)**:
$$
\text{CC}(v, m, f) = \left( \text{id}_v, \; m, \; f, \; B_v[f], \; t \right)
$$

其中:
- $m \in \mathcal{M}$: 观测者 (caller muscle)
- $f \in \text{Facets}_m$: 被读取的 facet name
- $B_v[f]$: 该 facet 的参数 tensor (lazy-init if absent)
- $t$: 时间戳

**关键性质**:
- CC 是 **ephemeral** 的 — 每次 forward 产生, 不持久化
- CC 携带了 $v$ 的**当下完整语义** (对 $m$ 而言)
- 同一 $v$ 在不同 $(m, f)$ 下产生**不同 CC**, 但共享 $\text{id}_v$

### 2.3 collapse 算子

$$
\text{collapse}: \text{ConceptNode} \times \text{Muscle} \times \text{FacetName} \to \text{ContextualizedConcept}
$$

**副作用**:
1. 若 $f$ 未初始化, 则按 $m$ 宣告的 shape lazy-init
2. 注册 $m \in C_v[f]$ (consumer registry)
3. 返回的 CC 包含可求导 $B_v[f]$ — 梯度流回

### 2.4 肌肉契约升级

肌肉不再直接访问 ConceptNode.bundle, 而是必须走 collapse:

```python
# 禁止
params = cg.concepts["concept:ans:3"].bundle.params["arithmetic_bias"]   # ✗

# 要求  
cc = cg.concepts["concept:ans:3"].collapse(caller="AddHead", facet="arithmetic_bias")
params = cc.facet_params   # ✓
```

这样 collapse 事件在**日志 / trace / gradient graph** 中都是显式的.

---

## 3. 架构后果

### 3.1 归因的语义升级

**D91 版本**:
> `consumed_by["arithmetic_bias"] = {"AddHead"}` — AddHead 读过这个参数.

**D92 版本**:
> `collapse_history["arithmetic_bias"] = [("AddHead", t1), ("AddHead", t2), ...]` — concept 以 (AddHead, arithmetic_bias) 的方式塌缩过 N 次.

**区别**: D91 记录"谁读过"; D92 记录"以什么模式存在过". 前者是 bookkeeping, 后者是**语义史**.

### 3.2 死概念 vs 活概念

**定义**: $v$ 的**活度 (liveness)** $\mathcal{L}(v) = |\bigcup_f C_v[f]|$ — 跨所有 facet 的唯一 consumer 数.

- $\mathcal{L}(v) = 0$: 空壳 concept, 只有 id, 没塌缩过 — 跟不存在无可操作区别 (**H4**)
- $\mathcal{L}(v) = 1$: 单义 concept, 只在一个 muscle 下有意义
- $\mathcal{L}(v) \geq 2$: **多义 (pluralistic) concept**, 同 id 跨 muscle 存在, 产生 cross-muscle 共识压力 (**H5**)

### 3.3 Concept 生命周期

```
[候选] → [首次 collapse] → [活概念 (L≥1)] → [多义概念 (L≥2)]
    ↓                           ↓                    ↓
  evict                     单义稳定              cross-muscle 一致性涌现
```

这给 D88 hypothesis concept 的**晋升**提供了客观标准: 被 ≥ 2 个肌肉以不同 facet 塌缩过的 hypothesis → 晋升为 "functional concept".

### 3.4 与 D85/D87 (hub-spoke) 的咬合

- **Hub (ConceptNode)**: 唯一 symbol id, 多 facet bundle. D92 让 hub 具有"空身份"本体论.
- **Spoke (SurfaceFormNode)**: Modality-specific 表面. **不** 带 bundle, **不** 参与 collapse — spoke 只是 hub 的锚点之一.
- **强化 D85**: Concept 作为 grounding hub 的地位更强 — 它是 collapse 的发源地, 语义的中转站.

---

## 4. 与 D91 的明确分工

| 问题 | D91 回答 | D92 回答 |
|---|---|---|
| 参数放哪? | ConceptNode.bundle | 同意 (D92 沿用) |
| 谁训练 bundle 参数? | 肌肉 forward 时梯度回流 | 同意 (D92 沿用) |
| 肌肉怎么拿参数? | `bundle.request(name, caller)` | 升级为 `node.collapse(caller, facet) → CC` |
| 归因怎么做? | `consumed_by` 记录读过谁 | 升级为 `collapse_history` 记录**以何方式存在过** |
| Concept 自己"是"什么? | **D91 不回答** (工程命题) | **空身份 + 多面 superposition** (本体论命题) |
| 同一 concept 在不同 muscle 下? | 同一 bundle, 读不同 name | **同一 id, 不同 CC, 不同"当下存在"** |
| 0 consumer 的 concept? | 合法存在 | 等价于不存在 (H4) |
| 多 consumer 的 concept? | 工程上允许 | **语义上涌现多义性 + cross-muscle 耦合** (H5) |

---

## 5. 可证伪命题 (决定是否推进 D92)

在 D91 的 H1, H2 之上, D92 加 H3, H4, H5:

| # | 命题 | 说明 | 验证 |
|---|---|---|---|
| H1 | Attribution Soundness | $m \notin C_v$ ⇒ ablate $B_v$ 不影响 $m$ | ablation × 显著性 |
| H2 | Attribution Completeness | $m \in C_v[f]$ ⇒ ablate $B_v[f]$ 显著降 $m$ 表现 | ablation × 显著性 |
| **H3** | **Collapse Independence** | $\text{CC}(v, m_1, f_1)$ 的变动不应传到 $\text{CC}(v, m_2, f_2)$ 的下游表现, $f_1 \neq f_2$ | 交叉 ablation: 擦 AddHead facet, 测 CompareHead accuracy |
| **H4** | **Void-Without-Collapse** | 从未 collapse 过的 concept, 行为等价于 `unknown` | 新建空 concept vs 已有 concept, 双肌肉任务表现对比 |
| **H5** | **Superposition Coherence** | $\mathcal{L}(v) \geq 2$ 的 concept, 其跨任务 consistency 涌现 > $\mathcal{L}(v) = 1$ | 单 vs 双 muscle 训练, 测 "同 concept 在两任务的 embedding distance" |

**决策规则**:
- H1-H4 全部通过 → D92 推进到扩展阶段
- H5 通过 → D92 具有**涌现性**, 值得论文
- 任一 H 不通过 → 回研究文档修订, 重新思考

### 5.1 H3 具体实验设计 (使用 PoC)

- 训练 AddHead × CompareHead 双肌肉 (都消费 concept:ans:1..7)
- AddHead 消费 `arithmetic_bias` facet
- CompareHead 消费 `ordinal_offset` facet
- **Ablation**: 把 concept:ans:3 的 `arithmetic_bias` 清零
- **测量**: AddHead accuracy (预计 ↓), CompareHead accuracy (预计几乎不变)
- **量化**: 交叉 ablation matrix, 对角主导性

### 5.2 H4 具体实验设计

- 在 graph 里新建 `concept:ans:void1`, 不做任何 collapse
- 推理时强制让 AddHead 用这个 concept (通过 manual override)
- 预测: 结果等价于 random / untrained

### 5.3 H5 具体实验设计

- 两组训练:
  - Group A: 只有 AddHead 训 (单 muscle)
  - Group B: AddHead + CompareHead 共训 (双 muscle, 共享 concept)
- 比较: 两组 concept centroid 的 cross-numerosity 排序稳定性
- 预测: Group B 的 concept 表征更"数字化" (排序更单调), 因为两个 task 同时在 anchor 同一 symbol id

---

## 6. 实现 (PoC · 本周)

### 6.1 API 增量

新增 `ContextualizedConcept` dataclass:

```python
@dataclass(frozen=True)
class ContextualizedConcept:
    """concept 在某次消费时的具体塌缩. Ephemeral."""
    concept_id: str
    caller: str
    facet: str
    facet_params: torch.Tensor   # 带 grad
    tick: int                    # collapse 发生时刻
    
    def as_tensor(self) -> torch.Tensor:
        return self.facet_params
```

ConceptNode 增 `collapse` 方法:

```python
def collapse(
    self,
    caller: str,
    facet: str,
    shape: tuple[int, ...],
    tick: int = 0,
    init: str = "normal_small",
) -> ContextualizedConcept:
    params = self.bundle.request(facet, shape, caller, init=init)
    self.bundle.collapse_history[facet].append((caller, tick))
    return ContextualizedConcept(
        concept_id=self.node_id,
        caller=caller,
        facet=facet,
        facet_params=params,
        tick=tick,
    )
```

### 6.2 双肌肉示范

见 §5.1. 实现位置:
- `mind/core/cognition/language/visual/arithmetic_head_v2.py`
- `mind/core/cognition/language/visual/comparison_head.py`

### 6.3 联合训练脚本

`mind/experiments/language/train_dual_muscle.py`:
- 载入冻结的 NumerosityEncoder + 已建立的 concept:ans:1..7
- 同步训两头, loss = loss_add + α · loss_cmp
- 图谱 bundle 参数进 optimizer
- 每 N step 记录 collapse_history 和 consumed_by

### 6.4 Ablation 脚本

`mind/experiments/language/eval_collapse.py`:
- H1, H2, H3, H4, H5 各一块, 全自动跑
- 产出 ablation_matrix.json + superposition_report.md

---

## 7. 风险 / 缓解

**新增风险 (D91 之外)**:

| 风险 | 等级 | 缓解 |
|---|---|---|
| Collapse API 增加冗余, 破坏现有代码 | ★★ | 保留 `bundle.request` 作为底层, `collapse` 是封装; 非 concept-aware 肌肉仍用旧接口 |
| 量子隐喻被误解 / 物理学家诟病 | ★★ | 文档反复声明"结构同构, 非物理同一"; 提供纯数学版本 |
| H3 collapse 独立性不成立 (facet 间梯度串扰) | ★★★ | 初期接受少量串扰; 若严重则加 facet-wise gradient masking |
| H4 void-without-collapse 不成立 (说明 lazy-init 偷偷注入先验) | ★★ | 用"恒等 identity" 初始化或 zero 初始化严格测 H4 |
| H5 不涌现 (多肌肉训练未产生 consistency) | ★★★ | 若不成立, D92 降级为"engineering-only", 剔除涌现 claim |
| Concept 数量暴增后 collapse_history 爆内存 | ★ | 只存最近 N 次 + 总计数 |

---

## 8. 路线图

### Phase 1 · PoC (本周, 今日启动)
- [x] 研究文档 + D92 决策
- [ ] ParamBundle + collapse_history
- [ ] ContextualizedConcept + ConceptNode.collapse
- [ ] ArithmeticHeadV2 + ComparisonHead 双肌肉
- [ ] 双肌肉联合训练收敛
- [ ] H1-H5 ablation 全跑, 产出 ablation_matrix

### Phase 2 · 扩展 (1-2 周)
- 加第三个 muscle (e.g., OrdinalClassifier / DotRenderer)
- 测试 $\mathcal{L}(v) \geq 3$ 下 H5 是否更强
- Viewer 升级: 节点展开显示 `collapse_history` + 活度徽章
- 自动死概念检测 + 剪枝提议

### Phase 3 · 论文化 (长期)
- 在 MNIST / Omniglot / ARC / bAbI 上推广
- 与 CLIP / CBM / LoRA baseline 对比
- 核心卖点:
  - 架构级归因 (D91 N2)
  - Superposition coherence (D92 H5) — **这个是论文 hook**
- 投稿目标: NeurIPS 2026 (engineering) + Cognitive Science / AJCS (theory)

---

## 9. 开放问题

1. **$\text{CC}$ 是否该持久化某些维度?** 比如"过去 1000 tick 里最常见的 collapse 模式" 是否该缓存到 ConceptNode?
2. **肌肉是否可以"自主选择" facet name?** 还是 facet name 必须是 concept 上声明的白名单?
3. **跨 instance 迁移**: 把 concept + bundle 导出到新 Percept 实例, 新实例肌肉契约不同, 如何处理 facet 对齐?
4. **与 D90 (stateful node, 未来) 关系**: state 跟 param 都住节点上, 谁读谁写, SRP 怎么划?
5. **量子隐喻可否扩展到非正交 facet?** 即 facet 之间可能存在"关联塌缩" (measuring A changes B).
6. **Wittgenstein 家族相似**: 两个 concept 共享大部分 facet → 是否自动产生"家族关系"边?
7. **是否支持反向 collapse**: muscle 报告 "我需要一个有 facet `arithmetic_bias` 的 concept", 图谱搜索匹配?

---

## 10. 参考文献

### 哲学 / 现象学
- **Wittgenstein, L. (1953/2009)** *Philosophical Investigations*. Blackwell. ★★★ 意义即使用
- **Heidegger, M. (1927/2008)** *Being and Time*. Harper. Zuhandenheit (上手性).
- **Merleau-Ponty, M. (1945/2012)** *Phenomenology of Perception*. Routledge.
- **Ryle, G. (1949/2009)** *The Concept of Mind*. knowing-that vs knowing-how.
- **Chemero, A. (2009)** *Radical Embodied Cognitive Science*. MIT Press.

### 认知科学
- **Bartlett, F. (1932)** *Remembering*. CUP. Schema theory.
- **Piaget, J. (1952)** *The Origins of Intelligence in Children*. Schema.
- **Rumelhart, D. (1980)** *Schemata: the building blocks of cognition*.
- **Fillmore, C. (1982)** *Frame Semantics*.
- **Lakoff, G. (1987)** *Women, Fire, and Dangerous Things*. 家族相似 + radial category.
- **Elman, J. (1990)** *Finding structure in time*. Cognitive Science. ★★★ context-dependent distributed rep.
- **Rogers, T. & McClelland, J. (2004)** *Semantic Cognition*. MIT Press.
- **Busemeyer, J. & Bruza, P. (2012)** *Quantum Models of Cognition and Decision*. CUP. ★★★ superposition-collapse in cognition.

### AI / 工程
- **Radford, A., et al. (2021)** *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*. ICML.
- **van den Oord, A., et al. (2017)** *Neural Discrete Representation Learning (VQ-VAE)*. NeurIPS.
- **Koh, P.W., et al. (2020)** *Concept Bottleneck Models*. ICML.
- **Elhage, N., et al. (Anthropic 2022)** *Toy Models of Superposition*. ★ polysemantic neurons (反向对照).
- **Rogers, T. et al. (2022)** *How Neural Networks See: Distributed Reps with Context*. 综述.
- **Finn, C., et al. (2017)** *MAML*. ICML.
- **Rusu, A., et al. (2018)** *Meta-Learning with Latent Embedding Optimization*.

### Percept 内部
- `DESIGN_DECISIONS.md#D85, D87, D88`: Grounding + Hub-Spoke + Pre-Symbolic
- `DESIGN_DECISIONS.md#D91`: Parametric Concept Memory (工程层)
- `research/PARAMETRIC_CONCEPT_MEMORY.md`: D91 调研
- `research/PROCEDURAL_MEMORY.md`: D89 (procedure 也可理解为特殊 facet)

---

## 11. 变更日志

| 日期 | 变更 |
|---|---|
| 2026-04-22 | 初版. 11 节 + 5 条可证伪命题 + 3 阶段路线图. 对应用户 "concept 如量子, 被消费时才塌缩" 原始 insight. |
