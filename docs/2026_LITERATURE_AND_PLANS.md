# 2026 文献调研 & PCM 实验方案路线图

**目的**：基于 2025–2026 年脑科学、发展心理、人类学、哲学、AI/ML 最新文献，列出可在 PCM 框架内**可证伪、可执行**的实验方案，**一方面探索项目潜力，一方面验证论文可靠性**。

**主轴**：寻求人类智能行为涌现的"本质"。

---

## Part A — 文献调研摘要（按领域）

### A1. 脑科学 · 概念神经元与范畴知觉

| Finding | Source | PCM 映射 |
| --- | --- | --- |
| MTL concept cells 编码**多级语义抽象**（不只单一 exemplar） | PLOS Bio (population coding); Nature 2025 amygdala/hippocampus region-based feature coding | Tier-G anchor 已是单层抽象，**应支持 anchor 上再 anchor**（层次化） |
| concept neurons 表示**抽象关系**（"flexibly represent abstract relations"，attention 重激活） | Nat Commun 2021 | PCM cooks 已对应；可加"silent reactivation"行为协议 |
| amygdala/hippocampus 用**区域基特征**编码物体，预测记忆性 | Nat Commun 2025 | 等价于 PCM "facets"；可量化 facet→memorability 关系 |

### A2. 脑科学 · 睡眠 replay 与抽象巩固（**最具操作意义**）

| Finding | Source | PCM 映射 |
| --- | --- | --- |
| **NREM 双相**：small-pupil substate replay 新记忆，large-pupil substate replay 老记忆，**避免 catastrophic forgetting** | Nat Neurosci 2025 (s41593-025-01886-6); Nature 2024 (s41586-024-08340-w) | 当前 Tier-G 是**单 pass**；应拆为 **fresh-replay + consolidate-replay** 双相 |
| **平行处理**：sleep 同时巩固过去 + 准备未来神经群组 | Nat Commun 2025 (s41467-025-58860-w) | 可在 sleep 后注入 "future-task" prefetch |
| **Spindle trains** → 特异性强化；**isolated spindles** → 泛化 | Commun Biol 2025 (s42003-025-09425-6) | 暗示 PCM 应有**两条 sleep 路径**：稳定化 vs 抽象化 |
| REM theta 贡献抽象与情感标签 | BMB Reports 2025 | 暂不直接对应 |

### A3. 发展心理学 · 颜色 / 数字 / 音素 时间线

| Finding | Source | PCM 映射 |
| --- | --- | --- |
| **2 月龄** ventrotemporal cortex 已有视觉范畴；**non-hierarchical** 涌现；DNN representational geometry 与婴儿对齐 | Nat Neurosci 2025 (s41593-025-02187-8) | PCM 颜色域 anchor 应能在小数据涌现；可加 "DNN-vs-PCM RSA" 对比 |
| 婴儿色彩分类由**低层 cone-opponent** 驱动，prelinguistic | PMC9314692 | 当前用 LMS centroids；可补 R-G / B-Y **拮抗轴** B 层 |
| **12 月数感**预测 4 岁数学；ANS 跨年龄稳定 | Dev Sci 2023 (Decarli) | number 域可加 ANS-style **ratio-based magnitude** 任务 |
| **8 月**婴儿可在词上下文中习得非母语对比 | Lipkind et al 2024 (S0163638324000407) | phoneme 域可加 **word-context-conditional** 聚类任务 |
| **9 月**对 phonotactic regularities 敏感 | Frontiers 2024 (fpsyg.2024.1367240) | phoneme 域已用 Zipf phonotactic；可加 contrast-specific 测 |

### A4. 人类学 · 颜色 / 语言普适

| Finding | Source | PCM 映射 |
| --- | --- | --- |
| WCS 110 unwritten languages 确认**universal 焦点色**与 6 Hering primaries 一致 | PNAS 2003 (1532837100); Kay & Cook WCS | PCM §6.8 已涌现 6 anchors → wedge alignment 可与 WCS 焦点色直接定量比较 |
| **Focal colors** 可由 category extension 用 representativeness 预测 | PNAS 2016 (1513298113) | 可计算 PCM anchor centroid 是否对应人类 focal hue |
| **IB 解释跨语言 spatial preposition**：人类翻译位于最优 frontier | arxiv 2603.19924 (2026) | 全新评估指标：测 PCM 学到的 codebook 是否近 IB-optimal |

### A5. 哲学 · 预测加工 / 主动推断 / 概念涌现

| Finding | Source | PCM 映射 |
| --- | --- | --- |
| Predictive Processing + 结构表征 + grounded cognition → 解释 abstract concepts via metaphor mapping & 模拟 | Minds&Machines 2026 (s11023-026-09773-0) | 为 §6 "geometric emergence" 提供哲学 grounding |
| **Active Inference**：耦合 AIF agents = higher-level AIF agent；free-energy 跨层一致 | arxiv 2603.20927 (2026) | PCM peer 模块可直接重构为 AIF 协调机制 |
| 意识 = 信息熵动态闭合 | Frontiers Hum Neurosci 2026 (1742084) | 不直接相关 |

### A6. AI / ML · 概念涌现 / 可解释 / 世界模型

| Finding | Source | PCM 映射 |
| --- | --- | --- |
| Transformer **概念涌现 3 维**（训练时间×层×scale）；early-layer features re-emerge later | arxiv 2505.19440 | 提供 emergence dynamics 评测协议 |
| **HSAE 层次化 SAE**：feature splitting → 父子 anchor 树 | arxiv 2602.11881 (2026) | 直接对应"anchor 上再 anchor"，与 A1 呼应 |
| **AlignSAE**：固定 latent slot 绑定概念，支持 causal swap | arxiv 2512.02004 | PCM 设计相同；可定量比较 |
| **SAE feature steering fragility**（layer/magnitude/context 敏感） | arxiv 2601.03047 | PCM 鲁棒性对照 baseline |
| **ZWM**：sparse temporally-factored predictors → 单童年视角生 physical understanding | arxiv 2604.10333 (2026) | PCM ConceptGraph 可视为同范式 |
| **Causal-JEPA**：object-level masking → counterfactual reasoning +20%，1% latent 即可规划 | arxiv 2602.11389 (2026) | 可加 "mask-one-bundle-row infer-from-others" D 层任务 |
| **Transformer 类比涌现 3 阶段**：memorize → compositional → analogical；similarity 必须先于 attribute | OpenReview aFCoTBGM4M (2026); arxiv 2602.01992 | curriculum design 直接借鉴；可作 PCM 类比里程碑 |
| 几何对齐 + functor mapping = 类比 | Nat Sci Rep 2026 (47123-3); OpenReview 806HUH0uQn | ρ 之外的"几何对齐度量"补充 |
| **IB 多 level 解释语言**（phonology→syntax→lexical 都最小化 predictive info） | Nat Hum Behav 2025 (s41562-025-02336-w) | 跨域 IB efficiency frontier 测试 |

### A7. 空间表征（rodent → human）

| Finding | Source | PCM 映射 |
| --- | --- | --- |
| Place + Grid cells 构建**拓扑表征 + 时间排序**，rather than rigid coords | Nat Rev Neurosci 2024 (s41583-024-00817-x); 2021 (s41583-021-00499-9) | PCM space 域当前是 rigid 2-D lattice；应升级到 ordered-experience graph |
| 海马 sharp-wave ripples 支持 offline consolidation 大量 uncorrelated info | Place/Grid review | 与 A2 NREM ripple 一致 |

---

## Part B — 八个实验方案（按"科学价值 × 工程可行性"排序）

每个方案给出：**假设 H** · **实验设计** · **可证伪预测 P** · **预计代价**（机器小时与代码量）· **对应论文章节**。

---

### S1 · 双相 NREM 睡眠（小瞳/大瞳）— 抗灾难性遗忘

**假设 H1**：把 PCM Tier-G 拆成 **"fresh-replay (small-pupil)"+ "consolidate-replay (large-pupil)"** 双相，新概念学习不破坏旧 anchors。

**设计**：
- 数据：连续学习 number→color→space 三个域（domain-incremental）
- A baseline：单 pass sleep（当前 Tier-G）
- B treatment：双相 sleep
  - small-pupil：仅 replay 最近 W 步内学到的 anchors
  - large-pupil：仅 replay W 步以前的 anchors（并阻塞新 cluster 写入）
- 评估：每域学习后跨域评估 ρ 与任务 acc，观察"忘了多少"

**预测 P1**：B 在旧域上的 acc drop 应显著 < A（差距 ≥ 5pp，5 seeds，paired t）。

**代价**：~3h；新增 ~150 LoC；改 `pcm/sleep.py` `run_sleep_pass` 加 `phase`。

**论文章节**：§8.1 Continual learning · 直接呼应 *Nat Neurosci 2025 (s41593-025-01886-6)*。

---

### S2 · 层次化 anchor（HSAE 风格）— anchor 上再 anchor

**假设 H2**：在 sleep pass 后对 anchors 自身再做 k-means → super-anchors，PCM 能涌现**两层范畴结构**（如 "warm/cool" 之上 of "red/orange" 之上 of 单一 hue）。

**设计**：
- 在 §6.8 颜色域 24-hue → k=6 anchors → super-k=2 super-anchors
- 在 §7.4 数字域 100-N → k=10 anchors（个位）→ super-k=2/5（小/大、奇/偶）
- 评估：super-anchor 的纯度（与人类 warm/cool / parity 标签）；下游 head 是否能用 super 而非 anchor 完成任务
- 证伪：若 super-anchor 纯度 ≤ 随机基线，则不涌现层级。

**预测 P2**：颜色域 super-anchor 与 warm/cool 标签 NMI ≥ 0.5；数字域 super-anchor 与 parity NMI ≥ 0.4。

**代价**：~4h；新增 ~250 LoC；扩展 `_phase_b_kmeans` 为递归 / 加 hierarchical assignment 索引。

**论文章节**：§9 Hierarchical abstraction · 呼应 *HSAE arxiv 2602.11881*、*MTL multi-level abstraction PLOS Bio*。

---

### S3 · Object-Centric Mask-Infer（Causal-JEPA 风格）— D 层因果干预

**假设 H3**：加一个 D 层任务"mask 一个 bundle row，从 facet 中其它 row 推断它"，可显著破除 §7.5-space mixed-OOD ceiling 与 §7.5 number length-OOD（不依赖 augmentation）。

**设计**：
- 新 head `MaskInferHead`：输入 N-1 个 collapsed bundle，预测被 mask 的那个 bundle 在 facet 中的 anchor id
- 在 number / space / color 三域分别加该 head；与已有 D 层任务并行训练
- 复跑 §7.4 / §7.5-space / §7.5 三个长度/分布外推
- 证伪：若 mixed-OOD acc 不超过 augmentation baseline (S37) 且不优于 BCD，则该任务无独立贡献

**预测 P3**：mixed-OOD ≥ 0.5（vs 当前 0.000，augmentation baseline ~0.85）；length-OOD ≥ +3pp（vs 当前 +1.1pp）。

**代价**：~5h；新增 ~300 LoC `experiments/_mask_infer.py` + 改 3 个 train.py。

**论文章节**：§10 Latent intervention · 呼应 *Causal-JEPA arxiv 2602.11389*。

---

### S4 · IB efficiency frontier — 跨域 codebook 评估

**假设 H4**：PCM 学到的 codebook（=anchor centroids）位于 information-bottleneck 最优 frontier 附近，且 BCD 条件下比 A 更接近最优。

**设计**：
- 计算 anchor 分配 P(anchor|item) 与 item 分布 P(item)
- 计算 IB 目标：min I(item;anchor) – β·I(anchor;target)（target = 任务标签）
- 跨 5 conditions × 4 域，绘制 (complexity, accuracy) frontier
- 与 WCS 110 语言色彩命名 frontier 直接对比（颜色域）
- 证伪：若 PCM 5 condition 的 codebook 全都远离 frontier，或 BCD 不优于 A

**预测 P4**：颜色域 BCD condition 距 frontier 距离最小；与 WCS 焦点色对应 NMI ≥ 0.6。

**代价**：~2h；新增 ~200 LoC `pcm/ib_eval.py`（纯分析）+ figure renderer。

**论文章节**：§11 Information-theoretic alignment · 呼应 *Nat Hum Behav 2025 (s41562-025-02336-w)*、*WCS PNAS*。

---

### S5 · Cone-Opponent B 层（颜色升级）— 更逼近婴儿

**假设 H5**：用 R-G、B-Y 拮抗轴（cone-opponent）替代 LMS 锥体作为 B 层先验，PCM 颜色域涌现的 anchors 应**更接近婴儿 prelinguistic 范畴**而非成人 11 色。

**设计**：
- `make_cone_opponent_centroids`（24-hue 投影到 RG/BY 平面 + 1 luminance）
- 在 §6.8 + §7.5-color 两个实验中比较 LMS-init vs Opponent-init
- 评估：anchor 与"婴儿 5–7 月 categorical boundary"（NIRS 文献）对齐度

**预测 P5**：Opponent-init 在 holdout hue 上 ρ 不降，且 anchor 数量稳定在 4（婴儿水平）而非 6（成人）。

**代价**：~2h；新增 ~80 LoC；改 `experiments/color_concept_study/graph_builder.py`。

**论文章节**：§6.8 prior 扩展 · 呼应 *PMC9314692 cone-opponent infant*。

---

### S6 · Curriculum Similarity → Attribute（类比涌现）

**假设 H6**：先学 similarity（哪些 bundle 是同类），后学 attribute（每类的属性），PCM 类比能力（A:B::C:?）涌现先于直接联合训练。

**设计**：
- 在 number 域加新 head `AnalogyHead`：输入 (a, b, c) 预测 d 使得 b-a ≈ d-c（向量类比）
- 三种训练顺序：（i）联合；（ii）先 similarity（聚类损失）→ 再 attribute；（iii）逆序
- 跟踪 analogy acc 随 step 演化
- 证伪：若三条曲线无显著差异

**预测 P6**：(ii) 在前 50% step 内类比 acc 早 ≥ 5pp，最终持平或更高。

**代价**：~3h；新增 ~200 LoC。

**论文章节**：§12 Curriculum effects · 呼应 *Feature Resemblance arxiv 2603.05143*、*OpenReview aFCoTBGM4M*。

---

### S7 · 拓扑空间（替代 rigid lattice）

**假设 H7**：把 space 域从 rigid 2-D lattice 改为**ordered-experience graph**（按访问顺序连边，而非欧氏邻接），length-OOD 可破。

**设计**：
- 新 graph builder：random-walk 生成 trajectory → 边按 transition 频率
- 在 §7.5-space 长度外推与 mixed-OOD 上比较 lattice vs ordered
- 证伪：若 ordered 不优于 lattice

**预测 P7**：length-OOD ≥ +3pp；mixed-OOD ≥ +3pp。

**代价**：~3h；新增 ~150 LoC。

**论文章节**：§13 Topology vs metric · 呼应 *Nat Rev Neurosci 2024 (s41583-024-00817-x)*。

---

### S8 · Word-Context Phoneme（婴儿 8 月）

**假设 H8**：在 phoneme 域加**词上下文条件聚类**（同一 phoneme 在不同 word 中的 instance 视为同类），可显著提升 cross-language transfer。

**设计**：
- 数据合成：phoneme × word-context (CV / CVC / VC)；同 phoneme 跨 context 视为同 anchor
- A：context-blind 聚类；B：context-aware 聚类
- 复跑 §6.9 cross-language transfer
- 证伪：若 B 在 target lang 上 acc 不优于 A

**预测 P8**：target-lang acc ≥ +4pp（与 §6.9 当前 baseline 比较）。

**代价**：~3h；新增 ~120 LoC。

**论文章节**：§14 Context-conditional categories · 呼应 *Lipkind 2024*。

---

## Part C — 推荐执行顺序

按"**先低成本破 ceiling → 再深化方法论 → 最后哲学整合**"分三波：

### 波次 1（fast wins，~10h，破 ceilings）
1. **S3 · Mask-Infer**（破 mixed-OOD 与 length-OOD，最大科学价值）
2. **S5 · Cone-Opponent**（最便宜，颜色 prior 升级）
3. **S1 · 双相 sleep**（continual learning 是 PCM 一直缺的功能）

### 波次 2（structural，~12h，方法论深化）
4. **S2 · 层次化 anchor**（直接对应 MTL multi-level abstraction）
5. **S4 · IB frontier**（提供"接近最优 codebook"这一全新评估维度，呼应 WCS）
6. **S6 · Curriculum**（类比能力 emergence dynamics）

### 波次 3（speculative，~6h）
7. **S7 · 拓扑空间**
8. **S8 · 词上下文音素**

**总预算**：~28 机器小时 + ~1500 LoC 新增。所有实验复用 `pcm.diagnostics.run_causal_ablation` API → 协议一致、统计可比。

---

## Part D — 全局产出（实验完成后）

- 新章节 §8–§14（中英双语）扩 PAPER ~30%
- 7 张新 figure：F18 (双相 sleep) · F19 (HSAE-tree) · F20 (Mask-Infer) · F21 (IB-frontier) · F22 (cone-opponent) · F23 (curriculum) · F24 (topology) · F25 (word-context)
- 1 个**统一论点**："PCM 把脑科学（NREM 双相、MTL 多级抽象、grid 拓扑）、发展心理（婴儿 cone-opponent、ANS、词上下文）、AI 解释（HSAE、Causal-JEPA、IB）三方独立证据**汇聚到同一可证伪框架**"。

这正是寻求"本质"的方向：不是重复某一学派，而是**把多学派的同一现象**（"概念如何从感觉中浮现"）**用同一个机制集**（B/C/D + sleep + anchor）**复现并证伪**。

---

## 引用论文一览（按文献年代）

| 序 | Year | Source | 核心贡献 |
| --- | --- | --- | --- |
| 1 | 2024 | Nat Rev Neurosci s41583-024-00817-x | Place/Grid topological remapping |
| 2 | 2024 | Nature s41586-024-08340-w | Sleep microstructure organizes replay |
| 3 | 2024 | Lipkind 2024 (S0163638324000407) | 8mo word-context phonetic learning |
| 4 | 2024 | Frontiers fpsyg.2024.1367240 | 9mo phonotactic sensitivity |
| 5 | 2025 | Nat Neurosci s41593-025-01886-6 | NREM 双相 separates new/old |
| 6 | 2025 | Nat Neurosci s41593-025-02187-8 | 2mo VTC visual categories |
| 7 | 2025 | Nat Commun s41467-025-58860-w | Sleep parallel past+future |
| 8 | 2025 | Nat Commun s41467-025-56793-y | MTL region-based feature coding |
| 9 | 2025 | Commun Biol s42003-025-09425-6 | Spindle trains vs isolated |
| 10 | 2025 | Nat Hum Behav s41562-025-02336-w | IB → linguistic structure |
| 11 | 2025 | arxiv 2505.19440 | LLM 概念涌现 3 维 |
| 12 | 2025 | arxiv 2512.02004 | AlignSAE concept-aligned |
| 13 | 2026 | Minds&Machines s11023-026-09773-0 | PP + structural reps + grounded |
| 14 | 2026 | arxiv 2603.20927 | Active Inference 跨层耦合 |
| 15 | 2026 | arxiv 2602.11881 | HSAE 层次化 SAE |
| 16 | 2026 | arxiv 2604.10333 | ZWM zero-shot world model |
| 17 | 2026 | arxiv 2602.11389 | Causal-JEPA object-level intervention |
| 18 | 2026 | arxiv 2603.29090 | HCLSM hierarchical causal latent |
| 19 | 2026 | OpenReview aFCoTBGM4M | Transformer 类比涌现 |
| 20 | 2026 | arxiv 2603.05143 | Feature Resemblance similarity-first |
| 21 | 2026 | Nat Sci Rep 2026 47123-3 | 几何视觉关系推理 |
| 22 | 2026 | arxiv 2603.19924 | IB 跨语言 spatial preposition |
| 23 | 2026 | arxiv 2601.03047 | SAE steering fragility |

---

*维护说明：每完成一个 Sn 方案后，在对应行追加"Status: ✅ §x.y" 链接到 PAPER 章节。*
