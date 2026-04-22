# Color Concept Study (A1, 2026-04-22)

**TL;DR — decisive positive result**

- 数字 domain 观察到的 concept bundle 几何 + 跨 muscle 对齐 (见
  `SINGLE_VS_DUAL_MUSCLE_FINDING.md`) **不是 ordinal 特例**, 在**循环拓扑**
  的颜色 domain 下同样出现.
- 5 seeds / 12 个 hue-wheel concept / mixing-muscle 单独训练下, bundle
  的 2D MDS 投影**是一个近乎完美的圆** (radial residual 3-11%), 12 个
  concept 严格按 hue 顺序循环排列 (5/5 seeds).
- Spearman 指标: **ρ_circular = 0.977 ± 0.007 ≫ ρ_linear = 0.681**, 差距
  稳定在 0.30 左右. Shuffle 反事实 |ρ| = 0.086, 完全塌缩.
- 跨 muscle (mixing ↔ adjacency) 对齐 mean = 0.376, permutation-test
  p = 0.016, significant. 但对齐方差较大 (0.29), 原因在 adjacency
  3-class loss 给 bundle 的 circular 压力不足.
- **结论**: D91 (Parametric Concept Memory) + D92 (Contextual Concept
  Collapse) 不依赖 ordinal 结构, 是**通用** parametric concept memory.
  Bundle 会 **自适应 adopt domain 拓扑** (linear for numbers, circular
  for colors) 作为几何表达.

---

## 1. 动机

之前整个 D91/D92 pipeline 都是在**数字 (1..N)** 上验证的. 数字的特殊性:
ordinal, linear, 有严格 `<` 关系. 无法排除一种可能性 — "ConceptGraph
bundle 表现出漂亮几何" 只是因为 **任务本身就是 ordinal-like**, 跟
D91/D92 框架关系不大.

A1 实验的核心问题: **同样的框架放进非 ordinal domain, 会发生什么?**

颜色是最清爽的 counter-example:
- **拓扑**: circular (hue wheel), 不是 linear.
- **特殊相等**: red (0°) 和 rose (330°) 物理上相邻, 但 `|0 - 11| = 11` 线性
  距离最远.
- **"加法"结构**: 颜色 mixing 在 hue wheel 上是**circular midpoint**, 而不是
  数字 `a + b`.

如果 bundle 在颜色下也 emerge 结构化几何, 且几何**反映的是 circular 而
不是 linear**, 那就证明 D91/D92 真的在做"通用 concept memory"的事.

## 2. 实验设计

### 2.1 Concepts

12 个 `concept:color:{0..11}`, 对应 hue wheel 每 30° 一个 concept. 无
任何视觉 encoder, 纯符号 concept_id → bundle.

### 2.2 Muscles (两个, 类比数字 domain 的 Arithmetic + Comparison)

**ColorMixingHead** (消费 `mixing_bias`, 64-dim, 类比 `ArithmeticHeadV2`):
- 输入: `(concept_a, concept_b)`, bundle concatenation.
- 输出: 128-dim pred vector.
- 监督: cross-entropy over **random orthogonal centroids** (12 × 128),
  target = `mix_pair(a, b)` = circular midpoint on hue wheel.
- Opposite pair (`circ_dist = 6`) 因 midpoint 歧义被排除, 留下 120 个训练
  triple.

**ColorAdjacencyHead** (消费 `adjacency_offset`, 8-dim, 类比
`ComparisonHead`):
- 输入: `(concept_a, concept_b)`.
- 输出: 3-class logits.
- Label: `0 = adjacent (circ_dist=1)`, `1 = near (2-3)`, `2 = far (≥4)`.
- 132 个训练 triple, bucket distribution = `{0: 24, 1: 48, 2: 60}`.

### 2.3 Centroid 选择

**Random orthogonal** (通过 QR 分解生成 12 个近正交单位向量). 关键设计:
这**剥离了一切 color similarity 先验** — centroid 之间两两几乎正交, 所以
如果 bundle 出现"颜色 a 和 b 在几何上更近"这种现象, 必须是**从 mix task
的组合结构中 emerge 的**, 不是从 supervision target 里透进来的.

### 2.4 训练超参

- 30 epochs × 200 steps × batch 32, LR = 1e-3, AdamW.
- 5 seeds (1000..1004).
- Wall time per seed: ~42s (single), ~57s (dual), total ~5 分钟 / 5 seeds.

### 2.5 指标

- **ρ_circular**: Spearman(off-diag cos, −circular_dist).
- **ρ_linear**: Spearman(off-diag cos, −|i−j|). Linear control; 如果 ρ_circ
  远大于 ρ_lin, 就证明 bundle emerge 的是**循环**而不是**线性**几何.
- **cross-facet alignment**: Spearman(vec(cos_mix), vec(cos_adj)).
- **MDS 2D 投影**: 把 bundle cos similarity 转 distance 后做 MDS → 2D, 看
  投影是否呈圆, 12 个点是否严格按 hue 顺序循环排列.
- **shuffle counterfactual**: 训练时把 concept_id → bundle 的映射随机
  打乱, 看 ρ 是否塌缩到 0.
- **permutation test**: 对 cross-facet alignment 做 1000-perm p-value.

## 3. 结果

### 3.1 E1 — Multi-seed (5 seeds, single vs dual)

| seed | single mix_acc | single ρ_circ | single ρ_lin | dual mix_acc | dual adj_acc | dual ρ_mix_circ | dual ρ_adj_circ | dual cross-facet align |
|---|---|---|---|---|---|---|---|---|
| 1000 | 1.000 | +0.964 | +0.659 | 1.000 | 1.000 | +0.977 | +0.340 | +0.345 |
| 1001 | 1.000 | +0.982 | +0.669 | 1.000 | 1.000 | +0.977 | −0.123 | −0.112 |
| 1002 | 1.000 | +0.980 | +0.700 | 1.000 | 1.000 | +0.979 | +0.621 | +0.562 |
| 1003 | 1.000 | +0.977 | +0.690 | 1.000 | 1.000 | +0.975 | +0.611 | +0.616 |
| 1004 | 1.000 | +0.981 | +0.686 | 1.000 | 1.000 | +0.982 | +0.443 | +0.470 |
| **mean** | 1.000 | **+0.977** | +0.681 | 1.000 | 1.000 | **+0.978** | +0.378 | **+0.376** |
| **std**  | 0.000 | **0.007** | 0.016 | 0.000 | 0.000 | **0.003** | 0.292 | 0.292 |

### 3.2 E2 — Shuffled concept counterfactual (5 seeds)

| seed | mix_acc | ρ_circ (raw order) | ρ_lin (raw order) |
|---|---|---|---|
| 2000 | 1.000 | −0.069 | −0.111 |
| 2001 | 1.000 | −0.075 | −0.118 |
| 2002 | 1.000 | +0.026 | +0.069 |
| 2003 | 1.000 | −0.132 | −0.086 |
| 2004 | 1.000 | −0.126 | −0.134 |
| **|ρ_circ| mean** |  | **0.086 ± 0.044** |  |

Shuffle 训练仍然达到 1.000 mix_acc (因为打乱只是 relabel, 任务仍然解得
出来), 但 bundle 的 natural-order ρ 塌缩到 ~0.09. **身份是必要条件**.

### 3.3 E4 — Cross-facet permutation test (1 seed × 1000 permutations)

- Observed cross-facet ρ = +0.345
- Null distribution mean = +0.003, std ≈ 0.17
- **p-value = 0.016, significant (p < 0.05)**

### 3.4 Geometric validation — 2D MDS 投影

对 5 seeds 的 mixing bundle 做 MDS:

| seed | radial residual (std / radius) | angular order (atan2-sorted) | cyclic check |
|---|---|---|---|
| 1000 | 0.113 | `[6,5,4,3,2,1,0,11,10,9,8,7]` | **reverse cyclic**, start=6 |
| 1001 | 0.037 | `[5,4,3,2,1,0,11,10,9,8,7,6]` | reverse cyclic, start=5 |
| 1002 | 0.054 | `[5,4,3,2,1,0,11,10,9,8,7,6]` | reverse cyclic, start=5 |
| 1003 | 0.061 | `[5,6,7,8,9,10,11,0,1,2,3,4]` | forward cyclic, start=5 |
| 1004 | 0.053 | `[5,6,7,8,9,10,11,0,1,2,3,4]` | forward cyclic, start=5 |

- **5/5 seeds**: 12 concept 在 MDS 2D 投影下**严格按 hue 顺序**循环排列.
- 正向 / 反向只是 chirality 的任意选择 (centroid 随机初始化决定).
- Radial residual (半径的 std 占比) 3-11%, 即 **bundle 几何是一个近乎
  完美的圆**.

## 4. 解读

### 4.1 Bundle 自适应 adopt domain 拓扑

数字 domain: bundle 呈 linear number line (ρ_linear ≈ 0.99, scale-study
已确立).

颜色 domain: bundle 呈 **circular hue ring** (ρ_circular = 0.977, MDS
残差 <10%).

**同一套 D91/D92 机制, 没有任何 domain-specific 修改**, 只是任务不同
(`a + b` → `mix(a, b)` circular midpoint), bundle 就 emerge 出了完全不同
但**与任务拓扑一致**的几何. 这是通用性的最强证据.

### 4.2 ρ_circular ≫ ρ_linear: 不是 "顺便" 线性的

0.977 - 0.681 = 0.30 的差距说明 bundle **确实抓住了 circular 结构**, 而
不是用一个 linear 近似凑合. 如果 bundle 是纯 linear, 两个 ρ 应该一致 (因
circular 在线性序上也有强相关). 差距越大, circular specific 越强.

### 4.3 Adjacency facet 的方差问题

5 seeds 里 ρ_adj_circ 分布: `[+0.34, −0.12, +0.62, +0.61, +0.44]`, 一个
seed 出现负值. 为什么不像 mixing 那么稳?

- Adjacency loss 是 3-class bucket (`adjacent / near / far`), 把真实的
  circular distance 粗粒化 → **bundle 的压力只需要区分 3 个 bucket, 不需要
  区分 "1 vs 2 vs 3 的细微差异"**.
- 同一 bucket 内的 concept pair 训练信号等价, bundle 可以任意排列, 只要
  bucket 边界对.
- 初始化随机性 → 某些 seed 陷入 "bucket 对但 global rotation 反的" 局部最
  优.

这和数字 domain 的 Comparison facet (3-class `<, =, >`) 表现不同, 因
数字 `<, =, >` 直接对应 ordinal, 全局 rotation 不能 trivially 解决. 颜色
的 circular adjacency 有 **N 个全局 rotation 不变方向**, 更难 regularize.

**启示**: facet 监督的 **信息密度** 决定 geometry emergence 的稳定性.
Mixing (实数取值 modulo 12) > Adjacency (3-bucket). 这可能是未来 facet
设计的一个 general 原则.

### 4.4 跨 muscle 对齐: 方差大 ≠ 无

mean align = 0.376, 方差 0.29. 四个 seed 正值 (0.34-0.62), 一个负值. 可是:

- 纯 null 下 align 应该 ~0 且方差更小 (见 E4 permutation: null mean=0.003,
  std≈0.17).
- Observed mean 0.376 比 null 分布高 2+ sigma.
- Permutation test 单 seed 已经 p=0.016 显著.

align 方差大本质上是 adj facet 本身的方差带进来的, 不是 alignment 机制
本身的问题. **修复路径**: 换一个信息密度更高的第二 muscle (比如回归版的
`predict circular_dist scalar`) 应该能把 alignment 和方差都收紧.

### 4.5 Shuffle 反事实

|ρ_circ| 从 0.977 塌缩到 0.086 (11× reduction). **Bundle 的几何完全来自
concept identity**, 不是 centroid 结构的 shadow. 这闭合了 "ConceptGraph
做的事 ≡ 符号记忆" 的论证.

## 5. Implications for Percept

### 5.1 D91/D92 从"数字特例"升级为"通用 parametric concept memory"

这是 Percept 宣言的关键支撑点. 之前 `SINGLE_VS_DUAL_MUSCLE_FINDING.md`,
`SCALE_STUDY.md`, `PURITY_AUDIT` 系列只能说"这个框架在数字上 work",
容易被质疑"数字太特殊". 本研究证明:

- **非 ordinal domain** (circular): work.
- **多 muscle 跨 facet 对齐**: work.
- **Bundle 自适应 domain 拓扑**: work.

### 5.2 Facet 监督信息密度是新设计维度

同一 framework 下, 不同 facet 带的信息密度决定 emergence 稳定性. 这给
未来 muscle 设计一个新 principle: **用连续 / 回归目标, 避免粗粒度 bucket**.

### 5.3 下一步 (A2 / A3)

- **A2**: 空间 concept (left/right/above/below 等) — 更复杂的拓扑, 部分
  有序部分对称.
- **A3**: 音素 concept — 离散无明显拓扑, 测试 bundle 是否仍能 emerge
  非平凡结构 (如区分元音/辅音群).
- **A系列完成后**: 可以把数字 + 颜色 + 空间 + 音素整合成一篇 "Universal
  Parametric Concept Memory" paper, novelty 在 framework 层面而不是
  任何单一 domain.

## 6. 实验产物

- `mind/experiments/language/color_concept_study.py` — 实验 harness
- `outputs/color_full/summary.json` — 5-seed 结果
- `outputs/color_full.log` — 训练 log
- `outputs/color_smoke/` — 单 seed smoke (弃)

## 7. 复现

```bash
python -m experiments.color_concept_study --n-seeds 5
```

Wall time: ~5 分钟 (CPU / 单 GPU 皆可). 完全 deterministic (seeded).
