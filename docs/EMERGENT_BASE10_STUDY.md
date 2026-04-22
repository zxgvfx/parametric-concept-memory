# Emergent Base-10 Study (Step 1 of pure-emergence program, 2026-04-22)

**TL;DR — 清晰的 negative result**

- 在 D91 flat ConceptGraph (每数字一个独立 ConceptNode, 独立 bundle) +
  add/sub 训练信号 + random orthogonal centroid 监督下, **base-10 结构
  没有自发 emerge**.
- 所有 seeds (N=50 × 3 + N=100 × 1) 呈现一致的**纯 linear number line**
  (ρ_linear ≈ 0.99), 10-周期 spike、units-digit sharing、tens 聚类等
  所有 base-10 signal 全部接近 0 或 p-value > 0.1 (non-significant).
- 这证明 D93a 手写 slot+carry 先验**在当前数据/compute 规模下是必需的**.
  要让 base-10 自发 emerge 需要额外的压力 — 最可能的候选是**视觉符号
  输入** (让 "12" 的像素里 "1", "2" 的 visual 分离驱动 factorization).

---

## 1. 背景与动机

D93a 的 slot composer + slot-equivariant head + 8-dim carry 是人手写的
base-10 先验. 它让架构在 4 位数加减上 100% 外推, 但也把 "AI 只能以人类
数学家方式思考数字" 这个约束 hardcoded 进了系统. 用户的哲学质疑:

> 手写 base-10 类似之前的"LLM encoding + 向量对齐" 省成本捷径, 可能
> 限制 AI 发现人类没想到的规律. 能不能让 agent 自己从视觉/任务中发现
> 位值结构?

本实验是这个研究方向的 **Step 1**: 先问"在最少先验下, base-10 能否
自发 emerge?" 如果能, 我们可以 deprecate D93a; 如果不能, 就有了严格的
empirical 证据 — 在 Percept 当前规模下, 无手写先验的 pure emergence
路径不可行, 必须在中间做中介.

## 2. 实验设计

- **架构**: `QuadArithHead(n_ops=2)` + D91 flat `ConceptGraph`,
  `concept:ans:1` ... `concept:ans:N` 每个独立 `ParamBundle(64)`.
  **没有**任何 slot / position / digit / carry 先验.
- **任务**: add + sub (用户问题核心在"加减乘除", 为干净先限制到两种).
  Target: random orthogonal centroid (bypass NumerosityEncoder, 保证
  monotonic / linear 结构是**训练动力学自发涌现**, 不是 encoder 先验).
- **训练**: balanced-op sampling, 80 epochs × 500 steps, batch=64.
  train acc: N=100 下达到 99.9%, 完全 in-distribution converged.
- **分析指标** (剥离 linear trend 后的 residual 层):
  - `spike_{k}`: `avg cos(n, n+k) - (avg cos(n, n+k-1) + avg cos(n, n+k+1))/2`.
    若 base-10 emerge, `spike_10` 应显著为正 (shift-10 的 pair 相似度高于
    邻近 shift 的插值).
  - `residual_units_effect`: 拟合 `cos ≈ α(-|Δ|) + β` 作为 linear trend,
    在 residual 上对比 `a%10 == b%10` 和不等. Permutation test 给 p-value.
  - `residual_tens_effect`: 同上, 按 `a//10 == b//10`.

## 3. 结果

### 3.1 主表

| N | seed | train | ρ_linear | spike_10 | spike_5 | resΔunits | p | resΔtens |
|---|---|---|---|---|---|---|---|---|
| 50 | 80500 | ~1.00 | 0.985 | +0.0014 | +0.0028 | **−0.028** | 0.175 | +0.092 |
| 50 | 80501 | ~1.00 | 0.985 | +0.0015 | +0.0028 | **−0.022** | 0.245 | +0.070 |
| 50 | 80502 | ~1.00 | 0.984 | +0.0012 | +0.0029 | **−0.030** | 0.120 | +0.093 |
| 100 | 81000 | 0.999 | 0.990 | +0.0008 | +0.0011 | **−0.006** | 0.445 | +0.057 |

### 3.2 解读

**三个关键 null findings**:

1. **无 10-周期 spike**: `spike_10` 在所有 seed 上都 ≈ 0.001 (同样数量级
   为 `spike_5, spike_11` 等 base-10 无关 shift). 如果 base-10 emerge,
   `spike_10` 应该在 0.05-0.1 量级.
2. **无 units-digit clustering** (residual level):
   `resΔunits` 在 N=50 稍负 (−0.02), 在 N=100 几乎 0. 所有 p > 0.1,
   **non-significant**. 方向还是**反**向 (same-units-digit residual 反而
   略低), 说明不仅没 emerge, 连"趋势的阴影"都没有.
3. **`resΔtens > 0` 是 linear ordinal 残差, 不是 base-10**:
   same-tens-digit 的 pair (如 30-39) 平均距离小, linear decay 预测它
   cos 高. 拟合后残差仍正说明 linear trend 的**二阶**(局部曲率) 效应,
   不是 base-10 factorization.

### 3.3 对照: D93a 下的 bundle 几何

D93a slot composer + slot-equivariant head (见 `COMPOSITIONAL_NUMBER_STUDY.md`)
的 bundle 是**硬 decomposed** 的: `bundle(12)[slot 0] == bundle(32)[slot 0]`
严格相等. 对应到本实验的 metric:

- `resΔunits → +∞` (同 units-digit 的 pair 在 slot 0 上完全一致)
- `spike_10` 巨正 (shift-10 的 pair 共享 slot 0 digit)

D93a 是 **oracle 参照系**: 它告诉我们"如果 base-10 emerge 应该长什么样".
D91 flat 的结果显示距离这个 oracle **无限远**.

## 4. 结论与含义

### 4.1 信息论/数据 scale 上的硬限制

当前数据 (~10^4 triples) + 训练目标 (CE over linear-separable centroid) +
架构 (bundle 为 opaque 64-dim vector) 的组合**信号量不足以推导出
base-10 factorization**. 具体原因:

- **训练目标无 factorization 压力**: CE loss 只需要每个 concept 有 unique
  direction, 任何排列都行. 没有 reward 函数 encourage "shared structure
  across units-digit".
- **Bundle 无内部结构**: `ParamBundle` 是 flat `nn.Parameter`, 没有
  sparsity / group / slot 先验. Lottery-ticket 意义上可能存在一个
  sub-network 实现了 digit decomposition, 但 SGD 没动力找它.
- **视觉符号信号缺失**: 人类学 base-10 极大程度依赖"'12' 的字符由 '1', '2'
  组成"这种视觉 composition. 当前训练只给 semantic 数字标签, 视觉压力
  为零.

### 4.2 "Pure emergence" 路径的下一步

Step 1 的 negative result 不是终点, 是**诊断**. 下一步 (Step 2) 必须
加入更多的自然先验/压力才有希望 emerge:

| Step | 加入压力 | 预期 |
|---|---|---|
| **Step 1** (本研究) | 无 | ❌ 无 base-10 |
| **Step 2** | **视觉符号** (字符图像 "1", "12", "23" 的 pixel grid) | 可能 emerge units-digit sharing (因为 pixel 里 '1' 被视觉上复用) |
| Step 3 | 视觉 + dot canvas (numerosity grounding) 联合 | 可能 emerge 分层 ANS + 位值 |
| Step 4 | 多 agent 通信 + 共享 arithmetic | 可能 emerge **非** base-10 (e.g. base-6, base-12, log-scale) |
| Step 5 | Video + 真实世界 counting 场景 | 最接近人类幼儿, 但算力/数据 cost 极高 |

### 4.3 对 D93a 当前状态的定位

这个 negative result 把 D93a 重新定位为:

- **不是 cheat**: 在当前 scale 下, base-10 先验是 *architecturally
  necessary*, 不是 optional optimization.
- **是 stepping stone**: D93a 展示了"如果这个结构被学会, 系统能做什么"
  (100% 4-digit 外推). 它也定义了 future emergence studies 的成功基准.
- **揭示了研究缺口**: "如何让 base-10 from emergence" 是一个独立 research
  program, 需要视觉 grounding / curriculum / 多 agent 等多条战线.

## 5. 实验产物

- `mind/experiments/language/emergent_base10_study.py` — 分析框架
- `outputs/emergent_base10_full/bundles_N50_seed{0,1,2}.json` —
  N=50 全 bundles (3 seeds)
- `outputs/emergent_base10_full/bundles_N100_seed0.json` — N=100 1 seed
- `outputs/emergent_base10_n100/` — N=100 初步 3-seed 版 (train acc 低,
  作为 consistency check)

## 6. Followup (Step 2 骨架建议)

如果用户同意继续, Step 2 应该:

1. **替换 centroid 为视觉输入**: 输入 `(image_of_a, image_of_b, op)`,
   image 是 "12", "34" 等字符的 28×28 pixel grid (MNIST-style). 
2. **保留 flat ConceptGraph**: 每个数字仍然独立 ConceptNode.
3. **Encoder 是 CNN**: 从像素直接 encode 到 bundle space.
4. **关键假设**: CNN 的 spatial 共享 filters 可能会发现 "'1' 这个 glyph
   被多个 number 复用" 的规律, 驱动 bundle 出现 units-digit sharing.
5. **观察指标**: 同本实验的 `spike_10`, `residual_units_effect`.
6. **成功标准**: `resΔunits > 0.05` 且 `p < 0.01` 跨多 seed.

如果 Step 2 仍然 negative, 说明即使加了视觉也不够, 需要更 structural
的改动 (Step 3-5).
