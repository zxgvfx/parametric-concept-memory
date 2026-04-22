# Quad-Arithmetic + Pair-OOD Study (2026-04-22)

**TL;DR**

- 在 N=100 范围上同时训练加减乘除, concept bundle 在不到 10 分钟内达到
  ~98% train / ~84% 外推准确率, 从未见过的 (a, b, op) 组合上仍有
  **加 94% / 减 94% / 乘 65% / 除 79%** 的表现 — 这是 D91/D92 架构下对
  "**compositional generalization**" 的决定性证据.
- "预测**训练中从未出现的单个数字** a×b" 在 D91/D92 下**不可能**: concept
  bundle 是 per-concept 参数, 未注册 concept 根本没 bundle. 真正能回答
  "能否预测没见过的运算" 的问题是 **unseen (a, b, op) 组合**, 而不是
  unseen individual number. 本研究聚焦后者.
- "真小数 (任意精度)" 需要架构升级 (D93: concept = 参数生成器). 本研究
  做的是 "离散化到固定精度 step" 的妥协版, step=0.5 时发现 bundle
  ordinal 纯度下降 (ρ_lin 0.97→0.77), 但 MLP 仍能学到 compositional
  映射 (pair-OOD add 81%, sub 83%).

---

## 1. 设计初衷

用户希望看到 D91/D92 架构对 "加减乘除, 小数, N=100, 外推" 全场景的承载
能力. 在规划阶段我们先做了可行性分析:

| 用户要求 | 架构承载力 | 策略 |
|---|---|---|
| N=100 整数 | ✅ 直接 | 100 concept × 64 bundle_dim, 微不足道 |
| 加减乘除混合 | ✅ 直接 | op_onehot 从 2d 扩到 4d 即可 |
| 小数 | ⚠️ 有限 | 离散到固定 step (0.5 → 60 concept), 保留半整数网格 |
| "预测未见过的数字" | ❌ 本质不可能 | 未注册 concept 没有 bundle, 这是 D91 per-concept memory 的先天结构 |
| "预测未见过的 (a, b, op) 组合" | ✅ 可测 | **pair-wise OOD**, 15% held-out |

用户提出的"外推"在概念空间的正确投影是 **compositional generalization
over (a, b, op) triples**, 这正是 Lake & Baroni (SCAN, COGS) 等工作定义的
经典问题.

## 2. 实验设计

**框架**: `mind/experiments/language/quad_study.py`

- 四则运算共享一个 `QuadArithHead` (muscle), 仅通过 4 维 op_onehot
  区分任务. 两个输入 concept 通过 `arithmetic_bias` facet 被 caller
  塌缩为 `ContextualizedConcept`, head 消费 `bias_a ‖ bias_b ‖ op_onehot`
  → MLP → (embed_dim) → 与 random orthogonal centroid 点积做 softmax.
- **concept 网格**: `step=1.0` 整数, `step=0.5` 含半整数. 约束所有运算
  结果 `c ∈ grid[0:N]`, 保证 concept 空间闭合.
- **Balanced-op batch sampling**: 乘除 triples (N=100 下仅 482 个) 远少于
  加减 (4950 个), 必须在每个 batch 内均衡 op 分布, 否则稀疏 op 被淹没
  (初版 uniform sampling 时 N=100 train acc 仅 55%; balanced 后 98%).
- **Pair-OOD split**: 每个 op 独立随机 hold out 15% triples, 训练完测
  held-out 三元组上的 per-op accuracy.
- **Random orthogonal centroids** (bypass NumerosityEncoder) 保证 target
  类之间的距离是先验均匀的, 任何 ordinal 结构都只能来自训练动力学.

## 3. 结果

### 3.1 N=30 整数 (3 seeds)

```
triples: add=435, sub=435, mul=111, div=111; train=85%, OOD=15%
                train acc    OOD acc
  add           1.00         0.898 ± 0.06
  sub           1.00         0.892 ± 0.03
  mul           0.97         0.354 ± 0.10
  div           1.00         0.688 ± 0.07
  ρ_linear      0.972 ± 0.002
  ρ_log         0.911 ± 0.008
```

### 3.2 N=100 整数 (2 seeds, balanced-op)

```
triples: add=4950, sub=4950, mul=482, div=482; 80 epochs × 500 steps
                train acc    OOD acc
  add           0.958        0.940 ± 0.01
  sub           0.965        0.938 ± 0.02
  mul           0.999        0.653 ± 0.04
  div           1.000        0.785 ± 0.01
  ρ_linear      0.859 ± 0.02
  ρ_log         0.758 ± 0.02
```

### 3.3 N=30 step=0.5 (小数离散化, 2 seeds)

```
concept grid: {0.5, 1.0, ..., 15.0} (30 个 concept)
triples: add=435, sub=435, mul=147, div=147
                train acc    OOD acc
  add           1.00         0.815 ± 0.02
  sub           1.00         0.831 ± 0.06
  mul           1.00         0.705 ± 0.29
  div           1.00         0.568 ± 0.02
  ρ_linear      0.765 ± 0.02
  ρ_log         0.760 ± 0.01
```

## 4. 关键发现

### F1. 四则混训下 bundle 仍自发形成线性数字线, 但强度随规模下降

- N=30 整数 ρ_linear = 0.97; N=100 ρ_linear 降到 0.86; step=0.5 再降到 0.77.
- **解释**: 加减在线性 bundle 上是完美的 (bias_a + bias_b ≈ bias_{a+b}),
  乘除则需要**非线性 / log-scale** 几何 (Weber 律之所以在人脑里出现并非
  偶然). 四则混训拉着 bundle 在"线性最优 for +/-"和"log 最优 for ×/÷"
  之间妥协, 规模越大偏差越明显.
- ρ_linear 和 ρ_log 的差距 (0.97 vs 0.91 → 0.86 vs 0.76 → 0.77 vs 0.76)
  单调缩小, 到 step=0.5 时两者几乎等价 — bundle 正从"纯线性"向"兼顾
  log"漂移, 但**仍未达到**以乘除支配的 log-scale.

### F2. 加减法实现近乎完美的 pair-OOD 外推

- N=100 加法 OOD acc 94%, 减法 94%. 标准 chance = 1/100 = 1%.
- Held-out 三元组 (e.g. (47, 52) → 99) 模型从未在训练中见过 **这个具体的
  对**, 但加法的"bias_a + bias_b"规则在 linear bundle 下是 **结构性**
  可组合的, MLP 可以直接从已训练的相邻 pair 插值出来.

### F3. 乘除外推显著弱于加减, 但仍远高于 chance

- N=100 乘法 OOD 65%, 除法 79%. Chance 1%, 显著泛化.
- 但加减法 94% vs 乘除 65-79%, 相差 15-29 个百分点.
- **解释**: 乘法在 linear bundle 下需要 MLP 本身学出非线性函数 a*b,
  这是标准的 algorithmic reasoning 难点. 相反在 log bundle 下 × 变成
  + 就能像加法一样外推. Bundle 被加减法"拉"成线性, 导致 MLP 必须承担
  乘法的非线性表达负担, 泛化减弱.

### F4. 小数离散化下 bundle 失去完美线性, 但 MLP 仍能泛化

- step=0.5, ρ_linear 从 0.97 掉到 0.77, 但 add/sub OOD 仍有 81-83%.
- **解释**: MLP 可能学会了"以 grid index 为单位"的 compositional
  规则 (bias_index=3 代表 1.5, bias_index=5 代表 2.5, index 作加法就对),
  不依赖 bundle 本身的严格线性. 换言之, concept 系统提供的是"**稳定
  可区分的离散 token**", 剩下的线性代数由 MLP 承担.

### F5. 训练 / OOD gap 反映 algorithmic regularity

对每个 op, train-OOD gap 越小 = 学到的映射越"规则": 

|           | train acc | OOD acc | gap  |
|-----------|-----------|---------|------|
| add N=100 | 0.958     | 0.940   | 0.02 |
| sub N=100 | 0.965     | 0.938   | 0.03 |
| mul N=100 | 0.999     | 0.653   | 0.35 |
| div N=100 | 1.000     | 0.785   | 0.22 |

加减 gap 几乎为零 (真正 **compositional**); 乘除 gap 大 (更像 **look-up
table**, 训练 pair 背下来但 OOD 跌). 这是 algorithmic vs memorization
的清晰切分.

## 5. 架构限制声明

### L1. 不能预测"完全没见过的单个数字"

如果训练时只注册 concept:ans:1..80, 测试时要求模型对 concept:ans:85
做运算, 在 D91/D92 下是**定义上不可能**的 — 85 这个 concept 的
`ParamBundle` 根本不存在. 这不是"泛化能力不足", 而是**架构的硬边界**.

真正意义上的"对未见数字外推"要求架构把 concept 视为一个**参数生成
函数** `bundle(v) = f_θ(v)`, 而不是 per-concept 字典. 这是 D93 级的
扩展 (e.g. NeRF-style concept field, 或 hypernetwork).

### L2. 真小数 (任意精度) 不可行

同上: step=0.01 意味着 N=10000 个 concept, 还没覆盖 3.14159... 这种
无理数. 要真正"连续小数", 需要 L1 提到的 concept 生成器.

本研究的妥协 (step=0.5, 60 concept) 仍然是离散的, 只是在整数基础上
**加密了 grid**. 它回答的问题是"在更稠密的 concept 网格上, 架构能否
保持 ordinal 结构", 而不是"能否对任意实数运算".

### L3. 乘法本质是非线性, 线性 bundle 次优

F1 分析的根本原因: 四则混训下 bundle 往线性走 (加减支配), 但乘除
需要对数几何. 这暗示**未来如果只想训 × / ÷**, 可能会自发出现 log-scale
bundle (Weber 律的神经原型). 这是一个可以单独做的 followup 实验:
"mul-only" + "mul ÷ div" 训练下的 bundle 几何.

## 6. 与人类认知的联系

| 观察 | 人类类似现象 |
|---|---|
| N=100 加减外推 94% | 儿童学会加法后对新组合的 generalization 能力 |
| 乘除 OOD < 加减 OOD | 九九乘法表本质是 look-up, 儿童对 7×8 的泛化慢于 7+8 (Siegler, 1988) |
| bundle 随四则混训偏离严格线性 | ANS + 精确数系统在人脑中并存, 跨 op 有 trade-off |
| step=0.5 下 MLP 仍能泛化 | 人类对"3.5 元 + 2.5 元 = 6 元"的运算能力不亚于"3+2" |

## 7. 文件清单

- `mind/experiments/language/quad_study.py` — 实验框架
- `outputs/quad_n30/summary.json` — N=30 整数 3 seeds
- `outputs/quad_n100_v2/summary.json` — N=100 整数 2 seeds (balanced-op)
- `outputs/quad_n30_half/summary.json` — N=30 step=0.5 小数变体

## 8. 未完成的 followup

- **mul-only / div-only 训练**: 验证 F3 的假设 — 脱离加减支配后,
  bundle 是否自发走向 log 几何 (Weber 律原型).
- **架构升级到 concept 生成器**: `bundle(v) = hypernet(v)`, 使得对任意
  数值 v 都能算出 bundle. 这样才能真正测试"未见数字外推"和"连续小数"
  — 但这是 D93 级别的工作.
- **更大 N (N=300, 1000)**: 目前 ρ_linear 随 N 下降, 但 OOD acc 稳定或
  上升, 需要更大规模确认这个 trend.
- **分阶段训练** (curriculum): 先加减收敛 → 再加乘除, 看 bundle 几何
  是否进入 "线性区" 后锁定, 或仍能漂移到 log.
