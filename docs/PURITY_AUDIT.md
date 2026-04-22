# PURITY AUDIT — D91/D92 实验环境纯净度检测 (2026-04-22)

> **目的**: 在 `SINGLE_VS_DUAL_MUSCLE_FINDING.md` 和 `robustness_study` 给出
> 强结论 (单 muscle 已足够产生 ρ≈0.97 ordinal coherence) 之后, 必须排查
> 实验环境是否存在"隐藏的共享信息通道", 让结论看起来比实际更干净.
>
> **产出**: `mind/experiments/language/purity_audit.py` 一键可复现 +
> `outputs/purity_audit/{summary.json, report.md}`.

---

## 0. TL;DR

扫码后识别 **9 个潜在的信息渠道**, 对最可疑的 4 个做了 5-seed ablation
(A1-A4). **3 个清白, 1 个揭示了一个**真正的**科学发现**:

| 渠道 | 风险评估 | 测试 | 结果 | 判定 |
|---|---|---|---|---|
| **C1** encoder 预训练 (ordinal 先验 → centroid) | 高 | **A1** 随机正交/高斯 centroid | 单 ρ=0.97, 双 ρ=0.96 | ✅ **清白** |
| **C4** `concept:ans:N` id 里的数字 | 需验证 | **A4** 随机 UUID 作 id | ρ=0.98 | ✅ **清白** |
| **C8** E2 的 shuffle 是 identity 破坏还是坐标破坏 | 高 | **A2** shuffle-inverse remap | inv ρ=0.97 vs nat ρ=0.12 | ⚠️ **重要反思** |
| **C6** `normal_small` init (std=0.01) | 需验证 | **A3** init scale 三档 | std=0.01/0 → ρ=0.97; std=1 → ρ=0.22 ± 0.21 | ⚠️ **真的依赖** |

**最关键的两个结论**:

1. **Ordinal coherence 不是 encoder 泄漏给 bundle 的** (A1).  
   即使 centroid 用纯随机正交向量 (cos ≈ 0, 没有任何 ordinal 结构),
   bundle 仍然在 arithmetic 任务驱动下自发形成 ρ≈0.97 的数量序.  
   → "任务结构驱动 identity 涌现" 是正确的, **与 encoder 的内禀 ordinal 无关**.

2. **Ordinal coherence 依赖 small-init 的 implicit bias** (A3).  
   `normal_small` (std=0.01) 和 `zero` 都给出 ρ=0.97;
   `normal` (std=1.0) 只给 ρ=0.22 ± 0.21 — 虽然 **acc 仍 ≈ 1.0**.  
   → 大初始化下网络直接"查表" (把 bundle 当 random feature), 不需要诱导 ordinal.  
   → 这不是 bug, 是 **implicit bias (AdamW + L2 + small init) 引导出的 low-complexity 解**.
   论文里应当明确声明这一点.

---

## 1. 所有潜在共享信息通道 (完整审计清单)

扫描路径:

```
mind/core/cognition/language/{param_bundle,concept_graph}.py
mind/core/cognition/language/visual/{numerosity_encoder,arithmetic_head_v2,comparison_head}.py
mind/experiments/language/{_graph_builder,robustness_study}.py
```

| # | 通道名 | 具体位置 | 泄漏方向 | 严重性 | 测试 |
|---|---|---|---|---|---|
| C1 | **预训练 NumerosityEncoder 权重** | `outputs/ans_encoder/final.pt` (contrastive + ordinal reg) | encoder state → centroids (带 ordinal 几何) → arith loss → bundle.params | 高 | **A1** ✅ |
| C2 | **centroid 固定 `manual_seed(0)`** | `_compute_centroids` | 所有训练 seed 共用同一 centroids | 中 | 声明 (非 bug) |
| C3 | **CE loss target 是 `c - n_min`, 本身 ordinal** | `train_one` | ordinal classification label | 中 | 内置 (即 arithmetic task 本身) |
| C4 | **`concept:ans:N` ID 里的数字** | `build_ans_graph`, muscle forward | 若代码 split 排序即泄漏 | 需验证 | **A4** ✅ |
| C5 | **Python dict 插入顺序** | `build_ans_graph` `range(n_min, n_max+1)` | 影响 optimizer param 顺序 | 低 | 理论无关 (Adam 无 state-order effect) |
| C6 | **init strategy `normal_small`** | `_init_parameter` | 所有 bundle 起点近零 | 需验证 | **A3** ⚠️ |
| C7 | **warmup collapse at tick=0** | `train_one` warmup | 全按 range 顺序, lazy-init 固定 | 低 | 理论无关 |
| C8 | **shuffle 是 identity 破坏还是坐标破坏** | E2 `shuffle_map` | — | 高 | **A2** ⚠️ |
| C9 | **encoder loss 本身带 `ordinal_weight=0.3`** | `contrastive_ordinal_loss` | encoder 设计就是学序 | 合法 | 已由 A1 间接覆盖 |

---

## 2. 四路 Ablation 设计与结果

### A1 · Random-Centroid (5 seeds × 2 centroid types)

**设计**: 用**随机正交单位向量** (或独立高斯单位向量) 代替 encoder 计算的
`centroids`, 完全剥离 encoder 的 ordinal 几何信息作为 supervision.

> 正交 centroids 的 cos 矩阵近似为 $I$: 不同 numerosity 之间**没有任何**
> ordinal/相似性信号. 如果 ρ 崩塌, 证明 coherence 来自 encoder; 如果 ρ
> 仍然 >0.9, 证明 coherence 来自 arithmetic 任务结构.

**结果**:

| centroid type | single \|ρ\| | dual \|ρ\| | dual \|ρ_ord\| |
|---|---|---|---|
| Random orthogonal | **0.9669 ± 0.0057** | **0.9628 ± 0.0037** | 0.9107 ± 0.0155 |
| Random gaussian   | **0.9689 ± 0.0054** | **0.9704 ± 0.0045** | — |

**判定**: ✅ 两种随机 centroid 下 ρ ≈ baseline (0.97), 对比 encoder centroid
结果几乎没区别. **encoder 不是 coherence 的来源**.

这是一个非常漂亮的结果: 即使 supervision 完全没有数量序提示
(正交 centroids), arithmetic 任务本身 (a+b 的 compositional structure) 迫使
bundle 自己"发明"数量序来解释任务.

### A2 · Shuffle-Inverse (5 seeds)

**设计**: 跟 E2 一样训练 (把 `concept:ans:n` 映射到随机 `concept:ans:sm[n]`
bundle), 但 eval 时**同时测两种** ρ:

- `natural-order ρ`: 按 `n=1..7` 直接取 bundle — 对应 E2 原结论
- `inverse-remapped ρ`: 按 `sm[n]` 取 bundle, 即把 shuffle 还原

**动机**: E2 的 ρ=0.18 可能有两种解释 —  
  (i) shuffle 破坏了 identity 学习 (bundle 没学到序)  
  (ii) shuffle 只是打乱了 (id↔bundle) 对应, bundle 实际学到了序, 只是坐标错位

**结果**:

| metric | mean ± std |
|---|---|
| natural-order \|ρ\| | 0.1228 ± 0.0562 |
| **inverse-remapped \|ρ\|** | **0.9711 ± 0.0074** |

**判定**: ⚠️ 反事实被**修正**: shuffle **并不破坏** identity 学习, 只是
**打乱了坐标系**. Bundle 仍然形成了完美的数量序 (ρ=0.97), 只是"住错了房间".

**对原 E2 结论的修正**:
- **原叙述**: "E2 的 ρ=0.18 证明 identity 是必要的" (弱结论)
- **修正叙述**: "无论 id 怎么重排, bundle 都会忠实地按训练时被用作的数量学出
  正确的序 — **identity 是 task-imposed, 不是 id-imposed**" (更强结论)

这实际上是 **H5' 的加强版**: concept identity 是 **任务驱动、坐标无关** 的.

### A3 · Init-Scale (5 seeds × 3 strategies)

**设计**: 同样 single-muscle arithmetic, 但 ParamBundle init 三种:

- `normal_small`: $\mathcal{N}(0, 0.01^2)$ (baseline)
- `normal`:       $\mathcal{N}(0, 1.0^2)$
- `zero`:         全零

**动机**: 怀疑 `normal_small` 的"近零起点 + 小扰动" 可能是 trivial 诱因
(bundles 初始几乎相同, 任何梯度都会把它们分开到任务要求的方向).

**结果**:

| init | \|ρ\| mean ± std | add acc |
|---|---|---|
| `normal_small` (std=0.01) | **0.9752 ± 0.0017** | 1.000 |
| `normal` (std=1.0)        | **0.2165 ± 0.2126** | 0.993 ± 0.010 |
| `zero`                    | **0.9745 ± 0.0019** | 1.000 |

**判定**: ⚠️ **这是一个真正的发现, 不是 bug**:

- `normal_small` 和 `zero` 都给出 ρ ≈ 0.97 — 说明"小初始化"普遍诱导 coherence.
- `normal` (std=1.0) 只给出 ρ ≈ 0.22, 然而 **accuracy 依然 99.3%** — 任务学会了,
  只是**没有**诱导 ordinal 结构.

**解释 (two interpretations)**:

1. **Lazy-regime / NTK-like**: 大 init 下 bundle 已是 high-dimensional random
   feature, 下游 MLP 可以直接作为 lookup table 拟合 arithmetic, 不需要改 bundle.
   → ρ 低但 acc 高, 完全合理.
2. **Feature-learning regime**: 小 init 下 MLP 表达能力不够, 必须让 bundle 学出
   "有用的 features", 最节省的 features 就是 ordinal 结构.
   → ρ 高.

**对 D91/D92 claim 的影响**:

- ✅ Coherence **不是** trivial (zero init 也给出 ρ=0.97, 证明它不是"初始扰动
  被放大")
- ⚠️ Coherence 的**涌现**依赖 implicit bias (small init + L2 + AdamW).
  这是 neural network **结构性**的特点, 不是数据泄漏.
- 📝 论文里必须声明: "ordinal coherence 在 lazy regime 下不涌现" — 这**本身**
  是一个有趣的 feature-learning / lazy regime 区分的例子, 和 NTK 理论的实证一致.

### A4 · Random Concept-ID (5 seeds)

**设计**: 把 concept_id 从 `concept:ans:1..7` 改成 `concept:opaque:<random_12hex>`,
保持 $(n \leftrightarrow id)$ 一对一映射.

**动机**: 排除 "ID 字符串本身的数字序列泄漏信息" (比如代码 split id 用到数字).

**结果**:

| metric | mean ± std |
|---|---|
| \|ρ\| | **0.9752 ± 0.0017** |
| acc   | 1.000 ± 0.000 |

**判定**: ✅ ID 完全可以用任意字符串, **ID 字符串没有泄漏信息**, identity
纯粹由训练时 `(n → bundle)` 的对应关系建立.

---

## 3. 综合影响

### 3.1 对 `SINGLE_VS_DUAL_MUSCLE_FINDING.md` 的结论修订

> 原结论: "single muscle 已足够产生 ρ≈0.97 ordinal coherence; dual muscle 无显著增益; H5 被证伪, H5' 被支持."

**修订后的更完整陈述**:

> 在 **small-init + gradient-based training** 的 implicit bias 下, 单 arithmetic
> muscle 任务足以诱导 ρ≈0.97 ordinal coherence; 该 coherence:
> 1. **不依赖 supervision target** 的 ordinal 几何 (A1)
> 2. **不依赖 concept ID 字符串** (A4)
> 3. **不依赖 (id ↔ bundle) 对应顺序** (A2): bundle 会按**实际被用作的数量**
>    学出序, 不管 id 怎么重排
> 4. **依赖 optimizer implicit bias** (A3): 大 init (lazy regime) 下不涌现,
>    但 arithmetic 任务仍能被解出 — 说明 coherence 是 feature-learning regime
>    的产物, 不是解任务的必要条件

### 3.2 对 PoC 工程层的建议

| 修改 | 位置 | 理由 |
|---|---|---|
| 文档化 init 依赖 | `WORKSHOP_PAPER_OUTLINE.md` | A3 的发现必须声明 |
| `robustness_study.py` E2 的叙述 | `SINGLE_VS_DUAL_MUSCLE_FINDING.md` | A2 修订 E2 结论 |
| `_compute_centroids` 增加 `seed` 参数 | `robustness_study.py` | 消除 C2 的共享 |
| workshop paper: 增 "Robustness to centroid and id permutation" 小节 | paper outline | A1/A2/A4 的发现值得正面陈述 |

### 3.3 三个"清白"渠道的明确排除

- ❌ C1 encoder 污染: **已排除** (A1)
- ❌ C4 ID 泄漏: **已排除** (A4)
- ❌ C8 shuffle-as-identity-break: **已排除**, shuffle 只是坐标变换 (A2)

### 3.4 两个"结构性依赖" (非 bug, 但要声明)

- ✔️ C6 init scale: `normal_small` 和 `zero` 给出 coherence, 大 init 不给
  — 这是 implicit bias, 不是数据泄漏. 应在 paper 明示.
- ✔️ C3 arithmetic task 的 target 本身是 ordinal: 这是 `a+b` 的内禀 —
  如果任务换成非 ordinal (比如 concept 之间的任意 bijection), A1 的结论
  应当失效. 留作 future work.

---

## 4. 复现

### 4.1 完整运行

```bash
python -m experiments.purity_audit \
    --encoder-ckpt outputs/ans_encoder/final.pt \
    --n-seeds 5 \
    --out outputs/purity_audit
```

预计 4 分钟 (单 A5000). 产出:

- `outputs/purity_audit/summary.json`: raw per-seed + stats
- `outputs/purity_audit/report.md`: 自动生成简报

### 4.2 仅跑某几个

```bash
# 只跑 A1 + A2 + A4 (快速验证数据泄漏相关的 3 项, 跳过 init-scale)
python -m experiments.purity_audit \
    --encoder-ckpt outputs/ans_encoder/final.pt \
    --skip a1b a3
```

### 4.3 Smoke test (1 seed)

```bash
python -m experiments.purity_audit \
    --encoder-ckpt outputs/ans_encoder/final.pt \
    --n-seeds 1 \
    --out outputs/purity_audit_smoke
```

---

## 5. 后续可做 (未在本轮执行)

- **A5 Random-encoder**: 用**完全随机权重**的 encoder 重算 centroids 再重训.
  是 A1 的更激进版本 — 但因为 A1 用 random orthogonal/gaussian 已经剥离了所有
  encoder-derived 信息, A5 的新增信息量有限, 暂缓.
- **A6 Non-ordinal task**: 把 arithmetic 换成"任意 bijection" (e.g. XOR-like
  permutation on 7 classes), 看 ρ 是否降为 ~0. 这能明确 **coherence 来自任务
  的 ordinal 结构本身, 而非学习动力学本身总是诱导序**.
- **A7 Facet 间相关性**: 测 `arithmetic_bias` 和 `ordinal_offset` 在 dual
  training 时是否通过共享 `concept_id` 产生相关 (D92 H5' 核心预言之一).
- **A8 Finer-grained init sweep**: std ∈ {0, 0.001, 0.01, 0.1, 0.3, 1.0} 给出
  **相变曲线**, 找出 lazy↔feature-learning 转变点.

---

## 6. 变更日志

| 日期 | 变更 |
|---|---|
| 2026-04-22 | 初版. 全量 9-channel audit + 4 ablation (A1-A4). 发现 encoder/id/shuffle 清白, init scale 是结构性依赖 (有趣 lazy-regime 现象). |
