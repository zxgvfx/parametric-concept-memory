# Scale Study · 数字范围扩展与 Weber's law 检验 (2026-04-22)

> **任务**: 在 `purity_audit` 证明 "ordinal coherence 是任务结构驱动, 不靠
> encoder / id / 对齐几何" 之后, 探究下列问题:
>
> **Q1**. N 扩大到 15 / 30 时 coherence 是否稳定?  
> **Q2**. bundle 学出的是**线性**数字线还是**对数**数字线 (Weber's law)?  
> **Q3**. add-only / sub-only / add+sub mixed 训出的几何是否同构?
>
> **脚本**: `mind/experiments/language/scale_study.py`  
> **产出**: `outputs/scale_study/{summary.json, report.md}`

---

## 0. TL;DR

四条定量结论 (3 seeds × 3 setups × 3 scales):

1. ✅ **Coherence 随 N 不崩反升**. ρ_linear (mix): N=7→**0.966**, N=15→**0.987**, N=30→**0.991**.
2. ⚠️ **Linear number line, NOT Weber's law**. 所有 scale + 所有 setup 下 `ρ_linear > ρ_log`, 在 N=30 mix 差距 Δ=0.107 (0.991 vs 0.884).
3. ✅ **任务不变几何**. cross-setup `add↔sub` ρ: N=7→0.74, N=15→0.90, N=30→**0.94**. 大 N 下数字概念的几何几乎**与任务无关**.
4. ✅ **add-only 是 hard case**. 数据稀疏导致小 N 时 ρ 低且 variance 大, 但 N 扩展自动缓解 (N=7 ρ=0.76±0.11 → N=30 ρ=0.94±0.005).

**最有趣的定性发现**: 神经网络 inductive bias **≠ 生物 ANS inductive bias**. 
生物 ANS 是 log-scale (人类/猴子 Weber's law), 我们的 bundle 是 linear. 
原因: `a+b=c` 任务要求 `embed(a+b) = f(bias_a, bias_b)` 可加分解, 最简解是 
`bias_n ∝ n · direction + noise`, 这正是线性结构. 
log-scale 下 `bias_{n+1}` 离 `bias_n` 越来越近, `f` 必须学非平凡的非线性.

---

## 1. 实验设计

### 1.1 为什么能扩 N

`NumerosityEncoder` 只训练到 N=7 (outputs/ans_encoder/final.pt). 直接扩 N 
会让 encoder 给出的 centroids 对 8..30 失真. **解决**: 用 `purity_audit` A1 
已证明等价的 **random orthogonal centroids** — bundle ρ 与 encoder centroid 
完全相同, 但 centroid 本身不限 N. 每 seed 生成独立正交 centroids.

### 1.2 三个 op_filter setup

| setup | 数据分布 | 动机 |
|---|---|---|
| `mix`  | 每 batch 50% add, 50% sub | 主 baseline, 跟 robustness / purity audit 一致 |
| `add`  | 100% add (a+b=c) | 观察单一操作几何; 小 N 时数据不平衡 |
| `sub`  | 100% sub (a-b=c) | 对比 add; 检查 add/sub 对称性 |

### 1.3 训练量按 N 调整

| N | epochs | steps/epoch | 总 samples | 每 pair 平均曝光 |
|---|---|---|---|---|
| 7 | 12 | 120 | 46 080 | ~1 097 |
| 15 | 16 | 160 | 81 920 | ~390 |
| 30 | 24 | 200 | 153 600 | ~176 |

(每 pair 曝光保持 O(10²), 避免 N 大时欠拟合.)

### 1.4 Spearman ρ 的三种距离假设

对每对 $(i,j)$ 比较 $\cos(\text{bundle}_i, \text{bundle}_j)$ 与下列三种"距离"的 
Spearman 排名相关:

| 名称 | 表达式 | 数字线形态 |
|---|---|---|
| `rho_linear` | $-\|i-j\|$ | 等间距数字线 |
| `rho_log` | $-\|\log i - \log j\|$ | Weber's law (log-spaced) |
| `rho_sqrt` | $-\sqrt{\|i-j\|}$ | 亚线性 (power-law) |

> **注**: Spearman ρ 对单调变换不变 ⇒ `rho_sqrt ≡ rho_linear` 恒成立
> (因 $\sqrt{\cdot}$ 在 |Δn| 上单调). 报告里保留是为了 sanity-check.
> 真正有区分度的只有 **linear vs log**.

---

## 2. 主表

| N | setup | acc | ρ_linear | ρ_log | Δ(lin − log) |
|---|---|---|---|---|---|
| 7  | `mix` | **1.000** ± .000 | **0.966** ± .012 | 0.865 ± .024 | **+0.101** |
| 7  | `add` | 1.000 ± .000 | 0.762 ± .113 | 0.731 ± .157 | +0.031 |
| 7  | `sub` | 1.000 ± .000 | 0.921 ± .037 | 0.777 ± .008 | +0.144 |
| 15 | `mix` | 1.000 ± .000 | **0.987** ± .001 | 0.905 ± .004 | **+0.082** |
| 15 | `add` | 1.000 ± .000 | 0.877 ± .017 | 0.800 ± .016 | +0.077 |
| 15 | `sub` | 0.996 ± .003 | 0.985 ± .009 | 0.815 ± .005 | +0.170 |
| 30 | `mix` | 0.998 ± .001 | **0.991** ± .001 | 0.884 ± .008 | **+0.107** |
| 30 | `add` | 0.963 ± .056 | 0.936 ± .005 | 0.850 ± .021 | +0.086 |
| 30 | `sub` | 0.976 ± .013 | 0.987 ± .004 | 0.768 ± .008 | +0.219 |

**三条可读结论**:

- **mix ρ_linear**: 0.966 → 0.987 → 0.991 **单调上升** (N↑ 约束越强)
- **Δ(lin − log)**: 所有 9 个 cell 都正, 无一例外 → **linear 碾压 log**
- **sub-only 比 add-only 稳定**: 见 §4.2.

## 3. 任务不变几何 (cross-setup cos-matrix ρ)

两个 setup 训出的 bundle 平均 (先对 seed L2-norm 再平均) 后, 它们的 cos 
matrix 跨 setup 的 Spearman ρ:

| N | mix ↔ add | mix ↔ sub | add ↔ sub |
|---|---|---|---|
| 7  | 0.761 | 0.933 | 0.736 |
| 15 | 0.928 | 0.962 | 0.903 |
| 30 | **0.938** | **0.954** | **0.937** |

**推论**: N 扩大后 **add-only 和 sub-only 训出的 bundle 几何 ρ=0.94 同构** — 
数字概念的几何是 task-invariant 的, 跟用什么操作把 ordinal 诱导出来几乎无关.

---

## 4. 逐点解读

### 4.1 没有 Weber's law — 为什么重要?

**Weber's law** (认知科学): 人/猴/鸽子对 numerosity 的辨别能力随两数之比恒定, 
等价于"log-spaced 数字线". 我们的 bundle **违反** Weber's law, 用 linear 
距离. 解释:

`ArithmeticHeadV2` 的前向本质是:
$$
\text{pred}_{c} \;=\; \text{MLP}\bigl(\text{bias}_a,\, \text{bias}_b,\, \text{op}\bigr)
$$
监督信号要求 $\text{pred}_c \approx \text{centroid}_c$ (random, 无序). 
梯度驱动 bundle 形成"可加法分解"的结构:

- **Linear 解**: $\text{bias}_n \approx n\cdot \vec{u} + \varepsilon_n$. 
  $\text{MLP}$ 只要学线性映射 $(\text{bias}_a + \text{bias}_b) \to \text{centroid}_{a+b}$ 
  就能解 task. **最简**.
- **Log 解**: $\text{bias}_n \approx \log(n)\cdot \vec{u}$. 
  要解 $a+b=c$, MLP 必须学 $\log^{-1}\!\bigl(\log a + \log b + \Delta\bigr)$ 
  — 非线性且不存在闭式. **贵**.

神经网络的 Occam 偏好 (implicit bias + L2 weight decay) 选择**线性解**. 
这与生物 ANS 不同, 但符合工程直觉.

**学术意义**: 这是一个**神经网络 inductive bias ≠ 生物 inductive bias** 的 
具象例证. 未来 paper 可以作为一个 "where ANN and brain differ" 的微型案例 
(对比 Dehaene 的 `log-scale NN finding` 等工作).

### 4.2 add-only 的 high variance 从哪来

add-only 下目标 $c$ 的分布极不均衡 (组合数 decreasing for small c):

| c | (a,b) 组合数 (N=7) | 概率 |
|---|---|---|
| 2 | 1 (1+1) | 1/21 ≈ 4.8% |
| 3 | 2 | 9.5% |
| 4 | 3 | 14% |
| 5 | 4 | 19% |
| 6 | 5 | 24% |
| 7 | 6 | 29% |

→ `concept:ans:1` 只在 `c=2` (a 或 b 为 1) 出现, 训练信号稀疏, 
`bias_1` 学得不好 → ρ 下降且 variance 大.

N 扩大后 *相对* 不平衡缓解 (N=30 下 c=2 仍只有 1 种, 但 c=30 有 29 种, 总 pair 435 个), 
稀疏的 extreme 点占比降低, ρ 恢复到 0.94.

→ **这个现象本身是可预测、可解释的**, 不挑战 H5' 主结论.

### 4.3 sub-only 为什么比 add-only 稳

sub-only 的分布相对均衡: 
$c = a-b$, $a \in [2, N]$, $b \in [1, a-1]$, 
每个 $c \in [1, N-1]$ 的组合数都 = N - c (递减但不归零), 且每个 `concept:ans:N` 
都会作为 a 出现至少 N-1 次 (a=N 时任何 b 都满足), `concept:ans:1` 也会作为 b 
出现频繁. 所以 extreme concept 的曝光远高于 add-only. ρ=0.99 也不奇怪.

### 4.4 acc 在 N=30 下降为什么没影响 ρ

N=30 add: acc=0.963 ± 0.056 (有 1 seed 掉到 0.899), 但 ρ=0.936 ± 0.005 
(variance 比 acc 低一个量级).

**解读**: bundle 的 ordinal 结构**比 task accuracy 更早收敛**. 即使 head 
还没把最后 5% 的难样本学会, bundle 的 geometry 已经到位. 这是因为梯度对 
bundle 的 ordinal 约束来自**所有** pair 的 loss, 不只是困难 pair.

## 5. 与前序 finding 的关系

| 文档 | 原结论 | 本次 Scale Study 带来的 |
|---|---|---|
| `SINGLE_VS_DUAL_MUSCLE_FINDING.md` | N=7: single ρ=0.915, dual ρ=0.892, no diff | **上推到 N=30**: single (mix) ρ=0.991, 仍无需 dual muscle |
| `robustness_study` E3 | N ∈ {5,7,9} ρ 稳定 | **N ∈ {7,15,30} ρ 仍稳定**, 且 N↑ 反而更强 |
| `purity_audit` A1 | random centroid ρ ≡ encoder centroid | 本研究全部用 random centroid, 同结论 |
| `purity_audit` A3 | small init → ρ=0.97, large init → ρ=0.22 | 未重复 A3, 默认 small init |

## 6. 复现

```bash
# 完整 (3 scales × 3 setups × 3 seeds, ~4 min on A5000)
python -m experiments.scale_study --n-seeds 3

# 只看 N=30
python -m experiments.scale_study --n-seeds 3 --ns 30

# Smoke test (N=7 × 1 seed × 3 setups, ~15s)
python -m experiments.scale_study --smoke
```

产物: `outputs/scale_study/{summary.json, report.md}`.

## 7. 后续建议

| # | 实验 | 目的 | 预期 |
|---|---|---|---|
| S1 | N ∈ {50, 100, 200} 继续扩 | 找 coherence 崩塌阈值 | 需加大 epochs, 或出现 capacity limit |
| S2 | 换任务: 把 arithmetic 替换成 **非加性** 操作 (e.g. XOR 7-class permutation) | 验证 linear number line 是 arithmetic-specific 还是通用 | 预计 ρ 崩, 几何变随机 |
| S3 | 注入 Weber-like noise (训练时 label 有 $\propto \log n$ 的 jitter) | 人工诱导 log-scale | 预计 ρ_log > ρ_linear |
| S4 | 在 bundle 前加一个 "cognitive log-prior" (BatchNorm over bundle) | 看能否把 linear 结构压成 log | 可能产出 Weber-compatible bundle |
| S5 | 比较 MLP head 和 Transformer head | 看 inductive bias 是否源于 MLP | Transformer 可能给出非线性数字线 |

---

## 8. 变更日志

| 日期 | 变更 |
|---|---|
| 2026-04-22 | 初版. N ∈ {7, 15, 30}, 3 setups, 3 seeds. 发现 linear number line, cross-setup invariance, add-only 的数据稀疏效应. |
