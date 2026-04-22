# Compositional Number Study (D93a 原型, 2026-04-22)

**TL;DR**

- 提出 **D93a**: 用 10 个 digit ConceptNode + 隐式 slot 位置合成任意数字的
  bundle, 替换 D91 flat 的 per-integer ConceptNode 字典.
- 单独用 D93a **composer** 不够, 必须配**slot-equivariant head** (类似
  硬件 ripple-carry adder 的 shared slot MLP + carry 传递). 二者合起来
  才支持真·外推.
- 在训练 a, b ∈ [1, 999] 的 3 位数加减, 测试 a, b ∈ [1000, 4999] 的 4 位
  数加减时, **3 seeds 全部达到 100% / |err|=0** — 这是 D91/D92 架构史上
  第一次"**未见过的数字组合**"的决定性外推.
- 更惊人的: 训练只见 [1, 99], digit×position 覆盖严重不完整时,
  slot-equivariant head 的 inductive bias 仍能把 add 外推到 [100, 299]
  达到 92-96% — 证明**正确的 equivariance 能绕过 combinatorial data
  coverage 要求**.

---

## 1. 动机

前期研究揭示 D91/D92 的一个**硬架构边界**: `ConceptNode` 是 per-concept
参数字典, 未注册的 concept 根本没 bundle 对象, 因此不能外推到 training
时从未见过的数字 (见 `QUAD_STUDY.md` §5).

"数字 20 是不是和 0-9 共享某种图谱结构?" — 用户的这个问题精准命中 D91 的
局限. 答案当时是: **不**, 20 是独立 atom 和 7, 83 平起平坐, 彼此没有
结构性联系.

D93a 首次让"20 = digit[2] at slot 1 + digit[0] at slot 0"成为 architectural
reality.

## 2. 架构

### 2.1 `DigitPosComposer` (位值合成器)

```
ConceptGraph 内只有 10 个 atom: concept:digit:0 ... concept:digit:9
每个 digit ConceptNode 拥有一个 per_digit_dim 维的 ParamBundle
bundle(N) = concat_over_i( digit_bundles[d_i(N)] )
           where d_i(N) = (N // 10^i) % 10
           result dim = max_pos × per_digit_dim   (= bias_dim)
```

- **Slot-by-position 填充**: slot i 放 N 在第 i 位的 digit 的 bundle.
  structural disentanglement 内建在架构里 — position 由 slot index 区分,
  digit_bundles 必须学 position-invariant 语义.
- 实验中使用 max_pos ∈ {3, 4}, per_digit_dim ∈ {24, 16}, bias_dim ∈ {72, 64}.

**和 Hadamard 版本对比**: 最初尝试 `bundle(N) = Σ_i digit[d_i] ⊙ pos[i]`,
但 pos_bundles 初始化近 1 导致 `bundle(12) ≈ bundle(21)`, 梯度 hard
打破对称. Slot concat 天然 non-commutative, 训练更快收敛.

### 2.2 `SlotEquivariantHead` (slot-共享 + carry)

```
carry = 0                          (carry_dim = 8)
for i in 0 .. max_pos-1:
    a_i = bias_a[slot i]
    b_i = bias_b[slot i]
    x   = concat(a_i, b_i, op_onehot, carry)
    y   = shared_MLP(x)            (same MLP across all slots)
    digit_out_i = y[:per_digit_dim]
    carry       = y[per_digit_dim:]
pred = concat(digit_out_0, ..., digit_out_{max_pos-1})
```

- **等价于硬件 ripple-carry 加法器**: slot-shared 运算规则 + 向高位
  单向传 carry.
- 关键 inductive bias: 训练时 slot 0 学会的 add rule **自动 transfer**
  到 slot 1..max_pos. 不需要每个 (digit, slot) 组合都见过.
- Carry 维度 8 给 borrow (减法) 和 overflow (加法) 留余量.

### 2.3 Loss: self-referential contrastive CE

```
candidates = [composer.compose(c) for c in 0..C_max_train]
pred_n, cand_n = L2_normalize(pred), L2_normalize(candidates)
logits = pred_n @ cand_n.T * temperature
loss = CE(logits, true_c_index)
```

- Target 也走 composer, 所以 digit_bundles 和 head 联合优化.
- 避免 degenerate (bundle 全塌缩) 的机制: softmax over C_max_train 个类,
  模型必须让相邻数字的 bundle 有区分度.
- Candidate set 在评估时扩展到 c_max_eval = 9998 (9999), 不需要训练见过
  target c 就能 score.

## 3. 实验矩阵

两个关键变量:

- **Composer**: slot-based (本文方法)
- **Head**: flat MLP vs slot-equivariant MLP + carry
- **Train 范围**: default [1, 99] vs extended [1, 999]

3 seeds per config, 30 epochs × 250 steps, 约 150s/seed on GPU.

### 3.1 主表

| Head | Train range | max_pos | train | L0 (in-range pair-OOD) | L1 (one-out-of-range) | L2 (both-out-of-range) |
|---|---|---|---|---|---|---|
| flat | [1, 99] | 3 | 0.997 | 0.988 | 0.003 | 0.112 |
| flat | [1, 999] | 4 | 0.976 | 0.971 | 0.001 | 0.011 |
| **slot** | **[1, 99]** | **3** | **1.000** | **1.000** | **0.926** | **0.964** |
| **slot** | **[1, 999]** | **4** | **1.000** | **1.000** | **1.000** | **1.000** |

### 3.2 Per-op breakdown (slot, default)

Slot head 训练 [1, 99]:

| | add L1 | add L2 | sub L1 | sub L2 |
|---|---|---|---|---|
| seed 0 | 1.000 | 1.000 | 0.801 | 0.895 |
| seed 1 | 1.000 | 1.000 | 0.760 | 0.879 |
| seed 2 | 0.984 | 1.000 | 0.800 | 0.893 |
| mean | 0.995 | 1.000 | 0.787 | 0.889 |

加法即使在 minimal training ([1,99]) 下外推到 [100, 299] **几乎完美**.
减法稍弱 (~80-90%), 怀疑 borrow propagation 比 carry 更难 learn 仅从
小范围训练. Extended training 下减法也达到 100%.

## 4. 核心发现

### F1. D91 flat 到 D93a atom: **100 个 atom → 10 个**, 且能外推

D91 要求 N 个独立 ConceptNode 表达 [1, N]. D93a 只用 10 个 digit atom
(外加一个 slot scheme), 配合正确 head, 能表达 O(10^max_pos) 个数字.
ConceptGraph 规模从 O(N) 降到 O(log N × 10).

### F2. Composer 必要但不充分

slot-based composer + flat head: L1/L2 全线崩溃 (0-11%). 原因: flat MLP
的 weight 对每个 slot 的每个 dim 是 specific 的, 训练时 slot 3 只见过
digit[0], digit[1] 激活, 测试时 digit[2..9] 激活等价于**分布外 input**.
即使 digit_bundles[2..9] 本身在 slot 0 训练良好, head 这层 kill 了
transfer.

### F3. Slot-equivariant head 是真正的外推关键

**架构 equivariance > 数据 coverage**. slot + extended [1,999] 达到
100%. 更重要的是 slot + [1,99] 训练下 add 仍 ~100% 外推到 [100, 299],
这时 digit[2..9] at slot=2 **从未联合激活过**. 通过 shared slot MLP,
"digit[5] 在 slot 0 的加法行为" 的训练信号**自动推广**到 slot 2, 绕过
了 combinatorial coverage 要求.

这跟 Lake (2017, SCAN) 及后续工作指出的"systematic generalization 需
要 structural inductive bias" 的结论一致, 但 Percept 把它 grounded 到
ConceptGraph 的 concrete architecture 里.

### F4. 架构类比: ripple-carry ALU

`SlotEquivariantHead` 的 carry 传递与硬件加法器几乎一比一. 8 维 carry
承载进位/借位/overflow 信号. 这暗示 neural compositional reasoning 架构
可能本质上需要硬件级别的**算法不变性 (algorithmic invariance)**, 而不
仅仅是 statistical pattern matching.

### F5. 与 LLM 数字推理的对比

GPT-4 类模型在"9999 + 1" 或"多位大数乘法" 这种 edge case 上经常失败,
因为它们学的是 tokenwise next-digit prediction, 对 digit-pos 没有
equivariance. D93a + slot head 只用 10 个 digit atom + ~10k training
triples 就达到 perfect extrapolation, 说明**正确的 inductive bias 能
戏剧性降低数据需求**.

## 5. 与 D91/D92 的哲学关系

- **D91 (parametric concept memory)**: 每 concept 一套参数. D93a 保留这个
  框架, 只是让"concept" 从 atomic integers 退回到 atomic digits.
- **D92 (contextual collapse)**: D93a 下每个 digit ConceptNode 仍然响应
  caller 和 facet. 本实验没显式测 dual-muscle, 但 shared digit bundles
  可以在多个 caller (head) 下被不同 facet 使用, 完全兼容.
- **D93a (positional composition)**: 首次在 ConceptGraph 框架里引入
  **structural composition**. concept N 的 bundle 不再是独立 row in a
  lookup table, 而是 10 个 primitive 的 programmatic composition.
- **D93b (predicted)**: bundle 生成器 `f_θ(v)`, 对任意实数 v 都能 ad hoc
  生成 bundle. D93a 需要 digit decomposition (离散 base), D93b 可能用
  neural field / sinusoidal positional encoding 支持真正的连续数.
- **D93c (predicted)**: 递归 concept — `ConceptNode` 引用其他 ConceptNode
  作为 parameter. 对 "双手", "一对" 等 compositional 普通名词也有用.

## 6. 架构限制

### L1. max_pos 上限

当前 slot composer 固定 max_pos=4, 支持 0..9999. 要处理更大数字需增
max_pos, 但 slot MLP 本身 slot-invariant, 所以**改 max_pos 无需 retrain**
— 只要 carry 机制 generalizing 到更多 slots. 这是一个 followup test.

### L2. Subtraction sub-optimal

default 模式下 sub L1=80% 而 add L1=100%. 怀疑 borrow 的 bidirectional
chain (low slot 可能需要 "problematic" borrow cascade 回高位) 更难 learn
from [1, 99] 训练. Extended 模式下 100% 说明 subtraction 需要更广范围
训练 才能 stabilize borrow dynamics.

### L3. 未测乘除

slot-equivariant 的 carry rule 对加减直观, 但乘法涉及 **partial products**
(a × b 分解为 Σ_i a × d_i(b) × 10^i, 类似小学笔算), 需要 2D slot
grid + accumulator. 除法更难. 这是 future work.

### L4. 只处理非负整数, 不支持小数

decimal point 本身需要一个特殊 slot 或 "小数点 concept", 架构需扩展.
同时 floating-point precision 问题出现.

## 7. 实验复现

```bash
# Baseline flat head, default range  
python -m experiments.compositional_number_study \
    --n-seeds 3 --head flat --epochs 30 \
    --out outputs/compositional_flat_default

# Slot head, default range ([1, 99])
python -m experiments.compositional_number_study \
    --n-seeds 3 --head slot --epochs 30 \
    --out outputs/compositional_slot_default

# Slot head, extended range ([1, 999])
python -m experiments.compositional_number_study \
    --n-seeds 3 --head slot --extended --epochs 30 \
    --out outputs/compositional_slot_ext_full
```

## 8. 文件

- `mind/experiments/language/compositional_number_study.py` — 完整实验框架
- `outputs/compositional_flat_ext_full/summary.json` — flat+extended baseline
- `outputs/compositional_slot_ext_full/summary.json` — slot+extended 关键结果 (100% × 3 seeds × 4 levels)
- `outputs/compositional_slot_default/summary.json` — slot+default (信息论上的
  hard case, 证明 equivariance 能绕过 coverage 要求)

## 9. 紧接着的 followup

1. **max_pos 外推**: 训练 max_pos=4, evaluate 时切换 max_pos=5 不 retrain,
   看 slot head 的 carry rule 是否真正 position-free.
2. **乘法 slot head**: 扩展 carry → 多 lane + partial products.
3. **融入主 ConceptGraph**: 把 10 digit atoms 注册为正式的 ConceptNode,
   让它们与 `concept:ans:N` 共存 (D93a + D91 wire-together 实验).
4. **D93b 连续 concept**: 当 bundle 由 `hypernet(value)` 生成时, 是否
   出现类似 slot-head 的 inductive bias 需求?
5. **与 Transformer 数字推理的量化对比**: 同样的 train/test 分布下 vs
   Transformer baseline 的外推 accuracy.
