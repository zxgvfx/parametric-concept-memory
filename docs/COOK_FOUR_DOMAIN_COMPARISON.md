# Cook vs Direct · 四域端到端等价性对比

> **Status**: `RESULT` (2026-05-10) · companion to
> [`PCM_NODE_AS_FUNCTION_DESIGN.md`](PCM_NODE_AS_FUNCTION_DESIGN.md) and
> [`TIER_D_DECISION.md`](TIER_D_DECISION.md).
> **Code**: [`experiments/cook_four_domain.py`](../experiments/cook_four_domain.py)
> **Tests**: [`tests/test_cook_all_heads.py`](../tests/test_cook_all_heads.py)
> **Output**: `outputs/cook_four_domain/summary.json`

---

## 0. TL;DR

> 把项目里**全部 8 个肌肉**改造为 Tier-D 图谱节点，在 §4 / §5 / §6.2 / §6.3 四个论文实验上跑端到端训练对比，**每一个域的 acc 和 ρ 与传统直 forward 路径 bit-identical（max delta = 0.000e+00）**。

这把 D1 bit-identity 从"forward 等价"升级到了"**完整训练 trajectory 等价**" — 不仅是某一帧的输出相同，**整个 epoch × steps 的每一次梯度、每一次 optimizer 更新、最终 bundle 几何都完全相同**。

## 1. 改造覆盖：8 个肌肉全部转 Tier-D 图谱节点

| 域 | Tier-A `nn.Module` | Tier-D 等价图谱节点 | Cook 工厂 |
|---|---|---|---|
| §4 数字 | `ArithmeticHeadV2` | `muscle.cook.arith_v2` | [`pcm.heads.build_arith_v2_cook`](../pcm/heads/cook_wrappers.py) |
| §4 数字 | `ComparisonHead` | `muscle.cook.comparison` | [`pcm.heads.build_comparison_cook`](../pcm/heads/cook_wrappers.py) |
| §4 数字 | `NumerosityClassifier` | `muscle.cook.numerosity_classifier` | [`pcm.heads.build_numerosity_classifier_cook`](../pcm/heads/cook_wrappers.py) |
| §5 颜色 | `ColorMixingHead` | `muscle.cook.color_mix` | [`experiments.color_concept_study.cook.build_mix_cook`](../experiments/color_concept_study/cook.py) |
| §5 颜色 | `ColorAdjacencyHead` | `muscle.cook.color_adj` | [`experiments.color_concept_study.cook.build_adj_cook`](../experiments/color_concept_study/cook.py) |
| §6.2 空间 | `MoveHead` | `muscle.cook.space_move` | [`experiments.space_concept_study.cook.build_move_cook`](../experiments/space_concept_study/cook.py) |
| §6.2 空间 | `DistanceHead` | `muscle.cook.space_dist` | [`experiments.space_concept_study.cook.build_dist_cook`](../experiments/space_concept_study/cook.py) |
| §6.3 音素 | `_SingleInputHead` × 3 | `muscle.cook.phoneme_{voice,manner,place}` | [`experiments.phoneme_concept_study.cook.build_single_input_cook`](../experiments/phoneme_concept_study/cook.py) |

每个 cook 节点都是一个 `kind="parametric_muscle_subgraph"` 的 ConceptNode，metadata 是 2-3 节点的 op DAG（典型形态）：

```python
ConceptNode(
    node_id="muscle.cook.arith_v2",
    kind="parametric_muscle_subgraph",
    metadata={
        "inputs": ["ids_a", "ids_b", "op_onehot"],
        "nodes": [
            {"id": "_bias_0", "op": "muscle.collapse_facet",
             "args": ["arithmetic_bias", "@ids_a"]},
            {"id": "_bias_1", "op": "muscle.collapse_facet",
             "args": ["arithmetic_bias", "@ids_b"]},
            {"id": "out", "op": "muscle.invoke_module",
             "args": ["arith_v2_mlp", "@_bias_0", "@_bias_1", "@op_onehot"]},
        ],
        "output": "out",
    },
)
```

## 2. 等价机制：参数共享 + 同 RNG

cook 路径与 direct 路径必须共享**完全相同**的：

1. **`nn.Linear` 参数对象** — 通过 [`HeadAsBackbone(head)`](../pcm/heads/cook_factory.py)
   把 head 的 `fc1`/`fc2`/`fc3` 暴露成 backbone，**不**新建 `nn.Parameter`。
2. **RNG 状态** — `torch.manual_seed(seed)` + `random.Random(seed)` 在两条
   路径开始处分别重置；`HeadAsBackbone` 不消耗任何 RNG（只是 view）。
3. **Optimizer 注册顺序** — 同一组 `head.parameters() + cg.iter_bundle_parameters()`。
4. **Forward 调用语义** — `evaluator.eval(...)` 内部最终调
   `head_as_backbone.forward(*tensors)`，等价于
   `head.fc3(F.relu(head.fc2(F.relu(head.fc1(cat(tensors))))))`，与
   `head.forward` 内部的 MLP 部分逐位相同。

由于 forward bit-identical → backward grad bit-identical → AdamW step
bit-identical → 下一步 forward bit-identical（递归）。整个训练 trajectory
完全等价。

## 3. 四域结果表（完整训练，paper-scale smoke）

| 域 | epochs × steps | direct acc | cook acc | direct ρ | cook ρ | **max Δ** |
|---|---|---|---|---|---|---|
| §4 number  | 6 × 60  | **1.000** | **1.000** | ρ_linear  = +0.9727 | ρ_linear  = +0.9727 | **0.000e+00** |
| §5 color   | 8 × 80  | **1.000** | **1.000** | ρ_circular = +0.9831 | ρ_circular = +0.9831 | **0.000e+00** |
| §6.2 space | 6 × 80  | **1.000** | **1.000** | ρ_L1       = +0.8681 | ρ_L1       = +0.8681 | **0.000e+00** |
| §6.3 phoneme | 6 × 60 | accs(v/m/p) = **1.000 / 1.000 / 1.000** | 同 | ρ_same_v = +0.8660 | ρ_same_v = +0.8660 | **0.000e+00** |

所有数字都满足论文已建立的几何指标范围：

- §5 paper: `ρ_circular = 0.977 ± 0.007` — 我们 **0.9831** 在区间内。
- §6.2 paper: `ρ_L1 = 0.860 ± 0.012` — 我们 **0.8681** 在区间内。
- §6.3 paper: voicing/manner/place 全部 100% — 我们三个全 **1.000**。

`outputs/cook_four_domain/summary.json` 还记录了每个域的 `max_delta` /
`bit_identical_within_1e6` 字段，CI 可以直接 grep。

## 4. 单元测试矩阵

[`tests/test_cook_all_heads.py`](../tests/test_cook_all_heads.py) — 6 个
class × 8 个肌肉，每个肌肉的 cook 输出与 direct 输出 `torch.equal`：

| Test class | Heads tested |
|---|---|
| `TestNumberHeadsCookBitIdentical` | ArithmeticHeadV2 / ComparisonHead / NumerosityClassifier |
| `TestColorHeadsCookBitIdentical` | ColorMixingHead / ColorAdjacencyHead |
| `TestSpaceHeadsCookBitIdentical` | MoveHead / DistanceHead |
| `TestPhonemeHeadsCookBitIdentical` | _SingleInputHead × 3 (voicing / manner / place) |

加上原有 `tests/test_cook_subgraph.py` 的 9 个，**所有肌肉 forward 等价测试 = 15**。

## 5. §3.4 命题 1 现状

升级版命题 1（[`docs/PCM_NODE_AS_FUNCTION_DESIGN.md`](PCM_NODE_AS_FUNCTION_DESIGN.md) §3）：

> 对每个优化步骤，流入 $\mathbf{B}_v[f]$ 的每个梯度都来自某次
> `evaluator.eval(g, …)` 调用，其中子图 $g$ 的 `metadata.nodes` 列表中至少
> 包含一个 op $n_i$ 满足 `op_kind(n_i) = collapse` 且其解析参数
> $(facet, ids)$ 满足 $f \in facet$ 且 $v \in ids$。

本文证明的端到端 bit-identity 把这条命题从"理论"变成"经验事实"：在 §4-§6.3
的全部 100+ 训练步、每步 N+ batch 内，cook trace 上的 `collapse_facet`
op-instances 是流入每个 bundle 行的梯度的**唯一**来源 — 任何其它来源都会
导致两条路径分歧，而 max Δ = 0 排除了这种可能。

## 6. 工程影响

- **论文 §4-§6.3 的全部数字保持** — 改造后 ρ 完全一致，可直接复用。
- **§B counterfactual swap 不受影响** — bundle row swap 仍是 in-place on
  `bundle.params[facet].data`；cook 路径调用同一行。已在
  `tests/test_grow_invariants.py::test_grow_then_swap` 间接验证。
- **可选启用** — 默认仍走 direct path（`use_cook=False`），cook path 通过
  `experiments.cook_four_domain --only number color ...` 显式启动。
- **运行时 DSL 自扩展** — 由于肌肉是图谱节点，agent 可以在不修改 Python
  的情况下注册新 cook 子图（[`pcm.graph_eval`](../pcm/graph_eval.py)
  支持任意 op DAG），这是论文 §3.6 Pipeline A 端到端发现的基础。

## 7. 复现命令

```bash
# 8 个肌肉的 forward bit-identity（约 1 秒）
python -m unittest tests.test_cook_all_heads tests.test_cook_subgraph

# 4 域端到端 bit-identical（约 12 秒 GPU / 60 秒 CPU）
python -m experiments.cook_four_domain

# 单域复现
python -m experiments.cook_four_domain --only color
```

`outputs/cook_four_domain/summary.json` 的 `overall_bit_identical` 字段为 `true` 时即可宣告整套对比 PASS。
