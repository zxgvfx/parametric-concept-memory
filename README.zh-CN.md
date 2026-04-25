# 参数概念记忆（PCM）

[![CI](https://github.com/zxgvfx/parametric-concept-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/zxgvfx/parametric-concept-memory/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Paper](https://img.shields.io/badge/paper-PAPER.md-brightgreen.svg)](./PAPER.md)

**语言**: [English](./README.md) | 简体中文

> *Concepts Collapse into Muscles — Domain-Topology-Adaptive Parametric
> Concept Memory.*
>
> 概念坍缩为肌肉：一种随领域拓扑自适应的参数概念记忆框架。

## 项目简介

**Parametric Concept Memory（PCM，参数概念记忆）** 是一个把"概念"
直接做成可训练参数对象的框架。传统可解释性通常会问："哪个神经元
表示概念 *X*？" PCM 换了一个问法：让每个 `ConceptNode` 自己拥有
一个多 facet 的 `ParamBundle`，再让任务模块（本文称为 "muscle"，
即肌肉）在需要时通过 `collapse()` 去消费这些参数。

这样一来，归因不再是事后推断，而是图结构里的一个一等事实：
某个概念的某个 facet 被哪些任务使用过，可以直接查看
`bundle.consumed_by[facet]`。换句话说，"概念属于哪里"从一个解释学
问题，变成了一个可查询的数据结构问题。

完整论文见 [`PAPER.md`](./PAPER.md)。论文图表见
[`docs/figures/`](./docs/figures/)。

![四领域 bundle 几何普适性](./docs/figures/F4_four_domain_universality.png)

## 核心结论

| # | 结论 | 证据 |
|---|---|---|
| 1 | 归因是查字典，不是事后推断 | 在 7 个数概念的 toy 实验中，H1-H4 全部 100% 通过（`experiments/purity_audit.py`）。 |
| 2 | Bundle 几何会贴合任务拓扑 | 数字：`ρ = 0.991`；颜色：`ρ_circ = 0.977`；空间：`ρ_L1 = 0.860` / Procrustes disparity `0.07`；音素：类内外余弦间距 `+1.2` 到 `+2.0`。 |
| 3 | 跨 muscle 对齐由 facet 级代数兼容性决定 | 同代数任务显著对齐；同领域但代数不兼容时不对齐；正交类别轴基本不对齐。 |
| 4 | Bundle 是概念语义身份的因果载体 | 训练后交换两个概念某个 facet 的 bundle，只会精准击中消费该 facet 的 muscle 和相关概念对。 |
| 5 | PCM 产生几何涌现，不自动产生算法涌现 | 纯 base-10 分解不会自发出现；加入手写位置先验后才能恢复任意位数泛化。 |

## 仓库结构

```text
pcm/                      核心框架
├── concept_graph.py      ConceptGraph + ConceptNode
├── param_bundle.py       ParamBundle + ContextualizedConcept
└── heads/                任务 muscle/head

experiments/              论文复现实验
├── robustness_study.py
├── purity_audit.py
├── scale_study.py
├── color_concept_study.py
├── space_concept_study.py
├── phoneme_concept_study.py
├── emergent_base10_study.py
└── counterfactual_swap_study.py

docs/figures/             论文图表
PAPER.md                  完整论文
```

## 安装

需要 Python 3.10+ 和可用的 PyTorch 环境（CPU / CUDA 均可）。

```bash
git clone https://github.com/zxgvfx/parametric-concept-memory.git
cd parametric-concept-memory
pip install -r requirements.txt

# 或安装为可导入的开发包
pip install -e .
```

依赖包括：`torch`、`numpy`、`scipy`、`scikit-learn`、`matplotlib`。

## 快速查看框架

```python
from pcm import ConceptGraph

cg = ConceptGraph(feat_dim=128)
for n in range(1, 8):
    cg.register_concept(
        node_id=f"concept:ans:{n}",
        label=f"ANS_{n}",
        scope="BASE",
        provenance=f"smoke:n={n}",
    )

c = cg.concepts["concept:ans:3"]
cc = c.collapse(
    caller="AddHead",
    facet="arithmetic_bias",
    shape=(64,),
    tick=0,
    init="normal_small",
)

print(cc.as_tensor().shape)  # torch.Size([64])
print(cg.concepts["concept:ans:3"].bundle.consumed_by)
# {'arithmetic_bias': {'AddHead'}}
```

## 复现实验

完整四领域复现实验在单张 RTX 4090 上约 25 分钟完成；除 `N = 100`
数值实验外，CPU 上也可在较短时间内运行。

```bash
# 数字：线性领域、归因、H5/H5' 测试
python -m experiments.robustness_study \
    --encoder-ckpt outputs/ans_encoder/final.pt --n-seeds 10

# 数字规模实验
python -m experiments.scale_study --n-seeds 3

# 颜色：圆形拓扑
python -m experiments.color_concept_study --n-seeds 5

# 空间：2-D 网格
python -m experiments.space_concept_study --n-seeds 3

# 音素：离散类别拓扑
python -m experiments.phoneme_concept_study --n-seeds 3

# 负结果：纯 base-10 不会自发涌现
python -m experiments.emergent_base10_study --scan 50 100 --n-seeds 3

# 因果实验：训练后 bundle swap
python -m experiments.counterfactual_swap_study --n-seeds 3
```

重新生成论文图表：

```bash
python -m experiments.render_paper_figures
```

## 与相关工作的区别

PCM 位于概念瓶颈模型、超网络、外部记忆、机制可解释性和多任务表征
学习的交叉处，但它不是其中任何一种：

| 方向 | 常见做法 | PCM 的不同点 |
|---|---|---|
| 概念瓶颈模型 | 概念是层中激活或标签 | 概念是一等图节点，并拥有参数 |
| 超网络 / fast weights | 根据上下文生成权重 | bundle 直接存储权重，不通过生成器 MLP |
| 外部 / episodic memory | 取回向量或 key-value | 记忆本身就是 `nn.Parameter` |
| 机制可解释性 / SAE | 事后从激活中发现特征 | 先构造概念，再检验几何是否涌现 |
| 多任务表征学习 | 关注共享表征带宽 | PCM 检验 facet 级代数兼容性 |

## 引用

如果你使用 PCM 或相关结果，请引用：

```bibtex
@misc{zhang2026pcm,
  title        = {Concepts Collapse into Muscles: Domain-Topology-Adaptive
                  Parametric Concept Memory},
  author       = {Zhang, Xugang},
  year         = {2026},
  howpublished = {\url{https://github.com/zxgvfx/parametric-concept-memory}},
  note         = {Framework + 4-domain empirical study; MIT licensed},
}
```

## 许可证

- 代码（`pcm/`、`experiments/`、`tests/`、`outputs/ans_encoder/`）使用
  MIT 许可证。
- 论文与图表（`PAPER.md`、`submission/`、`docs/`）使用 CC BY 4.0。
