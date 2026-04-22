"""numerosity_classifier.py — D91/D92 三肌肉 PoC 的第三肌肉 (negative control).

架构级作用: 与 ArithmeticHeadV2 / ComparisonHead 并列的独立消费者, 从 ConceptNode
的 ``identity_prototype`` facet 读参数, 执行**纯 1-of-K 分类** (无代数结构,
permutation-symmetric).

这是 H5'' 的关键 negative control:
- arithmetic_bias (AddHead) — 加法代数 → facet 自然形成数量单调排序
- ordinal_offset (CmpHead) — 总序 → facet 自然形成距离单调排序
- **identity_prototype (本 module)** — 分类 loss 对标签 permutation 不敏感 → facet
  **不应该**形成跨概念的几何一致性

triple-muscle 实验跑完后:
- ρ(arithmetic_bias, -|Δn|) = 0.912 ✓
- ρ(ordinal_offset, -|Δn|) = 0.924 ✓
- ρ(identity_prototype, -|Δn|) = 0.088 ✗  ← 如期随机, 验证 negative control 成立
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..concept_graph import ConceptGraph

CALLER = "NumerosityClassifier"
FACET_NAME = "identity_prototype"


class NumerosityClassifier(nn.Module):
    """1-of-K 分类肌肉, 从 ``identity_prototype`` facet 读参数.

    forward: concept_id list → logits (B, n_classes).
    训练目标: concept_id 对应的类索引 (N - n_min).
    """

    def __init__(
        self,
        n_classes: int,
        hidden_dim: int = 32,
        facet_dim: int = 16,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.facet_dim = facet_dim
        self.fc1 = nn.Linear(facet_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_classes)

    def forward(
        self,
        concept_ids: list[str],
        cg: "ConceptGraph",
        tick: int = 0,
    ) -> torch.Tensor:
        proto = self._collapse_proto_batch(concept_ids, cg, tick)
        h = F.relu(self.fc1(proto))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def _collapse_proto_batch(
        self,
        concept_ids: list[str],
        cg: "ConceptGraph",
        tick: int,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        rows: list[torch.Tensor] = []
        for cid in concept_ids:
            if cid not in cg.concepts:
                raise KeyError(f"concept {cid!r} not in ConceptGraph")
            cc = cg.concepts[cid].collapse(
                caller=CALLER,
                facet=FACET_NAME,
                shape=(self.facet_dim,),
                tick=tick,
                init="normal_small",
                device=device,
            )
            rows.append(cc.as_tensor())
        return torch.stack(rows, dim=0)
