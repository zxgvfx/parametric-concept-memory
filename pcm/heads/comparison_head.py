"""comparison_head.py — D91/D92 的 CmpHead: 消费 ``ordinal_offset`` facet.

作用: 给定 concept_a, concept_b, 输出 3 类 logits (<, =, >).
与 ArithmeticHeadV2 并列, 为 D91/D92 的 cross-muscle 归因实证提供第二消费者.

forward 里**不**接受感知 embedding, 与 ArithmeticHeadV2 同理.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..concept_graph import ConceptGraph

CALLER = "ComparisonHead"
FACET_NAME = "ordinal_offset"


class ComparisonHead(nn.Module):
    """二元比较肌肉 (<, =, >), 消费 ``ordinal_offset`` facet."""

    def __init__(
        self,
        embed_dim: int = 128,
        facet_dim: int = 8,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.facet_dim = facet_dim
        self.hidden_dim = hidden_dim
        in_dim = 2 * facet_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        emb_a: Optional[torch.Tensor],  # 占位
        emb_b: Optional[torch.Tensor],  # 占位
        concept_ids_a: list[str],
        concept_ids_b: list[str],
        cg: "ConceptGraph",
        tick: int = 0,
    ) -> torch.Tensor:
        off_a = self._collapse_offset_batch(concept_ids_a, cg, tick)
        off_b = self._collapse_offset_batch(concept_ids_b, cg, tick)
        x = torch.cat([off_a, off_b], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def _collapse_offset_batch(
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
