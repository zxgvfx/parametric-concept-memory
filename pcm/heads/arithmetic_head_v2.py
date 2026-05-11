"""arithmetic_head_v2.py — D91/D92 的 AddHead.

forward **只**消费 ConceptNode bundle 里的 ``arithmetic_bias`` facet + operation
one-hot; 接受的感知 embedding (``emb_a``, ``emb_b``) 在 forward 里**被忽略**,
保留只是为了与历史调用方签名兼容. 这是为了满足 D91/D92 的 "指纹归因" 硬约束 ——
所有 concept-specific 信息必须通过图谱的参数 bundle 流动, backbone 不得走私.

设计的来由 (旧版 D88 head 走 emb_a/emb_b 走私通路, H1-H3 ablation tests 失败)
见 ``mind/docs/research/PARAMETRIC_CONCEPT_MEMORY.md`` §"Ablation-Induced
Redesign".

Tier-D 等价物: ``pcm/heads/cook_wrappers.py::build_arith_v2_cook(cg, head)``,
返回的 cook 子图与本 forward bit-identical (claim D1, see
``tests/test_cook_all_heads.py``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..concept_graph import ConceptGraph

CALLER = "ArithmeticHeadV2"
FACET_NAME = "arithmetic_bias"


class ArithmeticHeadV2(nn.Module):
    """非符号加减法肌肉, 消费 ``arithmetic_bias`` facet.

    输入:
      - emb_a, emb_b : (B, embed_dim) — **占位参数, forward 里忽略**
      - op_onehot    : (B, 2)         — [add=1,0] / [sub=0,1]
      - concept_ids_a/b: list[str]    — 用于 bundle.collapse
    输出:
      - pred_emb     : (B, embed_dim) — 目标 numerosity embedding
    """

    def __init__(self, embed_dim: int = 128, bias_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.bias_dim = bias_dim
        in_dim = 2 * bias_dim + 2
        self.fc1 = nn.Linear(in_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        emb_a: torch.Tensor,   # 占位
        emb_b: torch.Tensor,   # 占位
        op_onehot: torch.Tensor,
        concept_ids_a: list[str],
        concept_ids_b: list[str],
        cg: "ConceptGraph",
        tick: int = 0,
    ) -> torch.Tensor:
        bias_a = self._collapse_bias_batch(concept_ids_a, cg, tick)
        bias_b = self._collapse_bias_batch(concept_ids_b, cg, tick)
        x = torch.cat([bias_a, bias_b, op_onehot], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def _collapse_bias_batch(
        self,
        concept_ids: list[str],
        cg: "ConceptGraph",
        tick: int,
    ) -> torch.Tensor:
        """Dense-pool fast path: single ``F.embedding`` over the whole batch.

        D93 invariant: bit-identical to the legacy ``for cid in ids: ...
        torch.stack(...)`` pattern — same row data, same autograd graph,
        same per-row gradient accumulation. Avoids B Python-level dict
        lookups and one ``stack`` per call.
        """
        device = next(self.parameters()).device
        return cg.collapse_batch(
            caller=CALLER,
            facet=FACET_NAME,
            concept_ids=concept_ids,
            shape=(self.bias_dim,),
            tick=tick,
            init="normal_small",
            device=device,
        )
