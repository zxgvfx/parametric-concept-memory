"""ArithmeticHead V1 — non-symbolic a ± b on numerosity embeddings.

D88 成果: 在 128-d ANS embedding 空间直接学加减法, 输出 pred_emb (128-d), 推理时
用 nearest-centroid 把 pred_emb 映射回具体 count.

训练成果 (outputs/arith_head/final.pt): overall_acc = 1.0 on n∈[1,7].

注意: 这是 V1 (D88). D91/D92 PoC 的 "muscle" 是 ArithmeticHeadV2 (另一文件),
后者强制绕过 emb_a/emb_b, 仅消费 bundle 参数.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArithmeticHead(nn.Module):
    """(emb_a, emb_b, op_onehot) → pred_emb. ckpt-compatible shapes.

    ckpt keys/shapes (outputs/arith_head/final.pt):
        fc1.weight: (128, 258)  [258 = 128 + 128 + 2]
        fc2.weight: (128, 128)
        fc3.weight: (128, 128)
    """

    EMBED_DIM = 128
    N_OPS = 2  # add, sub

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        in_dim = embed_dim * 2 + self.N_OPS
        self.fc1 = nn.Linear(in_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        op_onehot: torch.Tensor,
    ) -> torch.Tensor:
        if emb_a.shape != emb_b.shape:
            raise ValueError(f"emb_a.shape {tuple(emb_a.shape)} != emb_b.shape {tuple(emb_b.shape)}")
        if emb_a.shape[0] != op_onehot.shape[0]:
            raise ValueError(f"batch mismatch: emb_a {emb_a.shape[0]} vs op {op_onehot.shape[0]}")
        x = torch.cat([emb_a, emb_b, op_onehot], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return F.normalize(h, dim=-1)

    @staticmethod
    def op_to_onehot(
        op: str | list[str],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if isinstance(op, str):
            op = [op]
        v = [[1.0, 0.0] if o == "add" else [0.0, 1.0] for o in op]
        return torch.tensor(v, device=device)

    @staticmethod
    def classify_to_count(
        pred_emb: torch.Tensor,
        centroids: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """Nearest-centroid: pred (B,D), centroids (K,D), counts (K,) → (B,)."""
        sim = pred_emb @ centroids.t()
        idx = sim.argmax(dim=-1)
        return counts[idx]
