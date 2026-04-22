"""NumerosityEncoder — CNN for Approximate Number Sense (ANS).

D88 Pre-Symbolic Grounding: agent 在看到 dot canvas (1, 64, 64) 时, 不通过符号,
直接把 "数量感" encode 成 128-d embedding. 训练靠:
  (1) Contrastive loss (同 n 拉近, 不同 n 推远)
  (2) Ordinal regularization (|n_a - n_b| 越大 → embedding 越远, 符合 Weber's law)

ckpt 兼容: outputs/ans_encoder/final.pt (spatial_invariance=0.999, weber monotonic).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DatasetConfig:
    n_min: int = 1
    n_max: int = 7
    canvas_size: int = 64
    dot_size_range: Tuple[int, int] = (3, 6)
    min_dist: int = 7
    brightness_jitter: float = 0.1
    gaussian_noise_std: float = 0.02


class NumerosityEncoder(nn.Module):
    """3-layer conv CNN → 2-layer fc → 128-d L2-normalized embedding.

    精确 shape (来自 outputs/ans_encoder/final.pt):
        conv1: (16, 1, 5, 5)  bn1: (16,)
        conv2: (32, 16, 3, 3) bn2: (32,)
        conv3: (32, 32, 3, 3) bn3: (32,)
        fc1:   (64, 32)
        fc2:   (128, 64)
    """

    EMBED_DIM = 128

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 128)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 64, 64), values in [0,1]
        returns: (B, 128) L2-normalized embedding
        """
        if x.dim() != 4 or x.shape[-3] != 1 or x.shape[-1] != 64 or x.shape[-2] != 64:
            raise ValueError(f"expected (B, 1, 64, 64), got {tuple(x.shape)}")
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return F.normalize(h, dim=-1)


def contrastive_ordinal_loss(
    embeds: torch.Tensor,
    counts: torch.Tensor,
    temperature: float = 0.1,
    ordinal_weight: float = 0.3,
) -> Tuple[torch.Tensor, dict]:
    """Contrastive (SupCon-style) + Ordinal reg.

    embeds: (B, D) L2-norm; counts: (B,) int.
    """
    B = embeds.size(0)
    sim = embeds @ embeds.t() / temperature
    same = counts.view(-1, 1) == counts.view(1, -1)
    diag = torch.eye(B, dtype=torch.bool, device=embeds.device)
    pos_mask = same & ~diag
    logits_mask = ~diag
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True).clamp_min(1e-12))
    pos_counts = pos_mask.sum(1).clamp_min(1)
    con_loss = -(log_prob * pos_mask).sum(1) / pos_counts
    con_loss = con_loss.mean()

    cos = embeds @ embeds.t()
    diff = (counts.view(-1, 1) - counts.view(1, -1)).abs().float()
    target_cos = 1.0 - diff / diff.max().clamp_min(1.0)
    ord_loss = F.mse_loss(cos[logits_mask], target_cos[logits_mask])

    total = con_loss + ordinal_weight * ord_loss
    return total, {"con": con_loss.item(), "ord": ord_loss.item()}


def generate_dot_canvas(
    n_dots: int,
    cfg: DatasetConfig,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Rejection-sample n_dots circles with min_dist, return (1, H, W) canvas in [0,1]."""
    H = W = cfg.canvas_size
    canvas = torch.zeros(1, H, W)
    centers: list[tuple[int, int, int]] = []
    for _ in range(n_dots):
        for _try in range(200):
            r = int(
                torch.randint(
                    cfg.dot_size_range[0],
                    cfg.dot_size_range[1] + 1,
                    (1,),
                    generator=rng,
                ).item()
            )
            cx = int(torch.randint(r + 1, W - r - 1, (1,), generator=rng).item())
            cy = int(torch.randint(r + 1, H - r - 1, (1,), generator=rng).item())
            ok = all(
                ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5 >= cfg.min_dist
                for x, y, _ in centers
            )
            if ok:
                centers.append((cx, cy, r))
                break
        else:
            break
    yy, xx = torch.meshgrid(
        torch.arange(H), torch.arange(W), indexing="ij"
    )
    for cx, cy, r in centers:
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        brightness = 1.0 + (
            torch.rand(1, generator=rng).item() * 2 - 1
        ) * cfg.brightness_jitter
        canvas[0][mask] = min(max(brightness, 0.0), 1.0)
    noise = torch.randn(canvas.shape, generator=rng) * cfg.gaussian_noise_std
    return (canvas + noise).clamp(0, 1)


def encode_numerosity(encoder: NumerosityEncoder, img_or_imgs) -> torch.Tensor:
    """Encode an image or list of images into 128-d ANS embedding(s).

    Accepts any object with a ``.data`` attribute holding a numpy array
    of shape (H, W, 1) or (H, W, C) and dtype uint8, OR a raw torch
    tensor / numpy array of shape (C, H, W) or (H, W, C). Images are
    resized to 64×64 if needed.

    Returns a tensor of shape (128,) for a single input, or (N, 128)
    for a batch.
    """
    import numpy as np

    def _is_single(x) -> bool:
        if hasattr(x, "data") and hasattr(x, "shape"):
            return True
        if isinstance(x, (np.ndarray, torch.Tensor)) and x.ndim <= 3:
            return True
        return False

    single = _is_single(img_or_imgs)
    imgs = [img_or_imgs] if single else list(img_or_imgs)
    tensors = []
    for img in imgs:
        arr = img.data if hasattr(img, "data") else img
        t = arr if isinstance(arr, torch.Tensor) else torch.from_numpy(arr)
        t = t.float()
        if t.max() > 1.5:
            t = t / 255.0
        if t.dim() == 3 and t.shape[-1] <= 4:
            t = t.permute(2, 0, 1)
        if t.dim() == 2:
            t = t.unsqueeze(0)
        if t.shape[-2:] != (64, 64):
            t = F.interpolate(
                t.unsqueeze(0), size=(64, 64),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        tensors.append(t)
    batch = torch.stack(tensors)
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        out = encoder(batch)
    if was_training:
        encoder.train()
    return out[0] if single else out


def sample_batch(
    batch_size: int,
    cfg: DatasetConfig,
    rng: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (images (B,1,H,W), counts (B,))."""
    counts = torch.randint(cfg.n_min, cfg.n_max + 1, (batch_size,), generator=rng)
    imgs = torch.stack([generate_dot_canvas(int(c), cfg, rng) for c in counts])
    return imgs, counts
