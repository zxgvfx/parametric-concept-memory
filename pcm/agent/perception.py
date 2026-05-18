"""pcm.agent.perception — v6.4 perception layer.

F64 / F65 / F66 / F67 all used a *state-index → slot embedding*
encoder (``SlotStateEncoder``) that assumes the env exposes
state as a small integer. v6.4 demonstrates that the F62
universal operator works equally well when the slot is produced
by a **perception head** that consumes a token sequence — the
substrate of natural-language goal specification.

Architectural claim: the perception head can be a small
Transformer, and its mean-pooled hidden state drops into the F62
``UniversalCombiner`` unchanged. This mirrors F63g's per-
component analysis: attention patterns are universal (where each
head looks), hidden states are modality-specific (what the
residual stream encodes). v6.4 verifies the same picture survives
at the perception → agent boundary.

Public API::

    TextPerceptionHead   — token sequence → slot
    text_to_slot         — convenience inference helper
"""
from __future__ import annotations

import torch
import torch.nn as nn


__all__ = [
    "TextPerceptionHead",
    "text_to_slot",
    "ImagePerceptionHead",
    "image_to_slot",
]


class TextPerceptionHead(nn.Module):
    """Encode a token sequence into a single slot vector.

    Architecture: ``Embedding → 1-2 Transformer encoder layers →
    mean-pool → Linear → slot_dim``. The pooled mean is the
    most permutation-stable signal across short sequences and is
    cheap to compute. For longer / structured sequences, replace
    the mean pool with attention pooling — out of scope for
    v6.4 minimum-viable.
    """

    def __init__(
        self, vocab: int, slot_dim: int,
        *, d_model: int = 64, n_layers: int = 2,
        n_heads: int = 4, max_len: int = 16, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.slot_dim = slot_dim
        self.d_model = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_final = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, slot_dim)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``tokens``: ``(B, L)`` long indices. Optional
        ``mask``: ``(B, L)`` bool (True for valid positions).
        Returns ``(B, slot_dim)``."""
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device)
        h = self.tok_emb(tokens) + self.pos_emb(pos)
        h = self.encoder(h)
        h = self.ln_final(h)
        if mask is not None:
            m = mask.float().unsqueeze(-1)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            pooled = h.mean(dim=1)
        return self.proj(pooled)


@torch.no_grad()
def text_to_slot(
    perception: TextPerceptionHead, tokens: list[int],
    *, device: str = "cpu",
) -> torch.Tensor:
    """Inference helper: encode a 1-D Python list of token IDs
    into a single ``(1, slot_dim)`` slot vector."""
    perception.eval()
    t = torch.tensor([tokens], dtype=torch.long, device=device)
    return perception(t)


# ─────────────────────────────────────────────────────────────────
# v6.4-image — image-goal perception (F72)
# ─────────────────────────────────────────────────────────────────


class ImagePerceptionHead(nn.Module):
    """Encode a single greyscale image into a slot vector.

    Architecture: small 2-block CNN (Conv → ReLU → Pool ×2) →
    flatten → Linear → slot_dim. For ``image_size = 16``,
    after two 2× downsamplings the feature map is 4×4, giving a
    flat dim of ``channels[-1] · 16``.

    The CNN choice is deliberately simple — F72 tests whether
    the F62 universal-operator slot machinery works equally
    when the slot comes from a *spatial* feature extractor,
    not whether the extractor itself is state of the art.
    """

    def __init__(
        self, slot_dim: int, *,
        image_size: int = 16, in_channels: int = 1,
        channels: tuple[int, int] = (16, 32),
    ) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.image_size = image_size
        self.in_channels = in_channels
        self.channels = channels
        c1, c2 = channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # After two pools: image_size // 4 spatial dim
        feat_dim = c2 * (image_size // 4) ** 2
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, slot_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """``image``: ``(B, C, H, W)`` float in [0, 1].
        Returns ``(B, slot_dim)``."""
        if image.dim() == 3:
            image = image.unsqueeze(1)
        return self.proj(self.net(image))


@torch.no_grad()
def image_to_slot(
    perception: ImagePerceptionHead, image: torch.Tensor,
    *, device: str = "cpu",
) -> torch.Tensor:
    """Inference helper: encode a single image tensor
    ``(1, H, W)`` or ``(1, 1, H, W)`` into a ``(1, slot_dim)``
    slot vector."""
    perception.eval()
    if image.dim() == 3:
        image = image.unsqueeze(0)
    return perception(image.to(device))
