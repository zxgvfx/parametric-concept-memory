"""pcm.literacy — F88 cross-modal binding of printed glyphs.

Implements the **reading** capability for PCM: each token in
the vocabulary is rendered as a small grayscale image, and a
:class:`GlyphEncoder` is trained to map glyph images to the
LM's token embeddings. After training, the LM can consume
sequences of *images* (rendered text) and continue generating
text as if it had received tokens directly.

Two training stages live here as utilities:

* **Alignment stage** (Stage 2 in the F88 experiment) — the LM
  is frozen; the encoder is trained to match
  ``LM.tok_emb.weight[t]`` for every token ``t``. This is the
  cheap, supervised step.
* **Joint stage** (Stage 3) — the LM is *unfrozen*; sequences
  are passed through a :func:`multimodal_forward` that
  replaces a fraction ``mix_rate`` of token-embedding rows
  with their glyph-encoded counterparts. The standard next-
  token cross-entropy loss is back-propagated end-to-end.
  This is the step F87 V4 was missing — it makes the LM
  *learn to use* visual inputs, not just be aligned-to by
  them.

Cognitive parallel: by the end of Stage 3, the model
*reads*. The same downstream tasks (continuation, question
answering, the F85 teacher loop) work with glyph images as
input.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:  # pragma: no cover
    _PIL_OK = False


__all__ = [
    "GLYPH_SIZE",
    "FONT_CANDIDATES",
    "render_glyph",
    "build_glyph_table",
    "GlyphEncoder",
    "GlyphDecoder",
    "multimodal_forward",
    "alignment_loss",
    "reconstruction_loss",
]


# ─────────────────────────────────────────────────────────────────
# Rendering — small grayscale glyph per token (PIL + Courier)
# ─────────────────────────────────────────────────────────────────


GLYPH_SIZE: tuple[int, int] = (16, 64)  # (H, W) — small enough
# for a 1-conv-block CNN, large enough to disambiguate ~4 k
# English words at 12 pt Courier.

FONT_CANDIDATES: tuple[str, ...] = (
    # Try cross-platform monospace first
    "C:/Windows/Fonts/cour.ttf",      # Courier New (Windows)
    "C:/Windows/Fonts/consola.ttf",   # Consolas (Windows)
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/Library/Fonts/Courier New.ttf",  # macOS
)


@lru_cache(maxsize=1)
def _load_font(size: int = 12) -> "ImageFont.FreeTypeFont":
    if not _PIL_OK:
        raise RuntimeError(
            "PIL/Pillow is required for glyph rendering. "
            "Install via `pip install pillow`."
        )
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    # Fallback to PIL's bitmap default — rough but works
    return ImageFont.load_default()


def render_glyph(
    text: str, *, size: tuple[int, int] = GLYPH_SIZE,
    font_size: int = 12, padding: int = 1,
) -> np.ndarray:
    """Render ``text`` to a grayscale image of shape ``size``.

    Returns a ``(H, W)`` uint8 array in ``[0, 255]`` — black
    text on white background. If the text is wider than the
    image, it is clipped at the right edge (which is fine for
    our purposes since rare-word tokens are still
    distinguishable by their first letters).
    """
    if not _PIL_OK:
        raise RuntimeError("PIL/Pillow not available")
    H, W = size
    img = Image.new("L", (W, H), color=255)
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)
    draw.text((padding, padding - 1), text, font=font, fill=0)
    return np.asarray(img, dtype=np.uint8)


def render_glyph_tensor(
    text: str, *, size: tuple[int, int] = GLYPH_SIZE,
    font_size: int = 12,
) -> torch.Tensor:
    """Convenience: returns ``(1, H, W)`` float tensor in
    ``[0, 1]``."""
    arr = render_glyph(text, size=size, font_size=font_size)
    return torch.from_numpy(arr).float().unsqueeze(0) / 255.0


def build_glyph_table(
    itos: list[str], *, size: tuple[int, int] = GLYPH_SIZE,
    font_size: int = 12, special_tokens: int = 4,
) -> torch.Tensor:
    """Pre-render every vocab token to a glyph image.

    The first ``special_tokens`` entries (``<pad>``, ``<unk>``,
    ``<bos>``, ``<eos>``) get a *blank* image (no text) — these
    aren't visible printed words.

    Returns ``(V, 1, H, W)`` float tensor in ``[0, 1]``.
    """
    H, W = size
    V = len(itos)
    out = torch.zeros(V, 1, H, W)
    for i, tok in enumerate(itos):
        if i < special_tokens:
            # Leave as zeros / black (the model learns these
            # don't correspond to print)
            continue
        # Truncate words that won't fit — keep first chars
        max_chars = max(W // (font_size * 0.6), 4)
        truncated = tok[:int(max_chars)]
        arr = render_glyph(
            truncated, size=size, font_size=font_size,
        )
        out[i, 0] = torch.from_numpy(arr).float() / 255.0
    return out


# ─────────────────────────────────────────────────────────────────
# GlyphEncoder — CNN: grayscale glyph → token embedding
# ─────────────────────────────────────────────────────────────────


class GlyphEncoder(nn.Module):
    """ConvNet that maps a ``(1, H, W)`` glyph image to a
    single slot of dimension ``d_model``.

    The architecture is more aggressive than
    :class:`pcm.vision.VisualEncoder` because glyph
    discrimination requires resolving fine letter-shape
    differences:

        ``Conv(1 → c0, 3x3) → BN → SiLU``  (full resolution)
        ``Conv(c0 → c0·2, 3x3, stride 2) → BN → SiLU``
        ``Conv(c0·2 → c0·4, 3x3, stride 2) → BN → SiLU``
        ``AdaptiveAvgPool(1)``
        ``Linear(c0·4 → d_model)``

    ``base_channels=16`` (default) gives ~31 K params at
    ``d_model=128`` — the F88 baseline. For F92 / scaled
    settings, bump to 64 or 128 so the encoder's channel-
    pyramid scales with the decoder (symmetric cycle).
    """

    def __init__(
        self, d_model: int = 128, dropout: float = 0.0,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        if base_channels < 4 or base_channels % 4 != 0:
            raise ValueError(
                f"base_channels must be ≥4 and divisible by 4 "
                f"(got {base_channels})"
            )
        self.d_model = d_model
        c0 = base_channels
        c1 = c0 * 2
        c2 = c0 * 4
        self.features = nn.Sequential(
            nn.Conv2d(1, c0, kernel_size=3, padding=1),
            nn.BatchNorm2d(c0),
            nn.SiLU(),
            nn.Conv2d(c0, c1, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(c2, d_model)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, 1, H, W)`` float in ``[0, 1]``.
        Returns ``(B, d_model)`` slot."""
        h = self.features(x).flatten(1)
        h = self.dropout(h)
        return self.proj(h)


# ─────────────────────────────────────────────────────────────────
# GlyphDecoder — token embedding → grayscale glyph (F89 writing)
# ─────────────────────────────────────────────────────────────────


class GlyphDecoder(nn.Module):
    """ConvTranspose decoder that maps a token-embedding slot
    to a ``(1, H, W)`` glyph image (F89 writing capability).

    Architecture for ``GLYPH_SIZE = (16, 64)`` with
    ``seed_channels = c0``:

        ``Linear(d_model, c0·2·8)``  → reshape to ``(c0, 2, 8)``
        ``ConvT(c0 → c0/2, k=4, s=2) → BN → SiLU``  → ``(c0/2, 4, 16)``
        ``ConvT(c0/2 → c0/4, k=4, s=2) → BN → SiLU`` → ``(c0/4, 8, 32)``
        ``ConvT(c0/4 → 1, k=4, s=2) → Sigmoid``      → ``(1, 16, 64)``

    Parameter count is dominated by:

    * The ``proj`` linear (``d_model × c0 × 16``).
    * The first ConvT (``c0 × c0/2 × 16``).

    ``seed_channels=32`` (default) gives ~77 K params at
    ``d_model=128``. The F89 scaling-paradox finding (3.42)
    revealed that this is the bottleneck at large ``d_model``:
    the linear ``proj`` grows but the ConvT channel widths
    don't, so the model can land embeddings well but cannot
    render them as distinct pixels. ``seed_channels=128``
    (~1.2 M params) gives the ConvT path room to scale; F91
    uses this default at d_model=512.

    The final ``Sigmoid`` keeps output in ``[0, 1]`` matching
    the renderer's pixel range.
    """

    def __init__(
        self, d_model: int = 128,
        out_h: int = 16, out_w: int = 64,
        dropout: float = 0.0,
        seed_channels: int = 32,
    ) -> None:
        super().__init__()
        if out_h % 8 != 0 or out_w % 8 != 0:
            raise ValueError(
                f"out_h and out_w must each be divisible by 8 "
                f"(got {out_h}, {out_w})"
            )
        if seed_channels < 8 or seed_channels % 4 != 0:
            raise ValueError(
                f"seed_channels must be ≥8 and divisible by 4 "
                f"(got {seed_channels})"
            )
        self.d_model = d_model
        self.out_h = out_h
        self.out_w = out_w
        self._seed_h = out_h // 8
        self._seed_w = out_w // 8
        self._seed_c = seed_channels
        c0 = seed_channels
        c1 = max(c0 // 2, 4)
        c2 = max(c0 // 4, 4)
        self.proj = nn.Linear(
            d_model,
            c0 * self._seed_h * self._seed_w,
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                c0, c1, kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.ConvTranspose2d(
                c1, c2, kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
            nn.ConvTranspose2d(
                c2, 1, kernel_size=4, stride=2, padding=1,
            ),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, slot: torch.Tensor) -> torch.Tensor:
        """``slot``: ``(B, d_model)``. Returns ``(B, 1, H, W)``
        float tensor in ``[0, 1]``."""
        h = self.proj(slot)
        h = self.dropout(h)
        h = h.view(
            -1, self._seed_c, self._seed_h, self._seed_w,
        )
        return self.deconv(h)


# ─────────────────────────────────────────────────────────────────
# Reconstruction loss (F89)
# ─────────────────────────────────────────────────────────────────


def reconstruction_loss(
    predicted: torch.Tensor, target: torch.Tensor, *,
    mse_weight: float = 1.0, bce_weight: float = 0.0,
    contrastive_weight: float = 0.0,
    temperature: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Decoder reconstruction loss.

    Three optional components:

    * **MSE** (default) — soft pixel-wise L2 regression.
    * **BCE** — binary cross-entropy, sharper gradients for the
      quasi-binary glyph distributions.
    * **Contrastive** (in-batch InfoNCE over pixel L2) — forces
      decoded glyphs to be *discriminable* per-token. Without
      this, MSE alone collapses to a mean-glyph output (the
      classical L2 regression failure mode for high-dim
      structured outputs).

    Both ``predicted`` and ``target`` must be ``(B, 1, H, W)``
    floats in ``[0, 1]``.
    """
    diag: dict = {}
    mse = F.mse_loss(predicted, target)
    diag["mse"] = float(mse.item())
    loss = mse * mse_weight
    if bce_weight > 0:
        eps = 1e-6
        p = predicted.clamp(eps, 1 - eps)
        bce = F.binary_cross_entropy(p, target)
        diag["bce"] = float(bce.item())
        loss = loss + bce * bce_weight
    if contrastive_weight > 0 and predicted.shape[0] > 1:
        B = predicted.shape[0]
        p_flat = predicted.flatten(1)
        t_flat = target.flatten(1)
        # Pairwise negative L2 distances (B x B); diagonal is
        # the correct alignment.
        d = (
            (p_flat.unsqueeze(1) - t_flat.unsqueeze(0))
            .pow(2).mean(dim=-1)
        )
        # Convert to similarity-style logits: -d / temperature
        logits = -d / max(temperature, 1e-6)
        targets = torch.arange(
            B, device=predicted.device,
        )
        cont = 0.5 * (
            F.cross_entropy(logits, targets)
            + F.cross_entropy(logits.t(), targets)
        )
        diag["contrastive"] = float(cont.item())
        loss = loss + cont * contrastive_weight
    return loss, diag


# ─────────────────────────────────────────────────────────────────
# Alignment loss (Stage 2)
# ─────────────────────────────────────────────────────────────────


def alignment_loss(
    glyph_slots: torch.Tensor,
    target_embeddings: torch.Tensor, *,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """MSE + (1 − cosine) alignment loss between glyph encoder
    output and target token embeddings.

    Both arguments must be ``(B, d_model)`` of the same shape.
    """
    mse = F.mse_loss(glyph_slots, target_embeddings)
    g_n = F.normalize(glyph_slots, dim=-1)
    t_n = F.normalize(target_embeddings, dim=-1)
    cos = (g_n * t_n).sum(dim=-1).mean()
    loss = mse_weight * mse + cosine_weight * (1.0 - cos)
    return loss, {
        "mse": float(mse.item()),
        "cosine": float(cos.item()),
    }


# ─────────────────────────────────────────────────────────────────
# Multimodal forward (Stage 3)
# ─────────────────────────────────────────────────────────────────


def multimodal_forward(
    lm: nn.Module, glyph_encoder: GlyphEncoder,
    token_ids: torch.Tensor,
    glyph_table: torch.Tensor, *,
    mix_rate: float = 0.5,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``lm`` on ``token_ids`` but replace a fraction
    ``mix_rate`` of token embeddings with their
    glyph-encoded counterparts.

    Args:
        lm: a model exposing ``tok_emb`` (Embedding), an
            iterable ``layers``, and ``ln_final``. Standard
            :class:`pcm.lm.HybridPCMMiniLM`.
        glyph_encoder: a :class:`GlyphEncoder`.
        token_ids: ``(B, L)`` LongTensor.
        glyph_table: ``(V, 1, H, W)`` float tensor of pre-
            rendered glyphs (one per vocab id).
        mix_rate: probability of replacing each position's
            embedding with its glyph encoding. Default 0.5.
        rng: optional CPU generator.

    Returns:
        ``logits`` of shape ``(B, L, V)`` and the per-position
        ``mask`` (1 where a glyph was used, 0 where the regular
        token embedding was used).
    """
    B, L = token_ids.shape
    device = token_ids.device
    # Standard token embeddings
    text_slots = lm.tok_emb(token_ids)  # (B, L, D)
    # Decide which positions get glyph-encoded
    if mix_rate <= 0.0:
        mask = torch.zeros(B, L, device=device)
        slots = text_slots
    elif mix_rate >= 1.0:
        mask = torch.ones(B, L, device=device)
        flat_tokens = token_ids.reshape(-1)
        glyph_imgs = glyph_table[flat_tokens.cpu()].to(device)
        glyph_slots = glyph_encoder(glyph_imgs).view(B, L, -1)
        slots = glyph_slots
    else:
        if rng is None:
            mask = (torch.rand(B, L, device=device)
                    < mix_rate).float()
        else:
            mask = (torch.rand(B, L, generator=rng)
                    < mix_rate).float().to(device)
        # Compute glyph embeddings only for selected positions
        # (could be optimised; here we always compute all then
        # gate — simpler / cheap at this scale)
        flat_tokens = token_ids.reshape(-1)
        glyph_imgs = glyph_table[flat_tokens.cpu()].to(device)
        glyph_slots = glyph_encoder(glyph_imgs).view(B, L, -1)
        m = mask.unsqueeze(-1)
        slots = m * glyph_slots + (1.0 - m) * text_slots
    # Now run through the LM body manually
    h = slots
    for layer in lm.layers:
        h = h + layer(h)
    h = lm.ln_final(h)
    logits = h @ lm.tok_emb.weight.t()
    return logits, mask
