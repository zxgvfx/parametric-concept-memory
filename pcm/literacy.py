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
    """Small ConvNet that maps a ``(1, H, W)`` glyph image to
    a single slot of dimension ``d_model``.

    The architecture is more aggressive than
    :class:`pcm.vision.VisualEncoder` because glyph
    discrimination requires resolving fine letter-shape
    differences:

        ``Conv(1 → 16, 3x3) → BN → SiLU``  (full resolution)
        ``Conv(16 → 32, 3x3, stride 2) → BN → SiLU``
        ``Conv(32 → 64, 3x3, stride 2) → BN → SiLU``
        ``AdaptiveAvgPool(1)``
        ``Linear(64 → d_model)``

    For ``size = (16, 64)``, after two stride-2 convs the
    feature map is ``(64, 4, 16)`` before pooling. ~31 K
    parameters at d_model=128.
    """

    def __init__(
        self, d_model: int = 128, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(64, d_model)
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
    """Small ConvTranspose decoder that maps a token-embedding
    slot to a ``(1, H, W)`` glyph image.

    Mirror image of :class:`GlyphEncoder` for the F89 *writing*
    capability: given a token's embedding, reconstruct what its
    printed glyph looks like.

    Architecture for ``GLYPH_SIZE = (16, 64)``:

        ``Linear(d_model, 32·2·8)``  → reshape to ``(32, 2, 8)``
        ``ConvT(32 → 16, k=4, s=2) → BN → SiLU``  → ``(16, 4, 16)``
        ``ConvT(16 →  8, k=4, s=2) → BN → SiLU``  → ``(8, 8, 32)``
        ``ConvT( 8 →  1, k=4, s=2) → Sigmoid``    → ``(1, 16, 64)``

    ~77 K parameters at ``d_model=128`` — about twice the
    encoder, because expansion is harder than compression. The
    final ``Sigmoid`` keeps the output in ``[0, 1]`` matching
    the renderer's pixel range.
    """

    def __init__(
        self, d_model: int = 128,
        out_h: int = 16, out_w: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if out_h % 8 != 0 or out_w % 8 != 0:
            raise ValueError(
                f"out_h and out_w must each be divisible by 8 "
                f"(got {out_h}, {out_w})"
            )
        self.d_model = d_model
        self.out_h = out_h
        self.out_w = out_w
        self._seed_h = out_h // 8
        self._seed_w = out_w // 8
        self._seed_c = 32
        self.proj = nn.Linear(
            d_model,
            self._seed_c * self._seed_h * self._seed_w,
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                self._seed_c, 16,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.ConvTranspose2d(
                16, 8, kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(8),
            nn.SiLU(),
            nn.ConvTranspose2d(
                8, 1, kernel_size=4, stride=2, padding=1,
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
