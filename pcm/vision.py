"""pcm.vision — F87 visual perception of preschool stimuli.

Maps small RGB images of colour-shape stimuli (e.g. "red
circle on white") to **slots in the same space as the F81
language model's token embeddings**, so that perception and
language share a single concept-slot space.

The architecture has three pieces:

* :func:`render_colour_shape` — procedural renderer that
  generates a 32x32 RGB image of a single coloured shape on a
  neutral background. Reproducible from ``(color, shape, size,
  position, seed)``.
* :class:`VisualEncoder` — a small ConvNet (3 convs + adaptive
  pool + linear) mapping ``(3, 32, 32) → (d_model,)``.
* :func:`cross_modal_alignment_loss` — MSE between the
  ``VisualEncoder`` output and the mean of the LM's token
  embeddings for the caption tokens. Trains the encoder to
  place visual stimuli in the SAME slot space the LM uses for
  the words.

This is the **F87 grounding step**: after training, the
encoder's output for an image of a red circle should be close
in cosine distance to the LM's slot for the phrase
``"the red circle"``. The cross-modal universality of the F62
combiner can then be tested (V6 invariant in the F87 PoC).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "COLOR_RGB",
    "SHAPE_NAMES",
    "render_colour_shape",
    "VisualEncoder",
    "cross_modal_alignment_loss",
    "ColourShapeDataset",
]


# ─────────────────────────────────────────────────────────────────
# Constants — eight colours, eight shapes
# ─────────────────────────────────────────────────────────────────


COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red":    (220,  40,  40),
    "blue":   ( 40,  80, 220),
    "green":  ( 40, 180,  60),
    "yellow": (240, 220,  40),
    "black":  ( 20,  20,  20),
    "white":  (240, 240, 240),
    "pink":   (255, 150, 180),
    "orange": (240, 150,  40),
}

SHAPE_NAMES: tuple[str, ...] = (
    "circle", "square", "triangle", "star",
    "heart", "line", "dot", "cross",
)


# ─────────────────────────────────────────────────────────────────
# Procedural rendering
# ─────────────────────────────────────────────────────────────────


def _draw_circle(
    canvas: np.ndarray, *, cy: int, cx: int, radius: int,
    rgb: tuple[int, int, int],
) -> None:
    H, W, _ = canvas.shape
    yy, xx = np.meshgrid(
        np.arange(H), np.arange(W), indexing="ij",
    )
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    canvas[mask] = rgb


def _draw_square(
    canvas: np.ndarray, *, cy: int, cx: int, half: int,
    rgb: tuple[int, int, int],
) -> None:
    H, W, _ = canvas.shape
    y0 = max(0, cy - half)
    y1 = min(H, cy + half + 1)
    x0 = max(0, cx - half)
    x1 = min(W, cx + half + 1)
    canvas[y0:y1, x0:x1] = rgb


def _draw_triangle(
    canvas: np.ndarray, *, cy: int, cx: int, half: int,
    rgb: tuple[int, int, int],
) -> None:
    H, W, _ = canvas.shape
    yy, xx = np.meshgrid(
        np.arange(H), np.arange(W), indexing="ij",
    )
    # Equilateral-ish: |x - cx| / half <= (cy + half - y) / (2 * half)
    mask = (
        (yy >= cy - half)
        & (yy <= cy + half)
        & (np.abs(xx - cx)
           <= ((cy + half - yy).astype(np.float32)
               / max(1, 2 * half) * 2 * half).clip(min=0))
    )
    canvas[mask] = rgb


def _draw_star(
    canvas: np.ndarray, *, cy: int, cx: int, half: int,
    rgb: tuple[int, int, int],
) -> None:
    # 4-pointed star = two thin rectangles
    H, W, _ = canvas.shape
    half = max(2, half)
    thickness = max(1, half // 4)
    yy, xx = np.meshgrid(
        np.arange(H), np.arange(W), indexing="ij",
    )
    horiz = (np.abs(yy - cy) <= thickness) & (
        np.abs(xx - cx) <= half
    )
    vert = (np.abs(xx - cx) <= thickness) & (
        np.abs(yy - cy) <= half
    )
    diag1 = (
        np.abs((xx - cx) - (yy - cy)) <= thickness
    ) & (np.abs(xx - cx) <= half)
    diag2 = (
        np.abs((xx - cx) + (yy - cy)) <= thickness
    ) & (np.abs(xx - cx) <= half)
    mask = horiz | vert | diag1 | diag2
    canvas[mask] = rgb


def _draw_heart(
    canvas: np.ndarray, *, cy: int, cx: int, half: int,
    rgb: tuple[int, int, int],
) -> None:
    H, W, _ = canvas.shape
    half = max(3, half)
    # Two circles (lobes) + triangle (bottom)
    _draw_circle(
        canvas, cy=cy - half // 3, cx=cx - half // 2,
        radius=half // 2, rgb=rgb,
    )
    _draw_circle(
        canvas, cy=cy - half // 3, cx=cx + half // 2,
        radius=half // 2, rgb=rgb,
    )
    yy, xx = np.meshgrid(
        np.arange(H), np.arange(W), indexing="ij",
    )
    mask = (
        (yy >= cy - half // 3) & (yy <= cy + half)
        & (np.abs(xx - cx)
           <= (cy + half - yy).astype(np.float32))
    )
    canvas[mask] = rgb


def _draw_line(
    canvas: np.ndarray, *, cy: int, cx: int, half: int,
    rgb: tuple[int, int, int],
) -> None:
    H, W, _ = canvas.shape
    thickness = max(1, half // 8)
    y0 = max(0, cy - thickness)
    y1 = min(H, cy + thickness + 1)
    x0 = max(0, cx - half)
    x1 = min(W, cx + half + 1)
    canvas[y0:y1, x0:x1] = rgb


def _draw_dot(
    canvas: np.ndarray, *, cy: int, cx: int, half: int,
    rgb: tuple[int, int, int],
) -> None:
    radius = max(2, half // 3)
    _draw_circle(canvas, cy=cy, cx=cx, radius=radius, rgb=rgb)


def _draw_cross(
    canvas: np.ndarray, *, cy: int, cx: int, half: int,
    rgb: tuple[int, int, int],
) -> None:
    H, W, _ = canvas.shape
    thickness = max(1, half // 4)
    yy, xx = np.meshgrid(
        np.arange(H), np.arange(W), indexing="ij",
    )
    horiz = (np.abs(yy - cy) <= thickness) & (
        np.abs(xx - cx) <= half
    )
    vert = (np.abs(xx - cx) <= thickness) & (
        np.abs(yy - cy) <= half
    )
    canvas[horiz | vert] = rgb


_SHAPE_DRAWERS = {
    "circle": _draw_circle,
    "square": _draw_square,
    "triangle": _draw_triangle,
    "star": _draw_star,
    "heart": _draw_heart,
    "line": _draw_line,
    "dot": _draw_dot,
    "cross": _draw_cross,
}


def render_colour_shape(
    color: str, shape: str, *, size: str = "medium",
    position: str = "center", seed: int | None = None,
    image_size: int = 32, background: str = "white",
) -> np.ndarray:
    """Render a 32x32 RGB image of a coloured shape on a
    background.

    Args:
        color: one of :data:`COLOR_RGB` keys.
        shape: one of :data:`SHAPE_NAMES`.
        size: ``"small"`` / ``"medium"`` / ``"large"``.
        position: ``"center"`` / ``"top"`` / ``"bottom"`` /
            ``"left"`` / ``"right"``.
        seed: optional jitter (for non-deterministic
            positioning); ``None`` = exactly canonical.
        image_size: pixels per side (default 32).
        background: ``"white"`` / ``"black"`` / ``"gray"``.

    Returns a ``(image_size, image_size, 3)`` uint8 numpy array
    with the shape drawn in its colour over the background.
    """
    if color not in COLOR_RGB:
        raise ValueError(f"unknown color {color!r}")
    if shape == "circle":
        drawer = _draw_circle
    elif shape == "square":
        drawer = _draw_square
    else:
        drawer = _SHAPE_DRAWERS.get(shape)
    if drawer is None:
        raise ValueError(f"unknown shape {shape!r}")
    bg_rgb = {
        "white": (240, 240, 240),
        "black": (16, 16, 16),
        "gray":  (128, 128, 128),
    }.get(background, (240, 240, 240))
    canvas = np.full(
        (image_size, image_size, 3), bg_rgb, dtype=np.uint8,
    )
    size_to_half = {
        "small":  image_size // 6,
        "medium": image_size // 4,
        "large":  image_size // 3,
    }
    half = size_to_half.get(size, image_size // 4)
    position_offsets = {
        "center": (0, 0),
        "top": (-image_size // 5, 0),
        "bottom": (image_size // 5, 0),
        "left": (0, -image_size // 5),
        "right": (0, image_size // 5),
    }
    dy, dx = position_offsets.get(position, (0, 0))
    cy = image_size // 2 + dy
    cx = image_size // 2 + dx
    if seed is not None:
        rng = np.random.default_rng(seed)
        cy += int(rng.integers(-2, 3))
        cx += int(rng.integers(-2, 3))
    if shape == "circle":
        drawer(
            canvas, cy=cy, cx=cx, radius=half,
            rgb=COLOR_RGB[color],
        )
    else:
        drawer(
            canvas, cy=cy, cx=cx, half=half,
            rgb=COLOR_RGB[color],
        )
    return canvas


def render_colour_shape_tensor(
    color: str, shape: str, **kwargs,
) -> torch.Tensor:
    """Convenience wrapper: returns ``(3, H, W)`` float tensor
    in ``[0, 1]`` instead of uint8 (H, W, 3) array."""
    arr = render_colour_shape(color, shape, **kwargs)
    return (
        torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    )


# ─────────────────────────────────────────────────────────────────
# VisualEncoder — CNN producing PCM slots
# ─────────────────────────────────────────────────────────────────


class VisualEncoder(nn.Module):
    """Small ConvNet that maps a 32x32 RGB image to a single
    slot of dimension ``d_model``.

    Architecture:
        ``Conv(3→16, 3x3) → BN → SiLU → MaxPool(2)``
        ``Conv(16→32, 3x3) → BN → SiLU → MaxPool(2)``
        ``Conv(32→64, 3x3) → BN → SiLU``
        ``AdaptiveAvgPool(1)``
        ``Linear(64 → d_model)``

    Total: ~30k parameters at d_model=128.
    """

    def __init__(
        self, d_model: int = 128, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(64, d_model)
        # Init projection small (matches GPT-2 / F79 init style)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, 3, H, W)`` float in ``[0, 1]``.
        Returns ``(B, d_model)`` slot."""
        h = self.features(x).flatten(1)
        h = self.dropout(h)
        return self.proj(h)


# ─────────────────────────────────────────────────────────────────
# Cross-modal alignment loss
# ─────────────────────────────────────────────────────────────────


def cross_modal_alignment_loss(
    image_slots: torch.Tensor,
    caption_slots: torch.Tensor, *,
    contrastive_weight: float = 0.5,
    mse_weight: float = 1.0,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict]:
    """Train ``image_slots`` to align with ``caption_slots`` via
    a combined MSE + InfoNCE objective.

    Both inputs must be ``(B, d_model)`` tensors with the same
    batch size; row ``i`` of one corresponds to row ``i`` of the
    other.

    The MSE term forces direct equality (strong supervision),
    while the InfoNCE term ensures distinct items remain
    discriminable (CLIP-style). The combined loss preserves
    the F62 slot-space geometry: images and captions share the
    same metric.
    """
    mse = F.mse_loss(image_slots, caption_slots)
    # Contrastive (InfoNCE): each image should be most similar
    # to its OWN caption among the batch.
    img_n = F.normalize(image_slots, dim=-1)
    cap_n = F.normalize(caption_slots, dim=-1)
    sim = (img_n @ cap_n.t()) / max(temperature, 1e-6)
    targets = torch.arange(
        image_slots.shape[0], device=image_slots.device,
    )
    contrastive = 0.5 * (
        F.cross_entropy(sim, targets)
        + F.cross_entropy(sim.t(), targets)
    )
    loss = mse * mse_weight + contrastive * contrastive_weight
    return loss, {
        "mse": float(mse.item()),
        "contrastive": float(contrastive.item()),
    }


# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────


@dataclass
class ColourShapeDataset:
    """Procedurally generated set of (image, caption) pairs."""

    images: torch.Tensor  # (N, 3, H, W) float
    captions: list[list[str]]  # word lists
    color_labels: torch.Tensor  # (N,) long, 0..7
    shape_labels: torch.Tensor  # (N,) long, 0..7

    @classmethod
    def build(
        cls, *, n_per_combo: int = 8,
        image_size: int = 32,
        backgrounds: tuple[str, ...] = ("white",),
        sizes: tuple[str, ...] = ("small", "medium", "large"),
        positions: tuple[str, ...] = ("center",),
        seed: int = 0,
    ) -> "ColourShapeDataset":
        rng = np.random.default_rng(seed)
        colors = list(COLOR_RGB.keys())
        shapes = list(SHAPE_NAMES)
        records: list[tuple[
            np.ndarray, list[str], int, int,
        ]] = []
        for ci, color in enumerate(colors):
            for si, shape in enumerate(shapes):
                for _ in range(n_per_combo):
                    sz = rng.choice(sizes)
                    pos = rng.choice(positions)
                    bg = rng.choice(backgrounds)
                    seed_i = int(rng.integers(0, 1 << 30))
                    img = render_colour_shape(
                        color, shape, size=sz, position=pos,
                        seed=seed_i, image_size=image_size,
                        background=bg,
                    )
                    caption = ["the", color, shape, "."]
                    records.append((img, caption, ci, si))
        rng.shuffle(records)
        images = torch.from_numpy(
            np.stack([r[0] for r in records])
        ).permute(0, 3, 1, 2).float() / 255.0
        captions = [r[1] for r in records]
        color_labels = torch.tensor(
            [r[2] for r in records], dtype=torch.long,
        )
        shape_labels = torch.tensor(
            [r[3] for r in records], dtype=torch.long,
        )
        return cls(
            images=images, captions=captions,
            color_labels=color_labels, shape_labels=shape_labels,
        )

    def __len__(self) -> int:
        return len(self.images)
