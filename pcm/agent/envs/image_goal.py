"""pcm.agent.envs.image_goal — synthetic image-described goals.

For F72 (PCM v6.4-image), the goal is specified as a *small
synthetic image* of the target digit rather than as text tokens.
We synthesise 16×16 greyscale images of digits 0–9 on the fly
(``draw_digit_image``) so the experiment is self-contained
(no MNIST download dependency) while remaining a faithful test
of "image → slot" perception:

* Each digit has multiple rendered variants (font, position,
  noise) — synonymous *images* of the same goal-digit.
* Pixel-wise alias invariance is the analogue of F68's
  token-sequence alias invariance.

The env state and dynamics are unchanged from
``CyclicNavEnv``; only the goal-spec interface changes.

Public API::

    draw_digit_image(digit, *, size=16, variant=0, rng=None)
        → torch.Tensor of shape (1, size, size), float32 in [0, 1].
"""
from __future__ import annotations

import math

import torch


__all__ = [
    "draw_digit_image",
    "ImageGoalCyclicNavEnv",
]


# Compact 5×3 digit glyphs (rows top→bottom, columns left→right).
# Each glyph fits inside a 5×3 cell; we centre it in the 16×16
# canvas with optional offset jitter / Gaussian noise.
DIGIT_GLYPHS = {
    0: [
        " # ",
        "# #",
        "# #",
        "# #",
        " # ",
    ],
    1: [
        " # ",
        "## ",
        " # ",
        " # ",
        "###",
    ],
    2: [
        "## ",
        "  #",
        " # ",
        "#  ",
        "###",
    ],
    3: [
        "## ",
        "  #",
        " # ",
        "  #",
        "## ",
    ],
    4: [
        "# #",
        "# #",
        "###",
        "  #",
        "  #",
    ],
    5: [
        "###",
        "#  ",
        "## ",
        "  #",
        "## ",
    ],
    6: [
        " # ",
        "#  ",
        "## ",
        "# #",
        " # ",
    ],
    7: [
        "###",
        "  #",
        " # ",
        " # ",
        " # ",
    ],
    8: [
        " # ",
        "# #",
        " # ",
        "# #",
        " # ",
    ],
    9: [
        " # ",
        "# #",
        " ##",
        "  #",
        " # ",
    ],
}


def draw_digit_image(
    digit: int, *, size: int = 16, variant: int = 0,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Render ``digit`` as a (1, size, size) greyscale image.

    Variants 0..N-1 give different surface forms of the same
    digit (different rendered offsets, light blur, noise level)
    — they are pixel-distinct but should map to the same slot.
    """
    if digit not in DIGIT_GLYPHS:
        # For digits ≥ 10, render as two stacked single digits
        # — F72 uses digits 0..19 (N=20 states).
        d_tens, d_ones = divmod(digit, 10)
        img_t = draw_digit_image(
            d_tens, size=size, variant=variant, rng=rng,
        )
        img_o = draw_digit_image(
            d_ones, size=size, variant=variant, rng=rng,
        )
        # Stack horizontally by averaging shifted images
        canvas = img_t.clone()
        shift = size // 4
        canvas[:, :, shift:] = torch.maximum(
            canvas[:, :, shift:], img_o[:, :, :-shift],
        )
        return canvas
    canvas = torch.zeros(1, size, size, dtype=torch.float32)
    glyph = DIGIT_GLYPHS[digit]
    gh, gw = len(glyph), len(glyph[0])
    # Centre with small per-variant offset
    base_top = (size - gh) // 2
    base_left = (size - gw) // 2
    if rng is None:
        rng = torch.Generator(device="cpu").manual_seed(
            42 + variant
        )
    dy = int(torch.randint(-2, 3, (1,), generator=rng).item())
    dx = int(torch.randint(-2, 3, (1,), generator=rng).item())
    top = max(0, min(size - gh, base_top + dy))
    left = max(0, min(size - gw, base_left + dx))
    for r, row in enumerate(glyph):
        for c, ch in enumerate(row):
            if ch == "#":
                canvas[0, top + r, left + c] = 1.0
    # Optional Gaussian noise per variant
    noise_amp = 0.05 + 0.02 * variant
    canvas = canvas + noise_amp * torch.randn(
        canvas.shape, generator=rng,
    )
    canvas = canvas.clamp(0.0, 1.0)
    return canvas
