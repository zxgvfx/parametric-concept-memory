"""Unit tests for F87 ``pcm.vision`` module."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from pcm.vision import (
    COLOR_RGB,
    SHAPE_NAMES,
    ColourShapeDataset,
    VisualEncoder,
    cross_modal_alignment_loss,
    render_colour_shape,
    render_colour_shape_tensor,
)


# ─────────────────────────────────────────────────────────────────
# Renderer
# ─────────────────────────────────────────────────────────────────


def test_renderer_outputs_correct_shape() -> None:
    img = render_colour_shape("red", "circle", image_size=32)
    assert img.shape == (32, 32, 3)
    assert img.dtype == np.uint8


def test_renderer_all_color_shape_combos_succeed() -> None:
    for color in COLOR_RGB:
        for shape in SHAPE_NAMES:
            img = render_colour_shape(
                color, shape, image_size=32, seed=42,
            )
            assert img.shape == (32, 32, 3)


def test_renderer_color_visible_in_image() -> None:
    """If we render a red circle, the image should contain
    pixels close to the red RGB value."""
    img = render_colour_shape(
        "red", "circle", size="large", image_size=32,
    )
    target = np.array(COLOR_RGB["red"])
    flat = img.reshape(-1, 3)
    # Distance to the target colour
    dist = np.linalg.norm(
        flat.astype(np.float32) - target, axis=-1,
    )
    # At least 50 pixels should be very close to target
    close = (dist < 5).sum()
    assert close > 50, (
        f"only {close} pixels close to {target}"
    )


def test_renderer_seed_jitter_is_reproducible() -> None:
    a = render_colour_shape("blue", "square", seed=7)
    b = render_colour_shape("blue", "square", seed=7)
    assert np.array_equal(a, b)


def test_renderer_tensor_in_unit_range() -> None:
    t = render_colour_shape_tensor("green", "triangle")
    assert t.shape == (3, 32, 32)
    assert 0.0 <= t.min().item() <= t.max().item() <= 1.0


# ─────────────────────────────────────────────────────────────────
# VisualEncoder
# ─────────────────────────────────────────────────────────────────


def test_visual_encoder_forward_shape() -> None:
    torch.manual_seed(0)
    enc = VisualEncoder(d_model=64)
    x = torch.rand(4, 3, 32, 32)
    y = enc(x)
    assert y.shape == (4, 64)


def test_visual_encoder_init_loss_reasonable() -> None:
    """Random initial encoder should not produce NaN."""
    torch.manual_seed(0)
    enc = VisualEncoder(d_model=64)
    x = torch.rand(8, 3, 32, 32)
    y = enc(x)
    assert not torch.isnan(y).any()
    assert y.abs().max().item() < 100


def test_visual_encoder_trains() -> None:
    """One gradient step should reduce loss."""
    torch.manual_seed(0)
    enc = VisualEncoder(d_model=64)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-2)
    x = torch.rand(8, 3, 32, 32)
    target = torch.randn(8, 64)
    initial_loss = F.mse_loss(enc(x), target).item()
    for _ in range(5):
        out = enc(x)
        loss = F.mse_loss(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    final_loss = F.mse_loss(enc(x), target).item()
    assert final_loss < initial_loss


# ─────────────────────────────────────────────────────────────────
# Cross-modal alignment loss
# ─────────────────────────────────────────────────────────────────


def test_alignment_loss_perfect_pairs_zero_mse() -> None:
    torch.manual_seed(0)
    slots = torch.randn(8, 16)
    loss, diag = cross_modal_alignment_loss(slots, slots)
    assert diag["mse"] < 1e-6


def test_alignment_loss_random_pairs_finite() -> None:
    torch.manual_seed(0)
    a = torch.randn(8, 16)
    b = torch.randn(8, 16)
    loss, diag = cross_modal_alignment_loss(a, b)
    assert not torch.isnan(loss)
    assert diag["mse"] > 0
    assert diag["contrastive"] > 0


def test_alignment_loss_contrastive_penalises_misalignment() -> None:
    """When all image_slots are identical, contrastive loss
    should be high (each one can't be discriminated)."""
    img = torch.zeros(8, 16)
    cap = torch.randn(8, 16)
    _, diag_collapsed = cross_modal_alignment_loss(img, cap)
    img_distinct = torch.randn(8, 16) * 0.01
    _, diag_distinct = cross_modal_alignment_loss(
        img_distinct + cap, cap,
    )
    assert diag_collapsed["contrastive"] > diag_distinct["contrastive"]


# ─────────────────────────────────────────────────────────────────
# ColourShapeDataset
# ─────────────────────────────────────────────────────────────────


def test_dataset_builds_with_correct_count() -> None:
    ds = ColourShapeDataset.build(n_per_combo=3)
    expected = 8 * 8 * 3  # 8 colors × 8 shapes × 3 per combo
    assert len(ds) == expected
    assert ds.images.shape == (expected, 3, 32, 32)


def test_dataset_labels_cover_all_classes() -> None:
    ds = ColourShapeDataset.build(n_per_combo=2)
    assert sorted(ds.color_labels.unique().tolist()) == list(range(8))
    assert sorted(ds.shape_labels.unique().tolist()) == list(range(8))


def test_dataset_captions_format_correct() -> None:
    ds = ColourShapeDataset.build(n_per_combo=1)
    for c, cls_idx, sh_idx in zip(
        ds.captions, ds.color_labels.tolist(),
        ds.shape_labels.tolist(),
    ):
        assert c[0] == "the"
        assert c[-1] == "."
        assert len(c) == 4  # the COLOR SHAPE .
        # color and shape token must match labels
        expected_color = list(COLOR_RGB.keys())[cls_idx]
        expected_shape = SHAPE_NAMES[sh_idx]
        assert c[1] == expected_color
        assert c[2] == expected_shape


def test_dataset_images_in_unit_range() -> None:
    ds = ColourShapeDataset.build(n_per_combo=2)
    assert 0.0 <= ds.images.min().item()
    assert ds.images.max().item() <= 1.0
