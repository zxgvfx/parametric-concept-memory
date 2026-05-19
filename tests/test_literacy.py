"""Unit tests for F88 ``pcm.literacy`` module."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from pcm.lm import HybridPCMMiniLM
from pcm.literacy import (
    GLYPH_SIZE,
    GlyphDecoder,
    GlyphEncoder,
    alignment_loss,
    build_glyph_table,
    multimodal_forward,
    reconstruction_loss,
    render_glyph,
    render_glyph_tensor,
)


# ─────────────────────────────────────────────────────────────────
# Glyph rendering
# ─────────────────────────────────────────────────────────────────


def test_render_glyph_shape_and_dtype() -> None:
    arr = render_glyph("hello", size=(16, 64))
    assert arr.shape == (16, 64)
    assert arr.dtype == np.uint8


def test_render_glyph_distinct_words_distinct_pixels() -> None:
    a = render_glyph("cat")
    b = render_glyph("dog")
    diff = np.abs(a.astype(int) - b.astype(int)).sum()
    assert diff > 100, (
        f"two different words render to almost-identical glyphs "
        f"(diff = {diff})"
    )


def test_render_glyph_same_word_reproducible() -> None:
    a = render_glyph("apple")
    b = render_glyph("apple")
    assert np.array_equal(a, b)


def test_render_glyph_tensor_in_unit_range() -> None:
    t = render_glyph_tensor("hello")
    assert t.shape == (1, 16, 64)
    assert 0.0 <= t.min().item() <= t.max().item() <= 1.0


# ─────────────────────────────────────────────────────────────────
# Glyph table
# ─────────────────────────────────────────────────────────────────


def test_build_glyph_table_shape() -> None:
    itos = ["<pad>", "<unk>", "<bos>", "<eos>", "the", "cat"]
    t = build_glyph_table(itos, special_tokens=4)
    assert t.shape == (6, 1, *GLYPH_SIZE)


def test_build_glyph_table_special_tokens_blank() -> None:
    """First N entries (special tokens) should be all-zero
    images, since they don't correspond to printable words."""
    itos = ["<pad>", "<unk>", "<bos>", "<eos>", "the"]
    t = build_glyph_table(itos, special_tokens=4)
    # Specials are blank
    assert t[:4].abs().max().item() == 0.0
    # Content has ink (non-zero pixels)
    assert t[4].abs().max().item() > 0.0


def test_build_glyph_table_distinct_tokens_distinct_glyphs() -> None:
    itos = ["<pad>", "<unk>", "<bos>", "<eos>", "cat", "dog"]
    t = build_glyph_table(itos, special_tokens=4)
    diff = (t[4] - t[5]).abs().sum().item()
    assert diff > 0.1


# ─────────────────────────────────────────────────────────────────
# GlyphEncoder
# ─────────────────────────────────────────────────────────────────


def test_glyph_encoder_forward_shape() -> None:
    torch.manual_seed(0)
    enc = GlyphEncoder(d_model=64)
    x = torch.rand(4, 1, *GLYPH_SIZE)
    y = enc(x)
    assert y.shape == (4, 64)


def test_glyph_encoder_init_does_not_nan() -> None:
    torch.manual_seed(0)
    enc = GlyphEncoder(d_model=128)
    x = torch.rand(8, 1, *GLYPH_SIZE)
    y = enc(x)
    assert not torch.isnan(y).any()
    assert y.abs().max().item() < 100


def test_glyph_encoder_distinct_inputs_distinct_outputs() -> None:
    """At random init, two distinct glyphs produce outputs that
    differ by *some* amount (cosine < 1.0). After a brief MSE
    training they should differ substantially."""
    torch.manual_seed(0)
    enc = GlyphEncoder(d_model=32)
    enc.eval()
    g_cat = render_glyph_tensor("cat").unsqueeze(0)
    g_dog = render_glyph_tensor("dog").unsqueeze(0)
    out_cat = enc(g_cat)
    out_dog = enc(g_dog)
    # Even at random init: the outputs should not be IDENTICAL
    assert not torch.equal(out_cat, out_dog)
    # Brief MSE training to different targets → outputs diverge
    enc.train()
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-2)
    tgt_cat = torch.tensor([1.0] + [0.0] * 31).unsqueeze(0)
    tgt_dog = torch.tensor([0.0] * 31 + [1.0]).unsqueeze(0)
    for _ in range(30):
        loss = (
            F.mse_loss(enc(g_cat), tgt_cat)
            + F.mse_loss(enc(g_dog), tgt_dog)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    enc.eval()
    out_cat = enc(g_cat)
    out_dog = enc(g_dog)
    diff = (out_cat - out_dog).abs().sum().item()
    assert diff > 0.1, (
        f"after training, cat and dog outputs differ by only "
        f"{diff} (should be > 0.1)"
    )


def test_glyph_encoder_trains_on_mse() -> None:
    torch.manual_seed(0)
    enc = GlyphEncoder(d_model=16)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-2)
    x = torch.rand(8, 1, *GLYPH_SIZE)
    target = torch.randn(8, 16)
    initial = F.mse_loss(enc(x), target).item()
    for _ in range(10):
        out = enc(x)
        loss = F.mse_loss(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    final = F.mse_loss(enc(x), target).item()
    assert final < initial


# ─────────────────────────────────────────────────────────────────
# Alignment loss
# ─────────────────────────────────────────────────────────────────


def test_alignment_loss_zero_for_identical() -> None:
    a = torch.randn(8, 16)
    loss, diag = alignment_loss(a, a)
    assert diag["mse"] < 1e-6
    assert diag["cosine"] > 0.99


def test_alignment_loss_positive_for_random() -> None:
    torch.manual_seed(0)
    a = torch.randn(8, 16)
    b = torch.randn(8, 16)
    loss, diag = alignment_loss(a, b)
    assert loss.item() > 0


def test_alignment_loss_reduces_with_training() -> None:
    torch.manual_seed(0)
    enc = GlyphEncoder(d_model=32)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-2)
    x = torch.rand(8, 1, *GLYPH_SIZE)
    target = torch.randn(8, 32)
    losses = []
    for _ in range(10):
        out = enc(x)
        loss, _ = alignment_loss(out, target)
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]


# ─────────────────────────────────────────────────────────────────
# multimodal_forward
# ─────────────────────────────────────────────────────────────────


def _make_lm_and_encoder(vocab: int = 8, d_model: int = 16):
    torch.manual_seed(0)
    lm = HybridPCMMiniLM(
        vocab=vocab, d_model=d_model, n_layers=2,
        n_heads=4, attn_every=4,
    )
    enc = GlyphEncoder(d_model=d_model)
    itos = [f"tok{i}" for i in range(vocab)]
    table = build_glyph_table(itos, special_tokens=4)
    return lm, enc, table


def test_multimodal_forward_shape() -> None:
    lm, enc, table = _make_lm_and_encoder(vocab=8, d_model=16)
    lm.eval()
    enc.eval()
    x = torch.tensor([[4, 5, 6, 7]])
    logits, mask = multimodal_forward(
        lm, enc, x, table, mix_rate=0.5,
    )
    assert logits.shape == (1, 4, 8)
    assert mask.shape == (1, 4)


def test_multimodal_forward_mix_rate_0_equals_text_only() -> None:
    lm, enc, table = _make_lm_and_encoder(vocab=8, d_model=16)
    lm.eval()
    enc.eval()
    x = torch.tensor([[4, 5, 6, 7]])
    # mix_rate=0 means no glyph encoding, just text
    logits_mm, mask = multimodal_forward(
        lm, enc, x, table, mix_rate=0.0,
    )
    logits_text = lm(x)
    assert mask.sum().item() == 0
    # Numerically the two paths follow the same arithmetic; small
    # FP drift can appear depending on prior RNG state but the
    # outputs should match within 1e-3.
    assert torch.allclose(logits_mm, logits_text, atol=1e-3)


def test_multimodal_forward_mix_rate_1_uses_only_glyphs() -> None:
    lm, enc, table = _make_lm_and_encoder(vocab=8, d_model=16)
    lm.eval()
    enc.eval()
    x = torch.tensor([[4, 5, 6, 7]])
    logits, mask = multimodal_forward(
        lm, enc, x, table, mix_rate=1.0,
    )
    assert mask.sum().item() == 4
    # Compare to pure text path: should differ (untrained
    # encoder ≠ token embeddings)
    logits_text = lm(x)
    assert not torch.allclose(logits, logits_text, atol=1e-3)


def test_multimodal_forward_gradient_flows_to_encoder() -> None:
    lm, enc, table = _make_lm_and_encoder(vocab=8, d_model=16)
    x = torch.tensor([[4, 5, 6, 7]])
    y = torch.tensor([[5, 6, 7, 4]])
    logits, _ = multimodal_forward(
        lm, enc, x, table, mix_rate=1.0,
    )
    loss = F.cross_entropy(
        logits.reshape(-1, 8), y.reshape(-1),
    )
    loss.backward()
    # encoder must have gradient
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in enc.parameters()
    )
    assert has_grad


# ─────────────────────────────────────────────────────────────────
# F89 GlyphDecoder
# ─────────────────────────────────────────────────────────────────


def test_glyph_decoder_forward_shape() -> None:
    torch.manual_seed(0)
    dec = GlyphDecoder(d_model=32, out_h=16, out_w=64)
    slot = torch.randn(4, 32)
    img = dec(slot)
    assert img.shape == (4, 1, 16, 64)


def test_glyph_decoder_output_in_unit_range() -> None:
    """Sigmoid output must be in [0, 1] to match glyph pixel
    range."""
    torch.manual_seed(0)
    dec = GlyphDecoder(d_model=32)
    slot = torch.randn(8, 32)
    img = dec(slot)
    assert img.min().item() >= 0.0
    assert img.max().item() <= 1.0


def test_glyph_decoder_init_does_not_nan() -> None:
    torch.manual_seed(0)
    dec = GlyphDecoder(d_model=128)
    slot = torch.randn(4, 128)
    img = dec(slot)
    assert not torch.isnan(img).any()


def test_glyph_decoder_rejects_non_div8_output() -> None:
    """out_h and out_w must each be divisible by 8 for the
    three stride-2 ConvT layers to produce exact target size."""
    try:
        _ = GlyphDecoder(d_model=32, out_h=15, out_w=64)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_glyph_decoder_trains_on_reconstruction() -> None:
    torch.manual_seed(0)
    dec = GlyphDecoder(d_model=32)
    opt = torch.optim.AdamW(dec.parameters(), lr=1e-2)
    slot = torch.randn(8, 32)
    target = torch.rand(8, 1, 16, 64)
    losses = []
    for _ in range(20):
        out = dec(slot)
        loss, _ = reconstruction_loss(out, target)
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0] * 0.7


def test_glyph_decoder_distinct_slots_distinct_outputs() -> None:
    torch.manual_seed(0)
    dec = GlyphDecoder(d_model=32)
    dec.eval()
    a = torch.randn(1, 32)
    b = torch.randn(1, 32)
    out_a = dec(a)
    out_b = dec(b)
    diff = (out_a - out_b).abs().sum().item()
    assert diff > 0.01, (
        f"distinct slots produce nearly-identical glyphs "
        f"(diff = {diff})"
    )


# ─────────────────────────────────────────────────────────────────
# Reconstruction loss
# ─────────────────────────────────────────────────────────────────


def test_reconstruction_loss_zero_for_identical() -> None:
    a = torch.rand(4, 1, 16, 64)
    loss, diag = reconstruction_loss(a, a)
    assert diag["mse"] < 1e-6


def test_reconstruction_loss_increases_for_random() -> None:
    torch.manual_seed(0)
    a = torch.rand(4, 1, 16, 64)
    b = torch.rand(4, 1, 16, 64)
    loss, diag = reconstruction_loss(a, b)
    assert loss.item() > 0


def test_reconstruction_loss_with_bce_includes_both() -> None:
    a = torch.rand(4, 1, 16, 64)
    b = torch.rand(4, 1, 16, 64)
    _, diag = reconstruction_loss(
        a, b, mse_weight=1.0, bce_weight=0.5,
    )
    assert "mse" in diag
    assert "bce" in diag


# ─────────────────────────────────────────────────────────────────
# Encoder-decoder cycle smoke
# ─────────────────────────────────────────────────────────────────


def test_glyph_encoder_decoder_cycle_runs() -> None:
    """Smoke test: encoder(glyph) → decoder(slot) → glyph works
    without shape mismatches or NaNs."""
    torch.manual_seed(0)
    enc = GlyphEncoder(d_model=32)
    dec = GlyphDecoder(d_model=32)
    enc.eval()
    dec.eval()
    itos = ["<pad>", "<unk>", "<bos>", "<eos>", "cat", "dog"]
    table = build_glyph_table(itos, special_tokens=4)
    slots = enc(table[4:])
    rec = dec(slots)
    assert rec.shape == table[4:].shape
    assert not torch.isnan(rec).any()
