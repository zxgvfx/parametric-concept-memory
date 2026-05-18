"""Unit tests for F81 ``GatedAttentionLayer`` / ``HybridPCMMiniLM``
/ ``build_matched_pentad`` in ``pcm.lm``."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from pcm.lm import (
    GatedAttentionLayer,
    GatedPCMLayer,
    GatedPCMMiniLM,
    GPTMiniLM,
    HybridPCMMiniLM,
    PCMMiniLM,
    PCMTopKMiniLM,
    build_matched_pentad,
    count_params,
)


# ─────────────────────────────────────────────────────────────────
# GatedAttentionLayer
# ─────────────────────────────────────────────────────────────────


def test_gated_attention_forward_shape() -> None:
    torch.manual_seed(0)
    layer = GatedAttentionLayer(d_model=16, n_heads=4)
    x = torch.randn(2, 8, 16)
    y = layer(x)
    assert y.shape == (2, 8, 16)


def test_gated_attention_compute_gates_in_unit_range() -> None:
    torch.manual_seed(0)
    layer = GatedAttentionLayer(d_model=16, n_heads=4)
    x = torch.randn(3, 10, 16)
    g = layer.compute_gates(x)
    assert g.shape == (3, 10, 16)
    assert (g >= 0).all() and (g <= 1).all()


def test_gated_attention_dmodel_must_divide_nheads() -> None:
    with pytest.raises(ValueError, match="must divide"):
        _ = GatedAttentionLayer(d_model=13, n_heads=4)


def test_gated_attention_causal_mask_no_leak_from_future() -> None:
    """Token t output must not depend on tokens > t."""
    torch.manual_seed(0)
    layer = GatedAttentionLayer(d_model=16, n_heads=4)
    layer.eval()
    x = torch.randn(1, 8, 16)
    y_a = layer(x.clone())
    # Modify a future position only
    x_modified = x.clone()
    x_modified[:, -1] += torch.randn_like(x_modified[:, -1]) * 10
    y_b = layer(x_modified)
    # Outputs at positions [0..L-2] should be unchanged (only
    # the last position could differ)
    assert torch.allclose(y_a[:, :-1], y_b[:, :-1], atol=1e-5)


# ─────────────────────────────────────────────────────────────────
# HybridPCMMiniLM
# ─────────────────────────────────────────────────────────────────


def test_hybrid_pcm_mini_lm_layer_layout_3_to_1() -> None:
    m = HybridPCMMiniLM(
        vocab=64, d_model=16, n_layers=4, n_heads=4,
        attn_every=4,
    )
    assert m.layer_kinds == ["pcm", "pcm", "pcm", "attn"]
    assert m.n_pcm_layers() == 3
    assert m.n_attn_layers() == 1


def test_hybrid_pcm_mini_lm_eight_layer_layout() -> None:
    m = HybridPCMMiniLM(
        vocab=64, d_model=16, n_layers=8, n_heads=4,
        attn_every=4,
    )
    assert m.layer_kinds == [
        "pcm", "pcm", "pcm", "attn",
        "pcm", "pcm", "pcm", "attn",
    ]
    assert m.n_pcm_layers() == 6
    assert m.n_attn_layers() == 2


def test_hybrid_pcm_mini_lm_forward_shape() -> None:
    torch.manual_seed(0)
    m = HybridPCMMiniLM(
        vocab=128, d_model=16, n_layers=4, n_heads=4,
    )
    x = torch.randint(0, 128, (3, 12))
    out = m(x)
    assert out.shape == (3, 12, 128)


def test_hybrid_pcm_mini_lm_hidden_states_shape() -> None:
    torch.manual_seed(0)
    m = HybridPCMMiniLM(
        vocab=64, d_model=8, n_layers=4, n_heads=4,
    )
    x = torch.randint(0, 64, (2, 7))
    h = m.hidden_states(x)
    assert h.shape == (2, 7, 8)


def test_hybrid_pcm_mini_lm_init_loss_near_uniform() -> None:
    torch.manual_seed(0)
    vocab = 512
    m = HybridPCMMiniLM(
        vocab=vocab, d_model=32, n_layers=4, n_heads=4,
    )
    x = torch.randint(0, vocab, (4, 32))
    with torch.no_grad():
        logits = m(x)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab),
        x[:, 1:].reshape(-1),
    )
    uniform = math.log(vocab)
    assert abs(float(loss.item()) - uniform) < 1.5


def test_hybrid_pcm_mini_lm_attn_every_8() -> None:
    """attn_every=8 with n_layers=4 should yield no attention
    layers (the pattern only places attn at positions where
    (i+1) % attn_every == 0)."""
    m = HybridPCMMiniLM(
        vocab=64, d_model=16, n_layers=4, n_heads=4,
        attn_every=8,
    )
    assert m.n_attn_layers() == 0
    assert m.n_pcm_layers() == 4


def test_hybrid_pcm_mini_lm_trainable_step_runs() -> None:
    torch.manual_seed(0)
    vocab = 64
    m = HybridPCMMiniLM(
        vocab=vocab, d_model=16, n_layers=4, n_heads=4,
    )
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, vocab, (4, 16))
    y = torch.randint(0, vocab, (4, 16))
    logits = m(x)
    loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
    loss.backward()
    opt.step()
    # All layers should have non-None gradients
    for layer in m.layers:
        for p in layer.parameters():
            assert p.grad is not None


# ─────────────────────────────────────────────────────────────────
# build_matched_pentad
# ─────────────────────────────────────────────────────────────────


def test_build_matched_pentad_returns_five_models_plus_diag() -> None:
    gpt, pcm, topk, gated, hybrid, diag = build_matched_pentad(
        vocab=256, d_model=32, n_layers=4,
    )
    assert isinstance(gpt, GPTMiniLM)
    assert isinstance(pcm, PCMMiniLM)
    assert isinstance(topk, PCMTopKMiniLM)
    assert isinstance(gated, GatedPCMMiniLM)
    assert isinstance(hybrid, HybridPCMMiniLM)
    assert isinstance(diag, dict)
    # Diag should contain all five model sizes
    assert "hybrid_pcm_params" in diag
    assert "hybrid_pcm_n_pcm_layers" in diag
    assert "hybrid_pcm_n_attn_layers" in diag


def test_build_matched_pentad_within_tolerance() -> None:
    gpt, pcm, topk, gated, hybrid, diag = build_matched_pentad(
        vocab=512, d_model=32, n_layers=4, match_tol=0.30,
    )
    gpt_params = count_params(gpt)
    for name, m in (
        ("pcm", pcm), ("topk", topk),
        ("gated", gated), ("hybrid", hybrid),
    ):
        p = count_params(m)
        ratio = p / gpt_params
        # Allow generous bounds since hybrid uses attention
        assert 0.40 <= ratio <= 1.40, (
            f"{name} param ratio {ratio:.3f} out of [0.40, 1.40]"
        )


def test_build_matched_pentad_hybrid_layout_correct() -> None:
    _, _, _, _, hybrid, diag = build_matched_pentad(
        vocab=128, d_model=16, n_layers=4, attn_every=4,
    )
    assert hybrid.layer_kinds == ["pcm", "pcm", "pcm", "attn"]
    assert diag["hybrid_pcm_n_pcm_layers"] == 3
    assert diag["hybrid_pcm_n_attn_layers"] == 1
    assert diag["hybrid_pcm_attn_every"] == 4
