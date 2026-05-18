"""Unit tests for F80 ``GatedPCMLayer`` / ``GatedPCMMiniLM`` /
``build_matched_quad`` in ``pcm.lm``."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from pcm.lm import (
    GatedPCMLayer,
    GatedPCMMiniLM,
    GPTMiniLM,
    PCMMiniLM,
    PCMTopKMiniLM,
    build_matched_quad,
    count_params,
)


# ─────────────────────────────────────────────────────────────────
# GatedPCMLayer
# ─────────────────────────────────────────────────────────────────


def test_gated_pcm_layer_forward_shape() -> None:
    torch.manual_seed(0)
    layer = GatedPCMLayer(d_model=16, combiner_hidden=32)
    x = torch.randn(2, 8, 16)
    y = layer(x)
    assert y.shape == (2, 8, 16)


def test_gated_pcm_layer_compute_gates_shape_and_range() -> None:
    torch.manual_seed(0)
    layer = GatedPCMLayer(d_model=8, combiner_hidden=16)
    x = torch.randn(3, 10, 8)
    g = layer.compute_gates(x)
    assert g.shape == (3, 10, 8)
    assert (g >= 0).all() and (g <= 1).all()


def test_gated_pcm_layer_force_mean_matches_cumulative_mean() -> None:
    torch.manual_seed(0)
    layer = GatedPCMLayer(d_model=8, combiner_hidden=16)
    x = torch.randn(2, 6, 8)
    # In force_mean mode the context = cumulative mean explicitly
    expected_ctx = x.cumsum(dim=1) / torch.arange(
        1, 7, dtype=x.dtype,
    ).view(1, 6, 1)
    y_force = layer(x, force_mean=True)
    # The combiner output uses (x, expected_ctx). We can check by
    # running the combiner directly with the same context.
    with torch.no_grad():
        ref = layer.norm(layer.dropout(layer.combiner(x, expected_ctx)))
    # Eval mode handling is not strict, but with dropout=0 they
    # should match exactly.
    layer.eval()
    y_force_eval = layer(x, force_mean=True)
    ref_eval = layer.norm(layer.combiner(x, expected_ctx))
    assert torch.allclose(y_force_eval, ref_eval, atol=1e-5)


def test_gated_pcm_layer_gated_scan_differs_from_mean() -> None:
    torch.manual_seed(0)
    layer = GatedPCMLayer(d_model=8, combiner_hidden=16)
    layer.eval()
    x = torch.randn(2, 6, 8)
    y_gated = layer(x)
    y_mean = layer(x, force_mean=True)
    # Different paths should produce different outputs (with
    # default gate bias 0, gates ~0.5, so the recurrence is not
    # identical to cumulative mean).
    assert not torch.allclose(y_gated, y_mean, atol=1e-3)


def test_gated_pcm_layer_initial_gate_around_half() -> None:
    torch.manual_seed(0)
    layer = GatedPCMLayer(d_model=16, gate_bias_init=0.0)
    x = torch.randn(4, 8, 16)
    g = layer.compute_gates(x)
    # With bias 0 and small weight init, gates concentrate around
    # sigmoid(small_noise) ≈ 0.5
    assert 0.4 < g.mean().item() < 0.6


def test_gated_pcm_layer_gate_bias_init_shifts_mean() -> None:
    torch.manual_seed(0)
    layer_pos = GatedPCMLayer(d_model=16, gate_bias_init=2.0)
    layer_neg = GatedPCMLayer(d_model=16, gate_bias_init=-2.0)
    x = torch.randn(4, 8, 16)
    g_pos = layer_pos.compute_gates(x).mean().item()
    g_neg = layer_neg.compute_gates(x).mean().item()
    # bias +2 → sigmoid(2) ≈ 0.88, bias -2 → sigmoid(-2) ≈ 0.12
    assert g_pos > 0.8
    assert g_neg < 0.2


# ─────────────────────────────────────────────────────────────────
# GatedPCMMiniLM
# ─────────────────────────────────────────────────────────────────


def test_gated_pcm_mini_lm_forward_shape() -> None:
    torch.manual_seed(0)
    m = GatedPCMMiniLM(vocab=128, d_model=16, n_layers=2)
    x = torch.randint(0, 128, (3, 12))
    out = m(x)
    assert out.shape == (3, 12, 128)


def test_gated_pcm_mini_lm_hidden_states_shape() -> None:
    torch.manual_seed(0)
    m = GatedPCMMiniLM(vocab=64, d_model=8, n_layers=3)
    x = torch.randint(0, 64, (2, 7))
    h = m.hidden_states(x)
    assert h.shape == (2, 7, 8)


def test_gated_pcm_mini_lm_token_embeddings_shape() -> None:
    torch.manual_seed(0)
    m = GatedPCMMiniLM(vocab=20, d_model=4, n_layers=1)
    e = m.token_embeddings()
    assert e.shape == (20, 4)


def test_gated_pcm_mini_lm_all_layer_gates_shape() -> None:
    torch.manual_seed(0)
    m = GatedPCMMiniLM(vocab=32, d_model=8, n_layers=3)
    x = torch.randint(0, 32, (2, 6))
    gates = m.all_layer_gates(x)
    assert len(gates) == 3
    for g in gates:
        assert g.shape == (2, 6, 8)
        assert (g >= 0).all() and (g <= 1).all()


def test_gated_pcm_mini_lm_init_loss_near_uniform() -> None:
    torch.manual_seed(0)
    vocab = 512
    m = GatedPCMMiniLM(vocab=vocab, d_model=32, n_layers=2)
    x = torch.randint(0, vocab, (4, 32))
    with torch.no_grad():
        logits = m(x)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab),
        x[:, 1:].reshape(-1),
    )
    uniform = math.log(vocab)
    assert abs(float(loss.item()) - uniform) < 1.5


def test_gated_pcm_mini_lm_force_mean_no_crash() -> None:
    torch.manual_seed(0)
    m = GatedPCMMiniLM(vocab=64, d_model=8, n_layers=2)
    x = torch.randint(0, 64, (2, 10))
    y_gated = m(x)
    y_mean = m(x, force_mean=True)
    assert y_gated.shape == y_mean.shape
    # The two paths should produce different outputs at default
    # gate bias 0.
    assert not torch.allclose(y_gated, y_mean)


def test_gated_pcm_mini_lm_trainable_step_runs() -> None:
    torch.manual_seed(0)
    vocab = 64
    m = GatedPCMMiniLM(vocab=vocab, d_model=16, n_layers=2)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, vocab, (4, 16))
    y = torch.randint(0, vocab, (4, 16))
    logits = m(x)
    loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
    loss.backward()
    opt.step()
    # Gates should be inspectable after a training step
    g = m.all_layer_gates(x)
    assert all(gi.requires_grad is False for gi in g)


# ─────────────────────────────────────────────────────────────────
# build_matched_quad
# ─────────────────────────────────────────────────────────────────


def test_build_matched_quad_returns_four_models_plus_diag() -> None:
    gpt, pcm, topk, gated, diag = build_matched_quad(
        vocab=256, d_model=32, n_layers=2,
    )
    assert isinstance(gpt, GPTMiniLM)
    assert isinstance(pcm, PCMMiniLM)
    assert isinstance(topk, PCMTopKMiniLM)
    assert isinstance(gated, GatedPCMMiniLM)
    assert isinstance(diag, dict)


def test_build_matched_quad_params_within_tolerance() -> None:
    gpt, pcm, topk, gated, diag = build_matched_quad(
        vocab=512, d_model=32, n_layers=2, match_tol=0.25,
    )
    gpt_params = count_params(gpt)
    for name, m in (("pcm", pcm), ("topk", topk), ("gated", gated)):
        p = count_params(m)
        ratio = p / gpt_params
        assert 0.5 <= ratio <= 1.30, (
            f"{name} param ratio {ratio:.3f} out of [0.5, 1.30]"
        )


def test_build_matched_quad_diag_has_gated_keys() -> None:
    _, _, _, _, diag = build_matched_quad(
        vocab=256, d_model=16, n_layers=2,
    )
    assert "gated_pcm_params" in diag
    assert "gated_pcm_combiner_hidden" in diag
    assert "gated_pcm_gate_bias_init" in diag
    assert "ratio_gated_over_gpt" in diag


def test_build_matched_quad_gate_bias_propagates() -> None:
    _, _, _, gated, diag = build_matched_quad(
        vocab=128, d_model=8, n_layers=1, gate_bias_init=1.5,
    )
    # The first layer's gate_proj bias should reflect the init
    layer = gated.layers[0]
    assert torch.allclose(
        layer.gate_proj.bias.detach(),
        torch.full_like(layer.gate_proj.bias, 1.5),
        atol=1e-5,
    )
    assert diag["gated_pcm_gate_bias_init"] == 1.5
