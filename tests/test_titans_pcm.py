"""Unit tests for F83 ``HierarchicalMemoryLayer`` /
``TitansPCMMiniLM`` in ``pcm.lm``."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from pcm.lm import (
    HierarchicalMemoryLayer,
    HybridPCMMiniLM,
    TitansPCMMiniLM,
    count_params,
)


# ─────────────────────────────────────────────────────────────────
# HierarchicalMemoryLayer
# ─────────────────────────────────────────────────────────────────


def test_hierarchical_memory_empty_buffer_returns_zeros() -> None:
    torch.manual_seed(0)
    mem = HierarchicalMemoryLayer(
        d_model=16, buffer_capacity=8, top_k=4,
    )
    x = torch.randn(2, 6, 16)
    out = mem(x)
    assert out.shape == x.shape
    assert (out == 0).all()


def test_hierarchical_memory_imprint_grows_buffer() -> None:
    torch.manual_seed(0)
    mem = HierarchicalMemoryLayer(
        d_model=8, buffer_capacity=16, top_k=4,
    )
    x = torch.randn(2, 4, 8)
    n = mem.imprint(x, n_per_batch=2)
    assert n == 4  # 2 batches × 2 per batch
    assert int(mem.buf_size.item()) == 4
    assert int(mem.buf_ptr.item()) == 4


def test_hierarchical_memory_imprint_fifo_wraps_at_capacity() -> None:
    torch.manual_seed(0)
    mem = HierarchicalMemoryLayer(
        d_model=8, buffer_capacity=4, top_k=2,
    )
    x = torch.randn(2, 4, 8)
    # First imprint: 4 entries (fills the buffer)
    mem.imprint(x, n_per_batch=2)
    assert int(mem.buf_size.item()) == 4
    assert int(mem.buf_ptr.item()) == 0  # wrapped
    # Second imprint: writes 4 more, wraps again
    x2 = torch.randn(2, 4, 8)
    mem.imprint(x2, n_per_batch=2)
    assert int(mem.buf_size.item()) == 4  # still capped
    assert int(mem.buf_ptr.item()) == 0


def test_hierarchical_memory_forward_after_imprint_nonzero() -> None:
    torch.manual_seed(0)
    mem = HierarchicalMemoryLayer(
        d_model=16, buffer_capacity=32, top_k=4,
    )
    mem.imprint(torch.randn(2, 4, 16), n_per_batch=2)
    out = mem(torch.randn(2, 4, 16))
    assert (out.abs() > 0).any()


def test_hierarchical_memory_recall_diagnostics() -> None:
    torch.manual_seed(0)
    mem = HierarchicalMemoryLayer(
        d_model=16, buffer_capacity=32, top_k=4,
    )
    mem.imprint(torch.randn(2, 4, 16), n_per_batch=2)
    d = mem.recall_diagnostics(torch.randn(2, 4, 16))
    assert d["size"] == 4
    assert d["top_k"] == 4
    assert 0.0 <= d["mean_topk_mass"] <= 1.0


def test_hierarchical_memory_reset_clears_buffer() -> None:
    torch.manual_seed(0)
    mem = HierarchicalMemoryLayer(
        d_model=8, buffer_capacity=8, top_k=2,
    )
    mem.imprint(torch.randn(2, 4, 8), n_per_batch=2)
    assert int(mem.buf_size.item()) == 4
    mem.reset_buffer()
    assert int(mem.buf_size.item()) == 0
    assert int(mem.buf_ptr.item()) == 0
    assert (mem.buf_keys == 0).all()


# ─────────────────────────────────────────────────────────────────
# TitansPCMMiniLM
# ─────────────────────────────────────────────────────────────────


def test_titans_pcm_mini_lm_inherits_hybrid() -> None:
    torch.manual_seed(0)
    m = TitansPCMMiniLM(
        vocab=64, d_model=16, n_layers=4, n_heads=4,
    )
    assert isinstance(m, HybridPCMMiniLM)
    assert m.layer_kinds == ["pcm", "pcm", "pcm", "attn"]


def test_titans_pcm_mini_lm_forward_shape() -> None:
    torch.manual_seed(0)
    m = TitansPCMMiniLM(
        vocab=128, d_model=16, n_layers=4, n_heads=4,
        memory_capacity=16, memory_top_k=4,
    )
    x = torch.randint(0, 128, (3, 12))
    out = m(x)
    assert out.shape == (3, 12, 128)


def test_titans_pcm_mini_lm_imprint_only_in_training() -> None:
    torch.manual_seed(0)
    m = TitansPCMMiniLM(
        vocab=64, d_model=16, n_layers=4,
        memory_capacity=32, memory_top_k=4,
    )
    x = torch.randint(0, 64, (2, 8))
    # Train mode: imprint should happen
    m.train()
    m(x)
    assert int(m.memory_readout.buf_size.item()) > 0
    initial_size = int(m.memory_readout.buf_size.item())
    # Eval mode: imprint=True is silently ignored
    m.eval()
    m(x)
    assert int(m.memory_readout.buf_size.item()) == initial_size


def test_titans_pcm_mini_lm_use_memory_false_bypasses_readout() -> None:
    torch.manual_seed(0)
    m = TitansPCMMiniLM(
        vocab=64, d_model=16, n_layers=4,
        memory_capacity=32, memory_top_k=4,
    )
    x = torch.randint(0, 64, (2, 8))
    m.train()
    for _ in range(3):  # populate buffer
        m(x)
    m.eval()
    out_with = m(x, use_memory=True)
    out_without = m(x, use_memory=False)
    assert not torch.allclose(out_with, out_without)


def test_titans_pcm_mini_lm_reset_memory_works() -> None:
    torch.manual_seed(0)
    m = TitansPCMMiniLM(
        vocab=64, d_model=16, n_layers=4,
        memory_capacity=32, memory_top_k=4,
    )
    x = torch.randint(0, 64, (2, 8))
    m.train()
    m(x)
    assert int(m.memory_readout.buf_size.item()) > 0
    m.reset_memory()
    assert int(m.memory_readout.buf_size.item()) == 0


def test_titans_pcm_mini_lm_init_loss_near_uniform() -> None:
    torch.manual_seed(0)
    vocab = 512
    m = TitansPCMMiniLM(
        vocab=vocab, d_model=32, n_layers=4,
        memory_capacity=64, memory_top_k=4,
    )
    m.eval()
    x = torch.randint(0, vocab, (4, 32))
    with torch.no_grad():
        logits = m(x, use_memory=False, imprint=False)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab),
        x[:, 1:].reshape(-1),
    )
    uniform = math.log(vocab)
    assert abs(float(loss.item()) - uniform) < 1.5


def test_titans_pcm_mini_lm_param_count_above_hybrid() -> None:
    """Titans has the memory readout overhead."""
    torch.manual_seed(0)
    hyb = HybridPCMMiniLM(
        vocab=128, d_model=16, n_layers=4, n_heads=4,
    )
    tit = TitansPCMMiniLM(
        vocab=128, d_model=16, n_layers=4, n_heads=4,
        memory_capacity=8, memory_top_k=2,
    )
    # Titans has q/k/v + gate + LN ≈ 4 × (D² + D)
    extra = count_params(tit) - count_params(hyb)
    assert extra > 0


def test_titans_pcm_mini_lm_trainable_step_runs() -> None:
    torch.manual_seed(0)
    vocab = 64
    m = TitansPCMMiniLM(
        vocab=vocab, d_model=16, n_layers=4,
        memory_capacity=32, memory_top_k=4,
    )
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, vocab, (4, 16))
    y = torch.randint(0, vocab, (4, 16))
    m.train()
    # First forward populates buffer (readout not invoked on
    # empty buffer)
    with torch.no_grad():
        m(x)
    assert int(m.memory_readout.buf_size.item()) > 0
    # Second forward now flows gradient through the readout
    logits = m(x)
    loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
    loss.backward()
    opt.step()
    assert m.memory_readout.q_proj.weight.grad is not None
    assert m.memory_readout.out_gate.weight.grad is not None
