"""Unit tests for F77 PCMTopKMiniLM + multi-sentence discourse."""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from pcm.lm import (
    GPTMiniLM,
    PCMMiniLM,
    PCMTopKLayer,
    PCMTopKMiniLM,
    build_matched_pair,
    build_matched_triple,
    count_params,
)
from pcm.lm_synthetic import (
    Tokenizer,
    generate_multi_sentence_discourse,
    is_compatible_referent,
)


# ─────────────────────────────────────────────────────────────────
# PCMTopKLayer
# ─────────────────────────────────────────────────────────────────


def test_topk_layer_shape():
    layer = PCMTopKLayer(dim=16, combiner_hidden=32, top_k=3)
    slots = torch.randn(2, 8, 16)
    out = layer(slots)
    assert out.shape == (2, 8, 16)


def test_topk_layer_works_at_L1():
    """Edge case: sequence length 1 (no past to attend to).
    The all-masked positions should produce finite output via
    the cumulative-mean fallback."""
    layer = PCMTopKLayer(dim=16, top_k=4)
    slots = torch.randn(2, 1, 16)
    out = layer(slots)
    assert out.shape == (2, 1, 16)
    assert torch.isfinite(out).all()


def test_topk_layer_works_with_k_larger_than_seq():
    """If top_k > L, the layer should cap K at the available
    positions and not crash."""
    layer = PCMTopKLayer(dim=16, top_k=100)
    slots = torch.randn(2, 5, 16)
    out = layer(slots)
    assert out.shape == (2, 5, 16)
    assert torch.isfinite(out).all()


def test_topk_layer_is_causal():
    """Changing tokens at positions ≥ t should not change the
    layer's output at positions < t."""
    torch.manual_seed(0)
    layer = PCMTopKLayer(dim=16, top_k=3)
    layer.eval()
    slots_a = torch.randn(1, 6, 16)
    slots_b = slots_a.clone()
    slots_b[:, 3:] = torch.randn(1, 3, 16)  # change future
    with torch.no_grad():
        out_a = layer(slots_a)
        out_b = layer(slots_b)
    diff_past = (out_a[:, :3] - out_b[:, :3]).abs().max().item()
    assert diff_past < 1e-5, (
        f"causal violation: past changed by {diff_past}"
    )


# ─────────────────────────────────────────────────────────────────
# PCMTopKMiniLM
# ─────────────────────────────────────────────────────────────────


def test_topk_lm_forward_shape():
    tok = Tokenizer()
    lm = PCMTopKMiniLM(vocab=tok.vocab_size, d_model=32, n_layers=2,
                        top_k=3)
    x = torch.randint(0, tok.vocab_size, (4, 10))
    out = lm(x)
    assert out.shape == (4, 10, tok.vocab_size)


def test_topk_lm_hidden_states_shape():
    tok = Tokenizer()
    lm = PCMTopKMiniLM(vocab=tok.vocab_size, d_model=32, n_layers=2,
                        top_k=3)
    x = torch.randint(0, tok.vocab_size, (2, 6))
    h = lm.hidden_states(x)
    assert h.shape == (2, 6, 32)


def test_topk_lm_token_embeddings_shape():
    tok = Tokenizer()
    lm = PCMTopKMiniLM(vocab=tok.vocab_size, d_model=32, n_layers=2,
                        top_k=3)
    emb = lm.token_embeddings()
    assert emb.shape == (tok.vocab_size, 32)


def test_topk_lm_trainable_on_tiny_corpus():
    """The PCMTopK LM should converge on a tiny corpus."""
    torch.manual_seed(7)
    tok = Tokenizer()
    rng = random.Random(0)
    discourses = [
        generate_multi_sentence_discourse(rng, n_sentences=2)
        for _ in range(64)
    ]
    ids = torch.tensor(
        [tok.encode(d.tokens, max_len=10) for d in discourses],
        dtype=torch.long,
    )
    x = ids[:, :-1]
    y = ids[:, 1:]
    m = PCMTopKMiniLM(vocab=tok.vocab_size, d_model=32, n_layers=2,
                       combiner_hidden=64, top_k=3)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-3)
    initial = F.cross_entropy(
        m(x).reshape(-1, tok.vocab_size), y.reshape(-1),
        ignore_index=tok.pad_id,
    ).item()
    for _ in range(60):
        loss = F.cross_entropy(
            m(x).reshape(-1, tok.vocab_size), y.reshape(-1),
            ignore_index=tok.pad_id,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    final = F.cross_entropy(
        m(x).reshape(-1, tok.vocab_size), y.reshape(-1),
        ignore_index=tok.pad_id,
    ).item()
    assert final < initial * 0.5, (
        f"PCMTopK didn't converge: {initial:.3f} → {final:.3f}"
    )


# ─────────────────────────────────────────────────────────────────
# build_matched_triple
# ─────────────────────────────────────────────────────────────────


def test_build_matched_triple_param_counts():
    """The three models should have comparable param counts."""
    tok = Tokenizer()
    gpt, pcm, pcm_topk, diag = build_matched_triple(
        vocab=tok.vocab_size, d_model=64, n_layers=2, top_k=4,
    )
    # GPT and PCM are matched within match_tol (default 0.15)
    assert diag["ratio_pcm_over_gpt"] <= 1.15
    # PCMTopK has roughly the same params as PCM-mean (+ alpha scalar)
    assert abs(diag["pcm_topk_params"] - diag["pcm_params"]) <= 10
    assert isinstance(gpt, GPTMiniLM)
    assert isinstance(pcm, PCMMiniLM)
    assert isinstance(pcm_topk, PCMTopKMiniLM)


# ─────────────────────────────────────────────────────────────────
# Multi-sentence discourse generator
# ─────────────────────────────────────────────────────────────────


def test_multi_sentence_discourse_2():
    rng = random.Random(0)
    for _ in range(10):
        d = generate_multi_sentence_discourse(rng, n_sentences=2)
        assert d.referent == d.s1_subject
        assert d.tokens[d.pronoun_position] == d.pronoun
        # All candidates compatible
        for c, _ in d.candidates:
            assert is_compatible_referent(d.pronoun, c)


def test_multi_sentence_discourse_3():
    rng = random.Random(1)
    for _ in range(10):
        d = generate_multi_sentence_discourse(rng, n_sentences=3)
        assert d.referent == d.s1_subject
        # 3 sentences → at least 4 person candidates (S1 subj + obj,
        # S2 subj + obj)
        assert len(d.candidates) >= 4


def test_multi_sentence_discourse_4():
    rng = random.Random(2)
    for _ in range(10):
        d = generate_multi_sentence_discourse(rng, n_sentences=4)
        assert d.referent == d.s1_subject
        # Token count should be ~16 (4 sentences * 4 tokens avg)
        assert 12 <= len(d.tokens) <= 20


def test_multi_sentence_distractors_same_gender():
    """All candidates should share gender (forces the resolver
    to use cross-sentence context, not just gender filter)."""
    from pcm.lm_synthetic import person_gender_of
    rng = random.Random(3)
    for _ in range(20):
        d = generate_multi_sentence_discourse(rng, n_sentences=3)
        ref_gender = person_gender_of(d.referent)
        for c, _ in d.candidates:
            cg = person_gender_of(c)
            if cg is not None:  # all should be PERSON
                assert cg == ref_gender
