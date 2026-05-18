"""Unit tests for F79 ``experiments/tinystories_f79.py`` helpers
and ``pcm.lm`` GPT-2-style init fix."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from experiments.tinystories_f79 import (
    BOS_ID,
    EOS_ID,
    N_SPECIAL,
    PAD_ID,
    UNK_ID,
    _normalise_text,
    _tokenise_text,
    build_vocab,
    encode_story,
)
from pcm.lm import (
    GPTMiniLM,
    PCMMiniLM,
    PCMTopKMiniLM,
    build_matched_triple,
)


# ─────────────────────────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────────────────────────


def test_normalise_collapses_whitespace_and_lowers() -> None:
    s = "Hello,\n  WORLD!\t Yes."
    assert _normalise_text(s) == "hello, world! yes."


def test_tokenise_splits_words_and_punct() -> None:
    toks = _tokenise_text("once upon a time, lily.")
    assert toks == ["once", "upon", "a", "time", ",", "lily", "."]


def test_build_vocab_caps_to_vocab_cap() -> None:
    stories = ["the cat sat", "the dog ran", "the bird flew"]
    stoi, itos = build_vocab(stories, vocab_cap=6)
    assert len(itos) == 6
    assert len(stoi) == 6
    # First four are specials
    assert itos[:4] == ["<pad>", "<unk>", "<bos>", "<eos>"]


def test_build_vocab_orders_by_frequency() -> None:
    stories = ["the the the cat dog cat the the"]
    stoi, itos = build_vocab(stories, vocab_cap=10)
    # "the" must come before "cat"/"dog" (more frequent)
    assert stoi["the"] < stoi["cat"]
    assert stoi["the"] < stoi["dog"]


def test_encode_story_wraps_with_bos_eos() -> None:
    stories = ["hello world"]
    stoi, _ = build_vocab(stories, vocab_cap=16)
    ids = encode_story("hello world", stoi)
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID
    assert len(ids) == 4  # bos + 2 words + eos


def test_encode_unknown_word_becomes_unk() -> None:
    stories = ["hello"]
    stoi, _ = build_vocab(stories, vocab_cap=16)
    ids = encode_story("hello goodbye", stoi)
    assert UNK_ID in ids


def test_special_token_constants() -> None:
    assert PAD_ID == 0
    assert UNK_ID == 1
    assert BOS_ID == 2
    assert EOS_ID == 3
    assert N_SPECIAL == 4


# ─────────────────────────────────────────────────────────────────
# pcm.lm GPT-2-style init (regression for F79 stable training)
# ─────────────────────────────────────────────────────────────────


def _initial_shifted_loss(model, *, vocab: int, B: int = 4,
                          L: int = 32) -> float:
    torch.manual_seed(0)
    x = torch.randint(0, vocab, (B, L))
    with torch.no_grad():
        logits = model(x)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab),
        x[:, 1:].reshape(-1),
    )
    return float(loss.item())


def test_gpt_init_loss_near_uniform_baseline() -> None:
    """GPT init should produce loss ≈ ln(vocab), not orders of
    magnitude higher (the F79 bug fix)."""
    vocab = 1024
    m = GPTMiniLM(vocab=vocab, d_model=64, n_layers=2,
                  n_heads=4, max_len=32)
    loss = _initial_shifted_loss(m, vocab=vocab)
    uniform = math.log(vocab)
    assert abs(loss - uniform) < 1.5, (
        f"GPT init loss {loss:.3f} vs uniform {uniform:.3f} "
        f"differs by more than 1.5 nats — embeddings init "
        f"likely too large."
    )


def test_pcm_mini_init_loss_near_uniform_baseline() -> None:
    vocab = 1024
    m = PCMMiniLM(vocab=vocab, d_model=64, n_layers=2)
    loss = _initial_shifted_loss(m, vocab=vocab)
    uniform = math.log(vocab)
    assert abs(loss - uniform) < 1.5, (
        f"PCM-mean init loss {loss:.3f} vs uniform {uniform:.3f}"
    )


def test_pcm_topk_init_loss_near_uniform_baseline() -> None:
    vocab = 1024
    m = PCMTopKMiniLM(vocab=vocab, d_model=64, n_layers=2,
                       top_k=4)
    loss = _initial_shifted_loss(m, vocab=vocab)
    uniform = math.log(vocab)
    assert abs(loss - uniform) < 1.5, (
        f"PCM-topk init loss {loss:.3f} vs uniform {uniform:.3f}"
    )


def test_matched_triple_all_have_small_init_loss() -> None:
    vocab = 2048
    gpt, pcm, topk, _ = build_matched_triple(
        vocab=vocab, d_model=64, n_layers=2,
        n_heads=4, max_len=64, top_k=4,
    )
    uniform = math.log(vocab)
    for name, m in (("gpt", gpt), ("pcm-mean", pcm),
                     ("pcm-topk", topk)):
        loss = _initial_shifted_loss(m, vocab=vocab)
        assert abs(loss - uniform) < 1.5, (
            f"{name} init loss {loss:.3f} differs from "
            f"uniform {uniform:.3f}"
        )
