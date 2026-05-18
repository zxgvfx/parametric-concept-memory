"""Unit tests for F74 LM comparison: PCMMiniLM, GPTMiniLM,
synthetic language generator."""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from pcm.lm import (
    GPTMiniLM,
    PCMMiniLM,
    PCMUniversalCombiner,
    build_matched_pair,
    count_params,
    perplexity,
)
from pcm.lm_synthetic import (
    EOS_ID,
    NOUNS_BY_CLASS,
    PAD_ID,
    Tokenizer,
    VERBS_BY_CLASS,
    all_valid_verb_object_pairs,
    generate_corpus,
    generate_sentence,
    generate_violation_pair,
    noun_class_of,
    verb_class_of,
)


# ─────────────────────────────────────────────────────────────────
# Synthetic language
# ─────────────────────────────────────────────────────────────────


def test_tokenizer_roundtrip():
    tok = Tokenizer()
    s = ["alice", "eats", "bread", "."]
    ids = tok.encode(s, max_len=8)
    assert ids[0] == tok.stoi["alice"]
    assert ids[3] == tok.stoi["."]
    assert ids[4] == EOS_ID
    assert ids[5] == PAD_ID
    decoded = tok.decode(ids)
    assert decoded == "alice eats bread ."


def test_tokenizer_unk():
    tok = Tokenizer()
    ids = tok.encode(["alice", "wibbles", "bread", "."], max_len=8)
    assert ids[1] == tok.unk_id


def test_noun_class_lookup():
    assert noun_class_of("cat") == "ANIMAL"
    assert noun_class_of("bread") == "FOOD"
    assert noun_class_of("alice") == "PERSON"
    assert noun_class_of("home") == "PLACE"
    assert noun_class_of("wibble") is None


def test_verb_class_lookup():
    assert verb_class_of("walks") == "MOTION"
    assert verb_class_of("eats") == "CONSUMPTION"
    assert verb_class_of("sees") == "PERCEPTION"
    assert verb_class_of("has") == "POSSESSION"
    assert verb_class_of("says") == "COMMUNICATION"
    assert verb_class_of("wibbles") is None


def test_generate_sentence_respects_selectional_restrictions():
    """``eat`` should always have a FOOD object."""
    rng = random.Random(0)
    for _ in range(50):
        s = generate_sentence(rng, template="SVO")
        # Find the verb
        v = None
        for t in s:
            if verb_class_of(t) is not None:
                v = t
                break
        assert v is not None
        v_cls = verb_class_of(v)
        # Find first noun after verb
        after_v = False
        obj = None
        for t in s:
            if not after_v:
                if t == v:
                    after_v = True
                continue
            if noun_class_of(t) is not None:
                obj = t
                break
        assert obj is not None
        # Check selectional restriction
        from pcm.lm_synthetic import _SELECTIONAL
        sel = _SELECTIONAL[v_cls]
        if sel["object_classes"] is not None:
            assert noun_class_of(obj) in sel["object_classes"], (
                f"verb {v} ({v_cls}) got object {obj} "
                f"({noun_class_of(obj)}), expected one of "
                f"{sel['object_classes']}"
            )


def test_generate_corpus_size():
    corpus = generate_corpus(100, seed=42)
    assert len(corpus) == 100
    for s in corpus:
        assert isinstance(s, list)
        assert all(isinstance(t, str) for t in s)
        assert s[-1] == "."


def test_generate_corpus_with_holdout():
    holdout = {("eats", "bread"), ("drinks", "milk")}
    corpus = generate_corpus(500, seed=42,
                              holdout_verb_object_pairs=holdout)
    # No sentence in corpus should contain either held-out pair
    from pcm.lm_synthetic import _extract_verb_object
    for s in corpus:
        v, o = _extract_verb_object(s)
        if v is not None and o is not None:
            assert (v, o) not in holdout


def test_generate_violation_pair():
    rng = random.Random(0)
    for _ in range(20):
        valid, vio = generate_violation_pair(rng)
        # Both should be lists of strings ending with "."
        assert valid[-1] == "."
        assert vio[-1] == "."
        # Length should match
        assert len(valid) == len(vio)
        # They should differ by exactly one token (the object)
        differing = [i for i, (a, b) in enumerate(zip(valid, vio))
                     if a != b]
        assert len(differing) == 1
        # The differing position should be a noun in valid and
        # a noun in violation, but different *classes*.
        i = differing[0]
        v_class = noun_class_of(valid[i])
        b_class = noun_class_of(vio[i])
        assert v_class is not None
        assert b_class is not None
        assert v_class != b_class


def test_all_valid_pairs_size():
    pairs = all_valid_verb_object_pairs()
    # 6 CONSUMPTION × 15 FOOD = 90
    # 6 PERCEPTION × (15+15+15+15) = 360
    # 6 POSSESSION × (15+15+15) = 270
    # 6 COMMUNICATION × 15 PERSON = 90
    # Total = 90 + 360 + 270 + 90 = 810
    assert len(pairs) == 810
    # Spot-check: ("eats", "bread") is valid
    assert ("eats", "bread") in pairs
    # ("eats", "home") is NOT valid (eat needs FOOD, home is PLACE)
    assert ("eats", "home") not in pairs


# ─────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────


def test_pcm_universal_combiner_shape():
    c = PCMUniversalCombiner(dim=16, hidden=32)
    a = torch.randn(4, 16)
    b = torch.randn(4, 16)
    out = c(a, b)
    assert out.shape == (4, 16)


def test_gpt_mini_lm_forward_shape():
    tok = Tokenizer()
    m = GPTMiniLM(vocab=tok.vocab_size, d_model=32, n_layers=2)
    x = torch.randint(0, tok.vocab_size, (4, 10))
    out = m(x)
    assert out.shape == (4, 10, tok.vocab_size)


def test_pcm_mini_lm_forward_shape():
    tok = Tokenizer()
    m = PCMMiniLM(vocab=tok.vocab_size, d_model=32, n_layers=2)
    x = torch.randint(0, tok.vocab_size, (4, 10))
    out = m(x)
    assert out.shape == (4, 10, tok.vocab_size)


def test_models_have_hidden_states():
    tok = Tokenizer()
    g = GPTMiniLM(vocab=tok.vocab_size, d_model=32, n_layers=2)
    p = PCMMiniLM(vocab=tok.vocab_size, d_model=32, n_layers=2)
    x = torch.randint(0, tok.vocab_size, (2, 8))
    hg = g.hidden_states(x)
    hp = p.hidden_states(x)
    assert hg.shape == (2, 8, 32)
    assert hp.shape == (2, 8, 32)


def test_models_have_token_embeddings():
    tok = Tokenizer()
    g = GPTMiniLM(vocab=tok.vocab_size, d_model=32, n_layers=2)
    p = PCMMiniLM(vocab=tok.vocab_size, d_model=32, n_layers=2)
    eg = g.token_embeddings()
    ep = p.token_embeddings()
    assert eg.shape == (tok.vocab_size, 32)
    assert ep.shape == (tok.vocab_size, 32)


def test_build_matched_pair_ratio():
    """The matched pair should have PCM params ≤ 1.1 × GPT params."""
    tok = Tokenizer()
    gpt, pcm, diag = build_matched_pair(
        vocab=tok.vocab_size, d_model=64, n_layers=2, n_heads=4,
    )
    assert diag["ratio_pcm_over_gpt"] <= 1.10
    assert isinstance(gpt, GPTMiniLM)
    assert isinstance(pcm, PCMMiniLM)


def test_pcm_lm_trainable_on_tiny_corpus():
    """The PCM LM should successfully fit a tiny corpus
    (perplexity drops meaningfully)."""
    torch.manual_seed(7)
    tok = Tokenizer()
    rng = random.Random(0)
    corpus = [generate_sentence(rng, template="SV") for _ in range(64)]
    ids = torch.tensor(
        [tok.encode(s, max_len=8) for s in corpus],
        dtype=torch.long,
    )
    x = ids[:, :-1]
    y = ids[:, 1:]
    m = PCMMiniLM(vocab=tok.vocab_size, d_model=32, n_layers=2,
                   combiner_hidden=64)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-3)
    initial_ppl = perplexity(m, x, y, pad_id=tok.pad_id)
    for _ in range(50):
        logits = m(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1), ignore_index=tok.pad_id,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    final_ppl = perplexity(m, x, y, pad_id=tok.pad_id)
    assert final_ppl < initial_ppl * 0.5, (
        f"PCM LM didn't converge: {initial_ppl:.3f} → {final_ppl:.3f}"
    )


def test_gpt_lm_trainable_on_tiny_corpus():
    torch.manual_seed(7)
    tok = Tokenizer()
    rng = random.Random(0)
    corpus = [generate_sentence(rng, template="SV") for _ in range(64)]
    ids = torch.tensor(
        [tok.encode(s, max_len=8) for s in corpus],
        dtype=torch.long,
    )
    x = ids[:, :-1]
    y = ids[:, 1:]
    m = GPTMiniLM(vocab=tok.vocab_size, d_model=32, n_layers=2,
                   n_heads=4)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-3)
    initial_ppl = perplexity(m, x, y, pad_id=tok.pad_id)
    for _ in range(50):
        logits = m(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1), ignore_index=tok.pad_id,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    final_ppl = perplexity(m, x, y, pad_id=tok.pad_id)
    assert final_ppl < initial_ppl * 0.5, (
        f"GPT LM didn't converge: {initial_ppl:.3f} → {final_ppl:.3f}"
    )
