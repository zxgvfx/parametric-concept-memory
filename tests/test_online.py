"""Unit tests for F85 ``pcm.online`` module."""
from __future__ import annotations

import random

import torch

from pcm.lm import GatedPCMMiniLM, HybridPCMMiniLM, TitansPCMMiniLM
from pcm.lm_synthetic import (
    RESERVED_CONCEPTS,
    generate_concept_dataset,
    generate_concept_teaching_sentence,
    generate_concept_test_sentence,
)
from pcm.online import (
    MechanismCounters,
    OnlineTeacherSession,
    PretrainReplayBuffer,
)


# ─────────────────────────────────────────────────────────────────
# Concept dataset (F85 synthetic generator)
# ─────────────────────────────────────────────────────────────────


def test_reserved_concepts_has_eight() -> None:
    assert len(RESERVED_CONCEPTS) == 8
    tokens = {c[0] for c in RESERVED_CONCEPTS}
    classes = {c[1] for c in RESERVED_CONCEPTS}
    assert len(tokens) == 8  # all unique
    assert classes == {"ANIMAL", "FOOD"}


def test_reserved_concepts_balanced_classes() -> None:
    n_animal = sum(
        1 for _, cls, _ in RESERVED_CONCEPTS if cls == "ANIMAL"
    )
    n_food = sum(
        1 for _, cls, _ in RESERVED_CONCEPTS if cls == "FOOD"
    )
    assert n_animal == 4
    assert n_food == 4


def test_generate_concept_teaching_sentence_contains_concept() -> None:
    rng = random.Random(0)
    for tok, cls, _ in RESERVED_CONCEPTS[:4]:
        s = generate_concept_teaching_sentence(rng, tok, cls)
        assert tok in s


def test_generate_concept_test_sentence_contains_concept() -> None:
    rng = random.Random(0)
    for tok, cls, _ in RESERVED_CONCEPTS[:4]:
        s = generate_concept_test_sentence(rng, tok, cls)
        assert tok in s


def test_teaching_and_test_use_different_vocabularies() -> None:
    """Teaching and test sets must use disjoint filler verbs so
    that a PASS on test PPL requires generalisation, not
    memorisation."""
    rng = random.Random(0)
    teach_words: set[str] = set()
    test_words: set[str] = set()
    for tok, cls, _ in RESERVED_CONCEPTS:
        for _ in range(20):
            teach_words.update(
                generate_concept_teaching_sentence(rng, tok, cls)
            )
            test_words.update(
                generate_concept_test_sentence(rng, tok, cls)
            )
    teach_words.discard(".")
    test_words.discard(".")
    # Specifically the teaching verbs should NOT appear in test
    teach_specific = {"ran", "jumped", "ate", "saw", "liked"}
    test_specific = {"walked", "found", "hugged", "rested",
                     "wanted"}
    assert teach_specific & teach_words
    assert teach_specific.isdisjoint(test_words), (
        f"teaching verbs leaked into test: "
        f"{teach_specific & test_words}"
    )
    assert test_specific & test_words
    assert test_specific.isdisjoint(teach_words)


def test_generate_concept_dataset_structure() -> None:
    rng = random.Random(0)
    d = generate_concept_dataset(
        rng, n_teach_per_concept=5, n_test_per_concept=4,
    )
    assert "concepts" in d
    assert "teaching" in d
    assert "test" in d
    assert len(d["concepts"]) == 8
    for tok, _, _ in d["concepts"]:
        assert len(d["teaching"][tok]) == 5
        assert len(d["test"][tok]) == 4


# ─────────────────────────────────────────────────────────────────
# PretrainReplayBuffer
# ─────────────────────────────────────────────────────────────────


def test_replay_buffer_grows_then_caps() -> None:
    buf = PretrainReplayBuffer(capacity=4)
    xs = torch.arange(20).view(5, 4)
    ys = torch.arange(20, 40).view(5, 4)
    buf.add_batch(xs, ys)
    assert len(buf) == 4  # capped


def test_replay_buffer_sample_returns_correct_shape() -> None:
    buf = PretrainReplayBuffer(capacity=16)
    xs = torch.arange(40).view(10, 4)
    ys = torch.arange(40, 80).view(10, 4)
    buf.add_batch(xs, ys)
    xs_sampled, ys_sampled = buf.sample(5)
    assert xs_sampled.shape == (5, 4)
    assert ys_sampled.shape == (5, 4)


def test_replay_buffer_sample_empty_raises() -> None:
    buf = PretrainReplayBuffer(capacity=4)
    try:
        buf.sample(2)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


# ─────────────────────────────────────────────────────────────────
# MechanismCounters
# ─────────────────────────────────────────────────────────────────


def test_mechanism_counters_as_dict() -> None:
    c = MechanismCounters()
    c.n_corrections = 5
    c.n_m1_imprints = 5
    c.n_m3_grad_steps = 3
    d = c.as_dict()
    assert d["n_corrections"] == 5
    assert d["n_m3_grad_steps"] == 3


# ─────────────────────────────────────────────────────────────────
# OnlineTeacherSession
# ─────────────────────────────────────────────────────────────────


def _make_session(*, model_cls=HybridPCMMiniLM):
    torch.manual_seed(0)
    vocab = 32
    novel_ids = [28, 29, 30, 31]
    classes = {28: "ANIMAL", 29: "ANIMAL", 30: "FOOD", 31: "FOOD"}
    if model_cls is TitansPCMMiniLM:
        model = TitansPCMMiniLM(
            vocab=vocab, d_model=16, n_layers=4, n_heads=4,
            memory_capacity=32, memory_top_k=4,
        )
    else:
        model = model_cls(
            vocab=vocab, d_model=16, n_layers=4, n_heads=4,
        )
    replay = PretrainReplayBuffer(capacity=16)
    xs = torch.randint(0, 28, (4, 8))
    ys = torch.randint(0, 28, (4, 8))
    replay.add_batch(xs, ys)
    sess = OnlineTeacherSession(
        model=model, novel_concept_ids=novel_ids,
        novel_concept_classes=classes, replay=replay,
        m3_k_grad_threshold=1, m3_n_replay=2, m3_lr=1e-3,
        m3_inner_steps=1,
    )
    return sess, novel_ids


def test_session_receive_correction_increments_counters() -> None:
    sess, ids = _make_session()
    ctx = torch.tensor([1, 2, 3])
    tgt = torch.tensor([1, 2, 3, ids[0], 0])
    sess.receive_correction(ctx, tgt)
    assert sess.counters.n_corrections == 1
    assert sess.counters.n_m1_imprints == 1
    assert sess.counters.correction_counts[ids[0]] == 1


def test_session_skips_when_no_novel_concept_in_target() -> None:
    sess, _ = _make_session()
    ctx = torch.tensor([1, 2, 3])
    tgt = torch.tensor([1, 2, 3, 5, 0])  # no novel id
    result = sess.receive_correction(ctx, tgt)
    assert result["status"] == "no_novel_concept"
    assert sess.counters.n_m1_imprints == 0


def test_session_m3_fires_at_threshold() -> None:
    sess, ids = _make_session()
    ctx = torch.tensor([1, 2, 3])
    tgt = torch.tensor([1, 2, 3, ids[0], 0])
    # threshold=1, so first correction already fires M3
    sess.receive_correction(ctx, tgt)
    assert sess.counters.n_m3_grad_steps >= 1


def test_session_titans_imprints_to_memory_readout() -> None:
    sess, ids = _make_session(model_cls=TitansPCMMiniLM)
    ctx = torch.tensor([1, 2, 3])
    tgt = torch.tensor([1, 2, 3, ids[0], 0])
    sess.receive_correction(ctx, tgt)
    assert sess.counters.n_m1_titans_imprints == 1
    # Titans buffer should have grown
    assert int(sess.model.memory_readout.buf_size.item()) > 0


def test_session_gradient_only_touches_novel_rows() -> None:
    """After M3 fires, only novel-concept rows of tok_emb should
    have changed (other rows should be exactly unchanged)."""
    sess, ids = _make_session()
    initial_emb = sess.model.tok_emb.weight.detach().clone()
    ctx = torch.tensor([1, 2, 3])
    tgt = torch.tensor([1, 2, 3, ids[0], 0])
    for _ in range(3):
        sess.receive_correction(ctx, tgt)
    final_emb = sess.model.tok_emb.weight.detach()
    # Non-novel rows must be identical
    other_rows = [
        i for i in range(initial_emb.shape[0])
        if i not in set(ids)
    ]
    assert torch.allclose(
        initial_emb[other_rows], final_emb[other_rows],
    )
    # The trained novel row should have changed
    trained_id = ids[0]
    assert not torch.allclose(
        initial_emb[trained_id], final_emb[trained_id],
    )


def test_session_replay_prevents_other_rows_drifting() -> None:
    """Sanity: even after 10 corrections, all pretrained rows
    should be byte-identical to their pre-online values."""
    sess, ids = _make_session()
    initial_emb = sess.model.tok_emb.weight.detach().clone()
    initial_ln = sess.model.ln_final.weight.detach().clone()
    for _ in range(10):
        ctx = torch.tensor([1, 2, 3])
        tgt = torch.tensor([1, 2, 3, ids[0], 0])
        sess.receive_correction(ctx, tgt)
    final_emb = sess.model.tok_emb.weight.detach()
    final_ln = sess.model.ln_final.weight.detach()
    # All non-novel embedding rows unchanged
    other = [
        i for i in range(initial_emb.shape[0])
        if i not in set(ids)
    ]
    assert torch.allclose(initial_emb[other], final_emb[other])
    # ln_final IS updated though (it's also in the optimiser)
    # — just sanity-check that it doesn't NaN
    assert not torch.isnan(final_ln).any()


def test_session_sleep_returns_purity_dict() -> None:
    sess, ids = _make_session()
    ctx = torch.tensor([1, 2, 3])
    for cid in ids:
        tgt = torch.tensor([1, 2, 3, cid, 0])
        sess.receive_correction(ctx, tgt)
    diag = sess.sleep()
    assert diag["status"] == "ok"
    assert "purity_mean" in diag
    assert 0.0 <= diag["purity_mean"] <= 1.0


def test_session_eval_ppl_returns_finite_float() -> None:
    sess, ids = _make_session()
    seqs = [torch.tensor([1, 2, 3, ids[0], 0])]
    ppl = sess.eval_ppl_on_sequences(seqs)
    assert ppl > 0 and ppl < 1e10  # finite, non-NaN
