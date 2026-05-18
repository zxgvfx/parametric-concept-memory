"""Unit tests for F76 pronoun resolution."""
from __future__ import annotations

import random

import torch

from pcm import EpisodicBuffer
from pcm.coref import (
    ResolutionResult,
    candidates_from_buffer,
    candidates_from_token_history,
    resolve_pronoun,
    resolve_pronoun_class_only,
    resolve_pronoun_recency,
)
from pcm.lm import PCMMiniLM
from pcm.lm_synthetic import (
    PRONOUN_CLASSES,
    Tokenizer,
    generate_coreference_discourse,
    is_compatible_referent,
    person_gender_of,
    pronoun_compatible_classes,
)


# ─────────────────────────────────────────────────────────────────
# Vocabulary additions
# ─────────────────────────────────────────────────────────────────


def test_pronouns_in_vocab():
    tok = Tokenizer()
    for p in ("he", "she", "it", "they"):
        assert p in tok.stoi
    # Pronouns get unique ids
    ids = {tok.stoi[p] for p in ("he", "she", "it", "they")}
    assert len(ids) == 4


def test_person_gender_lookup():
    assert person_gender_of("alice") == "F"
    assert person_gender_of("bob") == "M"
    assert person_gender_of("dog") is None  # not a person


def test_pronoun_compatible_classes():
    assert pronoun_compatible_classes("he") == ("PERSON",)
    assert pronoun_compatible_classes("she") == ("PERSON",)
    assert "ANIMAL" in pronoun_compatible_classes("it")
    assert "PERSON" not in pronoun_compatible_classes("it")
    try:
        pronoun_compatible_classes("wibble")
    except ValueError:
        return
    assert False, "expected ValueError on unknown pronoun"


def test_is_compatible_referent_gender():
    """she/he obey gender, it obeys class but not gender."""
    assert is_compatible_referent("she", "alice")
    assert not is_compatible_referent("she", "bob")
    assert is_compatible_referent("he", "bob")
    assert not is_compatible_referent("he", "alice")
    assert is_compatible_referent("it", "dog")
    assert is_compatible_referent("it", "bread")
    assert not is_compatible_referent("it", "alice")


# ─────────────────────────────────────────────────────────────────
# Discourse generator
# ─────────────────────────────────────────────────────────────────


def test_subject_continuity_discourse_structure():
    rng = random.Random(0)
    for _ in range(20):
        d = generate_coreference_discourse(rng, rule="subject_continuity")
        # Referent must be the subject
        assert d.referent == d.s1_subject
        # Pronoun must be class+gender compatible with referent
        assert is_compatible_referent(d.pronoun, d.referent)
        # Pronoun position must point at a pronoun
        assert d.tokens[d.pronoun_position] == d.pronoun
        # At least two candidates (subject + object both same class+gender)
        assert len(d.candidates) >= 2
        # Both candidates are class-compatible
        for cand_noun, _ in d.candidates:
            assert is_compatible_referent(d.pronoun, cand_noun)


def test_object_continuity_discourse():
    rng = random.Random(1)
    for _ in range(10):
        d = generate_coreference_discourse(rng, rule="object_continuity")
        assert d.referent == d.s1_object


# ─────────────────────────────────────────────────────────────────
# Candidate extraction
# ─────────────────────────────────────────────────────────────────


def test_candidates_from_token_history_class_filter():
    tokens = ["alice", "gets", "bob", ".", "she", "is", "happy", "."]
    cands = candidates_from_token_history(
        tokens, pronoun_position=4, pronoun="she",
    )
    # Only alice is F-PERSON; bob is M-PERSON
    assert [c for c, _ in cands] == ["alice"]


def test_candidates_from_token_history_gender_filter():
    tokens = ["alice", "gets", "carol", ".", "she", "is", "happy", "."]
    cands = candidates_from_token_history(
        tokens, pronoun_position=4, pronoun="she",
    )
    # Both alice and carol are F-PERSON
    assert {c for c, _ in cands} == {"alice", "carol"}


def test_candidates_from_token_history_no_match():
    tokens = ["bob", "gets", "dave", ".", "she", "is", "happy", "."]
    cands = candidates_from_token_history(
        tokens, pronoun_position=4, pronoun="she",
    )
    # No female PERSON in S1 → empty candidates
    assert cands == []


def test_candidates_from_buffer_uses_entity_metadata():
    b = EpisodicBuffer(capacity=10, slot_dim=4)
    b.append(torch.randn(4), timestamp=0, metadata={"entity": "alice"})
    b.append(torch.randn(4), timestamp=1, metadata={"entity": "bob"})
    b.append(torch.randn(4), timestamp=2, metadata={"entity": "carol"})
    cands = candidates_from_buffer(b, "she")
    # alice + carol are F-PERSON
    assert {c for c, _ in cands} == {"alice", "carol"}


# ─────────────────────────────────────────────────────────────────
# Resolvers
# ─────────────────────────────────────────────────────────────────


def test_recency_picks_most_recent():
    cands = [("alice", 0), ("carol", 2)]
    r = resolve_pronoun_recency(
        ["alice", "gets", "carol", ".", "she", "is"],
        pronoun_position=4, candidates=cands,
    )
    assert r.predicted_referent == "carol"
    assert r.route == "recency"


def test_class_only_picks_first_candidate():
    cands = [("alice", 0), ("carol", 2)]
    r = resolve_pronoun_class_only(
        ["alice", "gets", "carol", ".", "she", "is"],
        pronoun_position=4, candidates=cands,
    )
    assert r.predicted_referent == "alice"
    assert r.route == "class_only"


def test_recency_empty_candidates():
    r = resolve_pronoun_recency(
        ["bob", "gets", "dave", ".", "she", "is"],
        pronoun_position=4, candidates=[],
    )
    assert r.predicted_referent is None
    assert r.route == "no_candidate"


def test_hypothesis_verify_target_position_smoke():
    """Smoke-only: untrained LM produces some prediction
    (not necessarily correct)."""
    torch.manual_seed(0)
    tok = Tokenizer()
    lm = PCMMiniLM(vocab=tok.vocab_size, d_model=32, n_layers=2)
    tokens = ["alice", "gets", "carol", ".", "she", "is", "happy", "."]
    r = resolve_pronoun(
        tokens, pronoun_position=4, lm=lm, tokenizer=tok,
        candidates=[("alice", 0), ("carol", 2)],
        scoring="target_position",
    )
    assert r.predicted_referent in {"alice", "carol"}
    assert r.route == "hypothesis_verify_target_position"


def test_hypothesis_verify_full_sentence_smoke():
    torch.manual_seed(0)
    tok = Tokenizer()
    lm = PCMMiniLM(vocab=tok.vocab_size, d_model=32, n_layers=2)
    tokens = ["alice", "gets", "carol", ".", "she", "is", "happy", "."]
    r = resolve_pronoun(
        tokens, pronoun_position=4, lm=lm, tokenizer=tok,
        candidates=[("alice", 0), ("carol", 2)],
        scoring="full_sentence",
    )
    assert r.predicted_referent in {"alice", "carol"}


def test_resolver_uses_buffer_when_provided():
    torch.manual_seed(0)
    tok = Tokenizer()
    lm = PCMMiniLM(vocab=tok.vocab_size, d_model=32, n_layers=2)
    b = EpisodicBuffer(capacity=10, slot_dim=4)
    b.append(torch.randn(4), timestamp=0, metadata={"entity": "alice"})
    b.append(torch.randn(4), timestamp=1, metadata={"entity": "carol"})
    tokens = ["alice", "gets", "carol", ".", "she", "is", "happy", "."]
    r = resolve_pronoun(
        tokens, pronoun_position=4, lm=lm, tokenizer=tok,
        buffer=b,  # no explicit candidates
    )
    assert r.predicted_referent in {"alice", "carol"}


def test_resolver_after_training_beats_chance():
    """After training on subject-continuity data with referent-
    substitution augmentation, hypothesis-verify should prefer
    the subject **above chance** (>= 0.60) and **above the
    anti-recency baseline** (which is 0.00 under this rule).

    The full F76 PoC reaches 0.95 at 4000 discourses + 15
    epochs; this unit test uses a tighter budget (800
    discourses, 30 epochs) and just verifies the mechanism is
    in the right direction. The PoC is the canonical
    quantitative result.
    """
    torch.manual_seed(42)
    tok = Tokenizer()
    rng = random.Random(0)
    discourses = []
    while len(discourses) < 2000:
        try:
            discourses.append(generate_coreference_discourse(
                rng, rule="subject_continuity",
            ))
        except RuntimeError:
            continue
    ids_list = []
    rng2 = random.Random(1)
    for d in discourses:
        toks = list(d.tokens)
        if rng2.random() < 0.5:
            toks[d.pronoun_position] = d.referent
        ids_list.append(tok.encode(toks, max_len=12))
    train_seq = torch.tensor(ids_list, dtype=torch.long)
    lm = PCMMiniLM(vocab=tok.vocab_size, d_model=48, n_layers=2,
                    combiner_hidden=96)
    opt = torch.optim.AdamW(lm.parameters(), lr=5e-3)
    # Mini-batch training, more epochs
    batch_size = 64
    N = train_seq.shape[0]
    for ep in range(40):
        idx = torch.randperm(N)
        for i in range(0, N, batch_size):
            j = idx[i:i + batch_size]
            batch = train_seq[j]
            x, y = batch[:, :-1], batch[:, 1:]
            logits = lm(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1), ignore_index=tok.pad_id,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    rng3 = random.Random(999)
    n_correct = 0
    n_total = 0
    for _ in range(100):
        try:
            d = generate_coreference_discourse(
                rng3, rule="subject_continuity",
            )
        except RuntimeError:
            continue
        cands = candidates_from_token_history(
            d.tokens, d.pronoun_position, d.pronoun,
        )
        r = resolve_pronoun(
            d.tokens, d.pronoun_position, lm, tok,
            candidates=cands, scoring="target_position",
        )
        n_total += 1
        if r.predicted_referent == d.referent:
            n_correct += 1
    acc = n_correct / max(n_total, 1)
    # 0.70 = 20pp above chance (0.5). PoC at 4000 disc + 15
    # epochs reaches 0.95.
    assert acc >= 0.70, f"trained resolver acc {acc} too low"
