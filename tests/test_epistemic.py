"""F96 — EpistemicAgent unit tests."""
from __future__ import annotations

import torch

from pcm.epistemic import (
    ArithmeticVerifier,
    BeliefStore,
    Claim,
    ClaimParser,
    ConsistencyVerifier,
    EpistemicAgent,
    LMPlausibilityVerifier,
    VerifyResult,
    WorldModel,
)
from pcm.lm import HybridPCMMiniLM


# ─────────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────────


def _tiny_lm(vocab: int = 64, d_model: int = 16):
    torch.manual_seed(0)
    return HybridPCMMiniLM(
        vocab=vocab, d_model=d_model, n_layers=2,
        n_heads=2, attn_every=2,
    )


def _tiny_vocab():
    words = [
        "<pad>", "<unk>", "<bos>", "<eos>",
        "i", "am", "is", "are", "the", "a", "an",
        "my", "name", "my", "love", "like",
        "cat", "dog", "bird", "rabbit", "horse",
        "apple", "bread", "cake", "fruit",
        "car", "truck", "bus", "train",
        "house", "tree", "stone",
        "boy", "girl", "child",
        "alex", "tim", "lily", "tom",
        "plus", "minus", "times", "equals",
        "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "not", "no", "yes", "and", "but", "or",
    ]
    stoi = {w: i for i, w in enumerate(words)}
    itos = list(words)
    return stoi, itos


# ─────────────────────────────────────────────────────────────────
# ClaimParser tests
# ─────────────────────────────────────────────────────────────────


def test_parse_arithmetic_symbolic():
    p = ClaimParser()
    claims = p.parse("1 + 1 = 2")
    assert len(claims) == 1
    c = claims[0]
    assert c.type == "arithmetic"
    assert c.lhs == "1"
    assert c.op == "+"
    assert c.rhs == "1"
    assert c.claimed_value == "2"


def test_parse_arithmetic_word():
    p = ClaimParser()
    claims = p.parse("2 plus 3 is 5")
    assert any(c.type == "arithmetic" for c in claims)
    c = next(c for c in claims if c.type == "arithmetic")
    assert c.lhs == "2"
    assert "plus" in (c.op or "").lower()
    assert c.rhs == "3"
    assert c.claimed_value == "5"


def test_parse_arithmetic_wrong():
    p = ClaimParser()
    claims = p.parse("1 + 1 = 3")
    assert claims[0].claimed_value == "3"


def test_parse_self_report_name():
    p = ClaimParser()
    claims = p.parse("my name is alex")
    assert any(
        c.type == "self_report" and c.predicate == "name"
        and c.value == "alex"
        for c in claims
    )


def test_parse_self_report_like():
    p = ClaimParser()
    claims = p.parse("i love coffee")
    assert any(
        c.type == "self_report" and c.predicate == "likes"
        and c.value == "coffee"
        for c in claims
    )


def test_parse_identity_positive():
    p = ClaimParser()
    claims = p.parse("a cat is an animal")
    assert any(
        c.type == "identity_positive"
        and c.subject == "cat" and c.object == "animal"
        for c in claims
    )


def test_parse_identity_negative():
    p = ClaimParser()
    claims = p.parse("a cat is not a car")
    assert any(
        c.type == "identity_negative"
        and c.subject == "cat" and c.object == "car"
        for c in claims
    )


def test_parse_ignores_attribute_predicates():
    p = ClaimParser()
    claims = p.parse("the cat is happy")
    types = [c.type for c in claims]
    assert "identity_positive" not in types


def test_parse_ignores_pronoun_subjects():
    p = ClaimParser()
    claims = p.parse("i am tired")
    types = [c.type for c in claims]
    assert "identity_positive" not in types
    assert "identity_negative" not in types


# ─────────────────────────────────────────────────────────────────
# ArithmeticVerifier tests
# ─────────────────────────────────────────────────────────────────


def test_arith_correct():
    c = Claim(
        type="arithmetic", raw_text="1+1=2",
        lhs="1", op="+", rhs="1", claimed_value="2",
    )
    r = ArithmeticVerifier.check(c)
    assert r.status == "verified"


def test_arith_wrong():
    c = Claim(
        type="arithmetic", raw_text="1+1=3",
        lhs="1", op="+", rhs="1", claimed_value="3",
    )
    r = ArithmeticVerifier.check(c)
    assert r.status == "contradicted"
    assert r.correct_value == "2"


def test_arith_word_op_plus():
    c = Claim(
        type="arithmetic", raw_text="2 plus 3 is 5",
        lhs="2", op="plus", rhs="3", claimed_value="5",
    )
    r = ArithmeticVerifier.check(c)
    assert r.status == "verified"


def test_arith_word_op_minus():
    c = Claim(
        type="arithmetic", raw_text="10 minus 3 is 7",
        lhs="10", op="minus", rhs="3", claimed_value="7",
    )
    r = ArithmeticVerifier.check(c)
    assert r.status == "verified"


def test_arith_times():
    c = Claim(
        type="arithmetic", raw_text="3*4=12",
        lhs="3", op="*", rhs="4", claimed_value="12",
    )
    r = ArithmeticVerifier.check(c)
    assert r.status == "verified"


def test_arith_division():
    c = Claim(
        type="arithmetic", raw_text="10/2=5",
        lhs="10", op="/", rhs="2", claimed_value="5",
    )
    r = ArithmeticVerifier.check(c)
    assert r.status == "verified"


def test_arith_division_by_zero_uncertain():
    c = Claim(
        type="arithmetic", raw_text="1/0=0",
        lhs="1", op="/", rhs="0", claimed_value="0",
    )
    r = ArithmeticVerifier.check(c)
    assert r.status == "uncertain"


def test_arith_skip_non_arithmetic():
    c = Claim(
        type="self_report", raw_text="hi",
        predicate="name", value="alex",
    )
    r = ArithmeticVerifier.check(c)
    assert r.status == "uncertain"


# ─────────────────────────────────────────────────────────────────
# BeliefStore tests
# ─────────────────────────────────────────────────────────────────


def test_belief_add_and_count():
    bs = BeliefStore()
    bs.add(Claim(
        type="identity_positive", raw_text="a cat is an animal",
        subject="cat", object="animal", t=1,
    ))
    assert len(bs) == 1


def test_belief_idempotent():
    bs = BeliefStore()
    c = Claim(
        type="identity_positive", raw_text="a cat is an animal",
        subject="cat", object="animal", t=1,
    )
    bs.add(c)
    bs.add(c)
    assert len(bs) == 1


def test_belief_find_conflict_negation():
    bs = BeliefStore()
    bs.add(Claim(
        type="identity_positive", raw_text="a cat is an animal",
        subject="cat", object="animal", t=1,
    ))
    neg = Claim(
        type="identity_negative", raw_text="a cat is not an animal",
        subject="cat", object="animal", t=2,
    )
    conflict = bs.find_conflict(neg)
    assert conflict is not None
    assert conflict.type == "identity_positive"


def test_belief_find_conflict_arith():
    bs = BeliefStore()
    bs.add(Claim(
        type="arithmetic", raw_text="1+1=2",
        lhs="1", op="+", rhs="1", claimed_value="2", t=1,
    ))
    new = Claim(
        type="arithmetic", raw_text="1+1=3",
        lhs="1", op="+", rhs="1", claimed_value="3", t=2,
    )
    conflict = bs.find_conflict(new)
    assert conflict is not None
    assert conflict.claimed_value == "2"


def test_belief_no_conflict_when_consistent():
    bs = BeliefStore()
    bs.add(Claim(
        type="identity_positive", raw_text="a cat is an animal",
        subject="cat", object="animal", t=1,
    ))
    next_claim = Claim(
        type="identity_positive", raw_text="a dog is an animal",
        subject="dog", object="animal", t=2,
    )
    conflict = bs.find_conflict(next_claim)
    assert conflict is None


# ─────────────────────────────────────────────────────────────────
# WorldModel tests
# ─────────────────────────────────────────────────────────────────


def test_world_model_builds_with_lm():
    lm = _tiny_lm()
    stoi, _ = _tiny_vocab()
    wm = WorldModel(lm, stoi)
    assert wm.sim_matrix.shape[0] == len(wm.words)
    assert wm.sim_matrix.shape[1] == len(wm.words)
    assert len(wm.words) >= 6


def test_world_model_same_category():
    lm = _tiny_lm()
    stoi, _ = _tiny_vocab()
    wm = WorldModel(lm, stoi)
    assert wm.are_known_same_category("cat", "dog") is True
    assert wm.are_known_same_category("cat", "apple") is False


def test_world_model_unknown_word_returns_none():
    lm = _tiny_lm()
    stoi, _ = _tiny_vocab()
    wm = WorldModel(lm, stoi)
    assert wm.are_likely_distinct("cat", "xyz") is None
    assert wm.are_likely_distinct("zzz", "dog") is None


# ─────────────────────────────────────────────────────────────────
# ConsistencyVerifier tests
# ─────────────────────────────────────────────────────────────────


def test_consistency_detects_belief_store_conflict():
    bs = BeliefStore()
    bs.add(Claim(
        type="identity_positive", raw_text="a cat is an animal",
        subject="cat", object="animal", t=1,
    ))
    cv = ConsistencyVerifier(bs, world_model=None)
    neg = Claim(
        type="identity_negative", raw_text="a cat is not an animal",
        subject="cat", object="animal", t=2,
    )
    r = cv.check(neg)
    assert r.status == "contradicted"
    assert r.conflict_with is not None


def test_consistency_passes_consistent_claim():
    bs = BeliefStore()
    cv = ConsistencyVerifier(bs, world_model=None)
    c = Claim(
        type="identity_positive", raw_text="a dog is an animal",
        subject="dog", object="animal", t=1,
    )
    r = cv.check(c)
    assert r.status == "verified"


def test_consistency_skips_non_identity():
    bs = BeliefStore()
    cv = ConsistencyVerifier(bs, world_model=None)
    c = Claim(
        type="self_report", raw_text="hi",
        predicate="name", value="alex",
    )
    r = cv.check(c)
    assert r.status == "uncertain"


# ─────────────────────────────────────────────────────────────────
# EpistemicAgent end-to-end tests
# ─────────────────────────────────────────────────────────────────


def test_epi_arith_correct_accepts():
    lm = _tiny_lm()
    stoi, _ = _tiny_vocab()
    epi = EpistemicAgent(
        lm=lm, stoi=stoi, use_lm_plausibility=False,
    )
    result = epi.process("1 + 1 = 2", t=1)
    assert result.action == "accept"


def test_epi_arith_wrong_pushes_back():
    lm = _tiny_lm()
    stoi, _ = _tiny_vocab()
    epi = EpistemicAgent(
        lm=lm, stoi=stoi, use_lm_plausibility=False,
    )
    result = epi.process("1 + 1 = 3", t=1)
    assert result.action == "pushback"
    assert result.pushback_text is not None
    assert "2" in result.pushback_text


def test_epi_self_report_accepts():
    lm = _tiny_lm()
    stoi, _ = _tiny_vocab()
    epi = EpistemicAgent(
        lm=lm, stoi=stoi, use_lm_plausibility=False,
    )
    result = epi.process("my name is alex", t=1)
    assert result.action == "accept"
    assert len(result.new_beliefs) >= 1


def test_epi_contradiction_pushes_back():
    lm = _tiny_lm()
    stoi, _ = _tiny_vocab()
    epi = EpistemicAgent(
        lm=lm, stoi=stoi, use_lm_plausibility=False,
    )
    epi.process("a cat is an animal", t=1)
    result = epi.process("a cat is not an animal", t=2)
    assert result.action == "pushback"
    assert "earlier" in result.pushback_text.lower()


def test_epi_no_claim_returns_none_action():
    lm = _tiny_lm()
    stoi, _ = _tiny_vocab()
    epi = EpistemicAgent(
        lm=lm, stoi=stoi, use_lm_plausibility=False,
    )
    result = epi.process("hello there", t=1)
    assert result.action == "no_claim"
    assert result.outcomes == []


def test_epi_belief_store_grows_on_accept():
    lm = _tiny_lm()
    stoi, _ = _tiny_vocab()
    epi = EpistemicAgent(
        lm=lm, stoi=stoi, use_lm_plausibility=False,
    )
    epi.process("my name is alex", t=1)
    epi.process("1 + 1 = 2", t=2)
    epi.process("a dog is an animal", t=3)
    assert len(epi.belief_store) >= 3


def test_epi_belief_store_does_not_grow_on_pushback():
    lm = _tiny_lm()
    stoi, _ = _tiny_vocab()
    epi = EpistemicAgent(
        lm=lm, stoi=stoi, use_lm_plausibility=False,
    )
    epi.process("1 + 1 = 3", t=1)
    arith_beliefs = [
        b for b in epi.belief_store
        if b["relation"] == "="
    ]
    assert arith_beliefs == []


def test_epi_world_model_catches_category_violation():
    """Use a hand-constructed WorldModel that forces
    cat <-> car cosine far below threshold, so the test
    measures the *logic* of the world model dispatcher
    instead of depending on a random LM's similarity."""
    lm = _tiny_lm()
    stoi, _ = _tiny_vocab()
    wm = WorldModel(lm, stoi, dissim_threshold=0.10)
    # Forcibly set cat <-> car similarity to ~0.0 so the
    # dissim check fires regardless of random init.
    if "cat" in wm._word_to_idx and "car" in wm._word_to_idx:
        i = wm._word_to_idx["cat"]
        j = wm._word_to_idx["car"]
        wm.sim_matrix[i, j] = 0.0
        wm.sim_matrix[j, i] = 0.0
    epi = EpistemicAgent(
        lm=lm, stoi=stoi, world_model=wm,
        use_lm_plausibility=False,
    )
    result = epi.process("a cat is a car", t=1)
    assert result.action in ("pushback", "uncertain")


# ─────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────


def test_epi_stats():
    lm = _tiny_lm()
    stoi, _ = _tiny_vocab()
    epi = EpistemicAgent(
        lm=lm, stoi=stoi, use_lm_plausibility=False,
    )
    epi.process("my name is alex", t=1)
    s = epi.stats()
    assert "n_beliefs" in s
    assert s["n_beliefs"] >= 1
