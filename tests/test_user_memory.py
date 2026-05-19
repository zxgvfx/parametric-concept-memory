"""F95 — UserFactMemory unit tests."""
from __future__ import annotations

from pcm.user_memory import UserFact, UserFactMemory


# ─────────────────────────────────────────────────────────────────
# Basic extraction
# ─────────────────────────────────────────────────────────────────


def test_name_extraction():
    mem = UserFactMemory()
    new = mem.ingest("Hi, my name is Alex.", t=1)
    assert any(
        f.predicate == "name" and f.value == "alex" for f in new
    )


def test_name_extraction_im_alex():
    mem = UserFactMemory()
    new = mem.ingest("I'm Alex", t=1)
    names = [f.value for f in new if f.predicate == "name"]
    assert "alex" in names


def test_likes_extraction():
    mem = UserFactMemory()
    new = mem.ingest("I love hiking on weekends.", t=1)
    likes = [f.value for f in new if f.predicate == "likes"]
    assert "hiking" in likes


def test_dislikes_extraction():
    mem = UserFactMemory()
    new = mem.ingest("I hate broccoli.", t=1)
    dislikes = [
        f.value for f in new if f.predicate == "dislikes"
    ]
    assert "broccoli" in dislikes


def test_is_a_extraction():
    mem = UserFactMemory()
    new = mem.ingest("I am a programmer.", t=1)
    iss = [f.value for f in new if f.predicate == "is"]
    assert "programmer" in iss


def test_age_extraction():
    mem = UserFactMemory()
    new = mem.ingest("I am 30 years old.", t=1)
    ages = [f.value for f in new if f.predicate == "age"]
    assert "30" in ages


def test_lives_in_extraction():
    mem = UserFactMemory()
    new = mem.ingest("I live in Boston.", t=1)
    places = [
        f.value for f in new if f.predicate == "lives_in"
    ]
    assert "boston" in places


# ─────────────────────────────────────────────────────────────────
# Multi-fact + dedup
# ─────────────────────────────────────────────────────────────────


def test_multi_fact_single_utterance():
    mem = UserFactMemory()
    new = mem.ingest(
        "My name is Alex and I love coffee.", t=1,
    )
    preds_values = {(f.predicate, f.value) for f in new}
    assert ("name", "alex") in preds_values
    assert ("likes", "coffee") in preds_values


def test_dedup_same_fact_twice():
    mem = UserFactMemory()
    mem.ingest("My name is Alex.", t=1)
    new2 = mem.ingest("My name is Alex.", t=2)
    assert new2 == []
    assert len(mem) == 1


def test_dedup_partial():
    mem = UserFactMemory()
    mem.ingest("I like hiking.", t=1)
    new2 = mem.ingest("I like coffee.", t=2)
    assert len(new2) == 1
    assert new2[0].value == "coffee"
    assert len(mem.get("likes")) == 2


# ─────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────


def test_get_predicate():
    mem = UserFactMemory()
    mem.ingest("My name is Alex.", t=1)
    mem.ingest("I love coffee.", t=2)
    names = [f.value for f in mem.get("name")]
    likes = [f.value for f in mem.get("likes")]
    assert names == ["alex"]
    assert likes == ["coffee"]


def test_get_latest():
    mem = UserFactMemory()
    mem.ingest("My name is Alex.", t=1)
    mem.ingest("My name is Bob.", t=2)
    latest = mem.get_latest("name")
    assert latest is not None
    assert latest.value == "bob"


def test_get_relevant_facts_name_question():
    mem = UserFactMemory()
    mem.ingest("My name is Alex.", t=1)
    mem.ingest("I love coffee.", t=2)
    relevant = mem.get_relevant_facts("what is my name?")
    pred_values = [(f.predicate, f.value) for f in relevant]
    assert ("name", "alex") in pred_values


def test_get_relevant_facts_likes_question():
    mem = UserFactMemory()
    mem.ingest("My name is Alex.", t=1)
    mem.ingest("I love coffee.", t=2)
    relevant = mem.get_relevant_facts(
        "what do I like to drink?",
    )
    pred_values = [(f.predicate, f.value) for f in relevant]
    assert ("likes", "coffee") in pred_values


def test_get_relevant_facts_empty_query():
    mem = UserFactMemory()
    mem.ingest("My name is Alex.", t=1)
    relevant = mem.get_relevant_facts(
        "the weather is nice today",
    )
    assert relevant == []


# ─────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────


def test_as_dict():
    mem = UserFactMemory()
    mem.ingest("My name is Alex.", t=1)
    mem.ingest("I love coffee.", t=2)
    d = mem.as_dict()
    assert d["n_facts"] == 2
    assert d["facts_by_predicate"]["name"] == ["alex"]
    assert d["facts_by_predicate"]["likes"] == ["coffee"]


# ─────────────────────────────────────────────────────────────────
# Robustness
# ─────────────────────────────────────────────────────────────────


def test_empty_input():
    mem = UserFactMemory()
    new = mem.ingest("", t=1)
    assert new == []


def test_no_match():
    mem = UserFactMemory()
    new = mem.ingest("Hello, how are you?", t=1)
    assert new == []


def test_iterable():
    mem = UserFactMemory()
    mem.ingest("My name is Alex.", t=1)
    facts = list(mem)
    assert len(facts) == 1
    assert facts[0].predicate == "name"
