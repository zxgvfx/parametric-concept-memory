"""F95 — ChatSession + generate() unit tests."""
from __future__ import annotations

import torch

from pcm.chat import ChatSession, ChatTurn, generate
from pcm.episodic import EpisodicBuffer
from pcm.lm import HybridPCMMiniLM
from pcm.user_memory import UserFactMemory


# ─────────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────────


def _tiny_lm(vocab: int = 32, d_model: int = 16):
    torch.manual_seed(0)
    return HybridPCMMiniLM(
        vocab=vocab, d_model=d_model, n_layers=2,
        n_heads=2, attn_every=2,
    )


def _tiny_vocab():
    words = [
        "<pad>", "<unk>", "<bos>", "<eos>",
        "hello", "world", "i", "am", "alex",
        "name", "is", "love", "like", "coffee",
        "user", "what", "the", "a", "an",
        "my", "you", "he", "she", "it",
        "and", "but", "or", "not", "yes",
        "no", "good", "bad",
    ]
    stoi = {w: i for i, w in enumerate(words)}
    itos = list(words)
    return stoi, itos


# ─────────────────────────────────────────────────────────────────
# generate() tests
# ─────────────────────────────────────────────────────────────────


def test_generate_returns_correct_length():
    model = _tiny_lm()
    prompt = torch.tensor([4, 5, 6], dtype=torch.long)
    out = generate(
        model, prompt, max_new_tokens=8, top_k=4,
    )
    assert out.dim() == 1
    assert out.shape[0] <= 8


def test_generate_stops_at_eos():
    model = _tiny_lm()
    prompt = torch.tensor([4, 5], dtype=torch.long)
    out = generate(
        model, prompt, max_new_tokens=64,
        top_k=1, temperature=1.0, eos_id=3,
    )
    if 3 in out.tolist():
        assert out.tolist().index(3) == out.shape[0] - 1


def test_generate_never_samples_pad():
    model = _tiny_lm()
    prompt = torch.tensor([4, 5], dtype=torch.long)
    for _ in range(5):
        out = generate(
            model, prompt, max_new_tokens=16,
            top_k=4, pad_id=0,
        )
        assert 0 not in out.tolist()


def test_generate_truncates_context():
    model = _tiny_lm()
    prompt = torch.arange(4, 20, dtype=torch.long)
    out = generate(
        model, prompt, max_new_tokens=4,
        top_k=4, max_context_len=8,
    )
    assert out.shape[0] <= 4


def test_generate_topk_zero_uses_full_distribution():
    model = _tiny_lm(vocab=8)
    prompt = torch.tensor([2], dtype=torch.long)
    out = generate(
        model, prompt, max_new_tokens=4, top_k=0,
        temperature=1.0,
    )
    assert out.shape[0] <= 4


def test_generate_2d_prompt_accepted():
    model = _tiny_lm()
    prompt = torch.tensor([[4, 5]], dtype=torch.long)
    out = generate(
        model, prompt, max_new_tokens=4, top_k=4,
    )
    assert out.dim() == 1


# ─────────────────────────────────────────────────────────────────
# ChatSession.encode_text / decode_ids tests
# ─────────────────────────────────────────────────────────────────


def test_encode_oov_to_unk():
    model = _tiny_lm()
    stoi, itos = _tiny_vocab()
    sess = ChatSession(
        model, stoi, itos,
        unk_id=1, pad_id=0,
    )
    ids = sess.encode_text("hello blah world")
    assert ids[0].item() == stoi["hello"]
    assert ids[1].item() == 1
    assert ids[2].item() == stoi["world"]


def test_encode_lowercase_and_strip_punct():
    model = _tiny_lm()
    stoi, itos = _tiny_vocab()
    sess = ChatSession(
        model, stoi, itos,
        unk_id=1, pad_id=0,
    )
    ids = sess.encode_text("Hello, World!")
    assert ids.tolist() == [stoi["hello"], stoi["world"]]


def test_encode_empty_yields_unk():
    model = _tiny_lm()
    stoi, itos = _tiny_vocab()
    sess = ChatSession(
        model, stoi, itos,
        unk_id=1, pad_id=0,
    )
    ids = sess.encode_text("    ")
    assert ids.tolist() == [1]


def test_decode_known_ids():
    model = _tiny_lm()
    stoi, itos = _tiny_vocab()
    sess = ChatSession(
        model, stoi, itos,
        unk_id=1, pad_id=0,
    )
    text = sess.decode_ids([
        stoi["hello"], stoi["world"],
    ])
    assert text == "hello world"


def test_decode_skips_pad():
    model = _tiny_lm()
    stoi, itos = _tiny_vocab()
    sess = ChatSession(
        model, stoi, itos,
        unk_id=1, pad_id=0,
    )
    text = sess.decode_ids([
        stoi["hello"], 0, stoi["world"],
    ])
    assert text == "hello world"


def test_decode_stops_at_eos():
    model = _tiny_lm()
    stoi, itos = _tiny_vocab()
    sess = ChatSession(
        model, stoi, itos,
        unk_id=1, pad_id=0, eos_id=3,
    )
    text = sess.decode_ids([
        stoi["hello"], 3, stoi["world"],
    ])
    assert text == "hello"


def test_decode_renders_unk():
    model = _tiny_lm()
    stoi, itos = _tiny_vocab()
    sess = ChatSession(
        model, stoi, itos,
        unk_id=1, pad_id=0,
    )
    text = sess.decode_ids([
        stoi["hello"], 1, stoi["world"],
    ])
    assert "<unk>" in text


# ─────────────────────────────────────────────────────────────────
# ChatSession.respond_to integration tests
# ─────────────────────────────────────────────────────────────────


def test_respond_to_produces_text():
    torch.manual_seed(0)
    model = _tiny_lm(vocab=32)
    stoi, itos = _tiny_vocab()
    sess = ChatSession(
        model, stoi, itos,
        unk_id=1, pad_id=0,
        max_response_tokens=8, top_k=4,
    )
    response = sess.respond_to("Hello world")
    assert isinstance(response, str)
    assert len(sess.turns) == 2
    assert sess.turns[0].role == "user"
    assert sess.turns[1].role == "agent"


def test_respond_to_extracts_user_facts():
    torch.manual_seed(0)
    model = _tiny_lm()
    stoi, itos = _tiny_vocab()
    sess = ChatSession(
        model, stoi, itos,
        unk_id=1, pad_id=0,
        max_response_tokens=4, top_k=4,
    )
    sess.respond_to("Hello, my name is Alex.")
    assert sess.user_memory.get_latest("name") is not None
    assert sess.user_memory.get_latest("name").value == "alex"


def test_respond_to_writes_episodic():
    torch.manual_seed(0)
    model = _tiny_lm()
    stoi, itos = _tiny_vocab()
    sess = ChatSession(
        model, stoi, itos,
        unk_id=1, pad_id=0,
        max_response_tokens=4, top_k=4,
    )
    assert len(sess.episodic) == 0
    sess.respond_to("hello world")
    assert len(sess.episodic) >= 1


def test_respond_to_persists_history():
    torch.manual_seed(0)
    model = _tiny_lm()
    stoi, itos = _tiny_vocab()
    sess = ChatSession(
        model, stoi, itos,
        unk_id=1, pad_id=0,
        max_response_tokens=4, top_k=4,
    )
    sess.respond_to("hello")
    sess.respond_to("world")
    history = sess.history()
    assert len(history) == 4
    assert history[0].text == "hello"
    assert history[2].text == "world"


def test_respond_to_oov_count():
    torch.manual_seed(0)
    model = _tiny_lm()
    stoi, itos = _tiny_vocab()
    sess = ChatSession(
        model, stoi, itos,
        unk_id=1, pad_id=0,
        max_response_tokens=4, top_k=4,
    )
    sess.respond_to("zzz blah qux")
    assert sess.turns[0].n_oov == 3


def test_stats_returns_dict():
    torch.manual_seed(0)
    model = _tiny_lm()
    stoi, itos = _tiny_vocab()
    sess = ChatSession(
        model, stoi, itos,
        unk_id=1, pad_id=0,
        max_response_tokens=4, top_k=4,
    )
    sess.respond_to("hello")
    s = sess.stats()
    assert s["n_user_turns"] == 1
    assert s["n_agent_turns"] == 1
    assert "oov_rate" in s


# ─────────────────────────────────────────────────────────────────
# Fact-prefix injection
# ─────────────────────────────────────────────────────────────────


def test_fact_prefix_injected_when_relevant():
    torch.manual_seed(0)
    model = _tiny_lm()
    stoi, itos = _tiny_vocab()
    sess = ChatSession(
        model, stoi, itos,
        unk_id=1, pad_id=0,
        max_response_tokens=4, top_k=4,
    )
    sess.user_memory.ingest("My name is Alex.", t=0)
    prefix = sess._fact_prefix("what is my name?")
    assert len(prefix) > 0
    decoded = sess.decode_ids(prefix)
    assert "name" in decoded
    assert "alex" in decoded


def test_fact_prefix_empty_when_no_relevant():
    torch.manual_seed(0)
    model = _tiny_lm()
    stoi, itos = _tiny_vocab()
    sess = ChatSession(
        model, stoi, itos,
        unk_id=1, pad_id=0,
        max_response_tokens=4, top_k=4,
    )
    sess.user_memory.ingest("My name is Alex.", t=0)
    prefix = sess._fact_prefix("the weather is nice")
    assert prefix == []
