"""pcm.chat — F95 chat-learning agent: conversational REPL +
online learning loop.

Integrates the existing PCM cognitive primitives into a
deployable chat agent:

* :func:`generate` — autoregressive sampling (top-K + temperature
  + EOS) over a :class:`~pcm.lm.HybridPCMMiniLM`.
* :class:`ChatSession` — stateful per-conversation object that:
  * maintains a turn-by-turn history,
  * encodes user text into the F90 vocab (with OOV → UNK
    fallback + per-turn OOV rate logging),
  * builds a sliding-window context across recent turns,
  * generates an agent response,
  * writes both turns into an :class:`~pcm.episodic.EpisodicBuffer`
    (F75 short-term hippocampal trace),
  * extracts user facts via :class:`~pcm.user_memory.UserFactMemory`
    (regex templates: "my name is X" / "I like Y" / "I am Z"),
  * injects relevant remembered facts into the prompt before
    generation (RAG-lite),
  * (optional) drives an :class:`~pcm.online.OnlineTeacherSession`
    to update the LM with one M3 micro-gradient step per turn
    (with pretrain replay to prevent catastrophic forgetting).

Cognitive parallels:

* The episodic write per turn corresponds to hippocampal
  encoding of "what just happened" (F75 M1).
* The fact-extraction step is a simplified semantic-memory
  consolidation (F75 LongTermEpisodicTrace).
* The optional M3 step corresponds to slow cortical re-tuning
  of word representations against the user's actual
  distribution.

This module is the F95 integration layer; it owns no new
architectural primitives. All cognitive components are
imported from existing modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .episodic import EpisodicBuffer
from .user_memory import UserFact, UserFactMemory


__all__ = [
    "generate",
    "ChatTurn",
    "ChatSession",
]


# ─────────────────────────────────────────────────────────────────
# Autoregressive sampling
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def generate(
    model: nn.Module, prompt_ids: torch.Tensor, *,
    max_new_tokens: int = 32,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    eos_id: int | None = None,
    pad_id: int = 0,
    max_context_len: int = 256,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Sample tokens autoregressively from a PCM language model.

    Args:
        model: any LM exposing ``forward(x) -> logits`` of shape
            ``(B, L, V)``. ``HybridPCMMiniLM`` is the canonical
            target.
        prompt_ids: 1-D or 2-D ``LongTensor``; ``(L,)`` or
            ``(1, L)``.
        max_new_tokens: maximum tokens to generate beyond the
            prompt.
        temperature: softmax temperature; lower = greedier.
        top_k: keep only the top ``k`` logits (``0`` disables).
        top_p: nucleus sampling cutoff (``1.0`` disables).
        repetition_penalty: divides logits of tokens already in
            the context by this factor (``1.0`` disables); a
            mild deterrent for the LM's tendency to repeat at
            low temperature.
        eos_id: stop generation if this token is sampled.
        pad_id: ignored in the prompt; never sampled.
        max_context_len: truncate context to the most recent
            ``max_context_len`` tokens per step (mini-LM context
            limit).
        device: target device.

    Returns:
        ``(N,)`` ``LongTensor`` of newly-generated token ids
        (does NOT include the prompt). ``N`` ≤ ``max_new_tokens``.
    """
    model.eval()
    if not isinstance(prompt_ids, torch.Tensor):
        prompt_ids = torch.tensor(
            prompt_ids, dtype=torch.long,
        )
    prompt = prompt_ids.to(device)
    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)
    ctx = prompt
    generated: list[int] = []
    for _ in range(max_new_tokens):
        if ctx.shape[1] > max_context_len:
            ctx_in = ctx[:, -max_context_len:]
        else:
            ctx_in = ctx
        logits = model(ctx_in)
        next_logits = logits[0, -1].clone()
        if repetition_penalty != 1.0:
            seen = set(ctx_in[0].tolist())
            for tid in seen:
                if next_logits[tid] > 0:
                    next_logits[tid] = (
                        next_logits[tid] / repetition_penalty
                    )
                else:
                    next_logits[tid] = (
                        next_logits[tid] * repetition_penalty
                    )
        if pad_id is not None:
            next_logits[pad_id] = float("-inf")
        next_logits = next_logits / max(temperature, 1e-6)
        if top_k > 0:
            k = min(top_k, next_logits.shape[-1])
            vals, idx = next_logits.topk(k)
            probs = F.softmax(vals, dim=-1)
            if top_p < 1.0:
                cum = probs.cumsum(dim=-1)
                keep = cum <= top_p
                keep[0] = True  # always keep the top-1
                probs = probs * keep.float()
                probs = probs / probs.sum().clamp_min(1e-9)
            choice = int(torch.multinomial(probs, 1).item())
            next_id = int(idx[choice].item())
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        ctx = torch.cat(
            [ctx, torch.tensor(
                [[next_id]], dtype=torch.long, device=device,
            )],
            dim=1,
        )
    return torch.tensor(generated, dtype=torch.long, device=device)


# ─────────────────────────────────────────────────────────────────
# ChatTurn — one user or agent turn
# ─────────────────────────────────────────────────────────────────


@dataclass
class ChatTurn:
    """One turn in a chat session."""
    role: str                       # "user" | "agent"
    text: str
    ids: list[int]
    t: int                          # turn index
    n_oov: int = 0                  # number of OOV tokens
    new_user_facts: list[UserFact] = field(default_factory=list)
    surprise: float | None = None   # cross-entropy of this turn
                                    # under the model (agent
                                    # responses use the surprise
                                    # of the *user's next* turn)


# ─────────────────────────────────────────────────────────────────
# ChatSession — top-level chat agent
# ─────────────────────────────────────────────────────────────────


class ChatSession:
    """Stateful chat session integrating LM + episodic memory +
    user-fact memory + (optional) online learning.

    Each call to :meth:`respond_to` performs four cognitive
    operations:

    1. **Encode + episodic-write the user turn.** The user text
       is tokenised; OOV tokens become ``unk_id``. The final
       hidden state at the last user-input token is appended to
       :attr:`episodic` with role-tagged metadata.
    2. **Extract user facts.** Regex templates pick up facts
       like "my name is Alex" / "I like hiking" / "I am a
       programmer". These persist for the lifetime of the
       :class:`UserFactMemory`.
    3. **Build context + generate.** Recent turns are
       concatenated into a sliding-window context. If any
       relevant remembered fact matches the user query
       (predicate-based heuristic), a short prefix
       like ``"user name alex"`` is prepended to the context
       *before* generation. This is RAG-lite: a non-parametric
       memory injection.
    4. **(Optional) Online learning.** If a teacher session is
       attached, the user's response is fed back as a
       correction: one M3 micro-gradient step on
       ``tok_emb + ln_final`` with pretrain replay (replay is
       the catastrophic-forgetting safeguard).

    Args:
        model: a PCM language model (any nn.Module with a
            ``forward(x) -> logits`` API and a
            ``hidden_states(x)`` method; ``HybridPCMMiniLM`` is
            the default target).
        stoi: word → id mapping (must be the same vocab the LM
            was pretrained with).
        itos: list ``id → word`` (length = vocab size).
        user_memory: a :class:`UserFactMemory` (created if
            ``None``).
        episodic: an :class:`EpisodicBuffer` (created if
            ``None``).
        teacher: optional :class:`~pcm.online.OnlineTeacherSession`
            wired with a pretrain replay buffer. If supplied,
            :meth:`respond_to` will call
            :meth:`OnlineTeacherSession.receive_chat_turn` on
            each turn after the first.
        pad_id / eos_id / unk_id: vocabulary specials.
        max_context_len: per-step context truncation.
        device: ``"cuda"`` or ``"cpu"``.
        max_response_tokens / temperature / top_k / top_p /
            repetition_penalty: generation knobs (see
            :func:`generate`).
        fact_prefix_max_tokens: cap on injected fact-prefix
            length per turn.
    """

    def __init__(
        self, model: nn.Module,
        stoi: dict[str, int], itos: list[str],
        *,
        user_memory: UserFactMemory | None = None,
        episodic: EpisodicBuffer | None = None,
        teacher=None,           # OnlineTeacherSession | None
        pad_id: int = 0,
        eos_id: int | None = None,
        unk_id: int = 1,
        max_context_len: int = 256,
        device: str | torch.device = "cpu",
        max_response_tokens: int = 24,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        fact_prefix_max_tokens: int = 16,
    ) -> None:
        self.model = model
        self.stoi = stoi
        self.itos = itos
        self.user_memory = user_memory or UserFactMemory()
        self.episodic = episodic or EpisodicBuffer(
            capacity=256,
            slot_dim=getattr(model, "d_model", 64),
            device=str(device),
        )
        self.teacher = teacher
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.max_context_len = max_context_len
        self.device = device
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.fact_prefix_max_tokens = fact_prefix_max_tokens
        self.turns: list[ChatTurn] = []
        self.t = 0
        self._last_agent_ids: torch.Tensor | None = None

    # ── Encoding / decoding ─────────────────────────────────────

    def encode_text(self, text: str) -> torch.Tensor:
        """Lowercase + whitespace-tokenise + map to ids with
        OOV → ``unk_id``. Punctuation is stripped from word
        boundaries."""
        words = text.lower().split()
        ids: list[int] = []
        for w in words:
            w = w.strip('.,!?;:"\'()[]{}')
            if not w:
                continue
            ids.append(self.stoi.get(w, self.unk_id))
        if not ids:
            ids = [self.unk_id]
        return torch.tensor(ids, dtype=torch.long)

    def decode_ids(self, ids: torch.Tensor | list[int]) -> str:
        """Detokenise back to a string. Drops EOS / PAD; UNK
        rendered as ``"<unk>"``."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        out: list[str] = []
        for i in ids:
            if self.eos_id is not None and i == self.eos_id:
                break
            if i == self.pad_id:
                continue
            if i == self.unk_id:
                out.append("<unk>")
            elif 0 <= i < len(self.itos):
                out.append(self.itos[i])
            else:
                out.append("<oob>")
        return " ".join(out)

    # ── Context construction ────────────────────────────────────

    def _build_context(self) -> torch.Tensor:
        """Concatenate token ids from the last few turns (most
        recent first, then reversed back to chronological). Cap
        to ``max_context_len`` by trimming the front."""
        ids: list[int] = []
        for turn in self.turns[-8:]:
            ids.extend(turn.ids)
        if not ids:
            ids = [self.unk_id]
        ids = ids[-self.max_context_len:]
        return torch.tensor(ids, dtype=torch.long)

    def _fact_prefix(self, user_text: str) -> list[int]:
        """If the user query references a remembered fact's
        predicate, build a short prefix that says e.g.
        ``"user name alex"`` (RAG-lite). Returns the token id
        list (possibly empty)."""
        relevant = self.user_memory.get_relevant_facts(user_text)
        if not relevant:
            return []
        ids: list[int] = []
        for fact in relevant:
            phrase = f"user {fact.predicate} {fact.value}"
            ids.extend(self.encode_text(phrase).tolist())
            if len(ids) >= self.fact_prefix_max_tokens:
                break
        return ids[:self.fact_prefix_max_tokens]

    # ── Episodic write ──────────────────────────────────────────

    def _episodic_write(
        self, ids: torch.Tensor, role: str,
    ) -> None:
        if ids.numel() == 0:
            return
        self.model.eval()
        with torch.no_grad():
            h = self.model.hidden_states(
                ids.unsqueeze(0).to(self.device)
            )
            last_slot = h[0, -1].detach()
        self.episodic.append(
            slot=last_slot,
            timestamp=self.t,
            salience=float(ids.numel()),
            metadata={
                "role": role,
                "ids": ids.cpu().tolist(),
            },
        )

    # ── Public API: one chat turn ───────────────────────────────

    def respond_to(self, user_text: str) -> str:
        """Run one turn end-to-end. Returns the agent's text
        response.

        Workflow:

        1. Encode user text → tokens.
        2. Run online learning on the *previous* agent response
           (if any), using the *current* user input as the
           correction signal.
        3. Episodic-write the user turn.
        4. Extract any new user facts from the user text.
        5. Build prompt = optional fact prefix + last 8 turns.
        6. Generate response tokens.
        7. Episodic-write the agent turn.
        """
        self.t += 1
        user_ids = self.encode_text(user_text)
        n_oov = int(
            (user_ids == self.unk_id).sum().item()
        )

        # 1. Online learning: feed previous agent context →
        #    real user response back to the teacher.
        if self.teacher is not None and self._last_agent_ids is not None:
            self.teacher.receive_chat_turn(
                context_ids=self._last_agent_ids.cpu(),
                target_ids=user_ids.cpu(),
            )

        # 2. Episodic write for the user turn.
        self._episodic_write(user_ids.to(self.device), role="user")

        # 3. Extract new user facts (semantic memory).
        new_facts = self.user_memory.ingest(user_text, t=self.t)

        # 4. Save turn before building the prompt that the
        #    response is conditioned on (so the latest user text
        #    is part of the context).
        self.turns.append(ChatTurn(
            role="user", text=user_text,
            ids=user_ids.tolist(),
            t=self.t, n_oov=n_oov,
            new_user_facts=new_facts,
        ))

        # 5. Build context. Optionally prepend a fact prefix.
        ctx_ids = self._build_context().tolist()
        prefix = self._fact_prefix(user_text)
        full_ids = prefix + ctx_ids
        full_ids = full_ids[-self.max_context_len:]
        ctx = torch.tensor(full_ids, dtype=torch.long)

        # 6. Generate agent response. When a fact prefix is
        # injected, we DISABLE repetition_penalty for this turn
        # so the LM can actually echo the injected fact word —
        # otherwise the penalty (which divides logits of seen
        # tokens) actively suppresses the fact's reappearance,
        # which is the opposite of what RAG-lite needs.
        rep_pen = (
            1.0 if prefix else self.repetition_penalty
        )
        new_ids = generate(
            self.model, ctx,
            max_new_tokens=self.max_response_tokens,
            temperature=self.temperature, top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=rep_pen,
            eos_id=self.eos_id, pad_id=self.pad_id,
            max_context_len=self.max_context_len,
            device=self.device,
        )
        agent_text = self.decode_ids(new_ids)

        # 7. Record + episodic-write agent turn.
        self.turns.append(ChatTurn(
            role="agent", text=agent_text,
            ids=new_ids.cpu().tolist(),
            t=self.t, n_oov=0,
            new_user_facts=[],
        ))
        self._episodic_write(new_ids, role="agent")
        self._last_agent_ids = new_ids

        return agent_text

    # ── Diagnostics ─────────────────────────────────────────────

    def history(self) -> list[ChatTurn]:
        """Return the full conversation history."""
        return list(self.turns)

    def stats(self) -> dict:
        """Return aggregate session statistics."""
        n_user = sum(1 for t in self.turns if t.role == "user")
        n_agent = sum(1 for t in self.turns if t.role == "agent")
        n_oov_total = sum(t.n_oov for t in self.turns)
        n_tokens_total = sum(
            len(t.ids) for t in self.turns if t.role == "user"
        )
        return {
            "n_turns_total": len(self.turns),
            "n_user_turns": n_user,
            "n_agent_turns": n_agent,
            "n_oov_tokens": n_oov_total,
            "n_user_tokens": n_tokens_total,
            "oov_rate": (
                n_oov_total / max(n_tokens_total, 1)
            ),
            "n_episodic_entries": len(self.episodic),
            "n_user_facts": len(self.user_memory),
            "n_online_grad_steps": (
                self.teacher.counters.n_m3_grad_steps
                if self.teacher else 0
            ),
        }


# ─────────────────────────────────────────────────────────────────
# CLI entry point — python -m pcm.chat
# ─────────────────────────────────────────────────────────────────


def _cli() -> None:
    """Interactive chat REPL with an F90-class LM checkpoint.

    Usage::

        python -m pcm.chat \\
            --corpus outputs/f79_data/tinystories_valid.txt \\
            --lm-checkpoint outputs/checkpoints/f91_lm_d512.pt

    Special commands inside the REPL:

    * ``/stats``      — print session statistics
    * ``/facts``      — list user facts known so far
    * ``/history N``  — print last N turns
    * ``/learn on|off`` — toggle online learning
    * ``/quit``       — exit
    """
    import argparse
    import random as _random
    from pathlib import Path

    from experiments.online_teacher_f85 import (
        build_vocab_with_reserved,
    )
    from experiments.tinystories_f79 import (
        BOS_ID as _BOS, EOS_ID as _EOS,
        PAD_ID as _PAD, UNK_ID as _UNK,
        _concatenate_ids, _sample_seq_batch, encode_story,
    )
    from pcm.lm import HybridPCMMiniLM
    from pcm.lm_synthetic import RESERVED_CONCEPTS
    from pcm.online import (
        OnlineTeacherSession, PretrainReplayBuffer,
    )

    ap = argparse.ArgumentParser(
        description=(
            "PCM v10.6 live chat REPL — F90 LM + episodic "
            "memory + online learning."
        ),
    )
    ap.add_argument(
        "--corpus", type=Path,
        default=Path(
            "outputs/f79_data/tinystories_valid.txt",
        ),
    )
    ap.add_argument("--vocab-cap", type=int, default=4096)
    ap.add_argument("--n-stories", type=int, default=10000)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--attn-every", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument(
        "--lm-checkpoint", type=Path,
        default=Path("outputs/checkpoints/f91_lm_d512.pt"),
    )
    ap.add_argument("--max-response-tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.3)
    ap.add_argument(
        "--learn", action="store_true", default=False,
        help="enable online learning (M3 grad step per turn)",
    )
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    print(f"  loading corpus from {args.corpus}...")
    raw = args.corpus.read_text(encoding="utf-8")
    stories = [
        s for s in raw.split("<|endoftext|>")
        if len(s.strip()) > 30
    ]
    rng = _random.Random(args.seed)
    rng.shuffle(stories)
    train_stories = stories[:args.n_stories]
    reserved = [c[0] for c in RESERVED_CONCEPTS]
    stoi, itos = build_vocab_with_reserved(
        train_stories, vocab_cap=args.vocab_cap,
        reserved_tokens=reserved,
    )
    train_ids = _concatenate_ids(
        [encode_story(s, stoi) for s in train_stories]
    )
    vocab = len(itos)
    print(f"  vocab: {vocab} tokens")

    print(f"  loading LM from {args.lm_checkpoint}...")
    torch.manual_seed(args.seed)
    lm = HybridPCMMiniLM(
        vocab=vocab, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
        attn_every=args.attn_every,
    )
    lm.load_state_dict(
        torch.load(args.lm_checkpoint, map_location="cpu")
    )
    lm.to(device)

    teacher: OnlineTeacherSession | None = None
    if args.learn:
        print("  populating replay buffer for online learning...")
        replay = PretrainReplayBuffer(
            capacity=64 * 32, device=device,
        )
        g = torch.Generator(device="cpu").manual_seed(2026)
        for _ in range(32):
            xs, ys = _sample_seq_batch(
                train_ids, seq_len=args.seq_len,
                batch_size=64, rng=g,
            )
            replay.add_batch(xs.to(device), ys.to(device))
        teacher = OnlineTeacherSession(
            model=lm, novel_concept_ids=[],
            novel_concept_classes={}, replay=replay,
            pad_id=_PAD,
            m3_k_grad_threshold=1,
            m3_salience_threshold=0.0,
            m3_n_replay=4, m3_lr=2e-4, m3_inner_steps=1,
            m1_buffer_capacity=128, device=device,
        )
        print("  online learning: ON")

    sess = ChatSession(
        lm, stoi, itos,
        unk_id=_UNK, pad_id=_PAD, eos_id=_EOS,
        teacher=teacher, device=device,
        max_response_tokens=args.max_response_tokens,
        temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    print(
        "\n  PCM v10.6 chat ready. Type /help for commands, "
        "/quit to exit.\n"
    )
    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  goodbye.")
            break
        if not user:
            continue
        if user.startswith("/"):
            cmd = user[1:].lower().split()
            if cmd[0] in ("quit", "exit", "q"):
                print("  goodbye.")
                break
            if cmd[0] in ("help", "h"):
                print(
                    "  commands: /stats /facts /history N "
                    "/learn on|off /quit",
                )
                continue
            if cmd[0] == "stats":
                for k, v in sess.stats().items():
                    print(f"    {k}: {v}")
                continue
            if cmd[0] == "facts":
                d = sess.user_memory.as_dict()
                if d["n_facts"] == 0:
                    print("    no facts yet.")
                else:
                    for pred, values in (
                        d["facts_by_predicate"].items()
                    ):
                        print(f"    {pred}: {values}")
                continue
            if cmd[0] == "history":
                n = (
                    int(cmd[1]) if len(cmd) > 1
                    and cmd[1].isdigit() else 6
                )
                for t in sess.turns[-n:]:
                    role_str = (
                        "you" if t.role == "user" else "bot"
                    )
                    print(f"    [{t.t:3d}] {role_str} > {t.text}")
                continue
            if cmd[0] == "learn":
                if len(cmd) > 1 and cmd[1] == "off":
                    sess.teacher = None
                    print("    online learning: OFF")
                elif len(cmd) > 1 and cmd[1] == "on" and teacher:
                    sess.teacher = teacher
                    print("    online learning: ON")
                else:
                    print(
                        f"    learn = "
                        f"{'ON' if sess.teacher else 'OFF'}"
                    )
                continue
            print("  unknown command. /help for help.")
            continue

        agent = sess.respond_to(user)
        n_oov = sess.turns[-2].n_oov
        n_user_tokens = len(sess.turns[-2].ids)
        oov_str = (
            f" ({n_oov}/{n_user_tokens} OOV)"
            if n_oov > 0 else ""
        )
        print(f"bot > {agent}{oov_str}")
        new_facts = sess.turns[-2].new_user_facts
        if new_facts:
            for f in new_facts:
                print(f"      (learned: {f.predicate}={f.value})")


if __name__ == "__main__":
    _cli()
