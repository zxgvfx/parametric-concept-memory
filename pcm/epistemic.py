"""pcm.epistemic — F96 epistemic agency: hypothesis-test
before accept.

The F95 chat agent is **gullible**: every user utterance
flows directly into the M3 micro-gradient and into the
fact memory. If the user says "1+1=3" or contradicts
something they said three turns ago, the model has no
mechanism to push back.

F96 inserts a *verification stage* between the user input
and the rest of the chat pipeline. The agent now:

1. **Parses** the user utterance into structured
   :class:`Claim` objects (arithmetic equations,
   ``X is Y`` ascriptions, self-reports, …).
2. **Verifies** each claim against three independent
   sources of evidence:
   * :class:`ArithmeticVerifier` — closed-form evaluation
     of the arithmetic. ``1 + 1 == 3`` is computable; no
     LM needed.
   * :class:`ConsistencyVerifier` — checks the claim
     against the :class:`BeliefStore` (explicit previous
     assertions) AND against the :class:`WorldModel`
     (pre-computed cosine similarity over a curated set
     of category words, e.g. ``"cat is car"`` →
     ``cos(emb[cat], emb[car])`` is very low → suspicious).
   * :class:`LMPlausibilityVerifier` — uses the LM itself
     as a world-model: the mean cross-entropy of the
     claim sentence is its *implausibility* signal.
3. **Decides**: ``"accept"`` (update beliefs), ``"reject"``
   (return a templated pushback as the agent response),
   or ``"uncertain"`` (let the LM produce a hedged
   response).

Hybrid pushback policy (per user choice):

* **Hard errors** (arithmetic mismatch, direct
  contradiction in the belief store) → fixed-template
  pushback. Clarity beats fluency for cases where the
  correct answer is determined.
* **Uncertain plausibility** (LM surprise above threshold,
  but no hard contradiction) → templated *prompt* fed
  through the LM, which rephrases in its TinyStories
  voice while preserving the doubt.

Cognitive parallels:

* The :class:`BeliefStore` is the agent's
  **declarative-memory** layer (Tulving's "I know that
  Paris is the capital of France") — complementary to the
  :class:`~pcm.user_memory.UserFactMemory` (which is
  *self-relevant* facts).
* :class:`WorldModel` is **prior knowledge** — what the
  agent considers obviously true / obviously false from
  pretraining alone.
* :class:`ArithmeticVerifier` is the **deductive
  reasoning** module — pure logic, no statistics.
* :class:`LMPlausibilityVerifier` is the **abductive
  reasoning** module — "would I have predicted this
  sentence?" If no, suspicion.

Together: deduction (arithmetic) + abduction (LM
plausibility) + bookkeeping (belief store) + prior
(world model) = rudimentary scientific method.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "Claim",
    "VerifyResult",
    "EpistemicOutcome",
    "EpistemicResult",
    "BeliefStore",
    "WorldModel",
    "ArithmeticVerifier",
    "ConsistencyVerifier",
    "LMPlausibilityVerifier",
    "ClaimParser",
    "EpistemicAgent",
]


# ─────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────


@dataclass
class Claim:
    """One structured claim extracted from a user utterance.

    Type is one of:

    * ``"arithmetic"`` — fields ``lhs``, ``op``, ``rhs``,
      ``claimed_value`` filled.
    * ``"identity_positive"`` — ``subject``, ``object``
      filled (e.g., ``"a cat is an animal"`` →
      ``subject="cat", object="animal"``).
    * ``"identity_negative"`` — same, but for ``X is not Y``.
    * ``"self_report"`` — ``predicate``, ``value`` filled
      (e.g., ``predicate="name", value="alex"``). Always
      accepted; never verified.
    * ``"generic"`` — fallback; not verified, only logged.
    """

    type: str
    raw_text: str
    t: int = 0
    lhs: str | None = None
    op: str | None = None
    rhs: str | None = None
    claimed_value: str | None = None
    subject: str | None = None
    object: str | None = None
    predicate: str | None = None
    value: str | None = None

    def short(self) -> str:
        """Human-readable one-liner."""
        if self.type == "arithmetic":
            return (
                f"{self.lhs} {self.op} {self.rhs} = "
                f"{self.claimed_value}"
            )
        if self.type == "identity_positive":
            return f"{self.subject} is {self.object}"
        if self.type == "identity_negative":
            return f"{self.subject} is not {self.object}"
        if self.type == "self_report":
            return f"user.{self.predicate} = {self.value}"
        return f"<{self.type}: {self.raw_text!r}>"


@dataclass
class VerifyResult:
    """Verifier output."""

    status: str            # "verified" | "contradicted" | "uncertain"
    confidence: float      # 0..1
    reason: str
    correct_value: str | None = None
    conflict_with: Claim | None = None


@dataclass
class EpistemicOutcome:
    """One claim + its verification result + the
    aggregated decision after combining all verifiers."""

    claim: Claim
    verdicts: list[VerifyResult] = field(default_factory=list)
    final_status: str = "uncertain"      # "verified" | "contradicted" | "uncertain"
    pushback_text: str | None = None      # if final_status != "verified"


@dataclass
class EpistemicResult:
    """Top-level output of :meth:`EpistemicAgent.process`."""

    action: str                            # "accept" | "pushback" | "uncertain" | "no_claim"
    outcomes: list[EpistemicOutcome]
    pushback_text: str | None = None
    hedge_prefix: str | None = None       # e.g., "hmm, i'm not sure but "
    new_beliefs: list[Claim] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────
# BeliefStore — symbolic (subject, relation, object) triples
# ─────────────────────────────────────────────────────────────────


class BeliefStore:
    """Sparse symbolic store of accepted ``Claim`` triples.

    Three triple kinds are stored:

    * Arithmetic: ``(lhs op rhs, "=", claimed_value)``.
    * Identity: ``(subject, "is", object)`` or
      ``(subject, "is_not", object)``.
    * Self-report: ``("user", predicate, value)``.

    The store is queried per-incoming-claim by
    :class:`ConsistencyVerifier` to detect contradictions
    (``X is Y`` then ``X is not Y``, or vice-versa).
    """

    def __init__(self) -> None:
        self.beliefs: list[dict] = []

    def __len__(self) -> int:
        return len(self.beliefs)

    def __iter__(self):
        return iter(self.beliefs)

    def add(self, claim: Claim) -> None:
        """Append a claim's triple. Idempotent on exact
        match."""
        triple = self._triple_for(claim)
        if triple is None:
            return
        for existing in self.beliefs:
            if (
                existing["subject"] == triple["subject"]
                and existing["relation"] == triple["relation"]
                and existing["object"] == triple["object"]
            ):
                return
        self.beliefs.append(triple)

    def _triple_for(self, claim: Claim) -> dict | None:
        if claim.type == "arithmetic":
            return {
                "subject": (
                    f"{claim.lhs}{claim.op}{claim.rhs}"
                ),
                "relation": "=",
                "object": claim.claimed_value,
                "t": claim.t,
                "polarity": True,
            }
        if claim.type == "identity_positive":
            return {
                "subject": (
                    claim.subject.lower()
                    if claim.subject else ""
                ),
                "relation": "is",
                "object": (
                    claim.object.lower()
                    if claim.object else ""
                ),
                "t": claim.t,
                "polarity": True,
            }
        if claim.type == "identity_negative":
            return {
                "subject": (
                    claim.subject.lower()
                    if claim.subject else ""
                ),
                "relation": "is_not",
                "object": (
                    claim.object.lower()
                    if claim.object else ""
                ),
                "t": claim.t,
                "polarity": False,
            }
        if claim.type == "self_report":
            return {
                "subject": "user",
                "relation": claim.predicate or "",
                "object": claim.value or "",
                "t": claim.t,
                "polarity": True,
            }
        return None

    def find_conflict(self, claim: Claim) -> Claim | None:
        """Return the *latest* prior claim that directly
        contradicts ``claim``, or ``None``. Only applicable
        to identity and arithmetic claims."""
        if claim.type == "identity_positive":
            target = (
                claim.subject.lower()
                if claim.subject else ""
            )
            obj = (
                claim.object.lower()
                if claim.object else ""
            )
            # Look for prior "X is not Y" with same X, Y.
            for b in reversed(self.beliefs):
                if (
                    b["subject"] == target
                    and b["relation"] == "is_not"
                    and b["object"] == obj
                ):
                    return Claim(
                        type="identity_negative",
                        raw_text="(from belief store)",
                        t=b["t"],
                        subject=b["subject"],
                        object=b["object"],
                    )
            return None
        if claim.type == "identity_negative":
            target = (
                claim.subject.lower()
                if claim.subject else ""
            )
            obj = (
                claim.object.lower()
                if claim.object else ""
            )
            for b in reversed(self.beliefs):
                if (
                    b["subject"] == target
                    and b["relation"] == "is"
                    and b["object"] == obj
                ):
                    return Claim(
                        type="identity_positive",
                        raw_text="(from belief store)",
                        t=b["t"],
                        subject=b["subject"],
                        object=b["object"],
                    )
            return None
        if claim.type == "arithmetic":
            key = f"{claim.lhs}{claim.op}{claim.rhs}"
            for b in reversed(self.beliefs):
                if (
                    b["subject"] == key
                    and b["relation"] == "="
                    and b["object"] != claim.claimed_value
                ):
                    return Claim(
                        type="arithmetic",
                        raw_text="(from belief store)",
                        t=b["t"],
                        lhs=claim.lhs, op=claim.op,
                        rhs=claim.rhs,
                        claimed_value=b["object"],
                    )
            return None
        return None

    def as_dict(self) -> dict:
        return {
            "n_beliefs": len(self.beliefs),
            "triples": [
                f"{b['subject']} {b['relation']} {b['object']}"
                for b in self.beliefs
            ],
        }


# ─────────────────────────────────────────────────────────────────
# WorldModel — bootstrapped commonsense from LM tok_emb
# ─────────────────────────────────────────────────────────────────


# Curated content-word categories. Words inside the same
# tuple are *intra-class* (cat-dog are both animals); words
# across tuples are *inter-class* (cat-apple are different
# categories). This is the supervised seed; the cosine
# similarities below it are computed from LM tok_emb.
_CATEGORY_SEEDS: list[tuple[str, ...]] = [
    ("cat", "dog", "bird", "rabbit", "horse", "fox",
     "bear", "frog", "duck", "fish"),
    ("apple", "bread", "cake", "soup", "fruit",
     "cookie", "candy", "milk"),
    ("ball", "doll", "book", "toy", "kite", "block"),
    ("house", "room", "garden", "park", "school",
     "kitchen", "bedroom"),
    ("car", "bus", "truck", "train", "boat", "bike"),
    ("tree", "stone", "river", "flower", "grass",
     "sun", "moon", "cloud", "rain", "snow"),
    ("boy", "girl", "mother", "father", "child",
     "baby", "friend", "sister", "brother"),
]


class WorldModel:
    """Pre-computed commonsense from LM ``tok_emb``.

    For each pair ``(word_i, word_j)`` in the curated
    :data:`_CATEGORY_SEEDS`, the cosine similarity of
    their embedding rows is pre-computed and stored. Two
    threshold functions are exposed:

    * :meth:`are_likely_distinct(a, b)` — both words known
      AND cosine below ``dissim_threshold`` AND the words
      are in *different* curated categories. Returns
      ``True`` ⇒ "X is Y" is suspicious.
    * :meth:`are_known_same_category(a, b)` — both words
      known AND in the *same* curated category. Returns
      ``True`` ⇒ "X is Y" can be considered intra-class
      (no pushback).

    For words outside the curated set the world model
    returns ``None`` (unknown), and the consistency
    verifier falls back to the belief store + LM
    plausibility only.
    """

    def __init__(
        self, lm: nn.Module | None, stoi: dict[str, int],
        *, dissim_threshold: float = 0.10,
    ) -> None:
        self.dissim_threshold = dissim_threshold
        flat: list[str] = []
        word_to_class: dict[str, int] = {}
        for cls_idx, cls in enumerate(_CATEGORY_SEEDS):
            for w in cls:
                if w in stoi:
                    word_to_class[w] = cls_idx
                    flat.append(w)
        self.words = flat
        self.word_to_class = word_to_class
        if lm is not None and self.words:
            ids = torch.tensor(
                [stoi[w] for w in self.words],
                dtype=torch.long,
            )
            with torch.no_grad():
                emb = lm.tok_emb.weight[ids].detach()
                emb_n = F.normalize(emb, dim=-1)
                self.sim_matrix = (emb_n @ emb_n.t()).cpu()
        else:
            self.sim_matrix = torch.zeros(
                len(self.words), len(self.words),
            )
        self._word_to_idx = {
            w: i for i, w in enumerate(self.words)
        }

    def cosine(
        self, a: str, b: str,
    ) -> float | None:
        a_l = a.lower().strip()
        b_l = b.lower().strip()
        if a_l not in self._word_to_idx:
            return None
        if b_l not in self._word_to_idx:
            return None
        i = self._word_to_idx[a_l]
        j = self._word_to_idx[b_l]
        return float(self.sim_matrix[i, j])

    def are_likely_distinct(
        self, a: str, b: str,
    ) -> bool | None:
        """Return ``True`` if ``a`` and ``b`` are in
        different curated classes *and* their cosine is
        below threshold. ``False`` if same class or cosine
        above threshold. ``None`` if either word is
        unknown."""
        a_l = a.lower().strip()
        b_l = b.lower().strip()
        if a_l not in self.word_to_class:
            return None
        if b_l not in self.word_to_class:
            return None
        same_class = (
            self.word_to_class[a_l]
            == self.word_to_class[b_l]
        )
        if same_class:
            return False
        sim = self.cosine(a_l, b_l)
        if sim is None:
            return None
        return sim < self.dissim_threshold

    def are_known_same_category(
        self, a: str, b: str,
    ) -> bool | None:
        a_l = a.lower().strip()
        b_l = b.lower().strip()
        if a_l not in self.word_to_class:
            return None
        if b_l not in self.word_to_class:
            return None
        return (
            self.word_to_class[a_l]
            == self.word_to_class[b_l]
        )

    def as_dict(self) -> dict:
        return {
            "n_words": len(self.words),
            "n_classes": len(_CATEGORY_SEEDS),
            "dissim_threshold": self.dissim_threshold,
        }


# ─────────────────────────────────────────────────────────────────
# ArithmeticVerifier — closed-form check
# ─────────────────────────────────────────────────────────────────


_OP_FN = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "x": lambda a, b: a * b,
    "/": lambda a, b: a / b if b != 0 else float("nan"),
    "plus": lambda a, b: a + b,
    "minus": lambda a, b: a - b,
    "times": lambda a, b: a * b,
    "divided by": lambda a, b: (
        a / b if b != 0 else float("nan")
    ),
}


def _try_float(s: str) -> float | None:
    if s is None:
        return None
    s = s.strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


class ArithmeticVerifier:
    """Closed-form arithmetic verification. Uses Python
    float arithmetic on parsed numeric literals; no eval.

    Tolerance for division: 1e-6 absolute, 1e-4 relative
    (lenient because the user can write "0.33" for 1/3).
    """

    @staticmethod
    def check(claim: Claim) -> VerifyResult:
        if claim.type != "arithmetic":
            return VerifyResult(
                status="uncertain", confidence=0.0,
                reason="not an arithmetic claim",
            )
        a = _try_float(claim.lhs)
        b = _try_float(claim.rhs)
        c = _try_float(claim.claimed_value)
        if a is None or b is None or c is None:
            return VerifyResult(
                status="uncertain", confidence=0.0,
                reason="non-numeric arguments",
            )
        op = (claim.op or "").lower().strip()
        if op not in _OP_FN:
            return VerifyResult(
                status="uncertain", confidence=0.0,
                reason=f"unknown operator {op!r}",
            )
        try:
            true_val = _OP_FN[op](a, b)
        except (ZeroDivisionError, OverflowError) as e:
            return VerifyResult(
                status="uncertain", confidence=0.0,
                reason=f"compute error: {e}",
            )
        if math.isnan(true_val) or math.isinf(true_val):
            return VerifyResult(
                status="uncertain", confidence=0.0,
                reason="non-finite result",
            )
        # Tolerance: absolute 1e-6 OR relative 1e-4
        diff = abs(true_val - c)
        rel = diff / max(abs(true_val), 1e-9)
        if diff < 1e-6 or rel < 1e-4:
            return VerifyResult(
                status="verified", confidence=1.0,
                reason=f"{a} {op} {b} = {true_val}",
            )
        # Format correct value cleanly
        if abs(true_val - round(true_val)) < 1e-9:
            correct_str = str(int(round(true_val)))
        else:
            correct_str = f"{true_val:g}"
        return VerifyResult(
            status="contradicted", confidence=1.0,
            reason=(
                f"computed {a} {op} {b} = {correct_str}, "
                f"not {c}"
            ),
            correct_value=correct_str,
        )


# ─────────────────────────────────────────────────────────────────
# ConsistencyVerifier — belief store + world model
# ─────────────────────────────────────────────────────────────────


class ConsistencyVerifier:
    """Consistency check against
    :class:`BeliefStore` (explicit prior assertions) and
    :class:`WorldModel` (commonsense from embeddings).

    Identity claims only. Self-reports and unknown types
    return ``uncertain`` (no signal)."""

    def __init__(
        self, belief_store: BeliefStore,
        world_model: WorldModel | None = None,
    ) -> None:
        self.belief_store = belief_store
        self.world_model = world_model

    def check(self, claim: Claim) -> VerifyResult:
        if claim.type not in (
            "identity_positive", "identity_negative",
        ):
            return VerifyResult(
                status="uncertain", confidence=0.0,
                reason="not an identity claim",
            )
        # 1. Explicit contradiction in belief store.
        conflict = self.belief_store.find_conflict(claim)
        if conflict is not None:
            return VerifyResult(
                status="contradicted", confidence=1.0,
                reason=(
                    f"contradicts earlier statement: "
                    f"{conflict.short()}"
                ),
                conflict_with=conflict,
            )
        # 2. World-model commonsense (positive identity
        #    only; "X is not Y" between unrelated words is
        #    trivially true commonsense-wise).
        if (
            claim.type == "identity_positive"
            and self.world_model is not None
            and claim.subject and claim.object
        ):
            distinct = self.world_model.are_likely_distinct(
                claim.subject, claim.object,
            )
            if distinct is True:
                cos = self.world_model.cosine(
                    claim.subject, claim.object,
                )
                return VerifyResult(
                    status="contradicted", confidence=0.7,
                    reason=(
                        f"world model says "
                        f"'{claim.subject}' and "
                        f"'{claim.object}' are in different "
                        f"categories (cos={cos:.3f})"
                    ),
                )
        return VerifyResult(
            status="verified", confidence=0.5,
            reason="no consistency conflict found",
        )


# ─────────────────────────────────────────────────────────────────
# LMPlausibilityVerifier — uses the LM as a world model
# ─────────────────────────────────────────────────────────────────


class LMPlausibilityVerifier:
    """Measures the mean cross-entropy of a claim
    sentence under the LM. High surprise ⇒ the claim is
    out-of-distribution for the LM's pretraining ⇒
    suspicious.

    This is intentionally a *soft* signal — it cannot
    push back on its own (we'd flag too many novel-but-
    correct claims). It only contributes to the
    ``"uncertain"`` bucket when the LM is surprised but
    no hard verifier fires."""

    def __init__(
        self, lm: nn.Module, stoi: dict[str, int],
        *,
        suspicion_threshold: float = 7.0,
        pad_id: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.lm = lm
        self.stoi = stoi
        self.suspicion_threshold = suspicion_threshold
        self.pad_id = pad_id
        self.device = device

    def _encode(self, text: str) -> torch.Tensor:
        words = text.lower().split()
        ids: list[int] = []
        for w in words:
            w = w.strip(".,!?;:\"'()[]{}")
            if not w:
                continue
            ids.append(self.stoi.get(w, 1))
        return torch.tensor(ids, dtype=torch.long)

    @torch.no_grad()
    def check(self, claim: Claim) -> VerifyResult:
        ids = self._encode(claim.raw_text)
        if ids.numel() < 3:
            return VerifyResult(
                status="uncertain", confidence=0.0,
                reason="too short for plausibility check",
            )
        seq = ids.to(self.device).unsqueeze(0)
        x = seq[:, :-1]
        y = seq[:, 1:]
        self.lm.eval()
        logits = self.lm(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            ignore_index=self.pad_id,
            reduction="mean",
        )
        surprise = float(loss.item())
        if surprise > self.suspicion_threshold:
            return VerifyResult(
                status="uncertain", confidence=0.6,
                reason=(
                    f"LM surprise {surprise:.2f} > "
                    f"threshold "
                    f"{self.suspicion_threshold:.2f}"
                ),
            )
        return VerifyResult(
            status="verified", confidence=0.5,
            reason=f"LM surprise {surprise:.2f} (plausible)",
        )


# ─────────────────────────────────────────────────────────────────
# ClaimParser — regex extraction
# ─────────────────────────────────────────────────────────────────


_ARITH_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"(?P<lhs>-?\d+(?:\.\d+)?)\s*"
            r"(?P<op>[+\-*/x])\s*"
            r"(?P<rhs>-?\d+(?:\.\d+)?)\s*"
            r"(?:=|is|equals?|equal\s+to)\s*"
            r"(?P<val>-?\d+(?:\.\d+)?)",
            re.I,
        ),
        "symbolic",
    ),
    (
        re.compile(
            r"(?P<lhs>-?\d+(?:\.\d+)?)\s+"
            r"(?P<op>plus|minus|times|divided\s+by)\s+"
            r"(?P<rhs>-?\d+(?:\.\d+)?)\s+"
            r"(?:is|equals?|equal\s+to|=)\s+"
            r"(?P<val>-?\d+(?:\.\d+)?)",
            re.I,
        ),
        "word",
    ),
]


_IDENT_NEG_PATTERN = re.compile(
    r"\ba\s+(?P<subj>[a-zA-Z][a-zA-Z\-]*)"
    r"\s+is\s+not\s+(?:a|an)\s+"
    r"(?P<obj>[a-zA-Z][a-zA-Z\-]*)\b",
    re.I,
)
_IDENT_NEG_THE_PATTERN = re.compile(
    r"\b(?:the\s+)?(?P<subj>[a-zA-Z][a-zA-Z\-]*)"
    r"\s+(?:is|are)\s+not\s+(?:a|an)?\s*"
    r"(?P<obj>[a-zA-Z][a-zA-Z\-]*)\b",
    re.I,
)
_IDENT_POS_PATTERN = re.compile(
    r"\ba\s+(?P<subj>[a-zA-Z][a-zA-Z\-]*)"
    r"\s+is\s+(?:a|an)\s+"
    r"(?P<obj>[a-zA-Z][a-zA-Z\-]*)\b",
    re.I,
)


# Words to skip as 'subject' for identity parsing (avoid
# treating self-reports / pronoun structures as identity
# claims).
_SUBJ_BLACKLIST = {
    "i", "you", "we", "they", "he", "she", "it",
    "this", "that", "there", "here",
    "my", "your", "our", "their", "his", "her", "its",
}

# Words to skip as identity 'object' to avoid extracting
# things like "happy" as an object class (those are
# attribute claims, not identity claims).
_OBJ_ATTRIBUTE_HINTS = {
    "happy", "sad", "tired", "hungry", "good", "bad",
    "nice", "warm", "cold", "hot", "big", "small",
    "fast", "slow", "soft", "hard", "old", "young",
    "new", "wet", "dry", "sorry", "ready",
}


_SELF_REPORT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"\bmy\s+name\s+is\s+"
            r"(?P<v>[a-zA-Z][a-zA-Z\-]*)\b",
            re.I,
        ),
        "name",
    ),
    (
        re.compile(
            r"\bi\s+(?:like|love|enjoy)\s+"
            r"(?P<v>[a-zA-Z][a-zA-Z\-]*)\b",
            re.I,
        ),
        "likes",
    ),
    (
        re.compile(
            r"\bi\s+(?:hate|dislike)\s+"
            r"(?P<v>[a-zA-Z][a-zA-Z\-]*)\b",
            re.I,
        ),
        "dislikes",
    ),
    (
        re.compile(
            r"\bi\s+am\s+(?:a|an)\s+"
            r"(?P<v>[a-zA-Z][a-zA-Z\-]*)\b",
            re.I,
        ),
        "is",
    ),
    (
        re.compile(
            r"\bi\s+live\s+in\s+"
            r"(?P<v>[a-zA-Z][a-zA-Z\-]*)\b",
            re.I,
        ),
        "lives_in",
    ),
    (
        re.compile(
            r"\bi\s+am\s+(?P<v>\d+)\s+years?\s+old\b",
            re.I,
        ),
        "age",
    ),
]


class ClaimParser:
    """Regex parser producing :class:`Claim` instances."""

    def parse(
        self, text: str, *, t: int = 0,
    ) -> list[Claim]:
        out: list[Claim] = []
        seen_spans: list[tuple[int, int]] = []

        def _add(claim: Claim, span: tuple[int, int]) -> None:
            for s, e in seen_spans:
                if not (span[1] <= s or span[0] >= e):
                    return
            seen_spans.append(span)
            out.append(claim)

        # 1. Arithmetic — try symbolic, then word forms.
        for pattern, _ in _ARITH_PATTERNS:
            for m in pattern.finditer(text):
                op_raw = m.group("op").lower().strip()
                op_norm = (
                    op_raw if op_raw in _OP_FN
                    else op_raw
                )
                _add(
                    Claim(
                        type="arithmetic",
                        raw_text=m.group(0),
                        t=t,
                        lhs=m.group("lhs"),
                        op=op_norm,
                        rhs=m.group("rhs"),
                        claimed_value=m.group("val"),
                    ),
                    m.span(),
                )

        # 2. Self-reports.
        for pattern, predicate in _SELF_REPORT_PATTERNS:
            for m in pattern.finditer(text):
                _add(
                    Claim(
                        type="self_report",
                        raw_text=m.group(0),
                        t=t,
                        predicate=predicate,
                        value=m.group("v").lower(),
                    ),
                    m.span(),
                )

        # 3. Negative identity.
        for pattern in (
            _IDENT_NEG_PATTERN, _IDENT_NEG_THE_PATTERN,
        ):
            for m in pattern.finditer(text):
                subj = m.group("subj").lower()
                obj = m.group("obj").lower()
                if subj in _SUBJ_BLACKLIST:
                    continue
                if obj in _OBJ_ATTRIBUTE_HINTS:
                    continue
                _add(
                    Claim(
                        type="identity_negative",
                        raw_text=m.group(0),
                        t=t,
                        subject=subj,
                        object=obj,
                    ),
                    m.span(),
                )

        # 4. Positive identity ("a cat is an animal" style;
        #    avoid plain "the cat is happy" by requiring
        #    "a/an" determiner on both subject and object).
        for m in _IDENT_POS_PATTERN.finditer(text):
            subj = m.group("subj").lower()
            obj = m.group("obj").lower()
            if subj in _SUBJ_BLACKLIST:
                continue
            if obj in _OBJ_ATTRIBUTE_HINTS:
                continue
            _add(
                Claim(
                    type="identity_positive",
                    raw_text=m.group(0),
                    t=t,
                    subject=subj,
                    object=obj,
                ),
                m.span(),
            )

        return out


# ─────────────────────────────────────────────────────────────────
# Pushback templates
# ─────────────────────────────────────────────────────────────────


def _pushback_arithmetic_template(
    claim: Claim, correct_value: str,
) -> str:
    return (
        f"i don't think {claim.lhs} {claim.op} "
        f"{claim.rhs} is {claim.claimed_value} . "
        f"i think it is {correct_value} ."
    )


def _pushback_contradiction_template(
    claim: Claim, prior: Claim,
) -> str:
    if claim.type == "identity_negative":
        return (
            f"wait , earlier you said "
            f"{prior.subject} is {prior.object} . "
            f"now you say {claim.subject} is not "
            f"{claim.object} . which one is right ?"
        )
    if claim.type == "identity_positive":
        return (
            f"wait , earlier you said "
            f"{prior.subject} is not {prior.object} . "
            f"now you say {claim.subject} is "
            f"{claim.object} . which one is right ?"
        )
    return (
        f"i thought you said something different "
        f"earlier . can you tell me again ?"
    )


def _pushback_world_model_template(claim: Claim) -> str:
    return (
        f"are you sure ? in my world a {claim.subject} "
        f"is not the same kind of thing as a "
        f"{claim.object} . can you tell me more ?"
    )


# ─────────────────────────────────────────────────────────────────
# EpistemicAgent — top-level orchestrator
# ─────────────────────────────────────────────────────────────────


class EpistemicAgent:
    """Hypothesis-test orchestrator.

    Holds the parser, the belief store, optionally a world
    model (LM-driven commonsense) and an LM-plausibility
    verifier. Per :meth:`process` call, parses the user
    text into claims, dispatches each claim to the
    appropriate verifier, aggregates the verdicts, and
    returns an :class:`EpistemicResult` telling the chat
    session what to do next.
    """

    def __init__(
        self, *, lm: nn.Module | None = None,
        stoi: dict[str, int] | None = None,
        world_model: WorldModel | None = None,
        belief_store: BeliefStore | None = None,
        lm_plausibility: LMPlausibilityVerifier | None = None,
        use_world_model: bool = True,
        use_lm_plausibility: bool = True,
        pad_id: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.parser = ClaimParser()
        self.belief_store = belief_store or BeliefStore()
        if world_model is not None:
            self.world_model = world_model
        elif use_world_model and lm is not None and stoi is not None:
            self.world_model = WorldModel(lm, stoi)
        else:
            self.world_model = None
        self.arith = ArithmeticVerifier()
        self.consist = ConsistencyVerifier(
            self.belief_store, self.world_model,
        )
        if lm_plausibility is not None:
            self.plaus: LMPlausibilityVerifier | None = (
                lm_plausibility
            )
        elif (
            use_lm_plausibility and lm is not None
            and stoi is not None
        ):
            self.plaus = LMPlausibilityVerifier(
                lm, stoi, pad_id=pad_id, device=device,
            )
        else:
            self.plaus = None

    # ── Main entrypoint ─────────────────────────────────────────

    def process(
        self, user_text: str, *, t: int = 0,
    ) -> EpistemicResult:
        claims = self.parser.parse(user_text, t=t)
        if not claims:
            return EpistemicResult(
                action="no_claim", outcomes=[],
            )

        outcomes: list[EpistemicOutcome] = []
        any_pushback = False
        any_uncertain = False
        accepted: list[Claim] = []

        for claim in claims:
            outcome = self._verify_single(claim)
            outcomes.append(outcome)
            if outcome.final_status == "contradicted":
                any_pushback = True
            elif outcome.final_status == "uncertain":
                any_uncertain = True
            elif outcome.final_status == "verified":
                accepted.append(claim)
                self.belief_store.add(claim)

        # Decision rule: any contradicted → pushback (first
        # such claim's message). Else any uncertain → hedge.
        # Else accept all.
        if any_pushback:
            first_bad = next(
                o for o in outcomes
                if o.final_status == "contradicted"
            )
            return EpistemicResult(
                action="pushback",
                outcomes=outcomes,
                pushback_text=first_bad.pushback_text,
                new_beliefs=accepted,
            )
        if any_uncertain:
            return EpistemicResult(
                action="uncertain",
                outcomes=outcomes,
                hedge_prefix="hmm , i am not sure but ",
                new_beliefs=accepted,
            )
        return EpistemicResult(
            action="accept",
            outcomes=outcomes,
            new_beliefs=accepted,
        )

    # ── Single-claim dispatch ──────────────────────────────────

    def _verify_single(
        self, claim: Claim,
    ) -> EpistemicOutcome:
        verdicts: list[VerifyResult] = []
        if claim.type == "self_report":
            # Self-reports are accepted unconditionally.
            verdicts.append(VerifyResult(
                status="verified", confidence=1.0,
                reason="self-report (no verification)",
            ))
            return EpistemicOutcome(
                claim=claim, verdicts=verdicts,
                final_status="verified",
            )

        if claim.type == "arithmetic":
            r_arith = self.arith.check(claim)
            verdicts.append(r_arith)
            if r_arith.status == "contradicted":
                return EpistemicOutcome(
                    claim=claim, verdicts=verdicts,
                    final_status="contradicted",
                    pushback_text=(
                        _pushback_arithmetic_template(
                            claim, r_arith.correct_value,
                        )
                    ),
                )
            if r_arith.status == "verified":
                return EpistemicOutcome(
                    claim=claim, verdicts=verdicts,
                    final_status="verified",
                )

        if claim.type in (
            "identity_positive", "identity_negative",
        ):
            r_con = self.consist.check(claim)
            verdicts.append(r_con)
            if r_con.status == "contradicted":
                if r_con.conflict_with is not None:
                    text = _pushback_contradiction_template(
                        claim, r_con.conflict_with,
                    )
                else:
                    text = _pushback_world_model_template(
                        claim,
                    )
                return EpistemicOutcome(
                    claim=claim, verdicts=verdicts,
                    final_status="contradicted",
                    pushback_text=text,
                )
            if r_con.status == "verified":
                return EpistemicOutcome(
                    claim=claim, verdicts=verdicts,
                    final_status="verified",
                )

        # Fall through: try LM plausibility for uncertain
        # claims (or non-arithmetic / non-identity ones).
        if self.plaus is not None:
            r_plaus = self.plaus.check(claim)
            verdicts.append(r_plaus)
            if r_plaus.status == "uncertain":
                return EpistemicOutcome(
                    claim=claim, verdicts=verdicts,
                    final_status="uncertain",
                )

        return EpistemicOutcome(
            claim=claim, verdicts=verdicts,
            final_status="uncertain",
        )

    # ── Convenience ─────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "n_beliefs": len(self.belief_store),
            "world_model": (
                self.world_model.as_dict()
                if self.world_model else None
            ),
            "has_lm_plausibility": self.plaus is not None,
        }
