"""pcm.user_memory — F95 UserFactMemory: declarative facts the
user has stated about themselves.

A complementary layer on top of :class:`~pcm.episodic.EpisodicBuffer`:

* The episodic buffer stores *every* event as a hidden-state
  vector (sub-symbolic, time-indexed).
* :class:`UserFactMemory` stores a sparse, symbolic *fact list*
  extracted by regex templates from user text. It is the
  semantic-memory analogue: "I know Alex likes hiking" rather
  than "I remember a vector trace from t=14".

F95 uses hand-coded regex templates. F96 will replace these
with learned extractors driven by the F85 online teacher
mechanism. The data structure :class:`UserFactMemory` is
designed to be stable across that transition.

Cognitive parallels:

* **EpisodicBuffer** (F75) = hippocampal episodic memory.
* **UserFactMemory** (F95) = ventromedial-prefrontal /
  temporal semantic memory of self-relevant facts.

Both are queried per-turn by :class:`~pcm.chat.ChatSession`,
with episodic memory feeding the model's slot space (via
M1 imprint) and user-fact memory feeding the prompt
(RAG-lite injection).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


__all__ = [
    "UserFact",
    "UserFactMemory",
]


# ─────────────────────────────────────────────────────────────────
# UserFact dataclass
# ─────────────────────────────────────────────────────────────────


@dataclass
class UserFact:
    """One declarative fact about the user.

    Attributes:
        predicate: a fixed predicate vocabulary
            (``"name"``, ``"likes"``, ``"dislikes"``,
            ``"is"``, ``"has"``, ``"lives_in"``, ``"age"``).
        value: the lower-cased extracted value
            (``"alex"``, ``"hiking"``, ``"programmer"``).
        raw_text: the raw user utterance from which the fact
            was extracted (kept for debugging + re-extraction
            in F96 when extractors become learned).
        t: turn index at which the fact was first observed.
    """

    predicate: str
    value: str
    raw_text: str
    t: int


# ─────────────────────────────────────────────────────────────────
# Regex extractor patterns (F95 hand-coded; F96 will replace
# with learned)
# ─────────────────────────────────────────────────────────────────


# Order matters slightly — more specific patterns come first.
_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"\bmy\s+name\s+is\s+([a-zA-Z][a-zA-Z\-]*)\b",
            re.I,
        ),
        "name",
    ),
    (
        re.compile(
            r"\bi\s*['']?\s*m\s+([a-zA-Z][a-zA-Z\-]*)\b"
            r"(?!\s+(?:a|an|the|happy|sad|tired|hungry|sorry"
            r"|here|going|doing|fine))",
            re.I,
        ),
        "name",
    ),
    (
        re.compile(
            r"\bi\s+am\s+([a-zA-Z][a-zA-Z\-]*)\b"
            r"(?!\s+(?:a|an|the|happy|sad|tired|hungry|sorry"
            r"|here|going|doing|fine))",
            re.I,
        ),
        "name",
    ),
    (
        re.compile(
            r"\bi\s+am\s+(?:a|an)\s+"
            r"([a-zA-Z][a-zA-Z\-]*)\b",
            re.I,
        ),
        "is",
    ),
    (
        re.compile(
            r"\bi\s*['']?\s*m\s+(?:a|an)\s+"
            r"([a-zA-Z][a-zA-Z\-]*)\b",
            re.I,
        ),
        "is",
    ),
    (
        re.compile(
            r"\bi\s+(?:like|love|enjoy)\s+"
            r"([a-zA-Z][a-zA-Z\-]*)\b",
            re.I,
        ),
        "likes",
    ),
    (
        re.compile(
            r"\bi\s+(?:hate|dislike)\s+"
            r"([a-zA-Z][a-zA-Z\-]*)\b",
            re.I,
        ),
        "dislikes",
    ),
    (
        re.compile(
            r"\bi\s+don\s*['']?\s*t\s+like\s+"
            r"([a-zA-Z][a-zA-Z\-]*)\b",
            re.I,
        ),
        "dislikes",
    ),
    (
        re.compile(
            r"\bi\s+have\s+(?:a|an)?\s*"
            r"([a-zA-Z][a-zA-Z\-]*)\b",
            re.I,
        ),
        "has",
    ),
    (
        re.compile(
            r"\bi\s+live\s+in\s+([a-zA-Z][a-zA-Z\-]*)\b",
            re.I,
        ),
        "lives_in",
    ),
    (
        re.compile(
            r"\bi\s*['']?\s*m\s+(\d+)\s+years?\s+old\b",
            re.I,
        ),
        "age",
    ),
    (
        re.compile(
            r"\bi\s+am\s+(\d+)\s+years?\s+old\b",
            re.I,
        ),
        "age",
    ),
]


# Keywords that signal the user is querying about a predicate
# (used by ChatSession to look up relevant facts before
# generating).
_PREDICATE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "name": (
        "name", "called", "who", "i'm", "i am",
    ),
    "likes": (
        "like", "love", "enjoy", "favorite", "favourite",
    ),
    "dislikes": (
        "hate", "dislike", "don't like",
    ),
    "is": (
        "am", "are", "do", "job", "work", "profession",
    ),
    "has": (
        "have", "got", "own",
    ),
    "lives_in": (
        "live", "from", "where",
    ),
    "age": (
        "age", "old", "years",
    ),
}


# ─────────────────────────────────────────────────────────────────
# UserFactMemory
# ─────────────────────────────────────────────────────────────────


class UserFactMemory:
    """Append-only store of facts the user has stated.

    Storage is a simple ``list[UserFact]``. Deduplication is
    on the ``(predicate, value)`` pair — re-stating the same
    fact later in the conversation does not create a second
    entry but does *not* update the original ``t`` either.
    """

    def __init__(self) -> None:
        self.facts: list[UserFact] = []

    def __len__(self) -> int:
        return len(self.facts)

    def __iter__(self):
        return iter(self.facts)

    def ingest(
        self, text: str, *, t: int = 0,
    ) -> list[UserFact]:
        """Extract any matching facts from ``text``. Returns
        only the *newly added* facts (duplicates are silently
        deduped)."""
        new_facts: list[UserFact] = []
        seen_keys = {
            (f.predicate, f.value) for f in self.facts
        }
        for pattern, predicate in _PATTERNS:
            for m in pattern.finditer(text):
                value = m.group(1).strip().lower()
                if not value:
                    continue
                key = (predicate, value)
                if key in seen_keys:
                    continue
                fact = UserFact(
                    predicate=predicate, value=value,
                    raw_text=text, t=t,
                )
                self.facts.append(fact)
                new_facts.append(fact)
                seen_keys.add(key)
        return new_facts

    def get(self, predicate: str) -> list[UserFact]:
        """Return all facts for a given predicate (chronological
        order)."""
        return [
            f for f in self.facts if f.predicate == predicate
        ]

    def get_latest(self, predicate: str) -> UserFact | None:
        """Return the most recent fact for a predicate, or
        ``None`` if absent."""
        items = self.get(predicate)
        return items[-1] if items else None

    def get_relevant_facts(self, text: str) -> list[UserFact]:
        """Return facts whose predicate's keywords appear in
        ``text``. The keyword sets are deliberately small and
        question-shaped (``"what is my name"`` →
        ``predicate="name"``)."""
        text_lower = text.lower()
        out: list[UserFact] = []
        for predicate, keywords in _PREDICATE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                preds_seen = {f.predicate for f in out}
                if predicate in preds_seen:
                    continue
                latest = self.get_latest(predicate)
                if latest is not None:
                    out.append(latest)
        return out

    def as_dict(self) -> dict:
        """Serialise to a plain dict (for logging / JSON)."""
        return {
            "n_facts": len(self.facts),
            "facts_by_predicate": {
                pred: [f.value for f in self.get(pred)]
                for pred in _PREDICATE_KEYWORDS.keys()
                if self.get(pred)
            },
        }
