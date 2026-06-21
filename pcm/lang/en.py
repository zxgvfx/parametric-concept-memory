r"""pcm.lang.en — the English language pack.

All English-specific resources for the F95/F96 chat + epistemic stack.
This module is pure data: the regexes, word-class seed lists, category
seeds, operator names, retrieval keywords, and pushback templates that
were previously hardcoded inside ``pcm.epistemic`` and
``pcm.user_memory``.

To add another language, copy this file (e.g. ``zh.py``), translate the
data, and ``register_pack`` it — no change to the cognitive core.
"""
from __future__ import annotations

import re

from . import LanguagePack, register_pack


# ─────────────────────────────────────────────────────────────────
# Commonsense category seeds → CommonsenseGraph membership
# ─────────────────────────────────────────────────────────────────
# Words inside the same category are intra-class (cat/dog are both
# animals); words across categories are inter-class (cat/apple differ).
# These seed the ConceptGraph built by pcm.knowledge.CommonsenseGraph.
_CATEGORY_SEEDS: dict[str, tuple[str, ...]] = {
    "animal": (
        "cat", "dog", "bird", "rabbit", "horse", "fox",
        "bear", "frog", "duck", "fish",
    ),
    "food": (
        "apple", "bread", "cake", "soup", "fruit",
        "cookie", "candy", "milk",
    ),
    "toy": ("ball", "doll", "book", "toy", "kite", "block"),
    "place": (
        "house", "room", "garden", "park", "school",
        "kitchen", "bedroom",
    ),
    "vehicle": ("car", "bus", "truck", "train", "boat", "bike"),
    "nature": (
        "tree", "stone", "river", "flower", "grass",
        "sun", "moon", "cloud", "rain", "snow",
    ),
    "person": (
        "boy", "girl", "mother", "father", "child",
        "baby", "friend", "sister", "brother",
    ),
}


# ─────────────────────────────────────────────────────────────────
# Word-class seeds → CommonsenseGraph word-class membership
# ─────────────────────────────────────────────────────────────────
_ATTRIBUTE_WORDS: frozenset[str] = frozenset({
    "happy", "sad", "tired", "hungry", "good", "bad",
    "nice", "warm", "cold", "hot", "big", "small",
    "fast", "slow", "soft", "hard", "old", "young",
    "new", "wet", "dry", "sorry", "ready",
})

_PRONOUN_WORDS: frozenset[str] = frozenset({
    "i", "you", "we", "they", "he", "she", "it",
    "this", "that", "there", "here",
    "my", "your", "our", "their", "his", "her", "its",
})


# ─────────────────────────────────────────────────────────────────
# Self-report patterns (shared by UserFactMemory + ClaimParser)
# ─────────────────────────────────────────────────────────────────
# Order matters slightly — more specific patterns come first.
_SELF_REPORT_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
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
            r"\bi\s+am\s+(?:a|an)\s+([a-zA-Z][a-zA-Z\-]*)\b",
            re.I,
        ),
        "is",
    ),
    (
        re.compile(
            r"\bi\s*['']?\s*m\s+(?:a|an)\s+([a-zA-Z][a-zA-Z\-]*)\b",
            re.I,
        ),
        "is",
    ),
    (
        re.compile(
            r"\bi\s+(?:like|love|enjoy)\s+([a-zA-Z][a-zA-Z\-]*)\b",
            re.I,
        ),
        "likes",
    ),
    (
        re.compile(
            r"\bi\s+(?:hate|dislike)\s+([a-zA-Z][a-zA-Z\-]*)\b",
            re.I,
        ),
        "dislikes",
    ),
    (
        re.compile(
            r"\bi\s+don\s*['']?\s*t\s+like\s+([a-zA-Z][a-zA-Z\-]*)\b",
            re.I,
        ),
        "dislikes",
    ),
    (
        re.compile(
            r"\bi\s+have\s+(?:a|an)?\s*([a-zA-Z][a-zA-Z\-]*)\b",
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
)


# ─────────────────────────────────────────────────────────────────
# Arithmetic patterns
# ─────────────────────────────────────────────────────────────────
_ARITH_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
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
)


# ─────────────────────────────────────────────────────────────────
# Identity ascription patterns
# ─────────────────────────────────────────────────────────────────
_IDENT_NEG_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(
        r"\ba\s+(?P<subj>[a-zA-Z][a-zA-Z\-]*)"
        r"\s+is\s+not\s+(?:a|an)\s+"
        r"(?P<obj>[a-zA-Z][a-zA-Z\-]*)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:the\s+)?(?P<subj>[a-zA-Z][a-zA-Z\-]*)"
        r"\s+(?:is|are)\s+not\s+(?:a|an)?\s*"
        r"(?P<obj>[a-zA-Z][a-zA-Z\-]*)\b",
        re.I,
    ),
)
_IDENT_POS_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(
        r"\ba\s+(?P<subj>[a-zA-Z][a-zA-Z\-]*)"
        r"\s+is\s+(?:a|an)\s+"
        r"(?P<obj>[a-zA-Z][a-zA-Z\-]*)\b",
        re.I,
    ),
)


# ─────────────────────────────────────────────────────────────────
# Arithmetic word-operator → symbol
# ─────────────────────────────────────────────────────────────────
_WORD_OPS: dict[str, str] = {
    "plus": "+",
    "minus": "-",
    "times": "*",
    "divided by": "/",
}


# ─────────────────────────────────────────────────────────────────
# Predicate retrieval keywords
# ─────────────────────────────────────────────────────────────────
_PREDICATE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "name": ("name", "called", "who", "i'm", "i am"),
    "likes": ("like", "love", "enjoy", "favorite", "favourite"),
    "dislikes": ("hate", "dislike", "don't like"),
    "is": ("am", "are", "do", "job", "work", "profession"),
    "has": ("have", "got", "own"),
    "lives_in": ("live", "from", "where"),
    "age": ("age", "old", "years"),
}


# ─────────────────────────────────────────────────────────────────
# Pushback / hedge templates
# ─────────────────────────────────────────────────────────────────
_PUSHBACK_ARITHMETIC = (
    "i don't think {lhs} {op} {rhs} is {claimed_value} . "
    "i think it is {correct_value} ."
)
_PUSHBACK_CONTRADICTION_NEG = (
    "wait , earlier you said {prior_subject} is {prior_object} . "
    "now you say {claim_subject} is not {claim_object} . "
    "which one is right ?"
)
_PUSHBACK_CONTRADICTION_POS = (
    "wait , earlier you said {prior_subject} is not {prior_object} . "
    "now you say {claim_subject} is {claim_object} . "
    "which one is right ?"
)
_PUSHBACK_CONTRADICTION_GENERIC = (
    "i thought you said something different earlier . "
    "can you tell me again ?"
)
_PUSHBACK_WORLD_MODEL = (
    "are you sure ? in my world a {subject} is not the same kind "
    "of thing as a {object} . can you tell me more ?"
)
_HEDGE_PREFIX = "hmm , i am not sure but "


EN_PACK = LanguagePack(
    name="en",
    category_seeds=_CATEGORY_SEEDS,
    attribute_words=_ATTRIBUTE_WORDS,
    pronoun_words=_PRONOUN_WORDS,
    self_report_patterns=_SELF_REPORT_PATTERNS,
    arith_patterns=_ARITH_PATTERNS,
    identity_neg_patterns=_IDENT_NEG_PATTERNS,
    identity_pos_patterns=_IDENT_POS_PATTERNS,
    word_ops=_WORD_OPS,
    predicate_keywords=_PREDICATE_KEYWORDS,
    pushback_arithmetic=_PUSHBACK_ARITHMETIC,
    pushback_contradiction_neg=_PUSHBACK_CONTRADICTION_NEG,
    pushback_contradiction_pos=_PUSHBACK_CONTRADICTION_POS,
    pushback_contradiction_generic=_PUSHBACK_CONTRADICTION_GENERIC,
    pushback_world_model=_PUSHBACK_WORLD_MODEL,
    hedge_prefix=_HEDGE_PREFIX,
)


register_pack(EN_PACK)
