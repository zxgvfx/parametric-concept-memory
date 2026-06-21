r"""pcm.lang — swappable per-language resource packs.

Everything language-specific that the F95/F96 chat + epistemic stack
needs (claim-extraction regexes, predicate keywords, commonsense
category seeds, word-class seed lists, and natural-language pushback
templates) lives in a :class:`LanguagePack`. The *code* in
``pcm.epistemic`` / ``pcm.user_memory`` / ``pcm.knowledge`` is
language-agnostic; switching to a new language means shipping a new
pack module (e.g. ``pcm/lang/zh.py``) and registering it here — no
change to the cognitive core.

This is the operational form of the README's standing claim that
"the architecture is language-agnostic; only the corpus, tokeniser,
and glyph table need to change". Before this module that claim was
falsified by hardcoded English regexes / word lists scattered through
``pcm.epistemic`` and ``pcm.user_memory``.

A pack is *data*. The runtime *structure* that consumes a pack's
``category_seeds`` / ``attribute_words`` / ``pronoun_words`` is the
graph built by :class:`pcm.knowledge.CommonsenseGraph` — i.e. the
seeds become ``ConceptNode`` + ``ConceptEdge`` membership, not a
module-level lookup table.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


__all__ = [
    "LanguagePack",
    "get_pack",
    "register_pack",
    "default_pack",
]


@dataclass(frozen=True)
class LanguagePack:
    """All language-specific resources for one natural language.

    Fields are deliberately plain data (compiled regexes, frozensets,
    dicts, format-string templates) so a pack can be authored without
    importing any PCM runtime code.
    """

    name: str

    # ── Commonsense category seeds (→ graph nodes + is_a edges) ──────
    # Maps a category label to the member words. Consumed by
    # :class:`pcm.knowledge.CommonsenseGraph` to build the category
    # membership graph; NOT used as a runtime lookup table directly.
    category_seeds: dict[str, tuple[str, ...]]

    # ── Word-class seeds (→ graph word-class membership) ────────────
    # ``attribute_words``  — predicate adjectives that must NOT be read
    #   as an identity object ("the cat is happy" is not "cat is_a happy").
    # ``pronoun_words``    — subjects that must NOT open an identity claim
    #   ("i am tired" is not an identity ascription).
    attribute_words: frozenset[str]
    pronoun_words: frozenset[str]

    # ── Claim / fact extraction patterns ────────────────────────────
    # Single source of truth shared by ``UserFactMemory`` (F95) and the
    # epistemic ``ClaimParser`` (F96). Each entry is (compiled, predicate).
    self_report_patterns: tuple[tuple[re.Pattern, str], ...]
    # Arithmetic: each entry is (compiled, form-tag). Named groups:
    # ``lhs`` / ``op`` / ``rhs`` / ``val``.
    arith_patterns: tuple[tuple[re.Pattern, str], ...]
    # Identity ascription patterns. Named groups: ``subj`` / ``obj``.
    identity_neg_patterns: tuple[re.Pattern, ...]
    identity_pos_patterns: tuple[re.Pattern, ...]

    # ── Arithmetic word-operator → symbol normalisation ─────────────
    # e.g. {"plus": "+", "minus": "-"}. Symbolic ops (+ - * / x) are
    # language-independent and live in ``pcm.epistemic``.
    word_ops: dict[str, str]

    # ── Predicate retrieval keywords (UserFactMemory.get_relevant) ──
    predicate_keywords: dict[str, tuple[str, ...]]

    # ── Natural-language pushback / hedge templates ─────────────────
    # ``str.format``-style templates; see ``pcm.epistemic`` for the
    # field names supplied at format time.
    pushback_arithmetic: str
    pushback_contradiction_neg: str
    pushback_contradiction_pos: str
    pushback_contradiction_generic: str
    pushback_world_model: str
    hedge_prefix: str


# ─────────────────────────────────────────────────────────────────
# Pack registry
# ─────────────────────────────────────────────────────────────────


_REGISTRY: dict[str, LanguagePack] = {}
_DEFAULT_NAME = "en"


def register_pack(pack: LanguagePack) -> None:
    """Register (or replace) a language pack by its ``name``."""
    _REGISTRY[pack.name] = pack


def get_pack(name: str | None = None) -> LanguagePack:
    """Return the pack for ``name`` (defaults to the English pack)."""
    key = name or _DEFAULT_NAME
    if key not in _REGISTRY:
        raise KeyError(
            f"no language pack registered for {key!r}; "
            f"available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def default_pack() -> LanguagePack:
    """Return the default (English) language pack."""
    return get_pack(_DEFAULT_NAME)


# Importing the English pack module registers it. Kept at the bottom to
# avoid a circular import at module-definition time.
from . import en as _en  # noqa: E402,F401
