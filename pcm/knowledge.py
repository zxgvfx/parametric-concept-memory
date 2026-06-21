r"""pcm.knowledge — commonsense category + word-class knowledge as a
ConceptGraph.

F96's original ``WorldModel`` held its commonsense as a module-level
``_CATEGORY_SEEDS`` list-of-tuples plus a parallel ``word_to_class``
dict — a small knowledge base living *outside* the ConceptGraph. That
contradicted the project's "everything is a graph" thesis (D91-D94):
concept knowledge is supposed to be ``ConceptNode`` + ``ConceptEdge``
so it is introspectable, attributable, and *growable* (teachable via
the F85 online loop), not frozen in a Python literal.

:class:`CommonsenseGraph` fixes that. It builds a real
:class:`~pcm.concept_graph.ConceptGraph` from a
:class:`~pcm.lang.LanguagePack`:

* every seed word is a ``ConceptNode`` (``concept:word:<w>``),
* every category is a ``ConceptNode`` (``concept:category:<c>``),
* every word-class (attribute / pronoun) is a ``ConceptNode``
  (``concept:wordclass:<k>``),
* membership is an ``is_a`` ``ConceptEdge`` from the word to its
  category / word-class node.

Category / word-class membership queries are then *graph traversals*,
and :meth:`add_member` lets the knowledge grow at runtime — the hook a
future F-step can wire to the online teacher so commonsense is learned,
not hardcoded.

The numeric cosine signal (used as a soft guard by the world model)
stays in :class:`pcm.epistemic.WorldModel`, sourced from the LM's
``tok_emb`` — that is an LM-*learned* signal, not hardcoded knowledge.
"""
from __future__ import annotations

from .concept_graph import ConceptGraph
from .lang import LanguagePack, default_pack


__all__ = ["CommonsenseGraph"]


_WORD_PREFIX = "concept:word:"
_CATEGORY_PREFIX = "concept:category:"
_WORDCLASS_PREFIX = "concept:wordclass:"
_IS_A = "is_a"


class CommonsenseGraph:
    """Category + word-class commonsense, stored as a ConceptGraph.

    Args:
        pack: the :class:`~pcm.lang.LanguagePack` whose
            ``category_seeds`` / ``attribute_words`` / ``pronoun_words``
            seed the graph. Defaults to the English pack.
        graph: an existing :class:`ConceptGraph` to populate (a fresh
            one is created if ``None``). Sharing a graph lets the
            commonsense nodes live alongside the rest of an agent's
            concepts.
    """

    def __init__(
        self,
        pack: LanguagePack | None = None,
        *,
        graph: ConceptGraph | None = None,
    ) -> None:
        self.pack = pack or default_pack()
        self.graph = graph if graph is not None else ConceptGraph()
        self._build_from_pack()

    # ── Construction ────────────────────────────────────────────

    def _build_from_pack(self) -> None:
        for category, words in self.pack.category_seeds.items():
            cat_id = _CATEGORY_PREFIX + category
            self.graph.register_concept(
                cat_id, label=category, scope="CORE",
                provenance="commonsense_seed",
            )
            for w in words:
                self._ensure_member(w, cat_id, scope="BASE")
        # Word-class membership (attribute / pronoun) as graph edges.
        self._build_word_class("attribute", self.pack.attribute_words)
        self._build_word_class("pronoun", self.pack.pronoun_words)

    def _build_word_class(self, klass: str, words) -> None:
        class_id = _WORDCLASS_PREFIX + klass
        self.graph.register_concept(
            class_id, label=klass, scope="CORE",
            provenance="wordclass_seed",
        )
        for w in words:
            self._ensure_member(w, class_id, scope="BASE")

    def _ensure_member(
        self, word: str, parent_id: str, *, scope: str,
    ) -> None:
        word_id = _WORD_PREFIX + word.lower()
        self.graph.register_concept(
            word_id, label=word.lower(), scope=scope,
            provenance="commonsense_seed",
        )
        self.graph.add_edge(word_id, parent_id, edge_type=_IS_A)

    # ── Membership queries (graph traversals) ───────────────────

    def _parents(self, word: str, prefix: str) -> list[str]:
        """Return the labels of ``is_a`` parents of ``word`` whose node
        id starts with ``prefix``."""
        word_id = _WORD_PREFIX + word.lower().strip()
        out: list[str] = []
        for tgt_id, edge_type in self.graph._adjacency.get(word_id, []):
            if edge_type == _IS_A and tgt_id.startswith(prefix):
                node = self.graph.concepts.get(tgt_id)
                if node is not None:
                    out.append(node.label)
        return out

    def knows_word(self, word: str) -> bool:
        """True if ``word`` is a node in the commonsense graph."""
        return (_WORD_PREFIX + word.lower().strip()) in self.graph.concepts

    def category_of(self, word: str) -> str | None:
        """Return the (first) commonsense category label of ``word``,
        or ``None`` if the word is unknown / uncategorised."""
        cats = self._parents(word, _CATEGORY_PREFIX)
        return cats[0] if cats else None

    def has_category(self, word: str) -> bool:
        """True if ``word`` belongs to at least one commonsense
        category (word-class-only words return ``False``)."""
        return bool(self._parents(word, _CATEGORY_PREFIX))

    def are_same_category(self, a: str, b: str) -> bool | None:
        """``True`` / ``False`` if both words have a category and they
        (do / don't) share one; ``None`` if either is uncategorised."""
        ca = self.category_of(a)
        cb = self.category_of(b)
        if ca is None or cb is None:
            return None
        return ca == cb

    def is_attribute(self, word: str) -> bool:
        """True if ``word`` is a predicate-adjective (word-class
        ``attribute``)."""
        return "attribute" in self._parents(word, _WORDCLASS_PREFIX)

    def is_pronoun(self, word: str) -> bool:
        """True if ``word`` is a pronoun-like subject (word-class
        ``pronoun``)."""
        return "pronoun" in self._parents(word, _WORDCLASS_PREFIX)

    def categorised_words(self) -> list[str]:
        """All words that belong to at least one commonsense category."""
        return [
            node.label
            for nid, node in self.graph.concepts.items()
            if nid.startswith(_WORD_PREFIX) and self.has_category(node.label)
        ]

    # ── Growth (the hook for teachable commonsense) ─────────────

    def add_member(
        self, word: str, category: str, *, tick: int = 0,
    ) -> None:
        """Add (or reinforce) ``word`` ∈ ``category`` at runtime.

        This is the growth path the original frozen ``_CATEGORY_SEEDS``
        could not offer: a future online-teacher step can extend
        commonsense without editing source code.
        """
        cat_id = _CATEGORY_PREFIX + category
        if cat_id not in self.graph.concepts:
            self.graph.register_concept(
                cat_id, label=category, scope="CORE",
                provenance="taught", tick=tick,
            )
        word_id = _WORD_PREFIX + word.lower()
        self.graph.register_concept(
            word_id, label=word.lower(), scope="BASE",
            provenance="taught", tick=tick,
        )
        self.graph.add_edge(
            word_id, cat_id, edge_type=_IS_A, tick=tick,
        )

    def as_dict(self) -> dict:
        n_cat = sum(
            1 for nid in self.graph.concepts
            if nid.startswith(_CATEGORY_PREFIX)
        )
        n_words = sum(
            1 for nid in self.graph.concepts
            if nid.startswith(_WORD_PREFIX)
        )
        return {
            "n_category_words": len(self.categorised_words()),
            "n_categories": n_cat,
            "n_word_nodes": n_words,
        }
