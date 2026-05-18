"""pcm.lm_synthetic — F74 synthetic language with ground-truth
semantics.

Built specifically to test *word-meaning understanding* (not
just probability matching). Every word has a **known** semantic
class; sentences are generated under **selectional restrictions**
(verbs require semantically-compatible arguments), so:

* We can probe whether the model recovers semantic class from
  its embeddings (U2 linear-probe test).
* We can construct semantically-violating sentences ("alice
  eats car") that a *probability-matching* model might still
  assign reasonable likelihood, but a *meaning-understanding*
  model assigns low likelihood (U3 violation test).
* We can hold out specific ``(verb, object)`` combinations and
  test compositional generalisation to unseen valid combos
  (U4 compositional test).
* We can train at increasing data scales and measure both
  perplexity *and* semantic probe accuracy as the training set
  grows (U5 sample-efficiency curve).

Vocabulary: ~150 content words + 10 closed-class. Generated
sentences are short (typically 3-8 tokens). All grammar
constraints are explicit and verifiable from this file.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


__all__ = [
    "NOUNS_BY_CLASS",
    "VERBS_BY_CLASS",
    "ADJECTIVES",
    "CLOSED_CLASS",
    "PRONOUNS",
    "PRONOUN_CLASSES",
    "MALE_PERSONS",
    "FEMALE_PERSONS",
    "VOCAB_ORDER",
    "PAD_ID",
    "UNK_ID",
    "EOS_ID",
    "noun_class_of",
    "verb_class_of",
    "person_gender_of",
    "pronoun_compatible_classes",
    "Tokenizer",
    "generate_sentence",
    "generate_corpus",
    "generate_violation_pair",
    "generate_coreference_discourse",
    "generate_multi_sentence_discourse",
    "all_valid_verb_object_pairs",
    "_SELECTIONAL",
    # F85
    "RESERVED_CONCEPTS",
    "generate_concept_teaching_sentence",
    "generate_concept_test_sentence",
    "generate_concept_dataset",
]


# ─────────────────────────────────────────────────────────────────
# Vocabulary with ground-truth semantic classes
# ─────────────────────────────────────────────────────────────────


NOUNS_BY_CLASS = {
    "ANIMAL": [
        "cat", "dog", "bird", "fish", "horse", "cow", "pig",
        "sheep", "rabbit", "mouse", "frog", "snake", "bee",
        "ant", "duck",
    ],
    "FOOD": [
        "bread", "apple", "milk", "water", "rice", "soup", "cake",
        "egg", "meat", "cheese", "salt", "sugar", "tea", "juice",
        "honey",
    ],
    "PERSON": [
        "alice", "bob", "carol", "dave", "eve", "frank", "grace",
        "henry", "iris", "jack", "kate", "leo", "mary", "nick",
        "olivia",
    ],
    "PLACE": [
        "home", "park", "school", "store", "river", "forest",
        "beach", "garden", "hill", "road", "kitchen", "room",
        "field", "pond", "yard",
    ],
}


# Gender subdivision for pronoun resolution (F76).
# Default convention: alice, carol, eve, grace, iris, kate, mary,
# olivia are FEMALE; bob, dave, frank, henry, jack, leo, nick are
# MALE. (Picked alphabetically to keep distinct from class itself.)
FEMALE_PERSONS = (
    "alice", "carol", "eve", "grace", "iris", "kate", "mary", "olivia",
)
MALE_PERSONS = ("bob", "dave", "frank", "henry", "jack", "leo", "nick")

VERBS_BY_CLASS = {
    # intransitive motion (subj only) or motion-to (subj + PLACE)
    "MOTION": ["walks", "runs", "jumps", "swims", "flies", "climbs"],
    # transitive, FOOD object, ANIMAL/PERSON subject
    "CONSUMPTION": ["eats", "drinks", "tastes", "swallows", "chews", "sips"],
    # transitive, ANIMAL/PERSON subject, any object
    "PERCEPTION": ["sees", "hears", "smells", "watches", "finds", "notices"],
    # transitive, PERSON subject, ANIMAL/FOOD/PERSON object
    "POSSESSION": ["has", "gets", "gives", "takes", "holds", "keeps"],
    # transitive, PERSON-PERSON
    "COMMUNICATION": ["says", "asks", "tells", "calls", "greets", "answers"],
}

ADJECTIVES = [
    "big", "small", "happy", "sad", "fast", "slow", "hot", "cold",
    "red", "blue", "old", "new", "soft", "hard", "kind", "smart",
    "rich", "poor", "tall", "short",
]

CLOSED_CLASS = [
    "the", "a", "and", "with", "to", "at", "in", "on", "is", ".",
]


# F85 — eight fictional concepts for the online teacher loop.
#
# These strings never occur in TinyStories (verified by string
# search over the validation corpus). They are added to the
# vocabulary at *build* time as the last 8 IDs and are *never*
# seen during pretraining. After pretraining they have random
# initial embeddings — the online teacher session then *teaches*
# the model what they mean by providing corrected examples.
#
# Tuple form: ``(token, class, ground_truth_animacy)``.
# Class is the F74-style selectional class; animacy distinguishes
# selectional restrictions like "ANIMAL can SLEEP, FOOD cannot".
RESERVED_CONCEPTS = (
    # Four animal-like — selects same as F74 ANIMAL class
    ("zorgon",  "ANIMAL", "animate"),
    ("floob",   "ANIMAL", "animate"),
    ("snerflo", "ANIMAL", "animate"),
    ("vooz",    "ANIMAL", "animate"),
    # Four food-like — selects same as F74 FOOD class
    ("glimber", "FOOD",   "inanimate"),
    ("quarp",   "FOOD",   "inanimate"),
    ("mibble",  "FOOD",   "inanimate"),
    ("prag",    "FOOD",   "inanimate"),
)

# Templates for teaching sentences (used in online interaction).
# These deliberately use *common* TinyStories verbs/objects so the
# model can leverage its pretrained selectional restrictions.
_CONCEPT_TEACHING_TEMPLATES = {
    "ANIMAL": (
        "the {C} {V_M} .",                       # zorgon ran .
        "a {C} {V_M} in the park .",
        "the {C} {V_P} the {N_F} .",             # zorgon ate the apple
        "the small {C} {V_M} .",
        "a {C} {V_P} the {N_F} in the box .",
    ),
    "FOOD": (
        "the {N_P} {V_P} the {C} .",             # the boy ate the glimber
        "the {N_P} {V_P} a {C} .",
        "a {C} is on the table .",
        "the {C} is in the {N_PL} .",
        "the {N_P} {V_P} the small {C} .",
    ),
}

# Templates for HELD-OUT test sentences. Use *different* verbs,
# adjectives, and surroundings than the teaching templates so a
# pass on the test set requires *selectional* / *compositional*
# generalisation, not memorisation.
_CONCEPT_TEST_TEMPLATES = {
    "ANIMAL": (
        "yesterday the {C} {V_M_TEST} .",        # zorgon walked
        "the {C} {V_P_TEST} the {N_F_TEST} .",   # zorgon found a flower
        "the happy {C} {V_M_TEST} in the {N_PL_TEST} .",
        "a {C} and a {N_P_TEST} {V_M_TEST} .",
    ),
    "FOOD": (
        "the {N_P_TEST} {V_P_TEST} the {C} .",
        "yesterday the {N_P_TEST} {V_P_TEST} a {C} .",
        "a small {C} is in the {N_PL_TEST} .",
        "the {N_P_TEST} found a {C} in the {N_PL_TEST} .",
    ),
}

# Sub-vocabulary for filling templates. We deliberately use
# different "training" vs "test" words to force generalisation.
_TEACH_FILL = {
    "V_M": ("ran", "jumped", "slept"),                 # MOTION
    "V_P": ("ate", "saw", "liked"),                    # POSSESSION/PERCEPTION
    "N_F": ("apple", "ball", "bowl"),                  # surrounding noun
    "N_P": ("boy", "girl", "child"),                   # PERSON subject
    "N_PL": ("box", "garden", "park"),                 # PLACE
}
_TEST_FILL = {
    "V_M_TEST": ("walked", "wandered", "rested"),
    "V_P_TEST": ("found", "wanted", "hugged"),
    "N_F_TEST": ("flower", "stone", "ribbon"),
    "N_P_TEST": ("mom", "lady", "uncle"),
    "N_PL_TEST": ("kitchen", "yard", "store"),
}


# F76 — pronouns. Each pronoun is compatible with a fixed set of
# noun classes (or genders within PERSON).
PRONOUNS = ("he", "she", "it", "they")
PRONOUN_CLASSES = {
    # "he" → male PERSON
    "he": {"classes": ("PERSON",), "gender": "M"},
    # "she" → female PERSON
    "she": {"classes": ("PERSON",), "gender": "F"},
    # "it" → non-person (ANIMAL/FOOD/PLACE)
    "it": {"classes": ("ANIMAL", "FOOD", "PLACE"), "gender": None},
    # "they" → coordinated entities (≥ 2, any class)
    "they": {"classes": ("ANIMAL", "PERSON", "FOOD", "PLACE"),
             "gender": None, "is_plural": True},
}

# Selectional restrictions per verb class
_SELECTIONAL = {
    "MOTION": {
        "subject_classes": ("ANIMAL", "PERSON"),
        "object_classes": None,
        "to_place_classes": ("PLACE",),
    },
    "CONSUMPTION": {
        "subject_classes": ("ANIMAL", "PERSON"),
        "object_classes": ("FOOD",),
        "to_place_classes": None,
    },
    "PERCEPTION": {
        "subject_classes": ("ANIMAL", "PERSON"),
        "object_classes": ("ANIMAL", "PERSON", "FOOD", "PLACE"),
        "to_place_classes": None,
    },
    "POSSESSION": {
        "subject_classes": ("PERSON",),
        "object_classes": ("ANIMAL", "FOOD", "PERSON"),
        "to_place_classes": None,
    },
    "COMMUNICATION": {
        "subject_classes": ("PERSON",),
        "object_classes": ("PERSON",),
        "to_place_classes": None,
    },
}


def noun_class_of(word: str) -> str | None:
    for cls, words in NOUNS_BY_CLASS.items():
        if word in words:
            return cls
    return None


def verb_class_of(word: str) -> str | None:
    for cls, words in VERBS_BY_CLASS.items():
        if word in words:
            return cls
    return None


def person_gender_of(word: str) -> str | None:
    """Return 'M' / 'F' / None depending on whether the word is
    a male / female PERSON or not a PERSON at all."""
    if word in FEMALE_PERSONS:
        return "F"
    if word in MALE_PERSONS:
        return "M"
    return None


def pronoun_compatible_classes(pronoun: str) -> tuple[str, ...]:
    """Return the noun classes a pronoun may refer to."""
    if pronoun not in PRONOUN_CLASSES:
        raise ValueError(f"unknown pronoun {pronoun!r}")
    return PRONOUN_CLASSES[pronoun]["classes"]


def is_compatible_referent(pronoun: str, noun: str) -> bool:
    """Return True iff ``noun`` could be the referent of
    ``pronoun`` under PCM's class + gender rules."""
    if pronoun not in PRONOUN_CLASSES:
        return False
    spec = PRONOUN_CLASSES[pronoun]
    noun_cls = noun_class_of(noun)
    if noun_cls is None:
        return False
    if noun_cls not in spec["classes"]:
        return False
    if spec.get("gender") is not None:
        if person_gender_of(noun) != spec["gender"]:
            return False
    return True


# ─────────────────────────────────────────────────────────────────
# Tokenizer (single-word tokens; tiny fixed vocab)
# ─────────────────────────────────────────────────────────────────


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
EOS_TOKEN = "<eos>"


def _build_vocab() -> list[str]:
    """Deterministic vocab ordering: specials first, then nouns
    (by class, by intra-class order), verbs, adjectives,
    closed-class, pronouns."""
    order: list[str] = [PAD_TOKEN, UNK_TOKEN, EOS_TOKEN]
    for cls in ("ANIMAL", "FOOD", "PERSON", "PLACE"):
        order.extend(NOUNS_BY_CLASS[cls])
    for cls in ("MOTION", "CONSUMPTION", "PERCEPTION", "POSSESSION",
                "COMMUNICATION"):
        order.extend(VERBS_BY_CLASS[cls])
    order.extend(ADJECTIVES)
    order.extend(CLOSED_CLASS)
    order.extend(PRONOUNS)
    seen = set()
    deduped = []
    for tok in order:
        if tok not in seen:
            deduped.append(tok)
            seen.add(tok)
    return deduped


VOCAB_ORDER = _build_vocab()
PAD_ID = VOCAB_ORDER.index(PAD_TOKEN)
UNK_ID = VOCAB_ORDER.index(UNK_TOKEN)
EOS_ID = VOCAB_ORDER.index(EOS_TOKEN)


class Tokenizer:
    """Trivial word-level tokenizer over the F74 fixed vocab."""

    def __init__(self) -> None:
        self.vocab = VOCAB_ORDER
        self.vocab_size = len(VOCAB_ORDER)
        self.stoi = {tok: i for i, tok in enumerate(self.vocab)}
        self.itos = list(self.vocab)
        self.pad_id = PAD_ID
        self.unk_id = UNK_ID
        self.eos_id = EOS_ID

    def encode(self, sentence: str | list[str], *,
               max_len: int | None = None,
               add_eos: bool = True) -> list[int]:
        if isinstance(sentence, str):
            tokens = sentence.strip().split()
        else:
            tokens = list(sentence)
        ids = [self.stoi.get(t, self.unk_id) for t in tokens]
        if add_eos:
            ids.append(self.eos_id)
        if max_len is not None:
            ids = ids[:max_len]
            ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i == self.pad_id:
                break
            tok = self.itos[i]
            if tok == EOS_TOKEN:
                break
            out.append(tok)
        return " ".join(out)


# ─────────────────────────────────────────────────────────────────
# Sentence generation
# ─────────────────────────────────────────────────────────────────


def _pick_noun(classes: tuple[str, ...], rng: random.Random) -> str:
    cls = rng.choice(classes)
    return rng.choice(NOUNS_BY_CLASS[cls])


def generate_sentence(
    rng: random.Random,
    *, template: str | None = None,
    with_det: float = 0.5,
    with_adj: float = 0.15,
) -> list[str]:
    """Generate one valid sentence respecting selectional
    restrictions.

    Templates (chosen uniformly if ``template`` is None):
    - ``"SV"``        — subject + intransitive motion
    - ``"SV_to_P"``   — subject + motion + to + PLACE
    - ``"SVO"``       — subject + transitive + object
    - ``"S_is_ADJ"``  — subject + is + adjective
    - ``"SVO_at_P"``  — subject + perception + object + at + PLACE
    - ``"S_and_S_V"`` — two-subject conjunction + intransitive
    """
    templates = ("SV", "SV_to_P", "SVO", "S_is_ADJ", "SVO_at_P", "S_and_S_V")
    if template is None:
        template = rng.choice(templates)

    def maybe_det(word: str, cls: str) -> list[str]:
        # PERSON names don't take 'the' / 'a' (proper nouns).
        if cls == "PERSON":
            return [word]
        if rng.random() < with_det:
            article = rng.choice(("the", "a"))
            return [article, word]
        return [word]

    if template == "SV":
        verb = rng.choice(VERBS_BY_CLASS["MOTION"])
        subj_classes = _SELECTIONAL["MOTION"]["subject_classes"]
        subj_cls = rng.choice(subj_classes)
        subj = rng.choice(NOUNS_BY_CLASS[subj_cls])
        tokens = maybe_det(subj, subj_cls) + [verb, "."]
        if rng.random() < with_adj:
            tokens = [rng.choice(ADJECTIVES)] + tokens
        return tokens
    if template == "SV_to_P":
        verb = rng.choice(VERBS_BY_CLASS["MOTION"])
        subj_classes = _SELECTIONAL["MOTION"]["subject_classes"]
        subj_cls = rng.choice(subj_classes)
        subj = rng.choice(NOUNS_BY_CLASS[subj_cls])
        place = rng.choice(NOUNS_BY_CLASS["PLACE"])
        tokens = (maybe_det(subj, subj_cls) + [verb, "to"]
                  + maybe_det(place, "PLACE") + ["."])
        return tokens
    if template == "S_is_ADJ":
        cls = rng.choice(("ANIMAL", "FOOD", "PERSON", "PLACE"))
        subj = rng.choice(NOUNS_BY_CLASS[cls])
        adj = rng.choice(ADJECTIVES)
        return maybe_det(subj, cls) + ["is", adj, "."]
    if template == "SVO":
        verb_cls = rng.choice(
            ("CONSUMPTION", "PERCEPTION", "POSSESSION", "COMMUNICATION")
        )
        sel = _SELECTIONAL[verb_cls]
        subj_cls = rng.choice(sel["subject_classes"])
        obj_cls = rng.choice(sel["object_classes"])
        verb = rng.choice(VERBS_BY_CLASS[verb_cls])
        subj = rng.choice(NOUNS_BY_CLASS[subj_cls])
        obj = rng.choice(NOUNS_BY_CLASS[obj_cls])
        return (maybe_det(subj, subj_cls) + [verb]
                + maybe_det(obj, obj_cls) + ["."])
    if template == "SVO_at_P":
        # Perception only; subject ∈ {ANIMAL, PERSON}, object any
        sel = _SELECTIONAL["PERCEPTION"]
        subj_cls = rng.choice(sel["subject_classes"])
        obj_cls = rng.choice(sel["object_classes"])
        verb = rng.choice(VERBS_BY_CLASS["PERCEPTION"])
        subj = rng.choice(NOUNS_BY_CLASS[subj_cls])
        obj = rng.choice(NOUNS_BY_CLASS[obj_cls])
        place = rng.choice(NOUNS_BY_CLASS["PLACE"])
        return (maybe_det(subj, subj_cls) + [verb]
                + maybe_det(obj, obj_cls)
                + ["at"] + maybe_det(place, "PLACE") + ["."])
    if template == "S_and_S_V":
        verb = rng.choice(VERBS_BY_CLASS["MOTION"])
        cls1, cls2 = rng.sample(("ANIMAL", "PERSON"), 2) \
            if rng.random() < 0.5 else (
                rng.choice(("ANIMAL", "PERSON")),
                rng.choice(("ANIMAL", "PERSON")),
            )
        n1 = rng.choice(NOUNS_BY_CLASS[cls1])
        n2 = rng.choice(NOUNS_BY_CLASS[cls2])
        return (maybe_det(n1, cls1) + ["and"]
                + maybe_det(n2, cls2) + [verb, "."])
    raise ValueError(f"Unknown template {template}")


def generate_corpus(
    n_sentences: int, *, seed: int = 0,
    holdout_verb_object_pairs: set[tuple[str, str]] | None = None,
) -> list[list[str]]:
    """Generate ``n_sentences`` valid sentences.

    If ``holdout_verb_object_pairs`` is provided, any sentence
    whose ``(verb, object)`` pair is in the set is **rejected**
    (so the model never sees those combos during training).
    The set is matched lemmatically: ``(verb, object_noun)``.
    """
    rng = random.Random(seed)
    out: list[list[str]] = []
    while len(out) < n_sentences:
        tokens = generate_sentence(rng)
        if holdout_verb_object_pairs is not None:
            v, o = _extract_verb_object(tokens)
            if v is not None and o is not None and (v, o) in holdout_verb_object_pairs:
                continue
        out.append(tokens)
    return out


def _extract_verb_object(tokens: list[str]) -> tuple[str | None, str | None]:
    """Find the first transitive (verb, object_noun) pair in
    ``tokens`` (skipping articles / adjectives between them).
    Returns ``(None, None)`` for intransitive / ``is ADJ`` /
    motion-to-place patterns."""
    v = None
    for t in tokens:
        if verb_class_of(t) in (
            "CONSUMPTION", "PERCEPTION", "POSSESSION", "COMMUNICATION"
        ):
            v = t
            break
    if v is None:
        return None, None
    # Object: first noun after the verb
    seen_verb = False
    for t in tokens:
        if not seen_verb:
            if t == v:
                seen_verb = True
            continue
        if noun_class_of(t) is not None:
            return v, t
    return v, None


def all_valid_verb_object_pairs() -> list[tuple[str, str]]:
    """Enumerate every legal (verb, object_noun) pair under the
    selectional restrictions. Used to construct the U4 held-out
    test set deterministically."""
    pairs = []
    for verb_cls, sel in _SELECTIONAL.items():
        if sel["object_classes"] is None:
            continue
        for verb in VERBS_BY_CLASS[verb_cls]:
            for noun_cls in sel["object_classes"]:
                for noun in NOUNS_BY_CLASS[noun_cls]:
                    pairs.append((verb, noun))
    return pairs


@dataclass
class CoreferenceDiscourse:
    """Generated 2-sentence discourse with ground-truth pronoun
    resolution annotation.

    * ``tokens``: full token sequence (S1 + S2, no separator —
      both end with "." which acts as a sentence boundary).
    * ``s1_subject``: first sentence's subject noun (string).
    * ``s1_object``: first sentence's object noun (string,
      or None if intransitive).
    * ``pronoun``: pronoun used in S2.
    * ``pronoun_position``: index of the pronoun token in
      ``tokens``.
    * ``referent``: ground-truth referent (one of ``s1_subject``
      or ``s1_object``).
    * ``candidates``: list of (noun, position) tuples — all
      class-compatible candidates the resolver should consider.
    """

    tokens: list[str]
    s1_subject: str
    s1_object: str | None
    pronoun: str
    pronoun_position: int
    referent: str
    candidates: list[tuple[str, int]] = field(default_factory=list)


def generate_coreference_discourse(
    rng: random.Random, *,
    rule: str = "subject_continuity",
    max_attempts: int = 32,
) -> CoreferenceDiscourse:
    """Generate a 2-sentence discourse where sentence 2 contains
    a pronoun whose ground-truth referent is fixed by the
    chosen ``rule``.

    Rules:
    * ``"subject_continuity"`` — pronoun = S1's subject (default).
      "alice eats bread . she is happy ." → she = alice.
    * ``"object_continuity"`` — pronoun = S1's object (when valid).
      "bob gives alice a flower . she smiles ." → she = alice.
    * ``"random_compatible"`` — pronoun = randomly chosen
      compatible candidate (for benchmark hard cases).

    The pronoun is chosen so that:
    1. Its class/gender is compatible with the chosen referent.
    2. At least one *competing* candidate of the same class
       exists in S1, so the resolution isn't trivial.
    """
    for _ in range(max_attempts):
        # Generate S1 via SVO template; ensure subject and object
        # are both PERSON so we have a non-trivial coreference task
        # by default. (Other patterns can be added later.)
        verb_cls = rng.choice(("POSSESSION", "COMMUNICATION", "PERCEPTION"))
        sel = _SELECTIONAL[verb_cls]
        subj_cls = rng.choice(sel["subject_classes"])
        obj_cls = rng.choice(sel["object_classes"])
        # Force both subj and obj to be PERSON so pronouns
        # have multiple compatible candidates
        if subj_cls != "PERSON" or obj_cls != "PERSON":
            continue
        verb = rng.choice(VERBS_BY_CLASS[verb_cls])
        subj = rng.choice(NOUNS_BY_CLASS["PERSON"])
        obj_pool = [n for n in NOUNS_BY_CLASS["PERSON"] if n != subj]
        obj = rng.choice(obj_pool)
        # S1 tokens
        s1 = [subj, verb, obj, "."]
        # Choose referent based on rule
        if rule == "subject_continuity":
            referent = subj
        elif rule == "object_continuity":
            referent = obj
        elif rule == "random_compatible":
            referent = rng.choice([subj, obj])
        else:
            raise ValueError(f"unknown rule {rule!r}")
        # Choose pronoun consistent with referent's gender
        ref_gender = person_gender_of(referent)
        if ref_gender is None:
            continue
        pronoun = "she" if ref_gender == "F" else "he"
        # S2: pronoun + is + adjective (simple continuation)
        adj = rng.choice(ADJECTIVES)
        s2 = [pronoun, "is", adj, "."]
        tokens = s1 + s2
        pronoun_pos = len(s1)  # first token of s2
        # Compatible candidates from S1: nouns matching pronoun
        # class+gender
        candidates: list[tuple[str, int]] = []
        for i, t in enumerate(s1):
            if is_compatible_referent(pronoun, t):
                candidates.append((t, i))
        if len(candidates) < 2:
            # Need a non-trivial choice; resample.
            continue
        return CoreferenceDiscourse(
            tokens=tokens,
            s1_subject=subj, s1_object=obj,
            pronoun=pronoun, pronoun_position=pronoun_pos,
            referent=referent, candidates=candidates,
        )
    raise RuntimeError("Could not construct coreference discourse")


def generate_multi_sentence_discourse(
    rng: random.Random, *,
    n_sentences: int = 3,
    rule: str = "subject_continuity_first",
    max_attempts: int = 64,
) -> CoreferenceDiscourse:
    """Generate an ``n_sentences``-long discourse with the
    pronoun in the **last** sentence referring to an entity in
    the **first** sentence.

    Used by F77 to test long-range coreference resolution
    where the referent is many tokens away from the pronoun.

    Rules:
    * ``"subject_continuity_first"`` (default): pronoun in S_n
      refers to S_1's subject. Intermediate sentences have
      *different* subjects of the same gender (distractors).
    * ``"object_continuity_first"``: pronoun refers to S_1's
      object.

    Sentence structure:
    * S_1: ``SUBJ VERB OBJ .`` (subject + object both PERSON,
      same gender → ambiguous for class-only resolver)
    * S_2, …, S_{n-1}: ``SUBJ' VERB' OBJ' .`` (distractor SVO
      sentences with *different* people but **same gender** as
      S_1's referent — to force the resolver to track *which*
      sentence the pronoun ties back to, not just gender)
    * S_n: ``PRONOUN is ADJ .``
    """
    for _ in range(max_attempts):
        # Pick S1 with both subject and object PERSON of same gender
        verb_cls_1 = rng.choice(
            ("POSSESSION", "COMMUNICATION", "PERCEPTION")
        )
        sel = _SELECTIONAL[verb_cls_1]
        subj_cls, obj_cls = sel["subject_classes"], sel["object_classes"]
        if "PERSON" not in subj_cls or "PERSON" not in obj_cls:
            continue
        # Choose gender for the referent's pool
        gender = rng.choice(("F", "M"))
        person_pool = (FEMALE_PERSONS if gender == "F"
                       else MALE_PERSONS)
        # We need at least 2 distinct people for S1 (subj + obj).
        # Distractor sentences may reuse people from the pool
        # (as long as they're not the S1 referent) or use
        # intransitive verbs (one person only).
        if len(person_pool) < 2:
            continue
        # S1: subj + obj different people, same gender
        s1_subj, s1_obj = rng.sample(person_pool, 2)
        verb_1 = rng.choice(VERBS_BY_CLASS[verb_cls_1])
        s1 = [s1_subj, verb_1, s1_obj, "."]
        # Decide referent
        if rule == "subject_continuity_first":
            referent = s1_subj
        elif rule == "object_continuity_first":
            referent = s1_obj
        else:
            raise ValueError(f"unknown rule {rule!r}")
        # Distractor sentences (S2 .. S_{n-1}): try to use fresh
        # people first, but fall back to reuse (with non-referent
        # people) or intransitive when the pool is exhausted.
        non_referent = [p for p in person_pool if p != referent]
        fresh_pool = [p for p in non_referent
                       if p not in (s1_subj, s1_obj)]
        all_sentences = [s1]
        for _ in range(n_sentences - 2):
            if len(fresh_pool) >= 2:
                # Transitive with fresh people
                sub_cls = rng.choice(
                    ("POSSESSION", "COMMUNICATION", "PERCEPTION")
                )
                v = rng.choice(VERBS_BY_CLASS[sub_cls])
                s_subj = fresh_pool.pop(0)
                s_obj = fresh_pool.pop(0)
                all_sentences.append([s_subj, v, s_obj, "."])
            elif len(fresh_pool) == 1:
                # Use the last fresh person + intransitive MOTION
                v = rng.choice(VERBS_BY_CLASS["MOTION"])
                s_subj = fresh_pool.pop(0)
                all_sentences.append([s_subj, v, "."])
            else:
                # Pool exhausted: reuse a non-referent person
                # intransitively. This makes the discourse
                # slightly easier (repeated people) but keeps it
                # well-formed.
                if not non_referent:
                    continue
                v = rng.choice(VERBS_BY_CLASS["MOTION"])
                s_subj = rng.choice(non_referent)
                all_sentences.append([s_subj, v, "."])
        # Last sentence with pronoun
        pronoun = "she" if gender == "F" else "he"
        adj = rng.choice(ADJECTIVES)
        all_sentences.append([pronoun, "is", adj, "."])
        # Flatten and find pronoun position
        tokens: list[str] = []
        for s in all_sentences:
            tokens.extend(s)
        # Find pronoun's index (last sentence's first token)
        pronoun_position = len(tokens) - 4
        assert tokens[pronoun_position] == pronoun
        # All compatible candidates from preceding tokens
        candidates: list[tuple[str, int]] = []
        seen_nouns: set[str] = set()
        for i, t in enumerate(tokens[:pronoun_position]):
            if t in seen_nouns:
                continue
            if is_compatible_referent(pronoun, t):
                candidates.append((t, i))
                seen_nouns.add(t)
        # Need at least 2 candidates for a non-trivial task
        if len(candidates) < 2:
            continue
        return CoreferenceDiscourse(
            tokens=tokens,
            s1_subject=s1_subj, s1_object=s1_obj,
            pronoun=pronoun, pronoun_position=pronoun_position,
            referent=referent, candidates=candidates,
        )
    raise RuntimeError("Could not construct multi-sentence discourse")


def generate_violation_pair(
    rng: random.Random, *, max_attempts: int = 64,
) -> tuple[list[str], list[str]]:
    """Return a (valid, violation) sentence pair sharing all
    structure except the object class.

    The violation swaps the object to a *type-incompatible*
    noun. E.g. valid ``"alice eats bread ."`` →
    violation ``"alice eats car ."`` (but we use a class from
    our vocab, so e.g. ``"alice eats home ."``).
    """
    for _ in range(max_attempts):
        # Generate an SVO sentence specifically
        valid = generate_sentence(rng, template="SVO", with_det=0.0,
                                   with_adj=0.0)
        v, o = _extract_verb_object(valid)
        if v is None or o is None:
            continue
        v_cls = verb_class_of(v)
        sel = _SELECTIONAL[v_cls]
        if sel["object_classes"] is None:
            continue
        all_classes = ("ANIMAL", "FOOD", "PERSON", "PLACE")
        bad_classes = [
            c for c in all_classes if c not in sel["object_classes"]
        ]
        if not bad_classes:
            continue
        bad_cls = rng.choice(bad_classes)
        bad_obj = rng.choice(NOUNS_BY_CLASS[bad_cls])
        # Replace only the *object position* (first noun after
        # the verb), not every occurrence of ``o`` — the subject
        # and object may share a word (e.g. ``alice asks alice``).
        violation = list(valid)
        seen_verb = False
        obj_position = None
        for i, t in enumerate(valid):
            if not seen_verb:
                if t == v:
                    seen_verb = True
                continue
            if noun_class_of(t) is not None:
                obj_position = i
                break
        if obj_position is None:
            continue
        violation[obj_position] = bad_obj
        return valid, violation
    raise RuntimeError("Could not construct violation pair within attempts")


# ─────────────────────────────────────────────────────────────────
# F85 — fictional concept teaching/test sentence generators
# ─────────────────────────────────────────────────────────────────


def _format_concept_template(
    template: str, *, concept: str, fill: dict,
    rng: random.Random,
) -> list[str]:
    """Format one template by sampling fillers from ``fill``.

    Returns a list of word tokens (already lowercased and split).
    """
    s = template.format(C=concept, **{
        k: rng.choice(v) for k, v in fill.items()
    })
    return s.split()


def generate_concept_teaching_sentence(
    rng: random.Random, concept_token: str, concept_class: str,
) -> list[str]:
    """One teaching sentence (used during the online teacher
    loop). Uses the *teaching* template + filler pools — the
    model will be corrected on these.

    Args:
        rng: ``random.Random`` for reproducibility.
        concept_token: e.g. ``"zorgon"`` from :data:`RESERVED_CONCEPTS`.
        concept_class: ``"ANIMAL"`` or ``"FOOD"``.
    """
    if concept_class not in _CONCEPT_TEACHING_TEMPLATES:
        raise ValueError(
            f"unknown concept_class {concept_class!r}; expected "
            f"ANIMAL or FOOD"
        )
    template = rng.choice(_CONCEPT_TEACHING_TEMPLATES[concept_class])
    return _format_concept_template(
        template, concept=concept_token, fill=_TEACH_FILL, rng=rng,
    )


def generate_concept_test_sentence(
    rng: random.Random, concept_token: str, concept_class: str,
) -> list[str]:
    """One **held-out** test sentence. Uses *different* templates
    and *different* filler vocabulary than the teaching set, so
    a PASS on test PPL requires selectional / compositional
    generalisation of the new concept's class — not just
    memorisation of training surface forms.
    """
    if concept_class not in _CONCEPT_TEST_TEMPLATES:
        raise ValueError(
            f"unknown concept_class {concept_class!r}"
        )
    template = rng.choice(_CONCEPT_TEST_TEMPLATES[concept_class])
    return _format_concept_template(
        template, concept=concept_token, fill=_TEST_FILL, rng=rng,
    )


def generate_concept_dataset(
    rng: random.Random, *,
    concepts: tuple[tuple[str, str, str], ...] | None = None,
    n_teach_per_concept: int = 30,
    n_test_per_concept: int = 30,
) -> dict:
    """Build the full F85 dataset: teaching set + held-out test
    set for each fictional concept.

    Returns a dict::

        {
            "concepts": [(token, class, animacy), ...],
            "teaching": {token: [list_of_word_lists]},
            "test":     {token: [list_of_word_lists]},
        }
    """
    if concepts is None:
        concepts = RESERVED_CONCEPTS
    teach: dict[str, list[list[str]]] = {}
    test: dict[str, list[list[str]]] = {}
    for tok, cls, _animacy in concepts:
        teach[tok] = [
            generate_concept_teaching_sentence(rng, tok, cls)
            for _ in range(n_teach_per_concept)
        ]
        test[tok] = [
            generate_concept_test_sentence(rng, tok, cls)
            for _ in range(n_test_per_concept)
        ]
    return {
        "concepts": list(concepts),
        "teaching": teach,
        "test": test,
    }
