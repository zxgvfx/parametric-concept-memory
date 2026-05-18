"""Assess the current PCM language substrate (F79 TinyStories
vocab) against the cognitive concept inventory needed for
"preschool" competence (~3-5 year-old).

Reports:
1. Which core cognitive concepts ARE present (in TinyStories
   top-4096 vocab).
2. Which are MISSING entirely.
3. Frequency stats for what's present.

Result drives the F86 augmentation design.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

# ─── Core cognitive concept inventory (target = preschool) ──

CONCEPT_INVENTORY = {
    "COLOR": [
        "red", "blue", "green", "yellow", "black", "white",
        "pink", "orange", "purple", "brown", "gray", "grey",
    ],
    "SHAPE": [
        "circle", "square", "triangle", "star", "heart",
        "line", "dot", "cross", "round", "flat",
    ],
    "NUMBER": [
        "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten",
        "zero", "first", "second", "many", "few",
    ],
    "SPATIAL": [
        "up", "down", "left", "right", "in", "out",
        "behind", "front", "near", "far", "above", "below",
        "inside", "outside", "between", "next",
    ],
    "EMOTION": [
        "happy", "sad", "scared", "angry", "surprised",
        "tired", "excited", "calm", "afraid", "glad",
        "worried", "proud", "shy",
    ],
    "FAMILY": [
        "mom", "dad", "mother", "father",
        "brother", "sister", "grandma", "grandpa",
        "baby", "friend", "mommy", "daddy",
    ],
    "BODY": [
        "hand", "foot", "head", "eye", "ear", "mouth",
        "nose", "arm", "leg", "finger", "hair", "face",
    ],
    "ACTION": [
        "push", "pull", "throw", "catch", "kick", "hug",
        "draw", "build", "run", "walk", "jump", "sleep",
        "eat", "drink", "play", "give", "take", "make",
        "look", "see", "hear", "say", "tell", "ask",
    ],
    "SIZE": [
        "big", "small", "little", "large", "tiny",
        "huge", "short", "tall", "long", "wide",
    ],
    "TIME": [
        "today", "tomorrow", "yesterday", "morning",
        "night", "day", "now", "later", "soon",
        "after", "before",
    ],
    "QUANTIFIER": [
        "all", "some", "none", "every", "any", "more",
        "less", "most", "least",
    ],
    "QUESTION": [
        "what", "where", "who", "why", "when", "how",
        "which",
    ],
    "NEGATION": [
        "no", "not", "never", "nothing", "nobody",
    ],
    "PRONOUN_ADV": [
        "i", "me", "my", "mine", "you", "your", "yours",
        "he", "him", "his", "she", "her", "hers",
        "we", "us", "our", "they", "them", "their",
        "this", "that", "these", "those", "here",
        "there",
    ],
}


_WORD_RE = re.compile(r"[a-z]+|[^\w\s]")


def _tokenise(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def main() -> None:
    corpus_path = Path("outputs/f79_data/tinystories_valid.txt")
    print(f"Loading {corpus_path}...")
    text = corpus_path.read_text(encoding="utf-8")
    print("Tokenising (counting)...")
    tokens = _tokenise(text)
    counts = Counter(tokens)
    total_tokens = sum(counts.values())
    print(f"Total tokens: {total_tokens:,}")
    print(f"Unique types: {len(counts):,}")

    # Top-4096 vocab simulates what the LM actually sees
    top4096 = set([w for w, _ in counts.most_common(4096)])
    print(f"Vocab cap = 4096 → smallest freq: "
          f"{min(counts[w] for w in top4096)}")
    print()

    print("=" * 76)
    print("  PCM LANGUAGE SUBSTRATE ASSESSMENT")
    print(
        "  (TinyStories valid, vocab=4096; assessing "
        "'preschool' coverage)"
    )
    print("=" * 76)

    total_classes_complete = 0
    overall_missing: list[tuple[str, str]] = []
    for cls, words in CONCEPT_INVENTORY.items():
        present = []
        missing = []
        for w in words:
            if w in top4096:
                present.append((w, counts[w]))
            else:
                missing.append(w)
        coverage_pct = 100 * len(present) / max(len(words), 1)
        marker = "OK" if not missing else "GAP"
        print(
            f"\n  {marker:4s}  {cls:14s}  "
            f"coverage = {len(present)}/{len(words)} "
            f"({coverage_pct:5.1f}%)"
        )
        if present:
            sample = sorted(present, key=lambda x: -x[1])[:6]
            samples_s = ", ".join(
                f"{w} ({c})" for w, c in sample
            )
            print(f"          present sample: {samples_s}")
        if missing:
            print(
                f"          MISSING ({len(missing)}): "
                f"{', '.join(missing)}"
            )
            for w in missing:
                overall_missing.append((cls, w))
        if not missing:
            total_classes_complete += 1

    print()
    print("=" * 76)
    n_classes = len(CONCEPT_INVENTORY)
    n_concepts = sum(len(v) for v in CONCEPT_INVENTORY.values())
    n_missing = len(overall_missing)
    n_present = n_concepts - n_missing
    print(
        f"  Summary: {total_classes_complete}/{n_classes} "
        f"concept classes complete; "
        f"{n_present}/{n_concepts} concept items present "
        f"({100*n_present/n_concepts:.1f}%)"
    )
    print(
        f"  Total missing concepts: {n_missing}"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
