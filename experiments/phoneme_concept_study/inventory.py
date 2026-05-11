"""Phoneme inventory (20-phoneme SPE-simplified set) + topology helpers.

The PHONEMES list is the data contract for the whole study; every
metric and runner re-derives features by indexing into it. Helpers
here are pure functions over that list.
"""
from __future__ import annotations

from pcm.concept_graph import ConceptGraph

from ._config import EMBED_DIM


__all__ = [
    "PHONEMES",
    "N_PH",
    "cid_of",
    "feat_of",
    "hamming",
    "build_phoneme_graph",
    "_apply_shuffle",
]


# 20 phonemes, 3 attributes: (label, voice, manner, place).
# voice: 0=voiceless, 1=voiced
# manner: 0=STOP, 1=FRIC, 2=NAS, 3=APR
# place:  0=LAB,  1=COR,  2=DOR,  3=GLT
PHONEMES: list[tuple[str, int, int, int]] = [
    ("p",  0, 0, 0),
    ("b",  1, 0, 0),
    ("t",  0, 0, 1),
    ("d",  1, 0, 1),
    ("k",  0, 0, 2),
    ("g",  1, 0, 2),
    ("q",  0, 0, 3),    # /ʔ/ glottal stop
    ("f",  0, 1, 0),
    ("v",  1, 1, 0),
    ("s",  0, 1, 1),
    ("z",  1, 1, 1),
    ("x",  0, 1, 2),    # velar fricative
    ("h",  0, 1, 3),
    ("m",  1, 2, 0),
    ("n",  1, 2, 1),
    ("N",  1, 2, 2),    # /ŋ/
    ("w",  1, 3, 0),
    ("l",  1, 3, 1),
    ("r",  1, 3, 1),
    ("j",  1, 3, 2),    # palatal approximant
]
N_PH = len(PHONEMES)


def cid_of(idx: int) -> str:
    return f"concept:phoneme:{PHONEMES[idx][0]}"


def feat_of(idx: int) -> tuple[int, int, int]:
    """→ (voice, manner, place)."""
    return PHONEMES[idx][1], PHONEMES[idx][2], PHONEMES[idx][3]


def hamming(i: int, j: int) -> int:
    """Count of differing attributes across (voice, manner, place), 0..3."""
    a = feat_of(i); b = feat_of(j)
    return sum(1 for x, y in zip(a, b) if x != y)


def build_phoneme_graph() -> ConceptGraph:
    cg = ConceptGraph(feat_dim=EMBED_DIM)
    for i, (lbl, v, m, p) in enumerate(PHONEMES):
        cg.register_concept(
            node_id=cid_of(i),
            label=f"PHONEME_{lbl}",
            scope="BASE",
            provenance=f"phoneme_study:voice={v},manner={m},place={p}",
        )
    return cg


def _apply_shuffle(ids: list[str], sm: dict[int, int] | None) -> list[str]:
    if sm is None:
        return ids
    out = []
    for cid in ids:
        lbl = cid.rsplit(":", 1)[-1]
        for i, row in enumerate(PHONEMES):
            if row[0] == lbl:
                out.append(cid_of(sm[i]))
                break
    return out
