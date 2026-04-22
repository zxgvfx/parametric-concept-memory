"""Shared helper: build the ANS ConceptGraph used by all muscle training runs."""
from __future__ import annotations

from pcm.concept_graph import ConceptGraph


def build_ans_graph(n_min: int = 1, n_max: int = 7, include_void: bool = True) -> ConceptGraph:
    """7 concepts for ANS numerosity + optional void concept (H4 control).

    Concept ids match the saved ckpts: ``concept:ans:1 .. concept:ans:7`` +
    ``concept:ans:void``. The void concept is registered but never collapsed
    during training — it must stay ``liveness=0`` at eval time (H4 check).
    """
    cg = ConceptGraph(feat_dim=128)
    for n in range(n_min, n_max + 1):
        cg.register_concept(
            node_id=f"concept:ans:{n}",
            label=f"ANS_{n}",
            scope="BASE",
            provenance=f"train_muscle:n={n}",
        )
    if include_void:
        cg.register_concept(
            node_id="concept:ans:void",
            label="ANS_VOID",
            scope="BASE",
            provenance="H4_void_control",
        )
    return cg


def concept_ids_for_counts(counts) -> list[str]:
    """map int tensor/list counts -> ['concept:ans:{n}', ...]."""
    try:
        return [f"concept:ans:{int(c)}" for c in counts.tolist()]
    except AttributeError:
        return [f"concept:ans:{int(c)}" for c in counts]
