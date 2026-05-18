"""pcm — Parametric Concept Memory.

A concept memory where each ``ConceptNode`` in a symbolic graph owns a
multi-facet parameter bundle, consumed on demand by task-specific
"muscle" modules via a ``contextual collapse`` operation.

Since v0.2 the per-concept bundle is a *view* into a pre-allocated dense
``ConceptGraph.bundle_pool`` — see ``docs/PCM_BIO_PREALLOC_UPGRADE.md``
for the bionic motivation and the G1-G6 capacity-grow invariants.

Paper: *Concepts Collapse into Muscles — Domain-Topology-Adaptive
Parametric Concept Memory.*
Repo: https://github.com/zxgvfx/parametric-concept-memory

Public API::

    ConceptGraph, ConceptNode          — graph-of-concepts container
    ParamBundle, BundleRowView         — per-concept parameter proxy
    ContextualizedConcept              — ephemeral collapse handle
    config                              — capacity / grow / gate defaults
"""
from . import (
    config, coref, diagnostics, dna_ops, episodic, gate, graph_eval,
    peer, sleep,
)
from .concept_graph import ConceptGraph, ConceptNode
from .episodic import (
    EpisodeRecord,
    EpisodicBuffer,
    LongTermEpisodicTrace,
    consolidate_to_concept_graph,
)
from .graph_eval import (
    PARAMETRIC_KIND,
    GraphEvaluator,
    SubgraphEvalError,
)
from .param_bundle import (
    BundleRowView,
    ContextualizedConcept,
    ParamBundle,
    init_row_,
    migrate_param_in_optimizer,
)

__version__ = "0.2.0"
__all__ = [
    "ConceptGraph",
    "ConceptNode",
    "ParamBundle",
    "BundleRowView",
    "ContextualizedConcept",
    "init_row_",
    "migrate_param_in_optimizer",
    "GraphEvaluator",
    "SubgraphEvalError",
    "PARAMETRIC_KIND",
    "config",
    "diagnostics",
    "dna_ops",
    "gate",
    "graph_eval",
    "peer",
    "sleep",
    # F75 — episodic memory
    "episodic",
    "EpisodeRecord",
    "EpisodicBuffer",
    "LongTermEpisodicTrace",
    "consolidate_to_concept_graph",
    # F76 — coreference resolution
    "coref",
]
