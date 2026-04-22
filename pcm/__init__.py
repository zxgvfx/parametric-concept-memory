"""pcm — Parametric Concept Memory.

A concept memory where each ConceptNode in a symbolic graph owns a
multi-facet parameter bundle, consumed on demand by task-specific
"muscle" modules via a `contextual collapse` operation.

Paper: *Concepts Collapse into Muscles — Domain-Topology-Adaptive
Parametric Concept Memory.*
Repo: https://github.com/zxgvfx/parametric-concept-memory

Public API:
    ConceptGraph, ConceptNode     — graph-of-concepts container
    ParamBundle, ContextualizedConcept — per-concept parameter memory
"""
from .concept_graph import ConceptGraph, ConceptNode
from .param_bundle import ContextualizedConcept, ParamBundle

__version__ = "0.1.0"
__all__ = [
    "ConceptGraph",
    "ConceptNode",
    "ParamBundle",
    "ContextualizedConcept",
]
