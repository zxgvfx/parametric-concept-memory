r"""ConceptGraph package — Hub-and-Spoke 概念图谱 (P1 架构核心).

Pre-2026-05 this lived in a single ``pcm/concept_graph.py`` (~1200 lines).
It has now been split across multiple files (one job each) so every
source file stays under the project's 500-line cap. The public API is
unchanged: every name that used to be importable from
``pcm.concept_graph`` is re-exported here.

Layout:

- :mod:`pcm.concept_graph.nodes`       — dataclasses + module constants.
- :mod:`pcm.concept_graph._core`       — ``__init__`` + register + grow.
- :mod:`pcm.concept_graph._collapse`   — collapse_batch + cook subgraph.
- :mod:`pcm.concept_graph._surface`    — surface form + has_surface_form edges.
- :mod:`pcm.concept_graph._edges`      — concept-to-concept edges.
- :mod:`pcm.concept_graph._procedures` — D89 procedural memory.
- :mod:`pcm.concept_graph._queries`    — read-only lookup helpers.
- :mod:`pcm.concept_graph._serialize`  — stats / iter_params / to_dict / save_atomic.
- :mod:`pcm.concept_graph.graph`       — final ``ConceptGraph`` class.

Pre-D87 v1 contract (see ``mind/`` history for the full table):

- ConceptNode = transmodal hub (Patterson 2007; Dehaene IPS).
- SurfaceFormNode = modality-specific spoke.
- 加新表象 / 新模态 / 新语言 = 加 SurfaceFormNode + has_surface_form 边, 不改架构.

Invariants (PHILOSOPHY 红线):

- 单文件 ≤ 500 行 (enforced by :mod:`scripts.check_file_size`).
- 不硬编码 game / 颜色 / 字符语义.
- 闭环可解释: ConceptNode.grounding_provenance 必填.
- BASE / CORE scope 节点不允许 evict (D86).
- centroid 全部在 SurfaceFormNode, ConceptNode 不存 centroid (D87).
"""
from __future__ import annotations

from .graph import ConceptGraph
from .nodes import (
    ConceptEdge,
    ConceptNode,
    DEFAULT_FEAT_DIM,
    DEFAULT_MAX_EDGES,
    DEFAULT_MAX_NODES,
    DEFAULT_MAX_SURFACES,
    HasSurfaceFormEdge,
    MAX_EDGE_WEIGHT,
    MAX_PROVENANCE_PER_NODE,
    PROTECTED_SCOPES,
    ProcedureNode,
    SurfaceFormNode,
)


__all__ = [
    # Public-API classes (the names downstream callers actually import).
    "ConceptGraph",
    "ConceptNode",
    "SurfaceFormNode",
    "HasSurfaceFormEdge",
    "ConceptEdge",
    "ProcedureNode",
    # Module-level constants (kept for back-compat with code that did
    # ``from pcm.concept_graph import PROTECTED_SCOPES`` etc.).
    "PROTECTED_SCOPES",
    "DEFAULT_FEAT_DIM",
    "DEFAULT_MAX_NODES",
    "DEFAULT_MAX_SURFACES",
    "DEFAULT_MAX_EDGES",
    "MAX_EDGE_WEIGHT",
    "MAX_PROVENANCE_PER_NODE",
]
