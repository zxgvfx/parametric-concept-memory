r"""Final ConceptGraph class — composes mixins from sibling modules.

This file is intentionally tiny: every behaviour lives in a focused
mixin (``_core`` / ``_collapse`` / ``_surface`` / ``_edges`` /
``_procedures`` / ``_queries`` / ``_serialize``). The combined class is
what callers import via :mod:`pcm.concept_graph` (the package).

MRO order matters for ``__init__`` resolution. We put ``_CoreMixin``
first so its ``__init__`` is the one that runs; the other mixins do
not define ``__init__``.
"""
from __future__ import annotations

from ._collapse import _CollapseMixin
from ._core import _CoreMixin
from ._edges import _EdgeMixin
from ._procedures import _ProcedureMixin
from ._queries import _QueryMixin
from ._serialize import _SerializeMixin
from ._surface import _SurfaceMixin


__all__ = ["ConceptGraph"]


class ConceptGraph(
    _CoreMixin,
    _CollapseMixin,
    _SurfaceMixin,
    _EdgeMixin,
    _ProcedureMixin,
    _QueryMixin,
    _SerializeMixin,
):
    """Hub-and-spoke 概念图谱 (D85 + D86 + D87) with D91-D94 extensions.

    架构::

        ConceptNode (hub, 抽象概念, 不存 centroid)
            │ has_surface_form 边
            ▼
        SurfaceFormNode (spoke, 具体表象, 一个 modality+surface_form 一个 centroid)

    跨 hub 关系: ``ConceptNode → ConceptNode`` 通过 :class:`ConceptEdge`
    (4 类, v1 借).

    The class is split across 7 mixins — see ``__all__`` of each ``_*.py``
    file under :mod:`pcm.concept_graph` for the per-file responsibility map.
    Splitting keeps every file under the project's 500-line cap while
    retaining a single ``ConceptGraph`` class identity for downstream code.
    """
