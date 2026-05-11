r"""Query mixin for ConceptGraph.

Owns the read-only lookup helpers:

- ``find_concept`` / ``find_surface``
- ``get_concept_for_surface`` / ``get_surfaces_for_concept``
- ``list_concepts`` / ``query_palace``
- ``nearest_surface`` (cosine top-K within a modality)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .nodes import ConceptNode, SurfaceFormNode


__all__ = ["_QueryMixin"]


class _QueryMixin:
    """Read-only query helpers."""

    def find_concept(
        self,
        label: str | None = None,
        node_id: str | None = None,
    ) -> ConceptNode | None:
        if node_id and node_id in self.concepts:
            return self.concepts[node_id]
        if label:
            for node in self.concepts.values():
                if node.label == label:
                    return node
        return None

    def find_surface(self, modality: str, surface_form: str) -> SurfaceFormNode | None:
        return self.surfaces.get(SurfaceFormNode.make_id(modality, surface_form))

    def get_concept_for_surface(self, surface_id: str) -> ConceptNode | None:
        cid = self._surface_to_concept.get(surface_id)
        return self.concepts.get(cid) if cid else None

    def get_surfaces_for_concept(self, concept_id: str) -> list[SurfaceFormNode]:
        ids = self._concept_to_surfaces.get(concept_id, [])
        return [self.surfaces[sid] for sid in ids if sid in self.surfaces]

    def list_concepts(
        self,
        scope: str | None = None,
        connected_to: str | None = None,
    ) -> list[ConceptNode]:
        result = []
        for node in self.concepts.values():
            if scope is not None and node.scope != scope:
                continue
            if connected_to is not None and connected_to not in node.connected_networks:
                continue
            result.append(node)
        return result

    def query_palace(
        self,
        scope: str = "level",
        connected_to: str | None = None,
    ) -> list[ConceptNode]:
        """跟 PHILOSOPHY §7 + cg 现有 query_palace 对齐."""
        return self.list_concepts(scope=scope, connected_to=connected_to)

    def nearest_surface(
        self,
        modality: str,
        embedding: Tensor,
        top_k: int = 5,
    ) -> list[tuple[SurfaceFormNode, float]]:
        """Cosine-nearest SurfaceFormNode in given modality."""
        query = embedding.detach().float().view(-1).cpu()
        if query.shape[-1] != self.feat_dim:
            raise ValueError(
                f"embedding dim {query.shape[-1]} != feat_dim {self.feat_dim}"
            )

        candidates = [sf for sf in self.surfaces.values() if sf.modality == modality]
        if not candidates:
            return []

        ids = [sf.node_id for sf in candidates]
        vecs = torch.stack([sf.centroid for sf in candidates])
        sims = F.cosine_similarity(query.unsqueeze(0), vecs, dim=1)
        k = min(top_k, len(ids))
        topk_vals, topk_idx = sims.topk(k)
        return [
            (self.surfaces[ids[i]], float(s))
            for i, s in zip(topk_idx.tolist(), topk_vals.tolist())
        ]
