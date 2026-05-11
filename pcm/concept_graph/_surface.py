r"""SurfaceFormNode + has_surface_form mixin (D87 hub-and-spoke spoke).

Owns:

- ``register_surface`` — add or EMA-update a SurfaceFormNode.
- ``_evict_oldest_surface`` — LRU eviction (no protected scope).
- ``link_surface_to_concept`` — establish / strengthen a
  ``has_surface_form`` edge (auto-rebinds if the surface was previously
  grounded to another concept).
"""
from __future__ import annotations

from torch import Tensor

from .nodes import (
    HasSurfaceFormEdge,
    MAX_EDGE_WEIGHT,
    SurfaceFormNode,
)


__all__ = ["_SurfaceMixin"]


class _SurfaceMixin:
    """Surface form node lifecycle + has_surface_form edge management."""

    def register_surface(
        self,
        modality: str,
        surface_form: str,
        centroid: Tensor,
        tick: int = 0,
        ema_alpha: float = 0.15,
        image_paths: dict[str, str] | None = None,
    ) -> SurfaceFormNode:
        """添加或 EMA 更新 SurfaceFormNode (spoke).

        ema_alpha: 0 = 不更新 (只读), 1.0 = 完全用新值, 0.15 = 默认 (跟 v1
        LiteracyTrainer 一致).
        image_paths: dict {type: path} (见 SurfaceFormNode.image_paths),
        非 None 时合并到 sf.image_paths (新 key 覆盖旧).
        """
        feat = centroid.detach().float().view(-1).cpu()
        if feat.shape[-1] != self.feat_dim:
            raise ValueError(
                f"SurfaceFormNode centroid dim {feat.shape[-1]} != feat_dim "
                f"{self.feat_dim}"
            )

        node_id = SurfaceFormNode.make_id(modality, surface_form)

        if node_id in self.surfaces:
            sf = self.surfaces[node_id]
            sf.centroid = ((1.0 - ema_alpha) * sf.centroid + ema_alpha * feat).detach()
            sf.hit_count += 1
            sf.last_tick = tick
            if image_paths:
                sf.image_paths.update(image_paths)
            return sf

        if len(self.surfaces) >= self.max_surfaces:
            self._evict_oldest_surface()

        sf = SurfaceFormNode(
            node_id=node_id,
            modality=modality,
            surface_form=surface_form,
            centroid=feat,
            hit_count=1,
            last_tick=tick,
            valid_from=tick,
            image_paths=dict(image_paths) if image_paths else {},
        )
        self.surfaces[node_id] = sf
        return sf

    def _evict_oldest_surface(self) -> None:
        """LRU evict SurfaceFormNode (无 protected scope 概念, 全可淘汰)."""
        if not self.surfaces:
            return
        oldest_id = min(self.surfaces, key=lambda k: self.surfaces[k].last_tick)
        sf = self.surfaces.pop(oldest_id)
        # 清理反向边
        if sf.grounded_to:
            concept = self.concepts.get(sf.grounded_to)
            if concept:
                concept.surface_forms.discard(oldest_id)
            self.has_surface_form.pop((sf.grounded_to, oldest_id), None)
        adj = self._concept_to_surfaces.get(sf.grounded_to or "", [])
        if oldest_id in adj:
            adj.remove(oldest_id)
        self._surface_to_concept.pop(oldest_id, None)

    def link_surface_to_concept(
        self,
        concept_id: str,
        surface_id: str,
        tick: int = 0,
    ) -> HasSurfaceFormEdge:
        """连接 ConceptNode (hub) 跟 SurfaceFormNode (spoke).

        如果 surface 已经 grounded 到另一 concept, 自动断开旧 link, 接到新 concept.
        """
        if concept_id not in self.concepts:
            raise KeyError(f"ConceptNode not found: {concept_id}")
        if surface_id not in self.surfaces:
            raise KeyError(f"SurfaceFormNode not found: {surface_id}")

        sf = self.surfaces[surface_id]
        # 若已 grounded 到别的 concept, 先断开
        if sf.grounded_to and sf.grounded_to != concept_id:
            old_concept = self.concepts.get(sf.grounded_to)
            if old_concept:
                old_concept.surface_forms.discard(surface_id)
            self.has_surface_form.pop((sf.grounded_to, surface_id), None)
            old_adj = self._concept_to_surfaces.get(sf.grounded_to, [])
            if surface_id in old_adj:
                old_adj.remove(surface_id)

        key = (concept_id, surface_id)
        if key in self.has_surface_form:
            old = self.has_surface_form[key]
            new = HasSurfaceFormEdge(
                concept_id=concept_id,
                surface_id=surface_id,
                weight=min(old.weight + 0.1, MAX_EDGE_WEIGHT),
                count=old.count + 1,
                last_tick=tick,
            )
            self.has_surface_form[key] = new
            sf.last_tick = tick
            return new

        edge = HasSurfaceFormEdge(
            concept_id=concept_id,
            surface_id=surface_id,
            weight=1.0,
            count=1,
            last_tick=tick,
        )
        self.has_surface_form[key] = edge
        self.concepts[concept_id].surface_forms.add(surface_id)
        sf.grounded_to = concept_id
        sf.last_tick = tick
        if surface_id not in self._concept_to_surfaces[concept_id]:
            self._concept_to_surfaces[concept_id].append(surface_id)
        self._surface_to_concept[surface_id] = concept_id
        return edge
