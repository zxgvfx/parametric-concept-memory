r"""Serialization + introspection mixin for ConceptGraph.

Owns:

- ``stats`` — high-level counters for viewer / debug.
- ``iter_bundle_parameters`` / ``iter_gate_parameters`` — feed an
  optimizer with the dense ``bundle_pool`` (and optional Tier-B gate)
  Parameters.
- ``bundles_to`` — move every facet pool to a target device.
- ``attribution_report`` — D91 + D92 combined audit.
- ``to_dict`` + ``save_atomic`` — JSON debug dump (NOT persistent
  storage; see ``STORAGE_ROADMAP`` stage C).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, Union

import torch
import torch.nn as nn

from ..param_bundle import migrate_param_in_optimizer


__all__ = ["_SerializeMixin"]


_PathLike = Union[str, Path]


class _SerializeMixin:
    """Stats, optimizer iteration, device migration, and JSON dump."""

    # ── 统计 / debug dump ────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        live_concepts = sum(1 for n in self.concepts.values() if n.liveness() > 0)
        plural_concepts = sum(1 for n in self.concepts.values() if n.liveness() >= 2)
        # D93: total tensor params == sum over facets of capacity*row_numel.
        total_bundle_params = 0
        for facet, pool in self.bundle_pool.items():
            total_bundle_params += int(pool.numel())
        total_collapses = sum(
            sum(len(h) for h in per_slot.values())
            for per_slot in self._collapse_history_by_slot.values()
        )
        return {
            "n_concepts": len(self.concepts),
            "n_surfaces": len(self.surfaces),
            "n_has_surface_edges": len(self.has_surface_form),
            "n_concept_edges": len(self.edges),
            "n_procedures": len(self.procedures),  # D89
            "n_base_concepts": sum(
                1 for n in self.concepts.values() if n.scope == "BASE"
            ),
            "n_core_concepts": sum(
                1 for n in self.concepts.values() if n.scope == "CORE"
            ),
            "n_verified_procedures": sum(
                1 for p in self.procedures.values() if p.verified
            ),
            # D91/D92 归因 + 语义史统计
            "n_live_concepts": live_concepts,
            "n_plural_concepts": plural_concepts,
            "n_bundle_params": total_bundle_params,
            "n_collapse_events": total_collapses,
            # D93 dense-pool stats
            "pool_capacity": self.capacity,
            "pool_n_facets": len(self.bundle_pool),
            "pool_grow_events": self._grow_events,
            "pool_used_slots": len(self.cid_to_slot),
        }

    # ── D91/D92/D93: bundle 参数聚合 ──────────────────────────────

    def iter_bundle_parameters(self) -> Iterator[nn.Parameter]:
        """Optimizer-feeder: yield every dense ``bundle_pool[facet]`` Parameter.

        With D93, this returns 1 ``Parameter`` per facet (not per concept).
        ``F.embedding`` indexes into the pool; sparse-grad updates fall out
        naturally from autograd. Pre-D93 callers that did
        ``list(cg.iter_bundle_parameters())`` keep working unchanged.
        """
        for facet in sorted(self.bundle_pool.keys()):
            yield self.bundle_pool[facet]

    def iter_gate_parameters(self) -> Iterator[nn.Parameter]:
        """Tier-B: yield every gate Parameter so it can join the optimizer.

        Returns an empty iterator when gates are disabled, so callers can
        unconditionally do ``list(cg.iter_bundle_parameters()) +
        list(cg.iter_gate_parameters())``.
        """
        if not self.gate_enabled:
            return
        for facet in sorted(self.bundle_gates.keys()):
            yield self.bundle_gates[facet]

    def bundles_to(self, device: torch.device | str) -> None:
        """Move every facet pool to ``device`` in-place. Re-binds the
        ``nn.Parameter`` so optimizer state migrates correctly when registered.
        """
        target = torch.device(device)
        for facet, pool in list(self.bundle_pool.items()):
            if pool.device == target:
                continue
            new_p = nn.Parameter(pool.data.to(target), requires_grad=pool.requires_grad)
            if self._registered_optimizer is not None:
                migrate_param_in_optimizer(
                    self._registered_optimizer, pool, new_p, self.capacity
                )
            self.bundle_pool[facet] = new_p

    def attribution_report(self) -> dict[str, dict]:
        """D91 归因 + D92 语义史合并报告.

        返回 ``{"by_concept": {cid: bundle.describe()}, "by_caller": {caller: {...}}}``.
        """
        by_concept = {
            cid: node.bundle.describe()
            for cid, node in self.concepts.items()
            if node.bundle is not None and node.bundle.facets()
        }
        by_caller: dict[str, dict] = {}
        for cid, node in self.concepts.items():
            if node.bundle is None:
                continue
            for facet, consumers in node.bundle.consumed_by.items():
                for caller in consumers:
                    slot = by_caller.setdefault(
                        caller,
                        {"concepts": set(), "facets": set(), "n_params": 0},
                    )
                    slot["concepts"].add(cid)
                    slot["facets"].add(facet)
                    p = node.bundle.params.get(facet)
                    if p is not None:
                        slot["n_params"] += p.numel()
        for caller, slot in by_caller.items():
            slot["concepts"] = sorted(slot["concepts"])
            slot["facets"] = sorted(slot["facets"])
        n_void = sum(1 for n in self.concepts.values() if n.liveness() == 0)
        n_single = sum(1 for n in self.concepts.values() if n.liveness() == 1)
        n_plural = sum(1 for n in self.concepts.values() if n.liveness() >= 2)
        return {
            "by_concept": by_concept,
            "by_caller": by_caller,
            "n_void": n_void,
            "n_single": n_single,
            "n_plural": n_plural,
        }

    # ── Serialize (debug / viewer / short-term checkpoint) ──────────────
    # ⚠️ 本节**不是** 生产级持久化方案 (那是 persistent_memory 的职责, 见
    # docs/research/STORAGE_ROADMAP.md 阶段 C). 只适合 < 10K 节点 debug 用.

    def to_dict(self) -> dict:
        """Dump 图谱结构到 JSON-serializable dict (仅 debug / visualize 用).

        NOT persistent storage — 那是 ``persistent_memory`` 的职责.
        centroid 只 dump 范数和前 5 维 (避免 JSON 体积爆炸).
        """
        return {
            "stats": self.stats(),
            "concepts": {
                nid: {
                    "node_id": node.node_id,
                    "label": node.label,
                    "scope": node.scope,
                    "surface_forms": sorted(node.surface_forms),
                    "connected_networks": sorted(node.connected_networks),
                    "grounding_provenance": list(node.grounding_provenance),
                    "last_tick": node.last_tick,
                    # D91/D92: bundle 元数据 (不含 tensor)
                    "bundle": node.bundle.describe() if node.bundle.facets() else None,
                    "liveness": node.liveness(),
                }
                for nid, node in self.concepts.items()
            },
            "surfaces": {
                sid: {
                    "node_id": sf.node_id,
                    "modality": sf.modality,
                    "surface_form": sf.surface_form,
                    "grounded_to": sf.grounded_to,
                    "hit_count": sf.hit_count,
                    "confidence": sf.confidence,
                    "last_tick": sf.last_tick,
                    "centroid_norm": float(sf.centroid.norm().item()),
                    "centroid_first_5": sf.centroid[:5].tolist(),
                    "image_paths": dict(sf.image_paths),
                }
                for sid, sf in self.surfaces.items()
            },
            "has_surface_form_edges": [
                {
                    "concept_id": e.concept_id,
                    "surface_id": e.surface_id,
                    "weight": e.weight,
                    "count": e.count,
                }
                for e in self.has_surface_form.values()
            ],
            "concept_edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "edge_type": e.edge_type,
                    "weight": e.weight,
                    "count": e.count,
                }
                for e in self.edges.values()
            ],
            # D89 程序性知识 (P1 schema 预留, P2 起 viewer / executor 会用)
            "procedures": {
                pid: {
                    "node_id": p.node_id,
                    "name": p.name,
                    "version": p.version,
                    "module_path": p.module_path,
                    "inline_source": (
                        p.inline_source if p.inline_source and len(p.inline_source) < 4096
                        else None
                    ),
                    "entry_symbol": p.entry_symbol,
                    "grounded_to": list(p.grounded_to),
                    "provenance": p.provenance,
                    "trust_stage": p.trust_stage,
                    "verified": p.verified,
                    "hit_count": p.hit_count,
                    "success_count": p.success_count,
                    "success_rate": p.success_rate,
                    "avg_latency_ms": p.avg_latency_ms,
                    "last_tick": p.last_tick,
                    "preconditions": list(p.preconditions),
                    "effects": list(p.effects),
                    "instruction_text": p.instruction_text,
                }
                for pid, p in self.procedures.items()
            },
        }

    def save_atomic(self, path: _PathLike, indent: int = 2) -> Path:
        """原子写 ``to_dict()`` 到 JSON 文件.

        用 ``tmp + rename`` 保证 viewer polling / 其他 reader **永远读不到半写**
        的 JSON (POSIX 原子 rename 语义). 是 STORAGE_ROADMAP 阶段 A 必备补丁.

        Args:
            path: 目标文件路径 (e.g. "outputs/m1_demo/concept_graph.json").
            indent: JSON 缩进 (默认 2, 便于人读; 0 会禁用缩进最快).

        Returns:
            解析后的 Path 对象 (方便链式调用).

        注意 (不是):
            - 不是跨进程事务 — 不保证两个 writer 同时 save_atomic 不冲突
              (P1 单进程模型下不会发生; 真需要时走 Kuzu, 见 STORAGE_ROADMAP §5 阶段 C)
            - 不是增量 append — 每次全量 dump, O(n_nodes) 开销
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=indent),
            encoding="utf-8",
        )
        os.replace(tmp, target)  # POSIX atomic rename
        return target
