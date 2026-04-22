r"""ConceptGraph — Hub-and-Spoke 概念图谱 (P1 架构核心).

跟 v1 ``src/consciousness/concept_graph.py`` 比较 (D87 修订):

| v1 字段 | mind/ 等价 | 来源 |
|---|---|---|
| ConceptNode.source: str ("spatial_cluster" \| "character") | SurfaceFormNode.modality (任意 modality) | D87 |
| ConceptNode.centroid: Tensor (单 centroid) | SurfaceFormNode.centroid (跟 ConceptNode 解耦) | D87 |
| (无) | ConceptNode.surface_forms: set[str] (指向 SurfaceFormNode) | D87 |
| (无) | ConceptNode.connected_networks: set[str] (lateralization) | D86 |
| (无) | ConceptNode.scope: str ("BASE"/"CORE"/"level"/"domain"/"global") | PHILOSOPHY §7 |
| ConceptEdge (4 类: co_occurrence/temporal/spatial_adjacency/character_pair) | 保留, 不重新发明 | v1 borrow |

学术对应:
  - ConceptNode = transmodal "hub" (Patterson 2007 hub-and-spoke; Dehaene IPS quantity)
  - SurfaceFormNode = modality-specific "spoke" (V1 area / VWFA / FFA / IPS)
  - 加新表象 / 新模态 / 新语言 = 加 SurfaceFormNode + has_surface_form 边, 不改架构

不变量 (PHILOSOPHY 红线):
  - 单文件 ≤ 500 行
  - 不硬编码 game / 颜色 / 字符语义
  - 闭环可解释: ConceptNode.grounding_provenance 必填
  - BASE / CORE scope 节点不允许 evict (D86)
  - centroid 全部在 SurfaceFormNode, ConceptNode 不存 centroid (D87)
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from .param_bundle import ContextualizedConcept, ParamBundle

_PathLike = Union[str, Path]

log = logging.getLogger(__name__)


# ============================================================================
# 数据类 (D87 hub-and-spoke)
# ============================================================================


@dataclass
class SurfaceFormNode:
    """同一 ConceptNode 在某个 (modality, surface_form) 下的具体表象 (D87 spoke).

    例:
      ConceptNode("concept:number:zero") 在 vision_text 下有:
        - SurfaceFormNode("surface:vision_text:arabic_0",     centroid=<>)
        - SurfaceFormNode("surface:vision_text:chinese_零",   centroid=<>)
        - SurfaceFormNode("surface:vision_text:english_zero", centroid=<>)

    image_paths (optional): debug / viewer 用, key → 相对路径. 约定 key:
      - "render": 原始渲染图 (e.g. "renders/0.png")
      - "fovea":  fovea 采样后的 patch (e.g. "fovea/0_fovea.png")
      - 其他: 自由扩展 (如 "saliency" / "annotated")
    path 相对 ``ConceptGraph.save_atomic`` 目标文件的父目录, viewer 据此找图.
    """

    node_id: str
    modality: str
    surface_form: str
    centroid: Tensor
    hit_count: int = 0
    confidence: float = 0.0
    last_tick: int = 0
    grounded_to: str | None = None
    valid_from: int = 0
    valid_until: int | None = None
    image_paths: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def make_id(modality: str, surface_form: str) -> str:
        return f"surface:{modality}:{surface_form}"


@dataclass
class ConceptNode:
    """A discrete concept (D85 hub).

    通过 has_surface_form 边连到 SurfaceFormNode (D87 spokes).
    centroid 全部在 SurfaceFormNode, ConceptNode 不存.

    D91 Parametric Concept Memory:
        ``bundle`` 挂一个 multi-facet 可学参数池, 被多个肌肉按 facet name 消费,
        梯度直接回流. ``bundle.consumed_by`` 自动产生"谁消费了我"的归因记录.

    D92 Contextual Concept Collapse:
        ConceptNode 无"当下状态" — 只有被 caller × facet 观测 (``collapse``) 时产生
        ephemeral ``ContextualizedConcept``. 同一 concept 在不同 caller 下的语义
        不同但 id 同 (Wittgenstein 式 "meaning as use").
    """

    node_id: str
    label: str
    surface_forms: set[str] = field(default_factory=set)
    connected_networks: set[str] = field(default_factory=set)
    scope: str = "level"
    grounding_provenance: list[str] = field(default_factory=list)
    last_tick: int = 0
    valid_from: int = 0
    valid_until: int | None = None
    # D91/D92: 可学参数池 (lazy init, 不占空间除非被消费过)
    bundle: ParamBundle = field(default_factory=ParamBundle)

    # ── D92 API: collapse ────────────────────────────────────────

    def collapse(
        self,
        caller: str,
        facet: str,
        shape: Iterable[int],
        tick: int = 0,
        init: str = "normal_small",
        device: torch.device | str | None = None,
    ) -> ContextualizedConcept:
        """D92 核心操作: 在 (caller, facet) 观测下产生一次塌缩.

        副作用:
            1. 若 ``facet`` 未初始化, lazy create nn.Parameter (shape 由 caller 定,
               device 可传入; 首次 collapse 决定 device)
            2. 注册 ``caller`` 为该 facet 的 consumer (D91 归因)
            3. 追加一条 ``collapse_history[facet]`` 事件 (D92 语义史)

        返回 ephemeral ``ContextualizedConcept``, 携带当下语义参数 (可求导).
        """
        cc = self.bundle.request(
            facet, shape, caller,
            concept_id=self.node_id, tick=tick,
            init=init, device=device,  # type: ignore[arg-type]
        )
        return cc

    def liveness(self) -> int:
        """D92: L(v) = 跨所有 facet 的 unique consumer 数 (死概念 = 0)."""
        return self.bundle.liveness()


@dataclass
class HasSurfaceFormEdge:
    """ConceptNode → SurfaceFormNode 边 (D87)."""

    concept_id: str
    surface_id: str
    weight: float = 1.0
    count: int = 1
    last_tick: int = 0


@dataclass
class ConceptEdge:
    """ConceptNode → ConceptNode 边 (v1 借, 4 类: co_occurrence/temporal/spatial_adjacency/character_pair)."""

    source_id: str
    target_id: str
    weight: float = 0.1
    edge_type: str = "co_occurrence"
    count: int = 1
    last_tick: int = 0


# ============================================================================
# ProcedureNode — 程序性知识 (D89, P1 预留 schema, P2 启用执行)
# ============================================================================


@dataclass
class ProcedureNode:
    """程序性知识节点 (D89): "how-to" 肌肉, 与 ConceptNode "what-is" 对偶.

    神经学类比:
      - basal ganglia procedural memory (Squire & Cohen 1981, H.M. 病例)
      - ACT-R production rule (Anderson 1993)
      - Soar operator + chunking (Laird 2012)

    AI 类比:
      - Voyager skill library (Wang 2023 NeurIPS)
      - HIPO programmatic option (Lin 2024 NeurIPS)
      - Deep Agents versioned skill (LangChain 2026)

    本节点只存**元数据 + code reference**. 实际执行在 ``mind/core/muscle/``
    的 skill_executor (P2+ 实施), 通过 module_path 或 inline_source 加载.

    Fitts-Posner 三阶段 (trust_stage):
      cognitive:    刚注册, verified=False, 每次执行都沙盒监督
      associative:  过 acceptance_tests + hit_count≥10 + success≥0.7, verified=True
      autonomous:   hit_count≥100 + success≥0.9, plan 层可直接 shortcut 调用

    获取途径 (provenance):
      bootstrap:    人写的 trusted primitive (P1)
      taught:       老师给 module_path, agent 挂上 (P2)
      chunked:      agent 观察自己 trajectory 抽出 subroutine (P3, Soar chunking)
      synthesized:  agent 从 primitive 组合 (P3/P4, DreamCoder library learning)

    P1 范围 (本 dataclass): 只预留 schema, 不实现 executor / registry / verifier.
    P2 起点: 见 ``docs/language/ROADMAP.md §P2-Skill-Library``.
    """

    # ── Identity ──────────────────────────────────────────────
    node_id: str                                  # "skill:replay_analysis:v1"
    name: str                                     # "replay_analysis"
    version: int = 1

    # ── Code 引用 (二选一) ─────────────────────────────────────
    inline_source: str | None = None              # 短 skill (<20 行) 直接存源码
    module_path: str | None = None                # "mind.skills.replay.analyze_v1"
    entry_symbol: str = "run"                     # 模块内的函数名

    # ── Interface contract (Soar operator 风格) ──────────────
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)   # 需满足的 concept_ids
    effects: list[str] = field(default_factory=list)         # 执行后成立的 concept_ids

    # ── 与 ConceptGraph 锚点 (grounded_to 是 concept_ids 列表) ─
    grounded_to: list[str] = field(default_factory=list)

    # ── Provenance (学习出处) ────────────────────────────────
    provenance: str = "bootstrap"                 # bootstrap / taught / chunked / synthesized
    learned_at_tick: int = 0
    instruction_text: str | None = None           # 原始自然语言指令 (Taught 模式)
    learned_from_episode: str | None = None       # episode/replay id (Chunked 模式)

    # ── Fitts-Posner 指标 + 经验统计 ──────────────────────────
    hit_count: int = 0
    success_count: int = 0
    avg_latency_ms: float = 0.0
    trust_stage: str = "cognitive"                # cognitive / associative / autonomous

    # ── 安全 (verification-first, PHILOSOPHY + D26 修订) ──────
    verified: bool = False
    max_runtime_seconds: float = 5.0
    requires_sandbox: bool = True
    acceptance_tests: list[str] = field(default_factory=list)   # pytest test ids

    last_tick: int = 0

    @staticmethod
    def make_id(name: str, version: int) -> str:
        """规范化的 ProcedureNode ID: ``skill:<name>:v<version>``."""
        return f"skill:{name}:v{version}"

    @property
    def success_rate(self) -> float:
        """hit_count>0 时的成功率, 否则 0.0."""
        return self.success_count / max(1, self.hit_count)

    def is_callable_by_plan(self) -> bool:
        """plan 层能直接调用此 skill 的判断.

        P1 阶段这个方法只作 schema 占位, P2 skill_executor 启用时才有真实调用语义.
        当前约定: 必须 verified=True 且 trust_stage 非 cognitive.
        """
        return self.verified and self.trust_stage != "cognitive"


# ============================================================================
# Scope 保护 + 默认配置
# ============================================================================

# BASE / CORE scope 节点不可 evict (D86)
PROTECTED_SCOPES = frozenset({"BASE", "CORE"})

# 默认配置
_DEFAULT_FEAT_DIM = 128
_DEFAULT_MAX_NODES = 10000
_DEFAULT_MAX_SURFACES = 50000
_DEFAULT_MAX_EDGES = 100000
_MAX_EDGE_WEIGHT = 10.0
_MAX_PROVENANCE_PER_NODE = 50  # 防 unbounded 增长 (PHILOSOPHY §7)


# ============================================================================
# ConceptGraph
# ============================================================================


class ConceptGraph:
    """Hub-and-spoke 概念图谱 (D85 + D86 + D87).

    架构:
      ConceptNode  (hub, 抽象概念, 不存 centroid)
          │ has_surface_form 边
          ▼
      SurfaceFormNode (spoke, 具体表象, 一个 modality + surface_form 一个 centroid)

    跨 hub 关系:
      ConceptNode → ConceptNode 通过 ConceptEdge (4 类, v1 借)
    """

    def __init__(
        self,
        feat_dim: int = _DEFAULT_FEAT_DIM,
        max_nodes: int = _DEFAULT_MAX_NODES,
        max_surfaces: int = _DEFAULT_MAX_SURFACES,
        max_edges: int = _DEFAULT_MAX_EDGES,
    ) -> None:
        self.feat_dim = feat_dim
        self.max_nodes = max_nodes
        self.max_surfaces = max_surfaces
        self.max_edges = max_edges

        self.concepts: dict[str, ConceptNode] = {}
        self.surfaces: dict[str, SurfaceFormNode] = {}
        self.has_surface_form: dict[tuple[str, str], HasSurfaceFormEdge] = {}
        self.edges: dict[tuple[str, str, str], ConceptEdge] = {}
        # D89 程序性知识 (P1 预留 schema, P2 起启用执行)
        self.procedures: dict[str, ProcedureNode] = {}

        # adjacency 索引 (查询加速; PHILOSOPHY §4 同一职责一种实现)
        self._concept_to_surfaces: dict[str, list[str]] = defaultdict(list)
        self._surface_to_concept: dict[str, str] = {}
        self._adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
        # ConceptNode → [ProcedureNode ids] (has_procedure 查询索引)
        self._concept_to_procedures: dict[str, list[str]] = defaultdict(list)

    # ── ConceptNode 操作 ─────────────────────────────────────────

    def register_concept(
        self,
        node_id: str,
        label: str,
        scope: str = "level",
        connected_networks: Iterable[str] | None = None,
        provenance: str | None = None,
        tick: int = 0,
    ) -> ConceptNode:
        """添加或更新 ConceptNode (hub). 已存在则 EMA-style 更新元数据."""
        if node_id in self.concepts:
            node = self.concepts[node_id]
            node.last_tick = tick
            if provenance:
                self._append_provenance(node, provenance)
            if connected_networks:
                node.connected_networks.update(connected_networks)
            return node

        if len(self.concepts) >= self.max_nodes:
            self._evict_oldest_concept()

        node = ConceptNode(
            node_id=node_id,
            label=label,
            scope=scope,
            connected_networks=set(connected_networks) if connected_networks else set(),
            grounding_provenance=[provenance] if provenance else [],
            last_tick=tick,
            valid_from=tick,
        )
        self.concepts[node_id] = node
        return node

    def _append_provenance(self, node: ConceptNode, provenance: str) -> None:
        """LRU-cap provenance list 防 unbounded 增长 (PHILOSOPHY §7)."""
        node.grounding_provenance.append(provenance)
        if len(node.grounding_provenance) > _MAX_PROVENANCE_PER_NODE:
            # 保留最早 1 条 (来源记录) + 最新 N-1 条
            node.grounding_provenance = (
                node.grounding_provenance[:1]
                + node.grounding_provenance[-(_MAX_PROVENANCE_PER_NODE - 1):]
            )

    def _evict_oldest_concept(self) -> None:
        """LRU evict, 但 BASE/CORE scope 节点不淘汰 (D86)."""
        candidates = [
            (nid, node)
            for nid, node in self.concepts.items()
            if node.scope not in PROTECTED_SCOPES
        ]
        if not candidates:
            log.warning(
                "ConceptGraph at capacity (%d) but only protected nodes exist; "
                "skipping evict (D86)",
                len(self.concepts),
            )
            return
        oldest_id = min(candidates, key=lambda x: x[1].last_tick)[0]
        self._remove_concept(oldest_id)

    def _remove_concept(self, node_id: str) -> None:
        """移除 ConceptNode + 相关 has_surface_form / ConceptEdge."""
        if node_id not in self.concepts:
            return
        node = self.concepts.pop(node_id)
        # 断开 SurfaceForm 反向指向 (但保留 SurfaceFormNode 本身, 等独立 evict)
        for sf_id in list(node.surface_forms):
            sf = self.surfaces.get(sf_id)
            if sf and sf.grounded_to == node_id:
                sf.grounded_to = None
            self.has_surface_form.pop((node_id, sf_id), None)
            self._surface_to_concept.pop(sf_id, None)
        self._concept_to_surfaces.pop(node_id, None)
        # 断开 ConceptEdge
        edge_keys = [k for k in self.edges if k[0] == node_id or k[1] == node_id]
        for k in edge_keys:
            self.edges.pop(k, None)
        self._adjacency.pop(node_id, None)
        for adj_list in self._adjacency.values():
            adj_list[:] = [(t, et) for t, et in adj_list if t != node_id]

    # ── SurfaceFormNode 操作 ─────────────────────────────────────

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

        ema_alpha: 0 = 不更新 (只读), 1.0 = 完全用新值, 0.15 = 默认 (跟 v1 LiteracyTrainer 一致).
        image_paths: dict {type: path} (见 SurfaceFormNode.image_paths), 非 None 时合并到 sf.image_paths (新 key 覆盖旧).
        """
        feat = centroid.detach().float().view(-1).cpu()
        if feat.shape[-1] != self.feat_dim:
            raise ValueError(
                f"SurfaceFormNode centroid dim {feat.shape[-1]} != feat_dim {self.feat_dim}"
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

    # ── has_surface_form 边 (D87 核心) ───────────────────────────

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
                weight=min(old.weight + 0.1, _MAX_EDGE_WEIGHT),
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

    # ── ProcedureNode 操作 (D89, P1 只支持 register/get) ──────────

    def register_procedure(
        self,
        name: str,
        version: int = 1,
        *,
        inline_source: str | None = None,
        module_path: str | None = None,
        entry_symbol: str = "run",
        grounded_to: Iterable[str] | None = None,
        provenance: str = "bootstrap",
        instruction_text: str | None = None,
        tick: int = 0,
        acceptance_tests: Iterable[str] | None = None,
    ) -> ProcedureNode:
        """注册 ProcedureNode (D89).

        P1 阶段只做**存储**: 不加载代码, 不执行, 不跑 acceptance_tests. 这些都等 P2
        起点的 ``mind/core/muscle/skill_registry.py`` 接管 (见 language/ROADMAP §P2).

        Args:
          name: skill 名, 唯一到 (name, version) 二元组
          version: 版本号; skill bug 修复永远造新版本 + supersedes 边, 不改 v1
          inline_source / module_path: 代码引用 (二选一)
          grounded_to: 这个 skill 关联到哪些 concept_ids
          provenance: bootstrap / taught / chunked / synthesized

        Raises:
          ValueError: inline_source 和 module_path 都是 None (至少一个必须给)
        """
        if inline_source is None and module_path is None:
            raise ValueError(
                "ProcedureNode needs at least one of inline_source / module_path"
            )
        if provenance not in {"bootstrap", "taught", "chunked", "synthesized"}:
            raise ValueError(f"Unknown provenance: {provenance}")

        node_id = ProcedureNode.make_id(name, version)
        if node_id in self.procedures:
            # 重复 register 同 id: 只更新 last_tick (幂等), 不覆盖代码 (code 必须换版本)
            p = self.procedures[node_id]
            p.last_tick = tick
            return p

        grounded_list = list(grounded_to) if grounded_to else []
        tests_list = list(acceptance_tests) if acceptance_tests else []

        p = ProcedureNode(
            node_id=node_id,
            name=name,
            version=version,
            inline_source=inline_source,
            module_path=module_path,
            entry_symbol=entry_symbol,
            grounded_to=grounded_list,
            provenance=provenance,
            instruction_text=instruction_text,
            learned_at_tick=tick,
            last_tick=tick,
            acceptance_tests=tests_list,
            # bootstrap 默认 verified: 人写的 trusted primitive, 不需要跑 acceptance
            # (但 taught / chunked / synthesized 默认 verified=False, 等过了 tests 再改)
            verified=(provenance == "bootstrap"),
            trust_stage="autonomous" if provenance == "bootstrap" else "cognitive",
        )
        self.procedures[node_id] = p
        for cid in grounded_list:
            if node_id not in self._concept_to_procedures[cid]:
                self._concept_to_procedures[cid].append(node_id)
        return p

    def get_procedure(self, node_id: str) -> ProcedureNode | None:
        """按 id 取 skill; 找不到返回 None."""
        return self.procedures.get(node_id)

    def procedures_for_concept(self, concept_id: str) -> list[ProcedureNode]:
        """查某 concept 下挂了哪些 skill (has_procedure 反向查询)."""
        ids = self._concept_to_procedures.get(concept_id, [])
        return [self.procedures[i] for i in ids if i in self.procedures]

    # ── ConceptNode → ConceptNode 边 (v1 4 类借) ────────────────

    def add_edge(
        self,
        src_id: str,
        tgt_id: str,
        edge_type: str,
        tick: int = 0,
        weight_delta: float = 0.1,
    ) -> ConceptEdge:
        """ConceptNode → ConceptNode 边 (co_occurrence/temporal/spatial_adjacency/character_pair)."""
        if src_id not in self.concepts or tgt_id not in self.concepts:
            raise KeyError(f"Both concept nodes must exist: {src_id}, {tgt_id}")

        key = (src_id, tgt_id, edge_type)
        if key in self.edges:
            edge = self.edges[key]
            edge.weight = min(edge.weight + weight_delta, _MAX_EDGE_WEIGHT)
            edge.count += 1
            edge.last_tick = tick
            return edge

        if len(self.edges) >= self.max_edges:
            self._evict_weakest_edge()

        edge = ConceptEdge(
            source_id=src_id,
            target_id=tgt_id,
            weight=weight_delta,
            edge_type=edge_type,
            count=1,
            last_tick=tick,
        )
        self.edges[key] = edge
        self._adjacency[src_id].append((tgt_id, edge_type))
        return edge

    def _evict_weakest_edge(self) -> None:
        if not self.edges:
            return
        weakest = min(self.edges, key=lambda k: self.edges[k].weight)
        edge = self.edges.pop(weakest)
        adj = self._adjacency.get(edge.source_id, [])
        adj[:] = [
            (t, et) for t, et in adj
            if not (t == edge.target_id and et == edge.edge_type)
        ]

    # ── 查询 ─────────────────────────────────────────────────────

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

    # ── 统计 / debug dump ────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        live_concepts = sum(1 for n in self.concepts.values() if n.liveness() > 0)
        plural_concepts = sum(1 for n in self.concepts.values() if n.liveness() >= 2)
        total_bundle_params = sum(
            n.bundle.n_parameters() for n in self.concepts.values()
        )
        total_collapses = sum(
            n.bundle.n_collapses() for n in self.concepts.values()
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
        }

    # ── D91/D92: bundle 参数聚合 ──────────────────────────────────

    def iter_bundle_parameters(self) -> Iterable:
        """给 optimizer 用: 遍历所有 ConceptNode.bundle 的 nn.Parameter.

        非空 bundle 才产出; 空 bundle (liveness=0) 无参数, 自动跳过.
        """
        for node in self.concepts.values():
            yield from node.bundle.parameters()

    def bundles_to(self, device: torch.device | str) -> None:
        """批量把所有 bundle 搬到目标 device."""
        for node in self.concepts.values():
            node.bundle.to(device)

    def attribution_report(self) -> dict[str, dict]:
        """D91 归因 + D92 语义史合并报告.

        返回 ``{"by_concept": {cid: bundle.describe()}, "by_caller": {caller: {...}}}``.
        """
        by_concept = {
            cid: node.bundle.describe()
            for cid, node in self.concepts.items()
            if node.bundle.facets()
        }
        by_caller: dict[str, dict] = {}
        for cid, node in self.concepts.items():
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
                        else None  # 过长不写 JSON (避免图谱爆炸), 按 module_path 定位
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

        用 `tmp + rename` 保证 viewer polling / 其他 reader **永远读不到半写**
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
