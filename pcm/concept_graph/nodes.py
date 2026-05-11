r"""Dataclass schema for the ConceptGraph package.

Lifted from the pre-split single-file ``pcm/concept_graph.py`` (line 60-289)
without behavioural changes:

- :class:`SurfaceFormNode` — D87 spoke (modality-specific representation).
- :class:`ConceptNode`     — D85 hub (transmodal concept) + D91/D92/D93 bundle
                             proxy + D94 cook ``kind``/``metadata``.
- :class:`HasSurfaceFormEdge` — concept ↔ surface edge.
- :class:`ConceptEdge`        — concept ↔ concept edge (4 kinds).
- :class:`ProcedureNode`      — D89 procedural memory schema.

Also re-exports the module-level constants (``PROTECTED_SCOPES`` and the
``_DEFAULT_*`` capacity defaults) so downstream mixins can import them
from a single location.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import Tensor

from ..param_bundle import ContextualizedConcept, ParamBundle


__all__ = [
    "SurfaceFormNode",
    "ConceptNode",
    "HasSurfaceFormEdge",
    "ConceptEdge",
    "ProcedureNode",
    "PROTECTED_SCOPES",
    "DEFAULT_FEAT_DIM",
    "DEFAULT_MAX_NODES",
    "DEFAULT_MAX_SURFACES",
    "DEFAULT_MAX_EDGES",
    "MAX_EDGE_WEIGHT",
    "MAX_PROVENANCE_PER_NODE",
]


# ============================================================================
# Module-level constants (D86 protected scopes + capacity defaults).
# ============================================================================

PROTECTED_SCOPES = frozenset({"BASE", "CORE"})
"""Scopes whose concepts may NOT be evicted (D86)."""

DEFAULT_FEAT_DIM: int = 128
DEFAULT_MAX_NODES: int = 10_000
DEFAULT_MAX_SURFACES: int = 50_000
DEFAULT_MAX_EDGES: int = 100_000
MAX_EDGE_WEIGHT: float = 10.0
MAX_PROVENANCE_PER_NODE: int = 50
"""Cap on per-node provenance list length (PHILOSOPHY §7 anti-bloat)."""


# ============================================================================
# SurfaceFormNode — D87 modality-specific spoke
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


# ============================================================================
# ConceptNode — D85 transmodal hub (with D91/D92/D93/D94 extensions)
# ============================================================================


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

    D93 Pre-allocated Dense Pool:
        ``bundle`` 不再 own 张量, 而是指向 ``ConceptGraph.bundle_pool[facet]``
        的 (slot_idx) 行视图. 这把 D91 storage 变成稠密 pool, 解锁 GPU 并行
        ``F.embedding`` lookup 与 bit-identical 容量 grow (G1-G6 不变量).

    D94 Concept-as-Function:
        ``kind`` discriminates plain data nodes from Houdini-style executable
        subgraphs; ``metadata`` carries the op DAG when
        ``kind == "parametric_muscle_subgraph"``. See
        ``docs/PCM_NODE_AS_FUNCTION_DESIGN.md`` for the schema.
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
    # D91/D92/D93: 指向 graph 共享 pool 的 bundle 代理 (graph.register_concept 时绑定).
    bundle: ParamBundle = field(default=None)  # type: ignore[assignment]
    # D94: cook semantics.
    kind: str = "data"
    metadata: dict = field(default_factory=dict)

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
            1. 若 ``facet`` 未初始化, lazy create 整个 pool (capacity x shape);
               若该 (facet, slot) 是首次, 按 ``init`` 策略 in-place 初始化此行
            2. 注册 ``caller`` 为该 (facet, slot) 的 consumer (D91 归因)
            3. 追加一条 ``collapse_history[facet]`` 事件 (D92 语义史)

        返回 ephemeral ``ContextualizedConcept``, 携带当下语义参数 (可求导).
        """
        if self.bundle is None:
            raise RuntimeError(
                f"ConceptNode {self.node_id!r} has no bundle bound; "
                "did you create it without going through ConceptGraph.register_concept?"
            )
        return self.bundle.request(
            facet, shape, caller,
            concept_id=self.node_id, tick=tick,
            init=init, device=device,
        )

    def liveness(self) -> int:
        """D92: L(v) = 跨所有 facet 的 unique consumer 数 (死概念 = 0)."""
        if self.bundle is None:
            return 0
        return self.bundle.liveness()


# ============================================================================
# Edge dataclasses
# ============================================================================


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
