r"""ProcedureNode mixin (D89 procedural memory).

Owns ``register_procedure`` / ``get_procedure`` /
``procedures_for_concept``. P1 stage: only stores schema; no executor /
verifier. P2 hooks live in ``mind/core/muscle/skill_registry``.
"""
from __future__ import annotations

from typing import Iterable

from .nodes import ProcedureNode


__all__ = ["_ProcedureMixin"]


_VALID_PROVENANCE = frozenset({"bootstrap", "taught", "chunked", "synthesized"})


class _ProcedureMixin:
    """ProcedureNode lifecycle (D89)."""

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

        P1 阶段只做**存储**: 不加载代码, 不执行, 不跑 acceptance_tests.
        这些都等 P2 起点的 ``mind/core/muscle/skill_registry.py`` 接管
        (见 ``language/ROADMAP §P2``).

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
        if provenance not in _VALID_PROVENANCE:
            raise ValueError(f"Unknown provenance: {provenance}")

        node_id = ProcedureNode.make_id(name, version)
        if node_id in self.procedures:
            # 重复 register 同 id: 只更新 last_tick (幂等), 不覆盖代码
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
