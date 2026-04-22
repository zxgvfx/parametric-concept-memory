"""param_bundle.py — D91/D92 核心原语: ConceptNode 的可学参数池 + 塌缩对象.

本文件实装两条研究决策:

- **D91 Parametric Concept Memory**: 每个 ConceptNode 挂一个 multi-facet 参数池,
  被多个肌肉按 facet name 消费, 梯度直接回流, ``consumed_by`` 自动归因.
- **D92 Contextual Concept Collapse**: ConceptNode 无 "当下状态", 只在被 caller ×
  facet 观测 (``collapse``) 时产生一个 ephemeral ``ContextualizedConcept`` —
  那是该 concept 当下对该 caller 的**全部语义**.

两者共用同一底层结构 (``ParamBundle``), 但语义边界清晰: D91 是工程
("参数放哪"), D92 是本体论 ("concept 本身是什么").

## 不变量 (PHILOSOPHY 红线)

- ``ContextualizedConcept`` ephemeral, 不持久化, 不进 ``to_dict``
- ``ParamBundle`` 参数 lazy init (未被 request 前不占空间)
- ``collapse_history`` 只记录事件 (caller, facet, tick), 不存 tensor
- 任何肌肉 forward 期间产生的 CC **必须**经 ``ConceptNode.collapse`` 创建, 不得
  绕道直接 ``node.bundle.params["..."]``
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

InitStrategy = Literal["zero", "normal_small", "normal", "identity"]

# ============================================================================
# ContextualizedConcept — D92 ephemeral collapse 对象
# ============================================================================


@dataclass
class ContextualizedConcept:
    """concept 在某次 caller × facet 观测下的具体塌缩 (D92).

    携带该 concept 对 caller 的**全部当下语义**. Ephemeral —
    每次 ``ConceptNode.collapse`` 产生一个, forward 结束后丢弃.
    """

    concept_id: str
    caller: str
    facet: str
    facet_params: torch.Tensor   # 保留 grad; 指向 ParamBundle 里的 nn.Parameter
    tick: int = 0

    def as_tensor(self) -> torch.Tensor:
        return self.facet_params

    def __repr__(self) -> str:
        return (
            f"CC(id={self.concept_id}, caller={self.caller}, facet={self.facet}, "
            f"shape={tuple(self.facet_params.shape)}, tick={self.tick})"
        )


# ============================================================================
# ParamBundle — D91 底层
# ============================================================================


class ParamBundle(nn.Module):
    """ConceptNode 的 multi-facet 可学参数池 (D91 底层).

    每个 facet 是 ``nn.Parameter``, 首次 ``request`` 时按调用方 shape lazy init.

    三个字典:

    - ``self.params`` (nn.ParameterDict): facet_name → Parameter
    - ``self.consumed_by`` (dict[str, set[str]]): facet_name → {caller_names}
    - ``self.collapse_history`` (dict[str, list[(caller, tick)]]): facet_name → 事件流

    ``consumed_by`` 和 ``collapse_history`` **不是 nn.Module 状态**, 不进
    state_dict. 它们是 Python 字典, 随实例存在.
    """

    def __init__(self) -> None:
        super().__init__()
        self.params: nn.ParameterDict = nn.ParameterDict()
        self.consumed_by: dict[str, set[str]] = defaultdict(set)
        self.collapse_history: dict[str, list[tuple[str, int]]] = defaultdict(list)

    # ── 核心 API ──────────────────────────────────────────────

    def request(
        self,
        facet: str,
        shape: Iterable[int],
        caller: str,
        concept_id: str = "",
        tick: int = 0,
        init: InitStrategy = "normal_small",
        device: torch.device | str | None = None,
    ) -> ContextualizedConcept:
        """按 (caller, facet) 拿 (必要时创建) 一个 facet 参数, 返回 ContextualizedConcept.

        副作用:
          1. 若 ``facet`` 不存在, 按 ``shape`` 和 ``init`` lazy 创建 nn.Parameter
          2. 注册 ``caller`` 为该 facet 的 consumer
          3. 把 (caller, tick) 加到 collapse_history[facet]
          4. 返回 ContextualizedConcept (D92 ephemeral)
        """
        if not facet or not facet.replace("_", "").isalnum():
            raise ValueError(
                f"facet name must be alnum+underscore (got {facet!r}); "
                "ParameterDict key restriction."
            )
        if facet not in self.params:
            shape_t = tuple(int(s) for s in shape)
            p = _init_parameter(shape_t, init)
            if device is not None:
                p = nn.Parameter(p.data.to(device), requires_grad=p.requires_grad)
            self.params[facet] = p
        else:
            existing = tuple(self.params[facet].shape)
            req = tuple(int(s) for s in shape)
            if existing != req:
                raise ValueError(
                    f"facet {facet!r} already initialized with shape {existing}, "
                    f"caller {caller!r} requested {req}"
                )
            if device is not None and self.params[facet].device != torch.device(device):
                self.params[facet].data = self.params[facet].data.to(device)
        self.consumed_by[facet].add(caller)
        self.collapse_history[facet].append((caller, int(tick)))
        return ContextualizedConcept(
            concept_id=concept_id,
            caller=caller,
            facet=facet,
            facet_params=self.params[facet],
            tick=int(tick),
        )

    def ablate(self, facet: str) -> None:
        """把某个 facet 参数原地清零 (归因实证用)."""
        if facet in self.params:
            with torch.no_grad():
                self.params[facet].zero_()

    def remove(self, facet: str) -> None:
        """完全移除一个 facet (调试/剪枝用)."""
        if facet in self.params:
            del self.params[facet]
        self.consumed_by.pop(facet, None)
        self.collapse_history.pop(facet, None)

    # ── 查询 ──────────────────────────────────────────────────

    def facets(self) -> list[str]:
        return list(self.params.keys())

    def consumers(self) -> set[str]:
        """跨所有 facet 的 union consumer (D92 liveness L(v))."""
        out: set[str] = set()
        for s in self.consumed_by.values():
            out |= s
        return out

    def liveness(self) -> int:
        """D92: L(v) = |unique consumers across all facets|.

        int 语义 (匹配 ckpt artifact attribution_report.json / history.json):
          0 → void; 1 → single; >=2 → plural. 上层可按阈值分类.
        """
        return len(self.consumers())

    def n_collapses(self) -> int:
        return sum(len(v) for v in self.collapse_history.values())

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.params.values())

    # ── 序列化 helper ────────────────────────────────────────

    def describe(self, recent_events: int = 20) -> dict:
        """给 viewer / 日志用: 元数据 (不含 tensor 值).

        Args:
            recent_events: 每 facet 在 ``facets[*].recent_collapses`` 里携带的
                最近塌缩事件数 (caller, tick). 0 表示不带事件. 默认 20 以适配
                viewer 时间线显示.
        """
        facets: dict[str, dict] = {}
        for name in self.params.keys():
            hist = self.collapse_history.get(name, [])
            tail = list(hist[-recent_events:]) if recent_events > 0 else []
            facets[name] = {
                "shape": tuple(self.params[name].shape),
                "n_params": self.params[name].numel(),
                "consumers": sorted(self.consumed_by.get(name, set())),
                "n_collapses": len(hist),
                "recent_collapses": [
                    {"caller": c, "tick": int(t)} for (c, t) in tail
                ],
            }
        return {
            "facets": facets,
            "n_facets": len(self.params),
            "n_params_total": self.n_parameters(),
            "liveness": self.liveness(),
            "n_collapses_total": self.n_collapses(),
        }


# ============================================================================
# 初始化策略 (纯函数, 方便 H4 严格 zero-init 实验)
# ============================================================================


def _init_parameter(shape: tuple[int, ...], strategy: InitStrategy) -> nn.Parameter:
    if strategy == "zero":
        t = torch.zeros(*shape)
    elif strategy == "normal_small":
        t = torch.randn(*shape) * 0.01
    elif strategy == "normal":
        t = torch.randn(*shape)
    elif strategy == "identity":
        t = torch.ones(*shape)
    else:
        raise ValueError(f"unknown init strategy: {strategy}")
    return nn.Parameter(t)


# ============================================================================
# 便利: 聚合多节点 bundle 参数 (给 optimizer)
# ============================================================================


def iter_bundle_parameters(bundles: Iterable[ParamBundle]) -> Iterator[nn.Parameter]:
    """遍历多个 bundle 的所有参数 (给 optim.AdamW 之类)."""
    for b in bundles:
        yield from b.parameters()


def aggregate_consumed_by(
    bundles: dict[str, ParamBundle],
) -> dict[str, dict[str, set[str]]]:
    """跨多 bundle 合并归因 (concept_id × facet × {callers}). 诊断用."""
    return {
        cid: {name: set(callers) for name, callers in b.consumed_by.items()}
        for cid, b in bundles.items()
    }
