"""Universal training loop reused by all four purity-audit assays."""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from pcm.heads.arithmetic_head_v2 import ArithmeticHeadV2
from pcm.heads.comparison_head import ComparisonHead
from pcm.heads.numerosity_encoder import DatasetConfig
from pcm.sleep import SleepConfig, attach_sleep, run_sleep_pass

from experiments.robustness_study import (
    BATCH_SIZE,
    BIAS_DIM,
    DEVICE,
    EPOCHS,
    LR,
    ORD_DIM,
    STEPS_PER_EPOCH,
    _op_onehot,
    _sample_arith,
    _sample_cmp,
)

from .graph_builder import build_graph_with_id_fn


__all__ = ["purity_train_one"]


def purity_train_one(
    mode: str,                                    # "single" / "dual"
    seed: int,
    cfg: DatasetConfig,
    centroids: torch.Tensor,
    *,
    id_fn: Callable[[int], str] | None = None,
    shuffle_map: dict[int, int] | None = None,
    init_strategy: str = "normal_small",
    sleep_every: int | None = None,  # Tier-G: run sleep pass every N epochs
) -> dict:
    """通用单轮训练. 返回 bundle_state (按 ``n`` 键存) 方便后续各种度量.

    Returns:
      dict:
        - mode, seed
        - add_acc (final)
        - bundle_by_n: {n: {"arithmetic_bias": tensor, "ordinal_offset": tensor?}}
          (按自然数 n 索引, 不依赖 concept_id 具体形式)
        - id_map: {n: concept_id_used}
        - shuffled: bool
    """
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    id_fn = id_fn or (lambda n: f"concept:ans:{n}")
    cg, id_map = build_graph_with_id_fn(cfg.n_min, cfg.n_max, id_fn)

    head_add = ArithmeticHeadV2(embed_dim=128, bias_dim=BIAS_DIM).to(DEVICE)
    head_cmp: ComparisonHead | None = None
    if mode == "dual":
        head_cmp = ComparisonHead(
            embed_dim=128, facet_dim=ORD_DIM, hidden_dim=64
        ).to(DEVICE)

    with torch.no_grad():
        for n in range(cfg.n_min, cfg.n_max + 1):
            c = cg.concepts[id_map[n]]
            c.collapse("ArithmeticHeadV2", "arithmetic_bias",
                       (BIAS_DIM,), tick=0, device=DEVICE, init=init_strategy)
            if mode == "dual":
                c.collapse("ComparisonHead", "ordinal_offset",
                           (ORD_DIM,), tick=0, device=DEVICE, init=init_strategy)
    cg.bundles_to(torch.device(DEVICE))

    params = list(head_add.parameters()) + list(cg.iter_bundle_parameters())
    if head_cmp is not None:
        params = (list(head_add.parameters())
                  + list(head_cmp.parameters())
                  + list(cg.iter_bundle_parameters()))
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    sleep_facets = ["arithmetic_bias"] + (["ordinal_offset"] if head_cmp is not None else [])
    if sleep_every is not None:
        attach_sleep(cg, facets=sleep_facets)
    sleep_reports: list[dict] = []

    def map_ids(ns_list: list[int]) -> list[str]:
        if shuffle_map is None:
            return [id_map[n] for n in ns_list]
        return [id_map[shuffle_map[n]] for n in ns_list]

    for epoch in range(1, EPOCHS + 1):
        head_add.train()
        if head_cmp is not None:
            head_cmp.train()
        for step in range(STEPS_PER_EPOCH):
            a_l, b_l, op_l, c_l = _sample_arith(cfg, BATCH_SIZE, rng)
            ids_a = map_ids(a_l)
            ids_b = map_ids(b_l)
            op = _op_onehot(op_l)
            tgt = torch.tensor(c_l, device=DEVICE) - cfg.n_min
            dummy = torch.zeros(BATCH_SIZE, 128, device=DEVICE)
            pred = head_add(dummy, dummy, op, ids_a, ids_b, cg,
                            tick=epoch * 10000 + step)
            la = F.cross_entropy(pred @ centroids.t(), tgt)
            total_loss = la
            if head_cmp is not None:
                ca_l, cb_l, clab = _sample_cmp(cfg, BATCH_SIZE, rng)
                cids_a = map_ids(ca_l)
                cids_b = map_ids(cb_l)
                ctgt = torch.tensor(clab, device=DEVICE)
                logits_c = head_cmp(None, None, cids_a, cids_b, cg,
                                    tick=epoch * 10000 + step)
                total_loss = la + F.cross_entropy(logits_c, ctgt)
            opt.zero_grad(); total_loss.backward(); opt.step()

        if sleep_every is not None and epoch % sleep_every == 0:
            report = run_sleep_pass(
                cg,
                optimizer=opt,
                facets=sleep_facets,
                config=SleepConfig(k_clusters="auto", replay_steps=0,
                                   seed=seed + epoch),
                tick=epoch * 10000 + 9999,
            )
            sleep_reports.append(report.to_dict())

    head_add.eval()
    if head_cmp is not None:
        head_cmp.eval()
    hits = total_n = 0
    with torch.no_grad():
        for _ in range(50):
            a_l, b_l, op_l, c_l = _sample_arith(cfg, 20, rng)
            ids_a = map_ids(a_l)
            ids_b = map_ids(b_l)
            op = _op_onehot(op_l)
            tgt = torch.tensor(c_l, device=DEVICE) - cfg.n_min
            dummy = torch.zeros(20, 128, device=DEVICE)
            pred = head_add(dummy, dummy, op, ids_a, ids_b, cg)
            hits += (pred @ centroids.t()).argmax(-1).eq(tgt).sum().item()
            total_n += 20
    add_acc = hits / max(total_n, 1)

    bundle_by_n: dict[int, dict] = {}
    for n in range(cfg.n_min, cfg.n_max + 1):
        cid = id_map[n]
        state = cg.concepts[cid].bundle.state_dict()
        entry: dict = {}
        for k, v in state.items():
            facet = k.split(".", 1)[-1]
            entry[facet] = v.detach().cpu().clone()
        bundle_by_n[n] = entry

    return {
        "mode": mode,
        "seed": seed,
        "shuffled": shuffle_map is not None,
        "init": init_strategy,
        "add_acc": add_acc,
        "bundle_by_n": bundle_by_n,
        "id_map": {str(n): cid for n, cid in id_map.items()},
        "sleep_reports": sleep_reports,
    }
