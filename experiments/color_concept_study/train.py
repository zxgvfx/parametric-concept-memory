"""Training loop for the color study (single = mix only / dual = mix+adj)."""
from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn.functional as F

from pcm.sleep import (
    SleepConfig,
    attach_sleep,
    materialize_effective_bundle_state,
    run_sleep_pass,
)

from ._config import (
    ADJ_DIM,
    BATCH_SIZE,
    BIAS_DIM,
    CALLER_ADJ,
    CALLER_MIX,
    DEVICE,
    EPOCHS,
    FACET_ADJ,
    FACET_MIX,
    LR,
    N_COLORS,
    STEPS_PER_EPOCH,
)
from .graph_builder import _apply_shuffle, build_color_graph
from .heads import ColorAdjacencyHead, ColorMixingHead, RipeFruitHead
from .topology import enumerate_adjacency_triples, enumerate_mixing_triples


__all__ = ["train_one"]


def train_one(
    mode: str,                       # "single" (mix only) / "dual" (mix + adj)
    seed: int,
    centroids: torch.Tensor,
    shuffle_map: dict[int, int] | None = None,
    epochs: int = EPOCHS,
    steps_per_epoch: int = STEPS_PER_EPOCH,
    ood_ratio: float = 0.0,  # F5: hold out fraction of mix triples for OOD eval
    mix_sample_weight: list[float] | None = None,  # §6.8 C: non-uniform hue sampling
    enable_ripe_head: bool = False,  # §6.8 D: red-fruit foraging head
    ripe_set: tuple[int, ...] = (0, 1, 11),  # default red wedge
    holdout_target_hues: tuple[int, ...] = (),  # §7.5-color G3: hold out triples whose target c ∈ holdout
    sleep_every: int | None = None,  # Tier-G: run sleep pass every N epochs
    sleep_warmup: int = 0,
    sleep_force_recluster: bool = False,
    sleep_k_clusters: int | str = "auto",
    sleep_anchor_ema: float = 1.0,
    sleep_assignment: str = "hard",
    sleep_soft_tau: float = 0.5,
    use_abstract: bool = False,
) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    mix_triples_all = enumerate_mixing_triples(N_COLORS)
    adj_triples = enumerate_adjacency_triples(N_COLORS)

    # §7.5-color G3: hold out triples whose target hue c ∈ holdout_target_hues.
    # Those triples become the held-out test set; remaining go into training.
    holdout_set = set(int(h) for h in holdout_target_hues)
    mix_holdout_triples: list[tuple[int, int, int]] = []
    if holdout_set:
        kept: list[tuple[int, int, int]] = []
        for t in mix_triples_all:
            if t[2] in holdout_set:
                mix_holdout_triples.append(t)
            else:
                kept.append(t)
        mix_triples_all = kept

    if ood_ratio > 0:
        rng_split = random.Random(seed + 7777)
        shuffled = list(mix_triples_all)
        rng_split.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * ood_ratio))
        mix_test_triples = shuffled[:n_test]
        mix_triples = shuffled[n_test:]
    else:
        mix_test_triples = []
        mix_triples = mix_triples_all

    cg = build_color_graph(N_COLORS)
    head_mix = ColorMixingHead(
        use_abstract=use_abstract,
        assignment=sleep_assignment,
        soft_tau=sleep_soft_tau,
    ).to(DEVICE)
    head_adj: ColorAdjacencyHead | None = None
    if mode == "dual":
        head_adj = ColorAdjacencyHead(
            use_abstract=use_abstract,
            assignment=sleep_assignment,
            soft_tau=sleep_soft_tau,
        ).to(DEVICE)
    head_ripe: RipeFruitHead | None = None
    if enable_ripe_head:
        head_ripe = RipeFruitHead(
            use_abstract=use_abstract,
            assignment=sleep_assignment,
            soft_tau=sleep_soft_tau,
        ).to(DEVICE)
    ripe_lookup = {h: 1 for h in ripe_set}

    if mix_sample_weight is not None:
        if len(mix_sample_weight) != N_COLORS:
            raise ValueError(
                f"mix_sample_weight length must equal N_COLORS={N_COLORS}, "
                f"got {len(mix_sample_weight)}"
            )
        # Pre-compute per-triple sampling weight = w[a] * w[b].
        triple_weights = [
            float(mix_sample_weight[t[0]]) * float(mix_sample_weight[t[1]])
            for t in mix_triples
        ]
    else:
        triple_weights = None

    with torch.no_grad():
        for i in range(N_COLORS):
            c = cg.concepts[f"concept:color:{i}"]
            c.collapse(CALLER_MIX, FACET_MIX, (BIAS_DIM,),
                       tick=0, device=DEVICE, init="normal_small")
            if mode == "dual":
                c.collapse(CALLER_ADJ, FACET_ADJ, (ADJ_DIM,),
                           tick=0, device=DEVICE, init="normal_small")
    cg.bundles_to(torch.device(DEVICE))

    params = list(head_mix.parameters())
    if head_adj is not None:
        params += list(head_adj.parameters())
    if head_ripe is not None:
        params += list(head_ripe.parameters())
    params += list(cg.iter_bundle_parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    sleep_facets = [FACET_MIX] + ([FACET_ADJ] if head_adj is not None else [])
    if sleep_every is not None:
        attach_sleep(cg, facets=sleep_facets)
    sleep_reports: list[dict] = []

    for epoch in range(1, epochs + 1):
        head_mix.train()
        if head_adj is not None:
            head_adj.train()
        if head_ripe is not None:
            head_ripe.train()
        for step_i in range(steps_per_epoch):
            if triple_weights is not None:
                batch = rng.choices(mix_triples, weights=triple_weights,
                                    k=BATCH_SIZE)
            else:
                batch = [mix_triples[rng.randrange(len(mix_triples))]
                         for _ in range(BATCH_SIZE)]
            ids_a = _apply_shuffle(
                [f"concept:color:{t[0]}" for t in batch], shuffle_map)
            ids_b = _apply_shuffle(
                [f"concept:color:{t[1]}" for t in batch], shuffle_map)
            tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
            pred = head_mix(ids_a, ids_b, cg, tick=epoch * 10000 + step_i)
            loss_mix = F.cross_entropy(pred @ centroids.t(), tgt)
            total = loss_mix

            if head_adj is not None:
                batch2 = [adj_triples[rng.randrange(len(adj_triples))]
                          for _ in range(BATCH_SIZE)]
                ids_a2 = _apply_shuffle(
                    [f"concept:color:{t[0]}" for t in batch2], shuffle_map)
                ids_b2 = _apply_shuffle(
                    [f"concept:color:{t[1]}" for t in batch2], shuffle_map)
                tgt2 = torch.tensor([t[2] for t in batch2], device=DEVICE)
                logits = head_adj(ids_a2, ids_b2, cg,
                                  tick=epoch * 10000 + step_i)
                total = total + F.cross_entropy(logits, tgt2)

            if head_ripe is not None:
                ripe_batch = [rng.randrange(N_COLORS)
                              for _ in range(BATCH_SIZE)]
                ids_r = _apply_shuffle(
                    [f"concept:color:{i}" for i in ripe_batch], shuffle_map)
                tgt_r = torch.tensor(
                    [ripe_lookup.get(i, 0) for i in ripe_batch],
                    device=DEVICE,
                )
                logits_r = head_ripe(
                    ids_r, cg, tick=epoch * 10000 + step_i,
                )
                total = total + F.cross_entropy(logits_r, tgt_r)

            opt.zero_grad(); total.backward(); opt.step()

        if (
            sleep_every is not None
            and epoch > sleep_warmup
            and epoch % sleep_every == 0
        ):
            report = run_sleep_pass(
                cg,
                optimizer=opt,
                facets=sleep_facets,
                config=SleepConfig(
                    k_clusters=sleep_k_clusters, replay_steps=0,
                    seed=seed + epoch,
                    anchor_ema=sleep_anchor_ema,
                    assignment=sleep_assignment,
                    soft_tau=sleep_soft_tau,
                ),
                tick=epoch * 10000 + 9999,
                force_recluster=sleep_force_recluster,
            )
            sleep_reports.append(report.to_dict())

    head_mix.eval()
    if head_adj is not None:
        head_adj.eval()

    def _eval_mix(triples: list) -> float:
        if not triples:
            return float("nan")
        hits = 0
        with torch.no_grad():
            for i in range(0, len(triples), 64):
                batch = triples[i:i + 64]
                ids_a = _apply_shuffle(
                    [f"concept:color:{t[0]}" for t in batch], shuffle_map)
                ids_b = _apply_shuffle(
                    [f"concept:color:{t[1]}" for t in batch], shuffle_map)
                tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
                pred = head_mix(ids_a, ids_b, cg)
                hits += (pred @ centroids.t()).argmax(-1).eq(tgt).sum().item()
        return hits / len(triples)

    mix_acc = _eval_mix(mix_triples)
    mix_ood_acc = _eval_mix(mix_test_triples) if mix_test_triples else None
    mix_holdout_acc = (_eval_mix(mix_holdout_triples)
                       if mix_holdout_triples else None)

    adj_acc: Optional[float] = None
    if head_adj is not None:
        hits_adj = 0
        with torch.no_grad():
            for i in range(0, len(adj_triples), 64):
                batch = adj_triples[i:i + 64]
                ids_a = _apply_shuffle(
                    [f"concept:color:{t[0]}" for t in batch], shuffle_map)
                ids_b = _apply_shuffle(
                    [f"concept:color:{t[1]}" for t in batch], shuffle_map)
                tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
                logits = head_adj(ids_a, ids_b, cg)
                hits_adj += logits.argmax(-1).eq(tgt).sum().item()
        adj_acc = hits_adj / len(adj_triples)

    bundle_state = {
        cid: {k: v.detach().cpu() for k, v in c.bundle.state_dict().items()}
        for cid, c in cg.concepts.items()
    }
    bundle_state = materialize_effective_bundle_state(
        cg, bundle_state, facets=sleep_facets,
        use_abstract=use_abstract,
        assignment=sleep_assignment, soft_tau=sleep_soft_tau,
    )
    return {
        "mode": mode,
        "seed": seed,
        "shuffled": shuffle_map is not None,
        "mix_acc": mix_acc,
        "mix_ood_acc": mix_ood_acc,
        "mix_holdout_acc": mix_holdout_acc,
        "adj_acc": adj_acc,
        "bundle_state": bundle_state,
        "sleep_reports": sleep_reports,
    }
