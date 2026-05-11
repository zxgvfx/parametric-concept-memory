"""Training loop for the 5×5 spatial study (single + dual modes)."""
from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn.functional as F

from pcm.concept_graph import ConceptGraph
from pcm.sleep import (
    SleepConfig,
    attach_sleep,
    materialize_effective_bundle_state,
    run_sleep_pass,
)

from ._config import (
    BATCH_SIZE,
    CALLER_DIST,
    CALLER_MOVE,
    DEVICE,
    DIST_DIM,
    EMBED_DIM,
    EPOCHS,
    FACET_DIST,
    FACET_MOVE,
    LR,
    MOTION_DIM,
    N_COLS,
    N_ROWS,
    STEPS_PER_EPOCH,
)
from .heads import DistanceHead, MoveHead
from .topology import (
    cid_of,
    enumerate_distance_triples,
    enumerate_move_triples,
    idx_of_rc,
    rc_of_idx,
)


__all__ = ["build_space_graph", "train_one"]


def build_space_graph() -> ConceptGraph:
    cg = ConceptGraph(feat_dim=EMBED_DIM)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            cg.register_concept(
                node_id=cid_of(r, c),
                label=f"SPACE_{r}_{c}",
                scope="BASE",
                provenance=f"space_study:r={r},c={c}",
            )
    return cg


def _apply_shuffle(ids: list[str], sm: dict[int, int] | None) -> list[str]:
    """shuffle map: flattened_index (0..24) → shuffled_flattened_index."""
    if sm is None:
        return ids
    out = []
    for cid in ids:
        _, rc = cid.rsplit(":", 1)
        r_s, c_s = rc.split("_")
        idx = idx_of_rc(int(r_s), int(c_s))
        new_idx = sm[idx]
        nr, nc = rc_of_idx(new_idx)
        out.append(cid_of(nr, nc))
    return out


def train_one(
    mode: str,                        # "single" (move only) or "dual"
    seed: int,
    shuffle_map: dict[int, int] | None = None,
    epochs: int = EPOCHS,
    steps_per_epoch: int = STEPS_PER_EPOCH,
    ood_ratio: float = 0.0,  # F5: hold out fraction of move triples for OOD
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

    move_triples_all = enumerate_move_triples()
    dist_triples = enumerate_distance_triples()
    if ood_ratio > 0:
        rng_split = random.Random(seed + 7777)
        shuffled = list(move_triples_all)
        rng_split.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * ood_ratio))
        move_test_triples = shuffled[:n_test]
        move_triples = shuffled[n_test:]
    else:
        move_test_triples = []
        move_triples = move_triples_all

    cg = build_space_graph()
    head_move = MoveHead(
        use_abstract=use_abstract,
        assignment=sleep_assignment,
        soft_tau=sleep_soft_tau,
    ).to(DEVICE)
    head_dist: DistanceHead | None = None
    if mode == "dual":
        head_dist = DistanceHead(
            use_abstract=use_abstract,
            assignment=sleep_assignment,
            soft_tau=sleep_soft_tau,
        ).to(DEVICE)

    with torch.no_grad():
        for r in range(N_ROWS):
            for c in range(N_COLS):
                cn = cg.concepts[cid_of(r, c)]
                cn.collapse(CALLER_MOVE, FACET_MOVE, (MOTION_DIM,),
                            tick=0, device=DEVICE, init="normal_small")
                if mode == "dual":
                    cn.collapse(CALLER_DIST, FACET_DIST, (DIST_DIM,),
                                tick=0, device=DEVICE, init="normal_small")
    cg.bundles_to(torch.device(DEVICE))

    params = list(head_move.parameters()) + list(cg.iter_bundle_parameters())
    if head_dist is not None:
        params = (list(head_move.parameters())
                  + list(head_dist.parameters())
                  + list(cg.iter_bundle_parameters()))
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    sleep_facets = [FACET_MOVE] + ([FACET_DIST] if head_dist is not None else [])
    if sleep_every is not None:
        attach_sleep(cg, facets=sleep_facets)
    sleep_reports: list[dict] = []

    for epoch in range(1, epochs + 1):
        head_move.train()
        if head_dist is not None:
            head_dist.train()
        for step_i in range(steps_per_epoch):
            batch = [move_triples[rng.randrange(len(move_triples))]
                     for _ in range(BATCH_SIZE)]
            ids_a = _apply_shuffle([cid_of(*t[0]) for t in batch], shuffle_map)
            ids_b = _apply_shuffle([cid_of(*t[1]) for t in batch], shuffle_map)
            tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
            pred = head_move(ids_a, ids_b, cg, tick=epoch * 10000 + step_i)
            loss_m = F.cross_entropy(pred, tgt)
            total = loss_m

            if head_dist is not None:
                batch2 = [dist_triples[rng.randrange(len(dist_triples))]
                          for _ in range(BATCH_SIZE)]
                ids_a2 = _apply_shuffle(
                    [cid_of(*t[0]) for t in batch2], shuffle_map)
                ids_b2 = _apply_shuffle(
                    [cid_of(*t[1]) for t in batch2], shuffle_map)
                tgt2 = torch.tensor([t[2] for t in batch2], device=DEVICE)
                logits = head_dist(ids_a2, ids_b2, cg,
                                   tick=epoch * 10000 + step_i)
                total = total + F.cross_entropy(logits, tgt2)

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

    head_move.eval()
    if head_dist is not None:
        head_dist.eval()

    def _eval_move(triples: list) -> float:
        if not triples:
            return float("nan")
        hits = 0
        with torch.no_grad():
            for i in range(0, len(triples), 64):
                batch = triples[i:i + 64]
                ids_a = _apply_shuffle([cid_of(*t[0]) for t in batch], shuffle_map)
                ids_b = _apply_shuffle([cid_of(*t[1]) for t in batch], shuffle_map)
                tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
                pred = head_move(ids_a, ids_b, cg)
                hits += pred.argmax(-1).eq(tgt).sum().item()
        return hits / len(triples)

    move_acc = _eval_move(move_triples)
    move_ood_acc = _eval_move(move_test_triples) if move_test_triples else None

    dist_acc: Optional[float] = None
    if head_dist is not None:
        hits_d = 0
        with torch.no_grad():
            for i in range(0, len(dist_triples), 64):
                batch = dist_triples[i:i + 64]
                ids_a = _apply_shuffle(
                    [cid_of(*t[0]) for t in batch], shuffle_map)
                ids_b = _apply_shuffle(
                    [cid_of(*t[1]) for t in batch], shuffle_map)
                tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
                logits = head_dist(ids_a, ids_b, cg)
                hits_d += logits.argmax(-1).eq(tgt).sum().item()
        dist_acc = hits_d / len(dist_triples)

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
        "mode": mode, "seed": seed,
        "shuffled": shuffle_map is not None,
        "move_acc": move_acc, "move_ood_acc": move_ood_acc,
        "dist_acc": dist_acc,
        "bundle_state": bundle_state,
        "sleep_reports": sleep_reports,
    }
