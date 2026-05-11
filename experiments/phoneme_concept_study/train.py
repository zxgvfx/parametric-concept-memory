"""Training loop for the phoneme study (single-axis or triple-orthogonal)."""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from pcm.sleep import (
    SleepConfig,
    attach_sleep,
    materialize_effective_bundle_state,
    run_sleep_pass,
)

from ._config import (
    BATCH_SIZE,
    CALLER_M,
    CALLER_P,
    CALLER_V,
    DEVICE,
    EPOCHS,
    FACET_M,
    FACET_P,
    FACET_V,
    LR,
    MANNER_DIM,
    PLACE_DIM,
    STEPS_PER_EPOCH,
    VOICE_DIM,
)
from .heads import (
    _SingleInputHead,
    build_manner_head,
    build_place_head,
    build_voicing_head,
)
from .inventory import (
    N_PH,
    _apply_shuffle,
    build_phoneme_graph,
    cid_of,
    feat_of,
)


__all__ = ["train_one"]


def train_one(
    mode: str,                          # "single_v" / "single_m" / "single_p" / "triple"
    seed: int,
    shuffle_map: dict[int, int] | None = None,
    epochs: int = EPOCHS,
    steps_per_epoch: int = STEPS_PER_EPOCH,
    sleep_every: int | None = None,  # Tier-G: run sleep pass every N epochs
    sleep_warmup: int = 0,
    sleep_force_recluster: bool = False,
    sleep_k_clusters: int | str = "auto",
    sleep_anchor_ema: float = 1.0,
    sleep_assignment: str = "hard",
    sleep_soft_tau: float = 0.5,
    use_abstract: bool = False,
    # §6.9 cross-language transfer hooks
    source_indices: list[int] | None = None,  # phonemes seen by V/M/P heads
    sample_weight: list[float] | None = None,  # per-phoneme sampling weight (over source)
    centroid_init: dict[str, torch.Tensor] | None = None,  # facet → (N_PH, dim) tensor for B layer init
    enable_minimal_pair_head: bool = False,  # D layer: pair-input head sees full inventory
    minimal_pair_facet: str = "voice_bias",  # which facet the pair head consumes
) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    cg = build_phoneme_graph()
    heads: dict[str, _SingleInputHead] = {}
    head_kw = dict(
        use_abstract=use_abstract,
        assignment=sleep_assignment,
        soft_tau=sleep_soft_tau,
    )
    if mode in ("single_v", "triple"):
        heads["v"] = build_voicing_head(**head_kw).to(DEVICE)
    if mode in ("single_m", "triple"):
        heads["m"] = build_manner_head(**head_kw).to(DEVICE)
    if mode in ("single_p", "triple"):
        heads["p"] = build_place_head(**head_kw).to(DEVICE)

    # D layer: minimal-pair head trained on full inventory pairs.
    pair_head = None
    if enable_minimal_pair_head:
        from experiments.phoneme_transfer_priors import (
            MinimalPairHead,
            minimal_pair_label,
        )
        pair_facet_dim = {
            FACET_V: VOICE_DIM, FACET_M: MANNER_DIM, FACET_P: PLACE_DIM,
        }[minimal_pair_facet]
        pair_head = MinimalPairHead(
            facet=minimal_pair_facet, facet_dim=pair_facet_dim,
            **head_kw,
        ).to(DEVICE)

    # lazy-init facets by calling collapse once per phoneme per active head
    with torch.no_grad():
        for i in range(N_PH):
            cn = cg.concepts[cid_of(i)]
            if "v" in heads:
                cn.collapse(CALLER_V, FACET_V, (VOICE_DIM,),
                            tick=0, device=DEVICE, init="normal_small")
            if "m" in heads:
                cn.collapse(CALLER_M, FACET_M, (MANNER_DIM,),
                            tick=0, device=DEVICE, init="normal_small")
            if "p" in heads:
                cn.collapse(CALLER_P, FACET_P, (PLACE_DIM,),
                            tick=0, device=DEVICE, init="normal_small")
    cg.bundles_to(torch.device(DEVICE))

    # B layer: overwrite bundle rows with provided centroid init (per facet).
    if centroid_init is not None:
        with torch.no_grad():
            for facet, rows in centroid_init.items():
                if facet not in cg.bundle_pool:
                    continue
                pool = cg.bundle_pool[facet]
                rows_dev = rows.to(pool.device)
                if rows_dev.shape[0] != N_PH:
                    raise ValueError(
                        f"centroid_init[{facet!r}] must have N_PH={N_PH} rows, "
                        f"got {rows_dev.shape[0]}"
                    )
                # bundle_pool is dim-trimmed by the facet shape;
                # truncate centroid_init last-dim if necessary.
                target_dim = pool.shape[-1]
                rows_trim = rows_dev[..., :target_dim]
                for i in range(N_PH):
                    slot = cg.cid_to_slot[cid_of(i)]
                    pool.data[slot] = rows_trim[i]

    params: list = []
    for h in heads.values():
        params += list(h.parameters())
    if pair_head is not None:
        params += list(pair_head.parameters())
    params += list(cg.iter_bundle_parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    # Source-language phoneme pool for V/M/P training. Default = full inventory.
    src_idx = list(source_indices) if source_indices is not None else list(range(N_PH))
    if sample_weight is not None and len(sample_weight) != len(src_idx):
        raise ValueError(
            f"sample_weight length ({len(sample_weight)}) must match "
            f"source_indices length ({len(src_idx)})"
        )

    facet_by_key = {"v": FACET_V, "m": FACET_M, "p": FACET_P}
    sleep_facets = [facet_by_key[k] for k in heads]
    if sleep_every is not None:
        attach_sleep(cg, facets=sleep_facets)
    sleep_reports: list[dict] = []

    for epoch in range(1, epochs + 1):
        for h in heads.values():
            h.train()
        if pair_head is not None:
            pair_head.train()
        for step_i in range(steps_per_epoch):
            # V/M/P heads: sample only over the source-language subset.
            if sample_weight is not None:
                idx_batch = rng.choices(
                    src_idx, weights=sample_weight, k=BATCH_SIZE,
                )
            else:
                idx_batch = [src_idx[rng.randrange(len(src_idx))]
                             for _ in range(BATCH_SIZE)]
            ids = _apply_shuffle([cid_of(i) for i in idx_batch], shuffle_map)

            total = 0.0
            count = 0
            if "v" in heads:
                tgt = torch.tensor(
                    [feat_of(i)[0] for i in idx_batch], device=DEVICE
                )
                logits = heads["v"](ids, cg, tick=epoch * 10000 + step_i)
                total = total + F.cross_entropy(logits, tgt); count += 1
            if "m" in heads:
                tgt = torch.tensor(
                    [feat_of(i)[1] for i in idx_batch], device=DEVICE
                )
                logits = heads["m"](ids, cg, tick=epoch * 10000 + step_i)
                total = total + F.cross_entropy(logits, tgt); count += 1
            if "p" in heads:
                tgt = torch.tensor(
                    [feat_of(i)[2] for i in idx_batch], device=DEVICE
                )
                logits = heads["p"](ids, cg, tick=epoch * 10000 + step_i)
                total = total + F.cross_entropy(logits, tgt); count += 1

            # D layer: minimal-pair head trains on FULL inventory pairs
            # (source + target), exposing target bundle rows to
            # axis-relevant gradient via a same/diff-axis classifier.
            if pair_head is not None:
                from experiments.phoneme_transfer_priors import (
                    minimal_pair_label,
                )
                # Each pair: sample two random indices over full inventory.
                pa = [rng.randrange(N_PH) for _ in range(BATCH_SIZE)]
                pb = [rng.randrange(N_PH) for _ in range(BATCH_SIZE)]
                ids_a = _apply_shuffle(
                    [cid_of(i) for i in pa], shuffle_map)
                ids_b = _apply_shuffle(
                    [cid_of(i) for i in pb], shuffle_map)
                pair_tgt = torch.tensor(
                    [minimal_pair_label(feat_of(a), feat_of(b))
                     for a, b in zip(pa, pb)],
                    device=DEVICE,
                )
                pair_logits = pair_head(
                    ids_a, ids_b, cg, tick=epoch * 10000 + step_i,
                )
                total = total + F.cross_entropy(pair_logits, pair_tgt)

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

    for h in heads.values():
        h.eval()
    if pair_head is not None:
        pair_head.eval()

    src_set = set(src_idx)
    tgt_idx = [i for i in range(N_PH) if i not in src_set]

    accs: dict[str, float] = {}
    accs_source: dict[str, float] = {}
    accs_target: dict[str, float] = {}
    with torch.no_grad():
        all_ids = _apply_shuffle([cid_of(i) for i in range(N_PH)], shuffle_map)
        for key, h in heads.items():
            logits = h(all_ids, cg)
            axis = {"v": 0, "m": 1, "p": 2}[key]
            preds = logits.argmax(-1)
            full_tgt = torch.tensor(
                [feat_of(i)[axis] for i in range(N_PH)], device=DEVICE,
            )
            accs[key] = float(preds.eq(full_tgt).sum().item()) / N_PH
            if src_idx:
                src_mask = torch.tensor(
                    [i in src_set for i in range(N_PH)], device=DEVICE,
                )
                accs_source[key] = float(
                    preds[src_mask].eq(full_tgt[src_mask]).sum().item()
                ) / max(int(src_mask.sum().item()), 1)
            if tgt_idx:
                tgt_mask = torch.tensor(
                    [i not in src_set for i in range(N_PH)], device=DEVICE,
                )
                accs_target[key] = float(
                    preds[tgt_mask].eq(full_tgt[tgt_mask]).sum().item()
                ) / max(int(tgt_mask.sum().item()), 1)

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
        "accs": accs,
        "accs_source": accs_source,
        "accs_target": accs_target,
        "n_source": len(src_idx),
        "n_target": len(tgt_idx),
        "bundle_state": bundle_state,
        "sleep_reports": sleep_reports,
    }
