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

    params: list = []
    for h in heads.values():
        params += list(h.parameters())
    params += list(cg.iter_bundle_parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    facet_by_key = {"v": FACET_V, "m": FACET_M, "p": FACET_P}
    sleep_facets = [facet_by_key[k] for k in heads]
    if sleep_every is not None:
        attach_sleep(cg, facets=sleep_facets)
    sleep_reports: list[dict] = []

    for epoch in range(1, epochs + 1):
        for h in heads.values():
            h.train()
        for step_i in range(steps_per_epoch):
            idx_batch = [rng.randrange(N_PH) for _ in range(BATCH_SIZE)]
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

    accs: dict[str, float] = {}
    with torch.no_grad():
        all_ids = _apply_shuffle([cid_of(i) for i in range(N_PH)], shuffle_map)
        for key, h in heads.items():
            logits = h(all_ids, cg)
            axis = {"v": 0, "m": 1, "p": 2}[key]
            tgt = torch.tensor([feat_of(i)[axis] for i in range(N_PH)], device=DEVICE)
            accs[key] = float(logits.argmax(-1).eq(tgt).sum().item()) / N_PH

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
        "bundle_state": bundle_state,
        "sleep_reports": sleep_reports,
    }
