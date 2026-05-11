"""Quick diagnostic for v2 attr_table geometry."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.number_dual_channel_poc import (
    ATTR_DIM, BASE_FACET, DEVICE, N, SLOT_DIM, _train_one_seed,
)
from pcm.dual_channel import collapse_dual_channel


def diagnose(seed: int = 92000) -> None:
    """Re-run training in-line, then dump attr_table stats."""
    torch.manual_seed(seed)
    print(f"diag seed {seed}")

    # We need a trained graph but _train_one_seed doesn't return it.
    # Re-implement a tiny version:
    from pcm.concept_graph import ConceptGraph
    from pcm.dual_channel import (
        arithmetic_consistency_loss, info_nce_loss,
        register_dual_channel_facet,
    )
    import random
    rng = random.Random(seed)

    cg = ConceptGraph(feat_dim=SLOT_DIM)
    cids = []
    for i in range(N):
        cid = f"concept:num:{i}"
        cg.register_concept(node_id=cid, label=f"NUM_{i}",
                            scope="BASE", provenance="diag")
        cids.append(cid)
    register_dual_channel_facet(cg, BASE_FACET,
                                slot_dim=SLOT_DIM, attr_dim=ATTR_DIM)
    with torch.no_grad():
        collapse_dual_channel(
            cg, caller="warmup", base_facet=BASE_FACET, concept_ids=cids,
            slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
            tick=0, device=DEVICE, init="normal_small",
        )
    cg.bundles_to(torch.device(DEVICE))

    quad = []
    for a in range(N):
        for b in range(N):
            if a == b:
                continue
            for c in range(N):
                d = a + c - b
                if 0 <= d < N and d != c:
                    quad.append((a, b, c, d))

    opt = torch.optim.AdamW(list(cg.iter_bundle_parameters()), lr=5e-3)

    # PURE arithmetic loss for 30 epochs.
    EPOCHS = 30
    STEPS = 200
    for epoch in range(1, EPOCHS + 1):
        for step in range(STEPS):
            _, attr = collapse_dual_channel(
                cg, caller="diag", base_facet=BASE_FACET, concept_ids=cids,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=epoch * 10000 + step, device=DEVICE,
            )
            qb = 64
            sample = [quad[rng.randrange(len(quad))] for _ in range(qb)]
            ia = torch.tensor([q[0] for q in sample], device=DEVICE)
            ib = torch.tensor([q[1] for q in sample], device=DEVICE)
            ic = torch.tensor([q[2] for q in sample], device=DEVICE)
            id_ = torch.tensor([q[3] for q in sample], device=DEVICE)
            loss = arithmetic_consistency_loss(attr, ia, ib, ic, id_)
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 5 == 0:
            print(f"  epoch {epoch}: arith_loss={loss.item():.6f}")

    with torch.no_grad():
        _, attr_table = collapse_dual_channel(
            cg, caller="diag-eval", base_facet=BASE_FACET, concept_ids=cids,
            slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
            tick=999999, device=DEVICE,
        )
    attr_table = attr_table.detach().cpu()

    print(f"\nattr_table shape = {tuple(attr_table.shape)}")
    print(f"row norms: {attr_table.norm(dim=-1).tolist()}")
    print(f"row means: {attr_table.mean(dim=-1).tolist()}")

    # Pairwise diffs: attr_{i+1} - attr_i should be roughly constant if linear.
    diffs = [attr_table[i + 1] - attr_table[i] for i in range(N - 1)]
    diff_norms = [d.norm().item() for d in diffs]
    print(f"diff norms: {[f'{x:.3f}' for x in diff_norms]}")
    # Cosine similarity between consecutive diffs.
    cos = []
    for i in range(N - 2):
        cs = F.cosine_similarity(diffs[i].unsqueeze(0),
                                 diffs[i + 1].unsqueeze(0)).item()
        cos.append(cs)
    print(f"diff cos sims: {[f'{x:.3f}' for x in cos]}  "
          f"(close to 1.0 = arithmetic structure)")

    # Vector analogy check.
    triples = [(a, b, c, a + c - b) for a in range(N) for b in range(N)
               for c in range(N) if a != b and 0 <= a + c - b < N
               and a + c - b != c]
    hits = 0
    for a, b, c, d in triples:
        q = attr_table[a] + attr_table[c] - attr_table[b]
        nn = (attr_table - q.unsqueeze(0)).pow(2).sum(dim=-1).argmin().item()
        if nn == d:
            hits += 1
    print(f"\nV2_top1 over {len(triples)} triples = {hits / len(triples):.3f}")


if __name__ == "__main__":
    diagnose()
