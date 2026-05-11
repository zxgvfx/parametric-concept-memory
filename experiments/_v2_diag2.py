"""Bottom-up diagnostic: write a known monotonic attr table directly,
then verify V2 evaluation gives 1.0. This pins down whether the
problem is in evaluation (which would be a bug) or in training
(which is the design problem we need to solve)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.number_dual_channel_poc import (
    ATTR_DIM, BASE_FACET, DEVICE, N, SLOT_DIM,
    v2_vector_analogy_top1,
)
from pcm.concept_graph import ConceptGraph
from pcm.dual_channel import (
    arithmetic_consistency_loss, collapse_dual_channel,
    register_dual_channel_facet, successor_consistency_loss,
)


def main() -> None:
    torch.manual_seed(7)
    cg = ConceptGraph(feat_dim=SLOT_DIM)
    cids = []
    for i in range(N):
        cid = f"concept:num:{i}"
        cg.register_concept(node_id=cid, label=f"NUM_{i}",
                            scope="BASE", provenance="diag2")
        cids.append(cid)
    register_dual_channel_facet(cg, BASE_FACET,
                                slot_dim=SLOT_DIM, attr_dim=ATTR_DIM)

    with torch.no_grad():
        collapse_dual_channel(
            cg, caller="warmup", base_facet=BASE_FACET, concept_ids=cids,
            slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
            tick=0, device=DEVICE,
        )

    # Manually write attr_i = i * v into bundle_pool.
    v = torch.randn(ATTR_DIM, device=DEVICE)
    v = v / v.norm()
    attr_facet_name = f"{BASE_FACET}_attr"
    pool = cg.bundle_pool[attr_facet_name]
    with torch.no_grad():
        for i, cid in enumerate(cids):
            slot = cg.cid_to_slot[cid]
            pool.data[slot] = float(i) * v

    # Read back via collapse_dual_channel (this is what _train_one_seed
    # uses for evaluation).
    with torch.no_grad():
        _, attr_eval = collapse_dual_channel(
            cg, caller="diag2-eval", base_facet=BASE_FACET, concept_ids=cids,
            slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
            tick=99, device=DEVICE, normalize_attr=False,
        )

    print("attr_eval row norms:", attr_eval.norm(dim=-1).tolist())
    v2 = v2_vector_analogy_top1(attr_eval.cpu(), N, seed=0)
    print("V2 with manually-written linear attr:", v2)

    # First reset attr to random-large init to test convergence
    # from cold start.
    print("\n--- resetting attr to random unit-var init ---")
    with torch.no_grad():
        for slot in range(pool.shape[0]):
            pool.data[slot] = torch.randn(ATTR_DIM, device=DEVICE)
    with torch.no_grad():
        _, attr_after_reset = collapse_dual_channel(
            cg, caller="reset", base_facet=BASE_FACET, concept_ids=cids,
            slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
            tick=2000, device=DEVICE,
        )
    norms = attr_after_reset.norm(dim=-1).tolist()
    print(f"  reset row norms: avg={sum(norms) / N:.3f}")
    v2 = v2_vector_analogy_top1(attr_after_reset.cpu(), N, seed=0)
    print(f"  V2 from random init (no train): {v2['v2_top1']:.3f}")

    print("\n--- training FROM RANDOM INIT, 30 epochs, arith+succ only ---")
    quad = []
    for a in range(N):
        for b in range(N):
            if a == b:
                continue
            for c in range(N):
                d = a + c - b
                if 0 <= d < N and d != c:
                    quad.append((a, b, c, d))
    import random
    rng = random.Random(0)
    opt = torch.optim.AdamW(list(cg.iter_bundle_parameters()), lr=5e-3)
    for epoch in range(1, 31):
        for step in range(100):
            _, attr_table = collapse_dual_channel(
                cg, caller="diag2-train", base_facet=BASE_FACET,
                concept_ids=cids,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=epoch * 100 + step, device=DEVICE,
            )
            qb = 64
            sample = [quad[rng.randrange(len(quad))] for _ in range(qb)]
            ia = torch.tensor([q[0] for q in sample], device=DEVICE)
            ib = torch.tensor([q[1] for q in sample], device=DEVICE)
            ic = torch.tensor([q[2] for q in sample], device=DEVICE)
            id_ = torch.tensor([q[3] for q in sample], device=DEVICE)
            l_a = arithmetic_consistency_loss(attr_table, ia, ib, ic, id_)
            l_s = successor_consistency_loss(attr_table)
            loss = l_a + 5.0 * l_s
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            _, attr_e = collapse_dual_channel(
                cg, caller="diag2-eval2", base_facet=BASE_FACET,
                concept_ids=cids,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=epoch * 1000, device=DEVICE,
            )
        v2 = v2_vector_analogy_top1(attr_e.cpu(), N, seed=0)
        norms = attr_e.norm(dim=-1).tolist()
        print(f"  epoch {epoch}: V2={v2['v2_top1']:.3f}, "
              f"row_norms_avg={sum(norms) / N:.3f}, "
              f"l_a={l_a.item():.4f}, l_s={l_s.item():.4f}")


if __name__ == "__main__":
    main()
