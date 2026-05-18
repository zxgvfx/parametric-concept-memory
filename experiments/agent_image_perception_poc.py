"""F72 — PCM v6.4-image perception layer: image-conditioned goals.

F68 demonstrated that the F62 universal operator works when the
goal is specified as a token sequence consumed by a Transformer
``TextPerceptionHead``. F72 extends to the *image* modality:
the goal is specified as a small synthetic image of the target
digit, consumed by a CNN ``ImagePerceptionHead``.

This is the **multimodal perception** test: the F62 combiner
and the rest of the F64 agent stack are unchanged; only the
goal-slot source switches from text-token-Transformer to
image-CNN.

Env: ``CyclicNavEnv`` (N=20 states, same as F68). Goal images
are 16×16 greyscale renderings of digits 0–19 (digits ≥ 10 are
rendered as stacked tens+ones digits). Each digit has multiple
*variants* (small position jitter + Gaussian noise) — pixel-
distinct images of the same goal that must map to the same
slot.

Five falsifiable invariants (mirror F68 P1–P5):

* **I1** in-domain success ≥ 0.85.
* **I2** image alias invariance: variants of the same digit
  produce slot cosine ≥ 0.70 within-target; across-target slot
  cosine should be near zero, with gap ≥ 0.30.
* **I3** wrong-digit image neg control: replace the goal image
  with an image of a *different* digit; success on the intended
  goal drops by ≥ 0.50.
* **I4** scrambled-pixel neg control: shuffle the pixels of the
  goal image; success drops by ≥ 0.30 (the CNN reads spatial
  structure, not pixel-marginals).
* **I5** transition_acc on state slots still works (≥ 0.90) —
  the F62 universal operator path is unchanged by the new
  perception modality.

Usage::

    python -m experiments.agent_image_perception_poc \\
        --N 20 --slot-dim 32 --epochs 40 \\
        --out outputs/f72_full
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcm.agent import (
    ImagePerceptionHead,
    PolicyHead,
    SlotStateEncoder,
    TransitionHead,
    transition_loss,
)
from pcm.agent.envs import (
    ACTION_DELTAS,
    CyclicNavEnv,
    bfs_optimal_action,
    draw_digit_image,
)


__all__ = ["main"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 16
N_VARIANTS_PER_DIGIT = 6


def _goal_image(
    goal: int, variant: int, rng_seed: int | None = None,
) -> torch.Tensor:
    """Render goal as a (1, IMAGE_SIZE, IMAGE_SIZE) image."""
    rng = (torch.Generator(device="cpu").manual_seed(rng_seed)
           if rng_seed is not None else None)
    return draw_digit_image(
        int(goal), size=IMAGE_SIZE, variant=variant, rng=rng,
    )


def _shuffle_pixels(image: torch.Tensor) -> torch.Tensor:
    """Spatial permutation of pixels — destroys digit shape but
    preserves marginal pixel histogram."""
    C, H, W = image.shape
    flat = image.flatten(start_dim=1)
    perm = torch.randperm(flat.shape[1])
    return flat[:, perm].reshape(C, H, W)


# ─────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────


def _sample_bc_batch(
    N: int, B: int, device: str,
    *, rng_seed: int = 0,
) -> tuple:
    """Sample (state_idx, goal_image, oracle_action)."""
    rng = torch.Generator(device="cpu").manual_seed(rng_seed)
    s_list, img_list, a_list = [], [], []
    while len(s_list) < B:
        s = int(torch.randint(0, N, (1,), generator=rng).item())
        g = int(torch.randint(0, N, (1,), generator=rng).item())
        if s == g:
            continue
        variant = int(torch.randint(
            0, N_VARIANTS_PER_DIGIT, (1,), generator=rng,
        ).item())
        img = _goal_image(g, variant)
        a = bfs_optimal_action(s, g, N)
        s_list.append(s)
        img_list.append(img)
        a_list.append(a)
    return (
        torch.tensor(s_list, dtype=torch.long, device=device),
        torch.stack(img_list, dim=0).to(device),
        torch.tensor(a_list, dtype=torch.long, device=device),
    )


def _sample_transition_batch(
    N: int, B: int, device: str,
) -> tuple:
    s = torch.randint(0, N, (B,), device=device)
    a = torch.randint(0, len(ACTION_DELTAS), (B,), device=device)
    deltas = torch.tensor(ACTION_DELTAS, device=device)[a]
    s_next = (s + deltas) % N
    return s, a, s_next


# ─────────────────────────────────────────────────────────────────
# Training + evaluation
# ─────────────────────────────────────────────────────────────────


def _train(
    encoder: SlotStateEncoder,
    perception: ImagePerceptionHead,
    transition: TransitionHead,
    policy: PolicyHead,
    *, N: int, epochs: int, batches_per_epoch: int,
    batch_size: int, lr: float,
) -> dict:
    params = (list(encoder.parameters())
              + list(perception.parameters())
              + list(transition.parameters())
              + list(policy.parameters()))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    history = []
    for ep in range(epochs):
        bc_total = trans_total = 0.0
        n = 0
        for bi in range(batches_per_epoch):
            s, img, a_star = _sample_bc_batch(
                N, batch_size, DEVICE,
                rng_seed=ep * 1000 + bi,
            )
            slot_s = encoder(s)
            slot_g = perception(img)
            logits = policy(slot_s, slot_g)
            bc = F.cross_entropy(logits, a_star)
            ts, ta, tsn = _sample_transition_batch(N, batch_size, DEVICE)
            t_loss = transition_loss(transition, encoder, ts, ta, tsn)
            loss = bc + t_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            bc_total += float(bc.item())
            trans_total += float(t_loss.item())
            n += 1
        history.append({"epoch": ep, "bc": bc_total / max(n, 1),
                        "trans": trans_total / max(n, 1)})
    return {"history": history}


@torch.no_grad()
def _evaluate(
    encoder: SlotStateEncoder,
    perception: ImagePerceptionHead,
    policy: PolicyHead,
    *, N: int, n_episodes: int, max_steps: int,
    image_transform=None,
    wrong_digit: bool = False,
) -> dict:
    """Evaluate the image-conditioned policy.

    Args:
        image_transform: optional callable on the goal image
            (e.g., pixel shuffle).
        wrong_digit: if True, render a different digit than the
            actual goal (lie-test).
    """
    encoder.eval()
    perception.eval()
    policy.eval()
    rng = torch.Generator(device="cpu").manual_seed(2026)
    total = succ = 0
    for _ in range(n_episodes):
        s0 = int(torch.randint(0, N, (1,), generator=rng).item())
        g = int(torch.randint(0, N, (1,), generator=rng).item())
        if s0 == g:
            g = (g + 1) % N
        variant = int(torch.randint(
            0, N_VARIANTS_PER_DIGIT, (1,), generator=rng,
        ).item())
        if wrong_digit:
            g_image_digit = g
            while g_image_digit == g:
                g_image_digit = int(torch.randint(
                    0, N, (1,), generator=rng,
                ).item())
            img = _goal_image(g_image_digit, variant)
        else:
            img = _goal_image(g, variant)
        if image_transform is not None:
            img = image_transform(img)
        img_t = img.unsqueeze(0).to(DEVICE)
        slot_g = perception(img_t)
        env = CyclicNavEnv(N=N, max_steps=max_steps)
        env.reset(s0)
        env.set_goal(g)
        s = s0
        success = False
        for _ in range(max_steps):
            s_t = torch.tensor([s], dtype=torch.long, device=DEVICE)
            slot_s = encoder(s_t)
            logits = policy(slot_s, slot_g)
            a = int(logits.argmax(-1).item())
            s_next, _, done = env.step(a)
            s = int(s_next)
            if s == g:
                success = True
                break
            if done:
                break
        total += 1
        if success:
            succ += 1
    return {"success_rate": succ / max(total, 1),
            "n_episodes": total, "n_success": succ}


@torch.no_grad()
def _alias_cos(
    perception: ImagePerceptionHead, *, N: int,
    n_variants: int = N_VARIANTS_PER_DIGIT,
) -> tuple[float, float]:
    perception.eval()
    per_digit_slots = []
    for d in range(N):
        slots = []
        for v in range(n_variants):
            img = _goal_image(d, v).unsqueeze(0).to(DEVICE)
            slots.append(perception(img)[0])
        per_digit_slots.append(torch.stack(slots, dim=0))
    within_means = []
    for slots in per_digit_slots:
        sn = F.normalize(slots, dim=-1)
        cos = sn @ sn.t()
        off = cos - torch.diag(torch.diagonal(cos))
        within_means.append(
            float(off.sum().item() / (cos.numel() - cos.shape[0]))
        )
    within = sum(within_means) / len(within_means)
    rng = torch.Generator(device="cpu").manual_seed(31337)
    across_cos = []
    for _ in range(200):
        i, j = 0, 0
        while i == j:
            i = int(torch.randint(0, N, (1,), generator=rng).item())
            j = int(torch.randint(0, N, (1,), generator=rng).item())
        si = per_digit_slots[i][
            int(torch.randint(0, n_variants, (1,), generator=rng).item())
        ]
        sj = per_digit_slots[j][
            int(torch.randint(0, n_variants, (1,), generator=rng).item())
        ]
        across_cos.append(float(F.cosine_similarity(
            si.unsqueeze(0), sj.unsqueeze(0),
        ).item()))
    across = sum(across_cos) / len(across_cos)
    return within, across


@torch.no_grad()
def _transition_accuracy(
    encoder: SlotStateEncoder, transition: TransitionHead,
    *, N: int, n_samples: int = 5000,
) -> float:
    encoder.eval()
    transition.eval()
    s = torch.randint(0, N, (n_samples,), device=DEVICE)
    a = torch.randint(0, len(ACTION_DELTAS), (n_samples,), device=DEVICE)
    deltas = torch.tensor(ACTION_DELTAS, device=DEVICE)[a]
    s_next = (s + deltas) % N
    slot_pred = transition(encoder(s), a)
    logits = slot_pred @ encoder.all_slots().t()
    return float((logits.argmax(-1) == s_next).float().mean().item())


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--slot-dim", type=int, default=32)
    ap.add_argument("--cnn-channels", type=str, default="16,32")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batches-per-epoch", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f72_full"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cnn_channels = tuple(int(c) for c in args.cnn_channels.split(","))

    print("=" * 76)
    print(f"  F72 PCM v6.4-image perception (image-conditioned goals "
          f"on cyclic ℤ_{args.N}, {IMAGE_SIZE}×{IMAGE_SIZE} digit images)")
    print("=" * 76)

    torch.manual_seed(11)
    encoder = SlotStateEncoder(args.N, args.slot_dim).to(DEVICE)
    perception = ImagePerceptionHead(
        slot_dim=args.slot_dim, image_size=IMAGE_SIZE,
        in_channels=1, channels=cnn_channels,
    ).to(DEVICE)
    transition = TransitionHead(args.slot_dim, len(ACTION_DELTAS)).to(DEVICE)
    policy = PolicyHead(args.slot_dim, len(ACTION_DELTAS)).to(DEVICE)

    n_params = sum(p.numel() for p in perception.parameters())
    print(f"\n  Image perception head: {n_params:,} params  "
          f"(CNN channels={cnn_channels})")

    print("\n[Train] joint BC + transition training, image-conditioned...")
    t0 = time.time()
    _train(
        encoder, perception, transition, policy,
        N=args.N, epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr,
    )
    print(f"    wall = {time.time()-t0:.1f}s")

    i1_eval = _evaluate(
        encoder, perception, policy,
        N=args.N, n_episodes=args.n_eval, max_steps=args.max_steps,
    )
    print(f"\n[I1] in-domain success = {i1_eval['success_rate']:.3f}")

    within, across = _alias_cos(perception, N=args.N)
    print(f"[I2] alias slot cos: within={within:.3f}  across={across:.3f}")

    i3_eval = _evaluate(
        encoder, perception, policy,
        N=args.N, n_episodes=args.n_eval, max_steps=args.max_steps,
        wrong_digit=True,
    )
    print(f"[I3] wrong-digit image success = {i3_eval['success_rate']:.3f}")

    i4_eval = _evaluate(
        encoder, perception, policy,
        N=args.N, n_episodes=args.n_eval, max_steps=args.max_steps,
        image_transform=_shuffle_pixels,
    )
    print(f"[I4] scrambled-pixel success = {i4_eval['success_rate']:.3f}")

    i5_trans = _transition_accuracy(encoder, transition, N=args.N)
    print(f"[I5] transition_acc = {i5_trans:.3f}")

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "image_size": IMAGE_SIZE,
        "n_variants_per_digit": N_VARIANTS_PER_DIGIT,
        "cnn_channels": list(cnn_channels),
        "perception_params": n_params,
        "I1_in_domain_eval": i1_eval,
        "I2_within_cos": within,
        "I2_across_cos": across,
        "I3_wrong_digit_eval": i3_eval,
        "I4_scrambled_pixel_eval": i4_eval,
        "I5_transition_acc": i5_trans,
    }
    summary["verdict"] = {
        "I1_pass": i1_eval["success_rate"] >= 0.85,
        "I2_pass": within >= 0.70 and within - across >= 0.30,
        "I3_pass": (
            i1_eval["success_rate"] - i3_eval["success_rate"] >= 0.50
            and i3_eval["success_rate"] <= 0.40
        ),
        "I4_pass": (
            i1_eval["success_rate"] - i4_eval["success_rate"] >= 0.30
            and i4_eval["success_rate"] <= 0.70
        ),
        "I5_pass": i5_trans >= 0.90,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print("  F72 v6.4-image perception verdict:")
    print("=" * 76)
    v = summary["verdict"]
    print(f"  I1 in-domain (success >=0.85)         : "
          f"{i1_eval['success_rate']:.3f}  "
          f"[{'PASS' if v['I1_pass'] else 'FAIL'}]")
    print(f"  I2 alias cos within>>across           : "
          f"within {within:.3f} − across {across:.3f} = "
          f"{within - across:+.3f}  "
          f"[{'PASS' if v['I2_pass'] else 'FAIL'}]")
    print(f"  I3 wrong-digit neg ctrl (gap>=.50)    : "
          f"in-domain {i1_eval['success_rate']:.3f} - wrong "
          f"{i3_eval['success_rate']:.3f} = "
          f"{i1_eval['success_rate'] - i3_eval['success_rate']:+.3f}  "
          f"[{'PASS' if v['I3_pass'] else 'FAIL'}]")
    print(f"  I4 scrambled-pixel neg ctrl (gap>=.30): "
          f"in-domain - scrambled = "
          f"{i1_eval['success_rate'] - i4_eval['success_rate']:+.3f}  "
          f"[{'PASS' if v['I4_pass'] else 'FAIL'}]")
    print(f"  I5 transition_acc (>=0.90)            : "
          f"{i5_trans:.3f}  "
          f"[{'PASS' if v['I5_pass'] else 'FAIL'}]")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
