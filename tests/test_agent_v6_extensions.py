"""Unit tests for F70 multi-arg tools, F71 gated policy,
F72 image perception extensions of the v6 agent base."""
from __future__ import annotations

import math

import torch

from pcm.agent import (
    ImagePerceptionHead,
    MultiArgActionTransitionHead,
    MultiArgPolicyHead,
    PolicyHead,
    SlotStateEncoder,
    image_to_slot,
    multi_arg_bc_loss,
    multi_arg_transition_loss,
)
from pcm.agent.envs import (
    MTC_ARG_DIM,
    MTC_TOOL_ARITY,
    MULTI_TOOL_CALC_TOOLS,
    MultiToolCalcEnv,
    draw_digit_image,
    mtc_apply_tool,
    mtc_bfs_optimal_action,
    mtc_bfs_optimal_steps,
    mtc_tool_arity,
)


# ─────────────────────────────────────────────────────────────────
# F70 — multi-tool env + oracle
# ─────────────────────────────────────────────────────────────────


def test_multi_tool_env_basic():
    env = MultiToolCalcEnv(S=20)
    assert env.n_states == 41
    assert env.n_tools == 6
    s = env.reset(7)
    assert s == env.state_to_idx(7)
    env.set_goal(15)
    s_next, r, done = env.step(0, (0.75, 0.0))  # SET(0.75 * 20) = 15
    assert s_next == env.state_to_idx(15)
    assert r == 1.0
    assert done


def test_multi_tool_lerp_two_args():
    """LERP(α, β): state = round(α·state + β·S)"""
    # state=10, LERP(0.5, 0.5) on S=20 → 0.5*10 + 0.5*20 = 15
    assert mtc_apply_tool(10, 5, (0.5, 0.5), S=20) == 15
    # LERP(-1, 0) negates
    assert mtc_apply_tool(7, 5, (-1.0, 0.0), S=20) == -7
    # LERP(0, 1) jumps to +S
    assert mtc_apply_tool(0, 5, (0.0, 1.0), S=20) == 20


def test_multi_tool_arities():
    assert MTC_TOOL_ARITY == (1, 1, 0, 0, 0, 2)
    assert MTC_ARG_DIM == 2
    assert mtc_tool_arity(0) == 1   # SET
    assert mtc_tool_arity(5) == 2   # LERP
    assert mtc_tool_arity(2) == 0   # NEG


def test_multi_tool_bfs_oracle_reachability():
    # All goals reachable from 0 in 1 step (SET or LERP can
    # jump anywhere in [-S, S]).
    for g in range(-20, 21):
        if g == 0:
            continue
        assert mtc_bfs_optimal_steps(0, g, 20) == 1


def test_multi_tool_bfs_oracle_picks_unary_for_ties():
    # 5 -> 15 in 1 step. SET(0.75) and LERP(0, 0.75) both work;
    # BFS prefers lower tool index → SET.
    t, args = mtc_bfs_optimal_action(5, 15, 20)
    assert t == 0
    assert math.isclose(args[0], 0.75, abs_tol=1e-6)


# ─────────────────────────────────────────────────────────────────
# F70 — multi-arg heads
# ─────────────────────────────────────────────────────────────────


def test_multi_arg_transition_head_shape():
    th = MultiArgActionTransitionHead(dim=16, n_tools=6, arg_dim=2)
    slot = torch.randn(4, 16)
    tool = torch.tensor([0, 5, 2, 5])
    args = torch.tensor([[0.5, 0.0], [0.5, 0.5], [0.0, 0.0], [-0.5, 0.5]])
    out = th(slot, tool, args)
    assert out.shape == (4, 16)


def test_multi_arg_policy_head_shape():
    ph = MultiArgPolicyHead(dim=16, n_tools=6, arg_dim=2)
    slot = torch.randn(4, 16)
    goal = torch.randn(4, 16)
    tool_logits, arg_mean, arg_log_std = ph(slot, goal)
    assert tool_logits.shape == (4, 6)
    assert arg_mean.shape == (4, 2)
    assert arg_log_std.shape == (4, 2)


def test_multi_arg_bc_loss_with_per_slot_mask():
    enc = SlotStateEncoder(41, 16)
    pol = MultiArgPolicyHead(dim=16, n_tools=6, arg_dim=2)
    s = torch.tensor([25, 30, 35])
    g = torch.tensor([30, 25, 20])
    tool = torch.tensor([0, 5, 2])  # SET, LERP, NEG
    args = torch.tensor([[0.25, 0.0], [0.5, 0.5], [0.0, 0.0]])
    # SET uses 1 arg, LERP uses 2, NEG uses 0.
    mask = torch.tensor([
        [True, False],
        [True, True],
        [False, False],
    ])
    loss, diag = multi_arg_bc_loss(pol, enc, s, g, tool, args, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert diag["n_arg_slots"] == 3


# ─────────────────────────────────────────────────────────────────
# F72 — image perception
# ─────────────────────────────────────────────────────────────────


def test_draw_digit_image_shape():
    img = draw_digit_image(5)
    assert img.shape == (1, 16, 16)
    assert img.dtype == torch.float32
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_draw_digit_image_variants_differ():
    """Different variants of the same digit should not be
    pixel-identical (jitter + noise should differentiate)."""
    a = draw_digit_image(5, variant=0)
    b = draw_digit_image(5, variant=1)
    assert not torch.allclose(a, b)


def test_draw_digit_image_two_digit():
    """Goals >= 10 render as stacked tens+ones digits."""
    img = draw_digit_image(15)
    assert img.shape == (1, 16, 16)


def test_image_perception_head_shape():
    head = ImagePerceptionHead(slot_dim=32, image_size=16)
    img = torch.zeros(4, 1, 16, 16)
    out = head(img)
    assert out.shape == (4, 32)


def test_image_perception_head_accepts_3d_input():
    head = ImagePerceptionHead(slot_dim=32, image_size=16)
    img = torch.zeros(4, 16, 16)  # (B, H, W) — head adds C dim
    out = head(img)
    assert out.shape == (4, 32)


def test_image_to_slot_inference():
    head = ImagePerceptionHead(slot_dim=16, image_size=16)
    img = draw_digit_image(7)
    slot = image_to_slot(head, img)
    assert slot.shape == (1, 16)


def test_image_perception_learns_to_distinguish_digits():
    """Smoke training: after a few hundred SGD steps to map two
    digit images to two distinct target slots, the perception
    head should produce well-separated outputs."""
    torch.manual_seed(0)
    head = ImagePerceptionHead(slot_dim=8, image_size=16,
                                channels=(8, 16))
    # Two target digits with different target slots
    img5 = torch.stack([draw_digit_image(5, variant=v)
                        for v in range(4)])
    img3 = torch.stack([draw_digit_image(3, variant=v)
                        for v in range(4)])
    t5 = torch.randn(1, 8)
    t3 = torch.randn(1, 8)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    for _ in range(300):
        out5 = head(img5)
        out3 = head(img3)
        loss = ((out5 - t5) ** 2 + (out3 - t3) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Within-class cos should exceed across-class cos
    out5_mean = head(img5).mean(0, keepdim=True)
    out3_mean = head(img3).mean(0, keepdim=True)
    cos_53 = torch.nn.functional.cosine_similarity(
        out5_mean, out3_mean,
    ).item()
    # Different targets → different slot directions
    assert cos_53 < 0.95, f"perception didn't differentiate: cos={cos_53}"
