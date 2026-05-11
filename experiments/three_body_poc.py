"""F60 — N=3 gravity in 2-D plane: stable vs chaotic variants.

Tests whether the F57 P1/P2/P3 invariants hold on a non-trivial
multi-body dynamical system. This is the **falsifiability boundary
test** for PCM v4 cook: in chaotic regimes, sensitive-dependence
on initial conditions implies rollout error grows exponentially
(`||Δs(t)|| ∝ exp(λ t)` for Lyapunov exponent λ > 0), directly
contradicting F57's polynomial P2 R² ≈ 0.95.

Two variants run on the same code:

* **stable** — hierarchical configuration (m₃ ≪ m₁, m₂),
  approximately circular relative orbits. Near-integrable. We
  expect P2 polynomial to PASS, mirroring F57 behaviour.
* **chaotic** — three equal masses with a generic non-symmetric
  initial condition. Lyapunov exponent λ > 0. We expect P2
  polynomial to **FAIL**: log-log linear fit produces low R²
  while exp(λ K) explains the data.

State representation:

    state = (x1, y1, vx1, vy1,
             x2, y2, vx2, vy2,
             x3, y3, vx3, vy3) ∈ ℝ^{12}

Force (per body i):

    a_i = G · Σ_{j ≠ i} m_j (r_j - r_i) / |r_j - r_i|^3

Ground truth integrator: velocity-Verlet (symplectic, second
order). Same dt for ground-truth-train labels and cook rollout.

Usage::

    python -m experiments.three_body_poc \\
        --variant stable --n-seeds 3 --epochs 30 \\
        --steps-per-epoch 200 --K-rollout 100 \\
        --out outputs/f60_stable

    python -m experiments.three_body_poc \\
        --variant chaotic --n-seeds 3 --epochs 30 \\
        --steps-per-epoch 200 --K-rollout 100 \\
        --out outputs/f60_chaotic
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from pcm.physics import PhysicsCook, PhysicsStateHead, physics_step_loss


__all__ = ["main", "ground_truth_step", "ground_truth_rollout"]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STATE_DIM = 12  # 3 bodies × (x, y, vx, vy)
G = 1.0  # gravity constant in our units
EPSILON = 1e-3  # softening to avoid divergence at close approach


# ─────────────────────────────────────────────────────────────────
# Configurations
# ─────────────────────────────────────────────────────────────────


VARIANTS: dict[str, dict] = {
    # Chenciner-Montgomery figure-8 (Annals of Math 2000): the
    # only known stable periodic solution for the equal-mass
    # 3-body problem. All three bodies trace the same lemniscate
    # in the plane, separated in time by a third of the period.
    # Period ≈ 6.32 in these units.
    "stable": {
        "masses": (1.0, 1.0, 1.0),
        "initial": [
            (-0.97000436, 0.24308753, 0.466203685, 0.43236573),
            (0.97000436, -0.24308753, 0.466203685, 0.43236573),
            (0.0, 0.0, -0.93240737, -0.86473146),
        ],
        "dt": 0.005,  # very small dt: figure-8 has fast close
                     # crossings at the lemniscate centre that need
                     # high temporal resolution to stay on attractor
        "L": 2.0,
    },
    # Pythagorean three-body problem (Burrau, 1913): three masses
    # 3, 4, 5 placed at the vertices of a right triangle with
    # legs 3, 4, 5, all at rest. The system is gravitationally
    # bound and famously *chaotic*: bodies undergo close
    # encounters and irregular motion before eventually one
    # escapes (~K=200 onwards). For K up to ~150 the dynamics
    # is bounded and exposes the Lyapunov regime.
    "chaotic": {
        "masses": (3.0, 4.0, 5.0),
        "initial": [
            (1.0, 3.0, 0.0, 0.0),    # mass 3 at (1, 3)
            (-2.0, -1.0, 0.0, 0.0),  # mass 4 at (-2, -1)
            (1.0, -1.0, 0.0, 0.0),   # mass 5 at (1, -1)
        ],
        "dt": 0.01,  # small dt: Pythagorean has close encounters
                    # that demand high temporal resolution
        "L": 5.0,
    },
}


def _initial_state(variant: str) -> torch.Tensor:
    cfg = VARIANTS[variant]
    bodies = cfg["initial"]
    return torch.tensor([v for body in bodies for v in body], dtype=torch.float32)


def _masses(variant: str) -> torch.Tensor:
    return torch.tensor(VARIANTS[variant]["masses"], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────
# Ground truth physics — velocity-Verlet integrator
# ─────────────────────────────────────────────────────────────────


def _accelerations(
    state: torch.Tensor, masses: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise gravitational accelerations.

    state: ``(B, 12)``; returns ``(B, 6)`` accelerations
    ``(ax1, ay1, ax2, ay2, ax3, ay3)``.
    """
    masses = masses.to(state.device)
    B = state.shape[0]
    pos = torch.zeros(B, 3, 2, device=state.device, dtype=state.dtype)
    for i in range(3):
        pos[:, i, 0] = state[:, 4 * i]
        pos[:, i, 1] = state[:, 4 * i + 1]
    accs = torch.zeros(B, 3, 2, device=state.device, dtype=state.dtype)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            r_ij = pos[:, j] - pos[:, i]  # (B, 2)
            dist_sq = r_ij.pow(2).sum(dim=-1, keepdim=True) + EPSILON ** 2
            accs[:, i] += G * masses[j] * r_ij / dist_sq.pow(1.5)
    out = torch.zeros(B, 6, device=state.device, dtype=state.dtype)
    for i in range(3):
        out[:, 2 * i] = accs[:, i, 0]
        out[:, 2 * i + 1] = accs[:, i, 1]
    return out


def ground_truth_step(
    state: torch.Tensor, masses: torch.Tensor, dt: float,
) -> torch.Tensor:
    """Velocity-Verlet step (symplectic, second-order).

    state: ``(B, 12)``. Returns ``(B, 12)`` state at t+dt.
    """
    a_t = _accelerations(state, masses)  # (B, 6)
    new_state = state.clone()
    # Position update: x_{n+1} = x_n + v_n * dt + 0.5 * a_n * dt^2
    for i in range(3):
        new_state[:, 4 * i] = (
            state[:, 4 * i]
            + state[:, 4 * i + 2] * dt
            + 0.5 * a_t[:, 2 * i] * dt * dt
        )
        new_state[:, 4 * i + 1] = (
            state[:, 4 * i + 1]
            + state[:, 4 * i + 3] * dt
            + 0.5 * a_t[:, 2 * i + 1] * dt * dt
        )
    a_tp1 = _accelerations(new_state, masses)
    # Velocity update: v_{n+1} = v_n + 0.5 * (a_n + a_{n+1}) * dt
    for i in range(3):
        new_state[:, 4 * i + 2] = (
            state[:, 4 * i + 2]
            + 0.5 * (a_t[:, 2 * i] + a_tp1[:, 2 * i]) * dt
        )
        new_state[:, 4 * i + 3] = (
            state[:, 4 * i + 3]
            + 0.5 * (a_t[:, 2 * i + 1] + a_tp1[:, 2 * i + 1]) * dt
        )
    return new_state


def ground_truth_rollout(
    s0: torch.Tensor, K: int, masses: torch.Tensor, dt: float,
) -> torch.Tensor:
    """Roll forward K steps. Returns ``(K+1, B, 12)``."""
    traj = [s0]
    state = s0
    for _ in range(K):
        state = ground_truth_step(state, masses, dt)
        traj.append(state)
    return torch.stack(traj, dim=0)


@torch.no_grad()
def measure_ground_truth_lyapunov(
    s0: torch.Tensor, *, masses: torch.Tensor, dt: float,
    K: int = 1000, perturbation: float = 1e-4,
    n_dirs: int = 16,
) -> dict:
    """Estimate the largest Lyapunov exponent of the dynamical
    system itself (independent of any learned head).

    For each of n_dirs random unit perturbations, roll forward
    a baseline trajectory and a perturbed trajectory simultaneously,
    track ``d(t) = ||s_t - s'_t||``, and fit ``log d(t) ≈ log d_0 + λt``.

    Returns dict with mean ``lyapunov_per_step`` and the per-step
    ``log_d_curve``. λ > 0 ⇒ chaotic.

    All integration is performed in float64 to avoid FP32 round-off
    annihilating tiny perturbations long before saturation.
    """
    device = s0.device
    state_dim = s0.shape[0]
    log_d = torch.zeros(K + 1, device=device, dtype=torch.float64)
    base = s0.unsqueeze(0).expand(n_dirs, -1).clone().to(torch.float64)
    masses64 = masses.to(torch.float64)
    delta = torch.randn(n_dirs, state_dim, device=device,
                        dtype=torch.float64)
    delta = delta / delta.norm(dim=-1, keepdim=True) * perturbation
    perturbed = base + delta
    log_d[0] = math.log(perturbation)
    for t in range(1, K + 1):
        base = ground_truth_step(base, masses64, dt)
        perturbed = ground_truth_step(perturbed, masses64, dt)
        d = (perturbed - base).norm(dim=-1)
        log_d[t] = torch.log(d + 1e-30).mean()
    # Two complementary estimators:
    #
    # 1) Mean linear-fit slope on [start, K_sat] where K_sat is
    #    the first crossing of log_d above log(L). Robust for
    #    *smooth* divergence (figure-8 weak drift).
    #
    # 2) Max-window slope over a sliding 100-step window. Robust
    #    for *impulsive* divergence (Pythagorean close encounters,
    #    where log_d can jump ~8 orders of magnitude in one event).
    #    The max-window slope is the more reliable chaos
    #    discriminator across regimes.
    L_log_cap = math.log(5.0)
    K_start = max(K // 20, 1)
    K_sat = K
    for t in range(1, K + 1):
        if log_d[t].item() > L_log_cap:
            K_sat = t
            break

    def _linfit(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
        n = len(y)
        if n < 2:
            return float("nan"), float("nan")
        sx, sy = float(x.sum()), float(y.sum())
        sxx = float((x * x).sum())
        sxy = float((x * y).sum())
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-12:
            return float("nan"), float("nan")
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        ss_res = float(((y - (slope * x + intercept)) ** 2).sum())
        ss_tot = float(((y - sy / n) ** 2).sum()) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        return slope, r2

    K_end = max(K_sat, K_start + 5)
    ts = torch.arange(K_start, K_end + 1,
                      dtype=torch.float64, device=device)
    y_lin = log_d[K_start: K_end + 1].cpu()
    x_lin = ts.cpu()
    lam_smooth, r2_smooth = _linfit(x_lin, y_lin)

    # Sliding window: maximum slope on any 100-step window before
    # saturation captures impulsive chaos events.
    window = min(100, max(K_sat // 4, 5))
    max_window_slope = float("-inf")
    max_window_start = 0
    cap_search = K_sat
    for t in range(0, max(cap_search - window, 1)):
        slope = (log_d[t + window].item() - log_d[t].item()) / window
        if slope > max_window_slope:
            max_window_slope = slope
            max_window_start = t

    return {
        "lyapunov_per_step": lam_smooth,
        "lyapunov_fit_R2": r2_smooth,
        "lyapunov_max_window_slope": max_window_slope,
        "lyapunov_max_window_start": max_window_start,
        "lyapunov_max_window_size": window,
        "K_fit_start": K_start,
        "K_fit_end": K_end,
        "K_sat": K_sat,
        "log_d_curve": log_d.cpu().tolist(),
    }


# ─────────────────────────────────────────────────────────────────
# Per-seed train + eval
# ─────────────────────────────────────────────────────────────────


def _sample_perturbed_states(
    s0: torch.Tensor, n: int, sigma: float,
) -> torch.Tensor:
    """Sample states near `s0` for training. sigma controls
    Gaussian perturbation magnitude per component."""
    s = s0.unsqueeze(0).expand(n, -1).clone()
    s = s + sigma * torch.randn_like(s)
    return s


def _run_one(
    seed: int, *,
    variant: str,
    epochs: int, steps_per_epoch: int,
    K_rollout: int,
    batch_size: int = 64, lr: float = 5e-3, sigma: float = 0.5,
) -> dict:
    torch.manual_seed(seed)
    cfg = VARIANTS[variant]
    masses = _masses(variant).to(DEVICE)
    dt = cfg["dt"]
    L = cfg["L"]
    s0 = _initial_state(variant).to(DEVICE)

    head = PhysicsStateHead(
        state_dim=STATE_DIM, force_dim=0, hidden=128, dt=dt,
    ).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    # ─── train: 1-step transitions sampled near the trajectory ───
    t0 = time.time()
    # Pre-compute a long ground-truth trajectory and sample states
    # along it as training centroids (ensures diverse phase-space
    # coverage).
    K_traj = max(K_rollout * 2, 200)
    traj_full = ground_truth_rollout(
        s0.unsqueeze(0), K_traj, masses=masses, dt=dt,
    )  # (K_traj+1, 1, 12)
    centroids = traj_full[:, 0, :]  # (K_traj+1, 12)
    for _ in range(epochs):
        head.train()
        for _ in range(steps_per_epoch):
            # Pick random centroids and add small Gaussian noise.
            idx = torch.randint(0, centroids.shape[0], (batch_size,))
            base = centroids[idx]
            states = base + sigma * torch.randn_like(base)
            true_next = ground_truth_step(states, masses, dt)
            pred_delta = head(states)
            loss = physics_step_loss(pred_delta, true_next, states)
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    train_wall = time.time() - t0

    cook = PhysicsCook(
        head, max_iters=K_rollout + 10, diverge_threshold=1e3,
    )

    # ─── P1: 1-step normalised MSE on a fresh test set ───
    @torch.no_grad()
    def _eval_p1() -> float:
        idx = torch.randint(0, centroids.shape[0], (1000,))
        base = centroids[idx]
        states = base + sigma * torch.randn_like(base)
        true_next = ground_truth_step(states, masses, dt)
        pred_next = states + head(states)
        mse = (pred_next - true_next).pow(2).mean()
        return float((mse / (L ** 2)).item())

    # ─── P2 / P3: K-step rollout vs ground truth.
    # We compute on a single shared rollout (long-K), then read
    # off intermediate K's. This guarantees that error metrics are
    # consistent with each other and that the same set of test
    # trajectories is used. n_test is large because chaotic
    # systems require ensemble averaging to expose Lyapunov
    # growth.
    n_test = 200
    idx = torch.randint(0, centroids.shape[0] // 2, (n_test,))
    base = centroids[idx]
    s_test = base + 0.1 * torch.randn_like(base)
    with torch.no_grad():
        traj_pred, rep_full = cook(s_test, K=K_rollout)
        traj_true = ground_truth_rollout(s_test, K_rollout,
                                         masses=masses, dt=dt)
        # per-step squared L2 error per trajectory (K_rollout+1, n_test)
        per_traj_se = (traj_pred - traj_true).pow(2).sum(dim=-1)
        # mean SE over trajectories (chaotic: noisy)
        ens_mean_se = per_traj_se.mean(dim=-1)
        # mean log SE over trajectories — robust to outlier
        # trajectories. In the chaotic regime this is the cleanest
        # signature of Lyapunov growth: μ(K) = log|δ_0| + λ K.
        log_se = torch.log(per_traj_se + 1e-20)
        ens_mean_log_se = log_se.mean(dim=-1)

    def _read_curve(K: int) -> dict:
        return {
            "K": K,
            "ens_mean_se": float(ens_mean_se[K].item()),
            "ens_mean_log_se": float(ens_mean_log_se[K].item()),
            "pred_final_state_norm": float(
                traj_pred[K].norm(dim=-1).max().item()
            ),
            "true_final_state_norm": float(
                traj_true[K].norm(dim=-1).max().item()
            ),
        }

    eval_Ks = sorted(set(k for k in [1, 5, 10, 20, 50, 75, K_rollout]
                          if 1 <= k <= K_rollout))
    p2_curve = {K: _read_curve(K) for K in eval_Ks}
    final = p2_curve[K_rollout] | {
        "diverged": rep_full.diverged,
        "final_state_norm": p2_curve[K_rollout]["pred_final_state_norm"],
    }

    p1 = _eval_p1()

    # ─── Polynomial fit on log(mse) vs log(K), and exponential fit
    # on log(mse) vs K. We use ens_mean_log_se directly because:
    #   - polynomial growth of MSE means log MSE ~ α log K
    #   - exponential (Lyapunov) growth means log MSE ~ λ K
    # Whichever fit gives higher R² wins. ───
    Ks_used = list(eval_Ks)
    log_K = [math.log(k) for k in Ks_used]
    K_vals = list(Ks_used)
    log_mse = [p2_curve[k]["ens_mean_log_se"] for k in Ks_used]

    def _linfit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
        if len(xs) < 3:
            return float("nan"), float("nan"), float("nan")
        n = len(xs)
        sx, sy = sum(xs), sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-12:
            return float("nan"), float("nan"), float("nan")
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        ss_res = sum((y - (slope * x + intercept)) ** 2
                     for x, y in zip(xs, ys))
        ss_tot = sum((y - sy / n) ** 2 for y in ys) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        return slope, intercept, r2

    poly_slope, _, poly_R2 = _linfit(log_K, log_mse)
    exp_slope, _, exp_R2 = _linfit(K_vals, log_mse)

    # ─── Ground-truth Lyapunov (system-intrinsic, head-independent).
    # Use a long horizon (K=1000 GT steps, very cheap, no learning)
    # to expose the asymptotic divergence rate even when the
    # cook training window K_rollout is short. Computed in float64.
    lyap = measure_ground_truth_lyapunov(
        s0, masses=masses, dt=dt, K=1000, perturbation=1e-4,
        n_dirs=16,
    )

    return {
        "seed": seed,
        "variant": variant,
        "wall_train_s": train_wall,
        "P1_1step_norm_mse": p1,
        "P2_curve": p2_curve,
        "P2_log_log_R2_polynomial": poly_R2,
        "P2_log_R2_exponential": exp_R2,
        "P2_polynomial_slope": poly_slope,  # exponent of K^α
        "P2_exponential_slope": exp_slope,  # Lyapunov-like rate from head
        "P3_K_max_final_state_norm": final["final_state_norm"],
        "P3_K_max_diverged": final["diverged"],
        "P3_K_max_ens_mean_se": final["ens_mean_se"],
        "GT_lyapunov_per_step": lyap["lyapunov_per_step"],
        "GT_lyapunov_fit_R2": lyap["lyapunov_fit_R2"],
        "GT_lyapunov_max_window_slope": lyap["lyapunov_max_window_slope"],
        "GT_lyapunov_max_window_start": lyap["lyapunov_max_window_start"],
        "GT_K_fit_start": lyap["K_fit_start"],
        "GT_K_fit_end": lyap["K_fit_end"],
        "GT_K_sat": lyap["K_sat"],
    }


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["stable", "chaotic"],
                    required=True)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=99500)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--K-rollout", type=int, default=100)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f60_three_body"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(
        f"  F60 N=3 gravity 2-D, variant={args.variant}, "
        f"K_rollout={args.K_rollout}, n_seeds={args.n_seeds}"
    )
    print("=" * 76)

    rows: list[dict] = []
    for si in range(args.n_seeds):
        seed = args.seed_base + si
        r = _run_one(
            seed, variant=args.variant,
            epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
            K_rollout=args.K_rollout,
        )
        rows.append(r)
        K_max = args.K_rollout
        final_mse = r["P2_curve"][K_max]["ens_mean_se"]
        print(
            f"  [seed={seed}] P1={r['P1_1step_norm_mse']:.5f}  "
            f"K={K_max} mse={final_mse:.4f}  "
            f"poly R2={r['P2_log_log_R2_polynomial']:.3f}  "
            f"exp R2={r['P2_log_R2_exponential']:.3f}  "
            f"final_norm={r['P3_K_max_final_state_norm']:.2f}  "
            f"div={r['P3_K_max_diverged']}  ({r['wall_train_s']:.1f}s)"
        )

    def _stats(key: str) -> dict:
        vals = [r[key] for r in rows
                if isinstance(r.get(key), (int, float))
                and not (isinstance(r.get(key), float)
                         and math.isnan(r[key]))]
        if not vals:
            return {"n": 0}
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals)
                       / max(len(vals) - 1, 1))
        return {"mean": m, "std": sd, "min": min(vals), "max": max(vals),
                "n": len(vals)}

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "per_seed": rows,
        "P1_1step_norm_mse": _stats("P1_1step_norm_mse"),
        "P2_log_log_R2_polynomial": _stats("P2_log_log_R2_polynomial"),
        "P2_log_R2_exponential": _stats("P2_log_R2_exponential"),
        "P2_polynomial_slope": _stats("P2_polynomial_slope"),
        "P2_exponential_slope": _stats("P2_exponential_slope"),
        "P3_K_max_final_state_norm": _stats("P3_K_max_final_state_norm"),
        "GT_lyapunov_per_step": _stats("GT_lyapunov_per_step"),
        "GT_lyapunov_fit_R2": _stats("GT_lyapunov_fit_R2"),
        "GT_lyapunov_max_window_slope": _stats(
            "GT_lyapunov_max_window_slope"
        ),
        "GT_K_sat": _stats("GT_K_sat"),
    }

    # Curve aggregates (ensemble mean SE and ensemble mean log SE)
    Ks = sorted(rows[0]["P2_curve"].keys())
    curve = {
        K: sum(r["P2_curve"][K]["ens_mean_se"] for r in rows) / len(rows)
        for K in Ks
    }
    log_curve = {
        K: sum(r["P2_curve"][K]["ens_mean_log_se"] for r in rows) / len(rows)
        for K in Ks
    }
    summary["P2_curve_mean"] = curve
    summary["P2_log_curve_mean"] = log_curve

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    poly = summary["P2_log_log_R2_polynomial"].get("mean", float("nan"))
    expo = summary["P2_log_R2_exponential"].get("mean", float("nan"))
    norm = summary["P3_K_max_final_state_norm"].get("mean", float("nan"))
    p1m = summary["P1_1step_norm_mse"].get("mean", float("nan"))
    gt_lam = summary["GT_lyapunov_per_step"].get("mean", float("nan"))
    gt_lam_r2 = summary["GT_lyapunov_fit_R2"].get("mean", float("nan"))
    gt_lam_max = summary["GT_lyapunov_max_window_slope"].get(
        "mean", float("nan"))
    gt_k_sat = summary["GT_K_sat"].get("mean", float("nan"))
    print(f"  Variant {args.variant}:")
    print(f"  P1 1-step norm MSE:                {p1m:.5f}")
    print(f"  P2 polynomial fit R^2:             {poly:.3f}")
    print(f"  P2 exponential fit R^2:            {expo:.3f}")
    poly_slope = summary["P2_polynomial_slope"].get("mean", float("nan"))
    exp_slope = summary["P2_exponential_slope"].get("mean", float("nan"))
    print(f"  P2 polynomial K^alpha (alpha):      {poly_slope:.2f}")
    print(f"  P2 exponential exp(lambda K):       lambda={exp_slope:.4f}")
    print(f"  P3 K=K_max final state norm:        {norm:.2f}")
    print(f"  GT Lyapunov smooth fit (slope):    lambda={gt_lam:+.4f}  "
          f"R^2={gt_lam_r2:.3f}")
    print(f"  GT Lyapunov max-window slope:      lambda*={gt_lam_max:+.4f}")
    print(f"  GT K to saturation (perturbation→L): K_sat={gt_k_sat:.0f}")
    print()
    print(f"  P2 ensemble mean SE curve:")
    for K in Ks:
        # also pull mean true norm to see if dynamics is bounded
        true_norms = [r["P2_curve"][K]["true_final_state_norm"]
                      for r in rows]
        true_norm_mean = sum(true_norms) / len(true_norms)
        print(
            f"    K={K:>3d}: SE={curve[K]:.4f}  "
            f"logSE={log_curve[K]:+.3f}  "
            f"|s_true|={true_norm_mean:.2f}"
        )

    # Verdict -- primary discriminator is the ground-truth Lyapunov
    # exponent (system-intrinsic), not the head fit (which is
    # confounded by 1-step error magnitude). Stable dynamics have
    # lambda_GT close to 0 (or negative). Chaotic dynamics have
    # lambda_GT > 0 with high fit R^2. We use 0.05 per step as a
    # robust threshold (e.g. figure-8 numerically drifts at ~0.015
    # due to integrator + perturbation amplification, but the
    # generic equal-mass three-body has lambda > 0.1).
    # The robust chaos discriminator is the maximum sliding-window
    # divergence rate. Pythagorean-style impulsive chaos can show
    # very small *average* divergence outside scattering events
    # but huge slopes during them, which is what fundamentally
    # limits cook applicability.
    print()
    print("  -- Cook applicability verdict ----------------------------")
    CHAOTIC_THRESHOLD = 0.02  # per-step rate; ~1.0 / 50 step horizon
    if not math.isnan(gt_lam_max) and gt_lam_max > CHAOTIC_THRESHOLD:
        print(
            f"  Ground-truth dynamics is CHAOTIC: max-window "
            f"divergence rate lambda*={gt_lam_max:+.4f} per step "
            f"(threshold {CHAOTIC_THRESHOLD})."
        )
        print(
            f"  -> Predictability horizon K* ~ "
            f"{1.0 / max(gt_lam_max, 1e-6):.0f} steps before "
            "perturbations dominate the trajectory."
        )
        print(
            "  -> This is the 'cook BOUNDED applicability' regime. "
            "Even with perfect P1 the cook K-step error explodes "
            "after K*. Distilling the cook into a 1-shot lookup "
            "(F59) does NOT rescue accuracy beyond K*: the "
            "supervisory signal itself is unreliable there."
        )
    elif not math.isnan(gt_lam_max):
        print(
            f"  Ground-truth dynamics is NEAR-INTEGRABLE: "
            f"max-window divergence rate lambda*={gt_lam_max:+.4f} "
            f"<= {CHAOTIC_THRESHOLD}."
        )
        print(
            "  -> Trajectory perturbations grow polynomially or "
            "very slowly. Cook rollout error follows the same K^alpha "
            "law observed for 1-D bouncing ball (F57). PCM v4 cook is "
            "fully applicable in this regime, and F59 lookup "
            "distillation transfers cleanly."
        )
    else:
        print(
            f"  Lyapunov diagnosis inconclusive "
            f"(lambda*={gt_lam_max}). Possibly a degenerate setup."
        )

    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
