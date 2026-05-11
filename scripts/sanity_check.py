"""scripts/sanity_check.py — 5-minute regression for the bio-inspired
dense-pool refactor on an 8GB GPU.

Runs (in order):

1. ``tests.test_grow_invariants`` (unittest, ~3s) — G1-G6 + 4 grow regressions
2. ``tests.test_smoke``           (unittest, ~1s) — public API + grad sanity
3. ``experiments.scale_study --smoke`` (~20s) — single seed, N=7 mix/add/sub
4. ``experiments.counterfactual_swap_study --smoke`` (~30s) — number+color
   double-dissociation. Asserts the canonical paper §B numbers
   (AddHead-inv 18.2%, MixHead-inv 5.3%) within tolerance.

Each step prints PASS/FAIL and the wall time. Peak VRAM is logged via
``torch.cuda.max_memory_allocated`` if CUDA is available; we expect to
stay well under 2GB on an 8GB card.

Usage::

    python -m scripts.sanity_check
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _run_module(args: list[str]) -> tuple[int, float]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-X", "utf8"] + args,
        cwd=str(ROOT), env=env,
    )
    return proc.returncode, time.time() - t0


def _peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def _check_swap_summary() -> tuple[bool, dict]:
    """Read outputs/counterfactual_swap/summary.json and verify the
    headline numbers from the paper §B (within tolerance).
    """
    path = ROOT / "outputs" / "counterfactual_swap" / "summary.json"
    if not path.exists():
        return False, {"error": "summary.json missing"}
    summary = json.loads(path.read_text(encoding="utf-8"))

    # paper §B.1: Add-inv on swap_arith_only ≈ 18.2%
    # paper §B.2: Mix-inv on swap_mix_only   ≈ 5.3%
    expected = {
        "add_inv_arith": (0.182, 0.05),     # value, tolerance
        "mix_inv_color": (0.053, 0.05),
    }
    found: dict[str, float] = {}

    # The swap study writes nested ``number_domain.per_seed[i]`` and
    # ``color_domain.per_seed[i]`` rows. Smoke uses a single seed.
    try:
        ns = summary.get("number_domain", {}).get("per_seed", [])
        cs = summary.get("color_domain", {}).get("per_seed", [])
        if ns:
            found["add_inv_arith"] = ns[0]["swap_arith_only"]["add"]["involving_swap"]["acc"]
        if cs:
            found["mix_inv_color"] = cs[0]["swap_mix_only"]["mix"]["involving_swap"]["acc"]
    except (KeyError, TypeError, IndexError) as e:
        return False, {"error": f"summary parse failed: {e}", "summary_keys": list(summary)}

    ok = True
    deltas: dict[str, float] = {}
    for k, (target, tol) in expected.items():
        if k not in found:
            ok = False
            deltas[k] = float("nan")
            continue
        deltas[k] = found[k] - target
        if abs(deltas[k]) > tol:
            ok = False

    return ok, {"found": found, "deltas": deltas}


def main() -> int:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    results = []

    _print_header("1/4 · grow invariants (tests.test_grow_invariants)")
    rc, dt = _run_module(["-m", "unittest", "tests.test_grow_invariants", "-v"])
    results.append(("grow_invariants", rc == 0, dt))

    _print_header("2/4 · public-API smoke (tests.test_smoke)")
    rc, dt = _run_module(["-m", "unittest", "tests.test_smoke", "-v"])
    results.append(("smoke", rc == 0, dt))

    _print_header("3/4 · §4 scale_study smoke (N=7 mix/add/sub × 1 seed)")
    rc, dt = _run_module([
        "-m", "experiments.scale_study", "--smoke",
        "--out", "outputs/scale_study_sanity",
    ])
    results.append(("scale_study", rc == 0, dt))

    _print_header("4/4 · §B counterfactual_swap smoke (number + color)")
    rc, dt = _run_module([
        "-m", "experiments.counterfactual_swap_study", "--smoke",
    ])
    swap_runs_ok = (rc == 0)
    swap_match, swap_report = _check_swap_summary()
    results.append(("counterfactual_swap", swap_runs_ok and swap_match, dt))

    peak_mb = _peak_vram_mb()

    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for name, ok, dt in results:
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name:<24} ({dt:5.1f}s)")
    print()
    print(f"  swap-claim check: {swap_report}")
    if torch.cuda.is_available():
        print(f"  peak VRAM allocated: {peak_mb:.1f} MB (budget: 8000 MB)")
    print()
    all_ok = all(ok for _, ok, _ in results)
    print(f"  overall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
