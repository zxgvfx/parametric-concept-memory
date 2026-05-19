"""Probe RTX 3070 (8 GB) memory at various model scales.

Finds the (d_model, n_layers) combination that hits ~70% VRAM
(~5.6 GB) at batch=64, seq_len=128 with Adam optimizer + FP32.
"""
import sys
import torch
import torch.nn.functional as F

from pcm.lm import HybridPCMMiniLM, count_params


def probe(d_model: int, n_layers: int, *,
          vocab: int = 4096, batch: int = 64, seq_len: int = 128,
          n_steps: int = 3) -> dict:
    """One forward + backward + step at given config; reports
    peak VRAM (GB) and per-step wall time (s)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    n_heads = 8 if d_model % 8 == 0 else 4
    try:
        m = HybridPCMMiniLM(
            vocab=vocab, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, attn_every=4,
        ).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        x = torch.randint(0, vocab, (batch, seq_len), device="cuda")
        y = torch.randint(0, vocab, (batch, seq_len), device="cuda")
        # Warm-up
        logits = m(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab), y.reshape(-1),
        )
        loss.backward()
        opt.step()
        opt.zero_grad()
        # Timed
        import time
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_steps):
            logits = m(x)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab), y.reshape(-1),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        wall = (time.time() - t0) / n_steps
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        n_params = count_params(m)
        del m, opt, x, y, logits, loss
        torch.cuda.empty_cache()
        return {
            "ok": True,
            "d_model": d_model, "n_layers": n_layers,
            "n_heads": n_heads,
            "batch": batch, "seq_len": seq_len,
            "params_M": n_params / 1e6,
            "peak_GB": peak_gb,
            "step_s": wall,
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {
            "ok": False, "d_model": d_model,
            "n_layers": n_layers,
            "error": "OOM",
        }
    except RuntimeError as e:
        torch.cuda.empty_cache()
        return {
            "ok": False, "d_model": d_model,
            "n_layers": n_layers,
            "error": str(e)[:100],
        }


def main():
    print("Probing HybridPCMMiniLM memory at various scales")
    print("=" * 78)
    print(f"  {'d':>5}  {'L':>3}  {'heads':>5}  "
          f"{'params (M)':>10}  {'peak (GB)':>10}  "
          f"{'step (ms)':>10}  {'status':<10}")
    print("-" * 78)
    # Sweep
    configs = [
        # baseline
        (128, 4), (128, 8),
        # 2x d
        (256, 4), (256, 8), (256, 12),
        # 3x d
        (384, 4), (384, 8), (384, 12),
        # 4x d
        (512, 4), (512, 8), (512, 12),
        # 6x d
        (768, 4), (768, 8), (768, 12),
        # 8x d
        (1024, 4), (1024, 8),
    ]
    results = []
    for d, L in configs:
        r = probe(d_model=d, n_layers=L, batch=64, seq_len=128)
        results.append(r)
        if r.get("ok"):
            print(
                f"  {r['d_model']:>5}  {r['n_layers']:>3}  "
                f"{r['n_heads']:>5}  {r['params_M']:>10.2f}  "
                f"{r['peak_GB']:>10.3f}  "
                f"{r['step_s']*1000:>10.1f}  OK",
                flush=True,
            )
        else:
            print(
                f"  {r['d_model']:>5}  {r['n_layers']:>3}      "
                f"          ---         ---         ---  "
                f"{r['error']}",
                flush=True,
            )
            # Skip the larger configs at this depth if OOM
    print()
    # Pick the largest config that fits in 5.6 GB
    target = 5.6
    fitting = [
        r for r in results
        if r.get("ok") and r["peak_GB"] <= target
    ]
    if fitting:
        best = max(fitting, key=lambda r: r["peak_GB"])
        print(f"Best fit ≤ {target} GB: d={best['d_model']}, "
              f"L={best['n_layers']}, peak={best['peak_GB']:.2f} GB, "
              f"step={best['step_s']*1000:.1f} ms")


if __name__ == "__main__":
    main()
