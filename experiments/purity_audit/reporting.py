"""Report rendering for the purity-audit summary.json."""
from __future__ import annotations


__all__ = ["render_report"]


def render_report(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# Purity Audit Report")
    lines.append("")
    lines.append(f"- encoder_ckpt: `{summary['encoder_ckpt']}`")
    lines.append(f"- n_seeds: {summary['n_seeds']}")
    lines.append(f"- device: {summary['device']}")
    lines.append(f"- epochs: {summary['epochs']}, "
                 f"steps/epoch: {summary['steps_per_epoch']}")
    lines.append("")

    if "A1_random_orthogonal" in summary:
        a = summary["A1_random_orthogonal"]
        lines.append("## A1 · Random Orthogonal Centroids (encoder 污染测试)")
        lines.append("")
        lines.append(f"- single |ρ| = **{a['single_rho_arith']['mean']:.4f} "
                     f"± {a['single_rho_arith']['std']:.4f}**")
        lines.append(f"- dual   |ρ| = **{a['dual_rho_arith']['mean']:.4f} "
                     f"± {a['dual_rho_arith']['std']:.4f}**")
        lines.append(f"- dual   |ρ_ord| = {a['dual_rho_ord']['mean']:.4f} "
                     f"± {a['dual_rho_ord']['std']:.4f}")
        lines.append("")

    if "A1_random_gaussian" in summary:
        a = summary["A1_random_gaussian"]
        lines.append("## A1b · Random Gaussian Centroids (对照)")
        lines.append("")
        lines.append(f"- single |ρ| = **{a['single_rho_arith']['mean']:.4f} "
                     f"± {a['single_rho_arith']['std']:.4f}**")
        lines.append(f"- dual   |ρ| = **{a['dual_rho_arith']['mean']:.4f} "
                     f"± {a['dual_rho_arith']['std']:.4f}**")
        lines.append("")

    if "A2_shuffle_inverse" in summary:
        a = summary["A2_shuffle_inverse"]
        lines.append("## A2 · Shuffle-Inverse (shuffle 是坐标破坏还是 identity 破坏)")
        lines.append("")
        lines.append(f"- natural-order |ρ|   = {a['rho_natural']['mean']:.4f} "
                     f"± {a['rho_natural']['std']:.4f}")
        lines.append(f"- inverse-remap |ρ|  = **{a['rho_inverse_remapped']['mean']:.4f} "
                     f"± {a['rho_inverse_remapped']['std']:.4f}**")
        lines.append("")

    if "A3_init_scale" in summary:
        a = summary["A3_init_scale"]
        lines.append("## A3 · Init-Scale (near-zero init trivial 解释)")
        lines.append("")
        for strat in ("normal_small", "normal", "zero"):
            if strat in a:
                s = a[strat]["abs_rho"]; acc = a[strat]["acc"]
                lines.append(f"- init=`{strat}`: |ρ| = {s['mean']:.4f} "
                             f"± {s['std']:.4f}, acc = {acc['mean']:.3f} "
                             f"± {acc['std']:.3f}")
        lines.append("")

    if "A4_random_id" in summary:
        a = summary["A4_random_id"]
        lines.append("## A4 · Random Concept-ID (ID 字符串泄漏)")
        lines.append("")
        lines.append(f"- |ρ| = **{a['abs_rho']['mean']:.4f} "
                     f"± {a['abs_rho']['std']:.4f}**")
        lines.append(f"- acc = {a['acc']['mean']:.3f} ± {a['acc']['std']:.3f}")
        lines.append("")

    return "\n".join(lines)
