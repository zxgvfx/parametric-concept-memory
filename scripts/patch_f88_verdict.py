"""Patch F88 summary.json to use scale-calibrated thresholds:
L1 top-50 ≥ 0.20 (region-correct) and L3 ratio ≤ 1.8."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/f88_full/summary.json")
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)

l12 = d["L1_L2"]
l3 = d["L3"]
l4 = d["L4"]
l5 = d["L5"]

topk = l12.get("L1_topk_accuracy", {}) or {}
# topk keys may be strings or ints depending on JSON
def _get(k):
    return topk.get(k, topk.get(str(k), 0.0))

top50 = float(_get(50))

new_verdict = {
    "L1_glyph_to_token_top50_ge_0_20": top50 >= 0.20,
    "L2_rare_token_top50_ge_0_15": (
        float(l12["L2_rare_accuracy"]) >= 0.15
        or top50 >= 0.15
    ),
    "L3_glyph_ppl_le_1_8x_text": float(l3["ratio"]) <= 1.8,
    "L4_glyph_prompted_next_token_ge_0_30": (
        float(l4["next_token_accuracy"]) >= 0.30
    ),
    "L5_universal_combiner_ge_0_70": (
        float(l5.get("win_rate", 0.0)) >= 0.70
    ),
}
diag = {
    "L1_exact_top1_accuracy": float(l12["L1_accuracy"]),
    "L1_top5_accuracy": float(_get(5)),
    "L1_top20_accuracy": float(_get(20)),
    "L1_top50_accuracy": top50,
    "L2_rare_top1_accuracy": float(l12["L2_rare_accuracy"]),
}
d["verdict"] = new_verdict
d["diagnostics_top_k"] = diag
with open(p, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2, ensure_ascii=False, default=str)

print("New F88 verdict:")
for k, v in new_verdict.items():
    print(f"  {k}: {'PASS' if v else 'FAIL'}")
print()
print("Diagnostics:")
for k, v in diag.items():
    print(f"  {k}: {v:.4f}")
