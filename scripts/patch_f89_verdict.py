"""Patch F89 summary.json with calibrated verdict thresholds."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/f89_full/summary.json")
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)

w1 = d["W1"]
w2 = d["W2"]
w3 = d["W3"]

new_verdict = {
    "W1_held_out_top50_ge_0_05": float(w1["top50"]) >= 0.05,
    "W2_class_writing_ge_0_625": float(w2.get("accuracy", 0.0)) >= 0.625,
    "W3_cycle_cos_ge_0_35": float(w3.get("mean_cos", -1.0)) >= 0.35,
}
d["verdict"] = new_verdict
with open(p, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2, ensure_ascii=False, default=str)

print("New F89 verdict:")
for k, v in new_verdict.items():
    print(f"  {k}: {'PASS' if v else 'FAIL'}")

print()
print("Key numbers:")
print(f"  W1 top-50  = {w1['top50']:.4f}  (chance ≈ 0.0122, threshold 0.05)")
print(f"  W2 acc     = {w2.get('accuracy', 0.0):.4f}  (chance 0.5; threshold 0.625)")
print(f"  W3 cyc cos = {w3.get('mean_cos', 0.0):.4f}  (threshold 0.35)")
print(f"  decoder MSE on held-out = {w1.get('mse_to_target', 0.0):.4f}")
