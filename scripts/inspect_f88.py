"""Inspect F88 summary.json — print top-K NN accuracies."""
import json
from pathlib import Path

p = Path("outputs/f88_full/summary.json")
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)

l12 = d["L1_L2"]
print("L1 top-K NN accuracies (encoder output → tok_emb):")
print(f"  top-1   (exact): {l12['L1_accuracy']:.4f}  (n={l12['L1_n']})")
for k, v in l12.get("L1_topk_accuracy", {}).items():
    print(f"  top-{int(k):>3}        : {v:.4f}")
print()
print(f"L2 rare-token top-1: {l12['L2_rare_accuracy']:.4f}")
print()
l5 = d["L5"]
print("F62 cross-modal universal combiner:")
print(f"  win_rate     : {l5['win_rate']:.4f}")
print(f"  cos correct  : {l5['mean_cos_correct']:.4f}")
print(f"  cos wrong    : {l5['mean_cos_wrong']:.4f}")
print(f"  margin       : {l5['mean_cos_correct'] - l5['mean_cos_wrong']:.4f}")
print()
l3 = d["L3"]
print(f"L3 PPL ratio: text={l3['text_ppl']:.2f}, glyph={l3['glyph_ppl']:.2f}, ratio={l3['ratio']:.3f}")
print(f"L4 glyph-prompted next-token accuracy: {d['L4']['next_token_accuracy']:.4f}")
