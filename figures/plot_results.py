#!/usr/bin/env python3
"""Generate result figures for the OpenVLA 1-bit quantization report."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent

# ── load results ────────────────────────────────────────────────────────────
with open(OUT.parent / "results" / "quant_results.json") as f:
    quant = json.load(f)
with open(OUT.parent / "results" / "action_accuracy.json") as f:
    action = json.load(f)

DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
BLUE  = "#4C72B0"
RED   = "#DD8452"
GREEN = "#55A868"
GRAY  = "#8E9BAA"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ── Figure 1: Perplexity comparison ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ppls   = [quant["fp16_ppl"] / 1e9, quant["quant_1bit_ppl"] / 1e9]
labels = ["FP16\n(bfloat16)", "1-bit\n(BiLLM BRAQ)"]
bars = ax.bar(labels, ppls, color=[BLUE, RED], width=0.45, edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, ppls):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{val:.0f}B", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("C4 Perplexity (billions)", fontsize=11)
ax.set_title("LLM Perplexity: FP16 vs 1-bit Quantized\n"
             "(high baseline = action fine-tuning replaces text knowledge)", fontsize=10)
ax.set_ylim(0, max(ppls) * 1.2)
pct = (quant["quant_1bit_ppl"] / quant["fp16_ppl"] - 1) * 100
ax.annotate(f"+{pct:.0f}%", xy=(1, ppls[1]), xytext=(1.35, ppls[1] * 0.7),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=11, color=RED, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "perplexity_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved perplexity_comparison.png")

# ── Figure 2: Per-joint L1 error ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
l1s  = action["per_dim_l1"]
stds = action["per_dim_std"]
x    = np.arange(len(DIM_LABELS))
bars = ax.bar(x, l1s, color=[RED if v > 0.05 else BLUE for v in l1s],
              yerr=stds, capsize=4, width=0.6, edgecolor="white", linewidth=0.8,
              error_kw={"ecolor": GRAY, "linewidth": 1.2})
for xi, (v, s) in enumerate(zip(l1s, stds)):
    ax.text(xi, v + s + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels(DIM_LABELS, fontsize=11)
ax.set_ylabel("Mean L1 Error", fontsize=11)
ax.set_title("1-bit vs FP16: Per-Dimension Action Error (50 robot scenes)\n"
             "Red = error > 0.05 (unsafe for robot control)", fontsize=10)
ax.axhline(0.05, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="0.05 threshold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "action_error_per_dim.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved action_error_per_dim.png")

# ── Figure 3: FP16 vs 1-bit action scatter (first 5 samples, all dims) ─────
fp16_s = np.array(action["fp16_actions_sample"])   # (5, 7)
qnt_s  = np.array(action["quant_actions_sample"])  # (5, 7)

fig, axes = plt.subplots(1, 7, figsize=(14, 3), sharey=False)
for di, (ax, label) in enumerate(zip(axes, DIM_LABELS)):
    ax.scatter(fp16_s[:, di], qnt_s[:, di], color=RED, s=60, alpha=0.85, zorder=3)
    lo = min(fp16_s[:, di].min(), qnt_s[:, di].min()) - 0.05
    hi = max(fp16_s[:, di].max(), qnt_s[:, di].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5, label="perfect")
    ax.set_xlabel("FP16", fontsize=9)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
axes[0].set_ylabel("1-bit", fontsize=9)
fig.suptitle("FP16 vs 1-bit Predicted Actions (5 samples each dimension)\n"
             "Points on diagonal = perfect agreement", fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "action_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved action_scatter.png")

# ── Figure 4: Quantization time per layer ───────────────────────────────────
times = quant["quant_timing"]["per_layer_seconds"]
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.bar(range(len(times)), times, color=BLUE, width=0.8, edgecolor="white", linewidth=0.4)
ax.axhline(np.mean(times), color=RED, linestyle="--", linewidth=1.2,
           label=f"Mean: {np.mean(times):.1f}s")
ax.set_xlabel("Llama Decoder Layer", fontsize=11)
ax.set_ylabel("Time (seconds)", fontsize=11)
ax.set_title(f"BiLLM BRAQ Quantization Time per Layer  "
             f"(total: {quant['quant_timing']['total_seconds']/60:.1f} min)", fontsize=10)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "quant_time_per_layer.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved quant_time_per_layer.png")

print("\nAll figures generated in", OUT)
