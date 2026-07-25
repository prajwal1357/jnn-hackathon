import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = r"p:\project\hackothon\jnn_shivamogga\output\stage2_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Metrics Data ──────────────────────────────────────────────────────────────
models = [
    "GitHub Baseline QFDet\n(Paper Reported)",
    "RGB-Only\n(Our Val)",
    "Thermal-Only\n(Our Val)",
    "Full QFDet Fused\n(Our Val)",
    "Full QFDet Fused\n(Our Test)"
]

mAP_vals   = [31.10, 6.90,  26.90, 33.80, 29.90]
mAP50_vals = [70.40, 23.10, 57.10, 72.10, 67.40]
mAP75_vals = [22.90, 2.30,  22.00, 27.30, 22.70]

metrics = ["mAP (0.50:0.95)", "mAP50 (IoU 0.50)", "mAP75 (IoU 0.75)"]
x = np.arange(len(metrics))

colors = ["#ffd166", "#3a86ff", "#ff4757", "#2ec4b6", "#a855f7"]

# ── Create Grouped Bar Chart ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7), facecolor='#0d1117')
ax.set_facecolor('#161b22')

n_models = len(models)
width = 0.15
offsets = np.linspace(-(n_models - 1)/2, (n_models - 1)/2, n_models) * width

for j in range(n_models):
    vals = [mAP_vals[j], mAP50_vals[j], mAP75_vals[j]]
    bars = ax.bar(x + offsets[j], vals, width, color=colors[j],
                  label=models[j].replace('\n', ' '), alpha=0.92,
                  edgecolor='white', linewidth=0.5, zorder=3)
    
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.8,
                f"{v:.1f}%", ha='center', va='bottom', fontsize=8.5,
                color='white', fontweight='bold')

ax.set_ylabel("Accuracy (%)", color='#c9d1d9', fontsize=12, labelpad=10)
ax.set_title("Baseline Performance Matrix: GitHub Baseline QFDet vs. Our Multimodal Baselines\n(Before Strategy A Implementation)",
             color='white', fontsize=14, fontweight='bold', pad=18)
ax.set_xticks(x)
ax.set_xticklabels(metrics, color='#c9d1d9', fontsize=11, fontweight='bold')
ax.tick_params(colors='#c9d1d9')

ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, color='#21262d', linestyle='--', alpha=0.7, zorder=0)
ax.set_ylim(0, 85)

# Highlight baseline reference line for mAP
ax.axhline(31.10, color='#ffd166', linestyle=':', linewidth=1.5, alpha=0.7, label='GitHub Baseline mAP (31.1%)')

ax.legend(loc='upper right', facecolor='#21262d', edgecolor='#30363d',
          labelcolor='white', fontsize=9.5, framealpha=0.9)

# Annotation Box
insight_text = (
    "Key Insights:\n"
    "• GitHub Baseline QFDet: mAP=31.10%, mAP50=70.40%, mAP75=22.90%\n"
    "• Our Full QFDet (Val): mAP=33.80% (+2.70% over paper baseline)\n"
    "• Thermal-Only dominates RGB-Only (26.9% vs 6.9% mAP) on drone scenes"
)
ax.text(0.02, 0.95, insight_text, transform=ax.transAxes,
        color='white', fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', edgecolor='#30363d', alpha=0.9))

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "github_vs_our_baselines_comparison.png")
plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='#0d1117')
plt.close()

print(f"Chart saved successfully to {out_path}")
