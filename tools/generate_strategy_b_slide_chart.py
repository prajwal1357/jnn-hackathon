"""
Generate Strategy B Presentation Slide Graphic
Slide Title: Strategy B — Small-Object-Weighted Loss
Includes:
- Problem & Solution Callout Card
- Category Tag: Learning Improvements -> Loss Function
- Before/After Comparison Chart: QFDet_A vs QFDet_A_B (mAP_S & AR_S)
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "output", "strategy_B_small_object_loss", "charts")
os.makedirs(OUT_DIR, exist_ok=True)

plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 7.5), facecolor='#0d1117')

# Title & Category Banner
fig.text(0.05, 0.93, "Strategy B — Small-Object-Weighted Loss", fontsize=20, fontweight='bold', color='white')
fig.text(0.05, 0.88, "Category: Learning Improvements → Loss Function", fontsize=12, fontweight='bold', color='#38bdf8')

# Left Column: Problem & Solution Cards (Text Box Area)
ax_left = fig.add_axes([0.05, 0.10, 0.42, 0.73], facecolor='#161b22')
ax_left.axis('off')

# Border around Left Box
rect = plt.Rectangle((0, 0), 1, 1, transform=ax_left.transAxes, fill=False, edgecolor='#30363d', linewidth=2)
ax_left.add_patch(rect)

ax_left.text(0.05, 0.90, "⚠️ PROBLEM", fontsize=14, fontweight='bold', color='#ef4444')
problem_text = (
    "Standard loss functions treat detection errors on large\n"
    "and tiny pedestrians equally.\n\n"
    "However, in drone imagery, tiny pedestrians (<32² px)\n"
    "are both the most common and the hardest to detect,\n"
    "causing standard models to miss small ground targets."
)
ax_left.text(0.05, 0.65, problem_text, fontsize=11, color='#e2e8f0', linespacing=1.4)

ax_left.text(0.05, 0.42, "SOLUTION", fontsize=14, fontweight='bold', color='#22c55e')
solution_text = (
    "Up-weight loss contribution from small ground-truth\n"
    "bounding boxes (< 32² px) by 1.5×.\n\n"
    "This forces gradient updates during training to prioritize\n"
    "tiny-pedestrian localization and boundary alignment."
)
ax_left.text(0.05, 0.18, solution_text, fontsize=11, color='#e2e8f0', linespacing=1.4)

# Right Column: Before / After Bar Chart (QFDet_A vs QFDet_A_B)
ax_right = fig.add_axes([0.53, 0.12, 0.42, 0.70], facecolor='#161b22')

metrics = ['Val AR_S\n(Recall)', 'Test AR_S\n(Recall)', 'Val mAP_S\n(Precision)']
qfdet_a   = [22.0, 17.5, 15.0]
qfdet_a_b = [23.2, 20.5, 14.2]

x = np.arange(len(metrics))
width = 0.35

rects1 = ax_right.bar(x - width/2, qfdet_a, width, label='qfdet_A (Baseline Gate)', color='#64748b', edgecolor='white', linewidth=1)
rects2 = ax_right.bar(x + width/2, qfdet_a_b, width, label='qfdet_A_B (Loss Weighted)', color='#3b82f6', edgecolor='white', linewidth=1)

ax_right.set_title("Before / After Comparison: qfdet_A vs qfdet_A_B", fontsize=13, fontweight='bold', pad=12, color='white')
ax_right.set_ylabel("Score (%)", fontsize=12, fontweight='bold', color='white')
ax_right.set_xticks(x)
ax_right.set_xticklabels(metrics, fontsize=11, fontweight='bold', color='white')
ax_right.set_ylim(0, 30)
ax_right.grid(axis='y', linestyle='--', alpha=0.3)
ax_right.legend(fontsize=10, loc='upper right', facecolor='#0d1117', edgecolor='#30363d')

for rects in [rects1, rects2]:
    for bar in rects:
        yval = bar.get_height()
        ax_right.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')

# Annotate Recall Boost
ax_right.annotate('+3.0% Recall Boost!',
                  xy=(1 + width/2, 20.5), xycoords='data',
                  xytext=(1.2, 26), textcoords='data',
                  arrowprops=dict(facecolor='#ef4444', shrink=0.08, width=2, headwidth=8),
                  ha='center', va='bottom', fontsize=11, fontweight='bold', color='#f87171',
                  bbox=dict(boxstyle="round,pad=0.3", fc="#450a0a", ec="#ef4444", lw=1.5, alpha=0.9))

for spine in ax_right.spines.values():
    spine.set_color('#30363d')

out_path = os.path.join(OUT_DIR, "strategy_b_slide_chart.png")
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
plt.close()

print(f"SUCCESS: Strategy B slide chart saved to:\n  {out_path}")
