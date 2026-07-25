"""
Generate Strategy C Highlight Comparison Chart:
Demonstrating Strategy C's clear superiority on:
1. Test Set mAP50 (69.8% vs 67.4% Baseline)
2. Test Set Small-Object Recall AR_S (23.7% vs 17.7% Baseline)
3. Test Set Small-Object Precision mAP_S (13.8% vs 12.9% Baseline)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT      = r"p:\project\hackothon\jnn_shivamogga"
OUT_DIR   = os.path.join(ROOT, "output", "strategy_C_highres_fpn", "charts")
os.makedirs(OUT_DIR, exist_ok=True)

stages = ['Baseline', 'Strategy A\n(ModalityGate)', 'Strategy A+B\n(Loss Weighting)', 'Strategy A+B+C\n(High-Res P2 FPN)']
map50_scores = [67.4, 67.6, 65.1, 69.8]
ar_s_scores  = [17.7, 17.5, 17.7, 23.7]
map_s_scores = [12.9, 12.4, 12.1, 13.8]

# Dark Theme Styling
plt.style.use('dark_background')
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor='#0d1117')

colors_map50 = ['#4a5568', '#4a5568', '#e53e3e', '#38a169']
colors_recall = ['#4a5568', '#4a5568', '#4a5568', '#3182ce']
colors_maps   = ['#4a5568', '#4a5568', '#4a5568', '#d69e2e']

# Subplot 1: Test mAP50 (Peak Accuracy)
ax1 = axes[0]
bars1 = ax1.bar(stages, map50_scores, color=colors_map50, width=0.55, edgecolor='white', linewidth=1)
ax1.set_title('Test Split mAP50 (%) — Peak Accuracy', fontsize=13, fontweight='bold', pad=12, color='white')
ax1.set_ylim(60, 74)
ax1.grid(axis='y', linestyle='--', alpha=0.3)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.annotate('WINNER\n+2.4% vs Base', xy=(3, 69.8), xytext=(2.2, 72.0),
             arrowprops=dict(facecolor='#38a169', shrink=0.08, width=2, headwidth=8),
             fontsize=10, fontweight='bold', color='#38a169', ha='center')

# Subplot 2: Test AR_S (Small Object Recall Jump)
ax2 = axes[1]
bars2 = ax2.bar(stages, ar_s_scores, color=colors_recall, width=0.55, edgecolor='white', linewidth=1)
ax2.set_title('Test Small Object Recall AR_S (%)', fontsize=13, fontweight='bold', pad=12, color='white')
ax2.set_ylim(14, 26)
ax2.grid(axis='y', linestyle='--', alpha=0.3)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.annotate('+6.0% Recall Jump!\n(Tiny Pedestrians)', xy=(3, 23.7), xytext=(2.1, 24.8),
             arrowprops=dict(facecolor='#3182ce', shrink=0.08, width=2, headwidth=8),
             fontsize=10, fontweight='bold', color='#63b3ed', ha='center')

# Subplot 3: Test mAP_S (Small Object Precision)
ax3 = axes[2]
bars3 = ax3.bar(stages, map_s_scores, color=colors_maps, width=0.55, edgecolor='white', linewidth=1)
ax3.set_title('Test Small Object Precision mAP_S (%)', fontsize=13, fontweight='bold', pad=12, color='white')
ax3.set_ylim(10, 15)
ax3.grid(axis='y', linestyle='--', alpha=0.3)
for bar in bars3:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
ax3.annotate('Best Small-Obj Precision', xy=(3, 13.8), xytext=(2.2, 14.4),
             arrowprops=dict(facecolor='#d69e2e', shrink=0.08, width=2, headwidth=8),
             fontsize=10, fontweight='bold', color='#f6e05e', ha='center')

for ax in axes:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='white', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#30363d')

plt.suptitle('Strategy C (P2 High-Res FPN) Clear Test Benchmark Wins', fontsize=16, fontweight='bold', color='white', y=1.03)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "C_test_highlights_chart.png")
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
plt.close()

print(f"SUCCESS: Strategy C highlight chart saved to:\n  {out_path}")
