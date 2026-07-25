import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = r"p:\project\hackothon\jnn_shivamogga"
GRAPHS_DIR = os.path.join(ROOT, "output", "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

def generate_stage1_graph():
    output_path = os.path.join(GRAPHS_DIR, "stage1_scale_distribution.png")

    splits = ['Train (8,138)', 'Val (2,337)', 'Test (2,068)']
    small_pct = [9.94, 18.10, 25.58]
    medium_pct = [66.96, 68.12, 61.41]
    large_pct = [23.10, 13.78, 13.01]

    x = np.arange(len(splits))
    width = 0.25

    plt.figure(figsize=(10, 6), dpi=150)
    rects1 = plt.bar(x - width, small_pct, width, label='Small (<32² px)', color='#e74c3c')
    rects2 = plt.bar(x, medium_pct, width, label='Medium (32²-96² px)', color='#f39c12')
    rects3 = plt.bar(x + width, large_pct, width, label='Large (≥96² px)', color='#2ecc71')

    plt.ylabel('Percentage of Pedestrian Instances (%)', fontsize=12, fontweight='bold')
    plt.title('Stage 1: Pedestrian Scale Distribution Across Dataset Splits', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(x, splits, fontsize=11, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.ylim(0, 80)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.1f}%',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[✓] Successfully generated Stage 1 graph: '{output_path}'")

def generate_stage2_graph():
    output_path = os.path.join(GRAPHS_DIR, "stage2_baseline_comparison.png")

    metrics = ['mAP', 'mAP50', 'mAP75', 'mAPS', 'mAPM', 'mAPL']
    x = np.arange(len(metrics))
    width = 0.25

    color_rgb = '#2b5c8f'
    color_thermal = '#e63946'
    color_qfdet = '#2a9d8f'

    val_rgb = [6.9, 23.1, 2.3, 0.5, 6.5, 17.3]
    val_thermal = [26.9, 57.1, 22.0, 8.7, 25.2, 56.6]
    val_qfdet = [33.8, 72.1, 27.3, 14.4, 32.4, 58.5]

    test_rgb = [5.5, 18.6, 1.9, 0.5, 5.2, 14.2]
    test_thermal = [22.0, 52.4, 15.6, 7.5, 21.7, 49.8]
    test_qfdet = [29.9, 67.4, 22.7, 12.9, 29.9, 55.5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    ax1.bar(x - width, val_rgb, width, label='RGB-Only', color=color_rgb)
    ax1.bar(x, val_thermal, width, label='Thermal-Only', color=color_thermal)
    ax1.bar(x + width, val_qfdet, width, label='QFDet Baseline', color=color_qfdet)

    ax1.set_title('Validation Dataset Performance (298 Images)', fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10)
    ax1.legend(loc='upper right', fontsize=9.5)
    ax1.set_ylim(0, 80)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    ax2.bar(x - width, test_rgb, width, label='RGB-Only', color=color_rgb)
    ax2.bar(x, test_thermal, width, label='Thermal-Only', color=color_thermal)
    ax2.bar(x + width, test_qfdet, width, label='QFDet Baseline', color=color_qfdet)

    ax2.set_title('Test Dataset Performance (199 Images)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=10)
    ax2.legend(loc='upper right', fontsize=9.5)
    ax2.set_ylim(0, 80)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[✓] Successfully generated Stage 2 baseline graph: '{output_path}'")

def generate_ablation_graph():
    output_path = os.path.join(GRAPHS_DIR, "strategy_ab_ablation_graph.png")

    models = ['Baseline', 'Strategy A\n(ModalityGate)', 'Strategy A+B\n(Small-Obj Loss)', 'Strategy A+B+C\n(High-Res P2)']
    map_overall = [29.9, 29.4, 28.6, 28.8]
    map_s = [12.9, 12.4, 12.0, 13.8]
    map50 = [67.4, 67.6, 65.1, 69.8]
    ar_s = [17.7, 17.5, 20.5, 23.7]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)

    rects1 = ax.bar(x - 1.5*width, map_overall, width, label='mAP (Overall)', color='#4c72b0')
    rects2 = ax.bar(x - 0.5*width, map50, width, label='mAP50 (Accuracy)', color='#2a9d8f')
    rects3 = ax.bar(x + 0.5*width, map_s, width, label='mAP_S (Tiny Precision)', color='#dd8452')
    rects4 = ax.bar(x + 1.5*width, ar_s, width, label='AR_S (Tiny Recall)', color='#e74c3c')

    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Progression across Strategies A, B, and C (Test Split)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, 80)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    ax.annotate('+6.0% Recall Boost!',
                xy=(3 + 1.5*width, 23.7), xycoords='data',
                xytext=(2.6, 32), textcoords='data',
                arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="red", lw=1.5, alpha=0.9))

    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[✓] Generated clean Ablation Graph: '{output_path}'")

if __name__ == '__main__':
    generate_stage1_graph()
    generate_stage2_graph()
    generate_ablation_graph()
