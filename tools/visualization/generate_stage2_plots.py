import os
import json
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = r"p:\project\hackothon\jnn_shivamogga\output\stage2_results"
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "stage2_final_summary.json")

def generate_plots():
    if not os.path.exists(SUMMARY_FILE):
        print(f"Summary file not found at {SUMMARY_FILE}")
        return

    with open(SUMMARY_FILE, "r") as f:
        data = json.load(f)

    modes = list(data.keys())
    splits = ["val", "test"]

    # Colors for RGB, Thermal, Fused
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 1. mAP Comparison Bar Chart (Val vs Test)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(modes))
    width = 0.35

    val_maps = [data[m]["val"]["mAP"] for m in modes]
    test_maps = [data[m]["test"]["mAP"] for m in modes]

    rects1 = ax.bar(x - width/2, val_maps, width, label='Val Split (mAP)', color='#3498db')
    rects2 = ax.bar(x + width/2, test_maps, width, label='Test Split (mAP)', color='#2ecc71')

    ax.set_ylabel('mAP (%)', fontsize=12, fontweight='bold')
    ax.set_title('VTUAV Benchmark: Multimodal Detection Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    chart1_path = os.path.join(OUTPUT_DIR, "benchmark_map_comparison.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"Saved {chart1_path}")

    # 2. Scale-wise Detection Performance (mAP_S, mAP_M, mAP_L) for Val Split
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(modes))
    width = 0.25

    maps_s = [data[m]["val"]["mAPS"] for m in modes]
    maps_m = [data[m]["val"]["mAPM"] for m in modes]
    maps_l = [data[m]["val"]["mAPL"] for m in modes]

    r1 = ax.bar(x - width, maps_s, width, label='Small (<32² px)', color='#e74c3c')
    r2 = ax.bar(x, maps_m, width, label='Medium (32²-96² px)', color='#f39c12')
    r3 = ax.bar(x + width, maps_l, width, label='Large (≥96² px)', color='#9b59b6')

    ax.set_ylabel('mAP (%)', fontsize=12, fontweight='bold')
    ax.set_title('Detection Performance Across Pedestrian Scale Categories (Val Split)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    autolabel(r1)
    autolabel(r2)
    autolabel(r3)

    plt.tight_layout()
    chart2_path = os.path.join(OUTPUT_DIR, "benchmark_scale_performance.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"Saved {chart2_path}")

if __name__ == "__main__":
    generate_plots()
