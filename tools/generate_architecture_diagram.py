"""
Generate High-Resolution System Architecture Diagram Graphic:
Visually illustrating the modified QFDet pipeline with 3 highlighted modifications:
  1. Strategy A: Spatially-Aware ModalityGate (injected at QCE Fusion)
  2. Strategy C: High-Resolution P2 Feature Level (injected at FPN Neck)
  3. Strategy B: Small-Object-Weighted Loss (Training Loss Enhancement)
Saved in: output/system_architecture_diagram.png
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(16, 9), facecolor='#0d1117')
ax.set_facecolor('#0d1117')
ax.axis('off')
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)

# Title
ax.text(8, 8.5, "System Architecture: Modified QFDet Pipeline", fontsize=20, fontweight='bold', ha='center', color='white')
ax.text(8, 8.1, "RGB-T Fusion Architecture with 3 Progressive Architectural & Loss Enhancements", fontsize=12, ha='center', color='#94a3b8')

def draw_box(ax, x, y, w, h, text, subtext="", bg_color="#1e293b", border_color="#475569", text_color="white", highlight=False, tag=None):
    if highlight:
        border_color = "#38bdf8"
        bg_color = "#0f172a"
        linewidth = 2.5
    else:
        linewidth = 1.5

    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                 facecolor=bg_color, edgecolor=border_color, linewidth=linewidth)
    ax.add_patch(rect)

    if tag:
        tag_box = patches.FancyBboxPatch((x + w - 1.8, y + h - 0.35), 1.7, 0.3, boxstyle="round,pad=0.05",
                                         facecolor="#0284c7", edgecolor="white", linewidth=1)
        ax.add_patch(tag_box)
        ax.text(x + w - 0.9, y + h - 0.2, tag, fontsize=8, fontweight='bold', ha='center', color='white')

    if subtext:
        ax.text(x + w/2, y + h/2 + 0.15, text, fontsize=11, fontweight='bold', ha='center', va='center', color=text_color)
        ax.text(x + w/2, y + h/2 - 0.20, subtext, fontsize=9, ha='center', va='center', color='#94a3b8')
    else:
        ax.text(x + w/2, y + h/2, text, fontsize=11, fontweight='bold', ha='center', va='center', color=text_color)

def draw_arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=2, mutation_scale=15))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.15, label, fontsize=9, fontweight='bold', ha='center', color='#cbd5e1')

# 1. Inputs (Left Column)
draw_box(ax, 0.5, 5.5, 2.2, 1.4, "RGB Frame", "(512 × 640)", bg_color="#1e1b4b", border_color="#6366f1")
draw_box(ax, 0.5, 2.5, 2.2, 1.4, "Thermal Frame", "(512 × 640)", bg_color="#450a0a", border_color="#ef4444")

# 2. Backbones (Dual Stream)
draw_box(ax, 3.4, 5.5, 2.5, 1.4, "ResNet-50 (RGB)", "Extracts C2, C3, C4, C5")
draw_box(ax, 3.4, 2.5, 2.5, 1.4, "ResNet-50 (Thermal)", "Extracts C2, C3, C4, C5")

draw_arrow(ax, 2.7, 6.2, 3.4, 6.2)
draw_arrow(ax, 2.7, 3.2, 3.4, 3.2)

# 3. Strategy A: ModalityGate + QCE Fusion
draw_box(ax, 6.6, 3.8, 3.0, 1.8, "QCE Fusion + ModalityGate", "Pixel-Wise Spatial Trust Meter\nW ∈ [0, 1]^(1×H×W)",
         bg_color="#0c4a6e", border_color="#38bdf8", highlight=True, tag="CHANGE #1 (A)")

draw_arrow(ax, 5.9, 6.2, 6.6, 4.9)
draw_arrow(ax, 5.9, 3.2, 6.6, 4.5)

# 4. Strategy C: FPN Neck + High-Res P2 Level
draw_box(ax, 10.3, 3.8, 2.8, 1.8, "FPN Neck + High-Res P2", "Stride-4 Pyramid Level (P2)\nPreserves Sub-16px Detail",
         bg_color="#064e3b", border_color="#22c55e", highlight=True, tag="CHANGE #2 (C)")

draw_arrow(ax, 9.6, 4.7, 10.3, 4.7, label="Fused Feats")

# 5. ATSS Detection Head
draw_box(ax, 13.8, 3.8, 1.8, 1.8, "ATSS Head", "Cls + BBox Reg\nPedestrian Boxes")

draw_arrow(ax, 13.1, 4.7, 13.8, 4.7)

# 6. Off-Diagram Training Loss Box (Strategy B)
draw_box(ax, 5.0, 0.5, 6.0, 1.3, "Strategy B — Small-Object-Weighted Loss (Training Only)",
         "Loss scaling (1.5× boost) for ground-truth boxes < 32² px",
         bg_color="#3b0764", border_color="#c084fc", highlight=True, tag="CHANGE #3 (B)")

# Legend / Notes at bottom right
ax.text(14.5, 1.3, "Legend:", fontsize=10, fontweight='bold', color='white')
ax.text(14.5, 0.9, "■ Baseline QFDet Modules", fontsize=9, color='#94a3b8')
ax.text(14.5, 0.5, "■ Highlighted Innovations", fontsize=9, color='#38bdf8')

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "system_architecture_diagram.png")
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
plt.close()

print(f"SUCCESS: System Architecture Diagram saved to:\n  {out_path}")
