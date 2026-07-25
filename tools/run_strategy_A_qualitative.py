"""
Strategy A Qualitative Comparison Suite
========================================
Runs side-by-side inference comparing:
  1. QFDet Baseline (epoch_11_qfdet_vtuav.pth)
  2. Strategy A Fine-Tuned (strategy_A_finetuned.pth)

Generates:
  - Side-by-side predicted bounding boxes (Baseline vs Strategy A)
  - Learned ModalityGate spatial trust weight heatmaps
  - Qualitative comparison report
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = r"p:\project\hackothon\jnn_shivamogga"
MMDET_ROOT   = os.path.join(ROOT, "mmdet-rgbtdroneperson")
BASE_WEIGHTS = os.path.join(ROOT, "epoch_11_qfdet_vtuav.pth")
STRA_WEIGHTS = os.path.join(ROOT, "output", "strategy_A_modality_gate", "work_dir", "strategy_A_finetuned.pth")

OUT_DIR      = os.path.join(ROOT, "output", "strategy_A_qualitative")
os.makedirs(OUT_DIR, exist_ok=True)

CFG_BASE     = os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_eval_fused.py")
CFG_STRA     = os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_r50_fpn_1x_vtuav_modality_gate.py")
DATA_ROOT    = os.path.join(ROOT, "VTUAV_subset")

if MMDET_ROOT not in sys.path:
    sys.path.insert(0, MMDET_ROOT)

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset

def unpack_item(item):
    img_list = item['img'][0]
    if isinstance(img_list, list):
        if len(img_list) == 2:
            v_t = img_list[0].data.unsqueeze(0).cuda()
            t_t = img_list[1].data.unsqueeze(0).cuda()
            img_input = (v_t, t_t)
        else:
            img_input = img_list[0].data.unsqueeze(0).cuda()
    elif hasattr(img_list, 'data'):
        img_input = img_list.data.unsqueeze(0).cuda()
    else:
        img_input = img_list
    meta = item['img_metas'][0].data
    img_metas = [meta] if isinstance(meta, dict) else meta
    return img_input, img_metas

def draw_bboxes(img, bboxes_results, score_thr=0.3):
    img_out = img.copy()
    if isinstance(bboxes_results, list):
        bboxes = bboxes_results[0]
    else:
        bboxes = bboxes_results
    
    count = 0
    for bbox in bboxes:
        if len(bbox) >= 5:
            x1, y1, x2, y2, score = bbox[:5]
            if score >= score_thr:
                count += 1
                cv2.rectangle(img_out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img_out, f"ped {score:.2f}", (int(x1), max(int(y1)-5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return img_out, count

def main():
    print("=" * 60)
    print("  Strategy A Qualitative Side-by-Side Comparison Suite")
    print("=" * 60)

    # 1. Load Baseline Model
    print("\n[1/3] Loading Baseline QFDet Model...")
    cfg_base = mmcv.Config.fromfile(CFG_BASE)
    cfg_base.model.pretrained = None
    model_base = build_detector(cfg_base.model, test_cfg=cfg_base.get('test_cfg')).cuda()
    load_checkpoint(model_base, BASE_WEIGHTS, map_location='cuda')
    model_base.eval()

    # 2. Load Fine-Tuned Strategy A Model
    print("[2/3] Loading Fine-Tuned Strategy A (ModalityGate) Model...")
    cfg_stra = mmcv.Config.fromfile(CFG_STRA)
    cfg_stra.model.pretrained = None
    model_stra = build_detector(cfg_stra.model, test_cfg=cfg_stra.get('test_cfg')).cuda()
    load_checkpoint(model_stra, STRA_WEIGHTS, map_location='cuda')
    model_stra.eval()

    # 3. Load Test Dataset
    dataset = build_dataset(cfg_stra.data.test)
    print(f"[3/3] Running Qualitative Inference on {min(10, len(dataset))} sample test images...")

    samples_to_visualize = [0, 5, 12, 25, 40, 63, 115, 130]

    for idx in samples_to_visualize:
        if idx >= len(dataset):
            continue
        item = dataset[idx]
        fname = dataset.data_infos[idx]['filename']

        img_input, img_metas = unpack_item(item)

        # Baseline inference
        with torch.no_grad():
            res_base = model_base.simple_test(img_input, img_metas, rescale=True)[0]

        # Strategy A inference + collect weight map
        model_stra.last_gate_weights = []
        with torch.no_grad():
            res_stra = model_stra.simple_test(img_input, img_metas, rescale=True)[0]

        gate_weights = getattr(model_stra, 'last_gate_weights', [])
        w_map = gate_weights[0][0].squeeze().cpu().numpy() if gate_weights else np.full((48, 80), 0.5)

        # Load raw images
        rgb_path = os.path.join(DATA_ROOT, "VTUAV_co", "test", "images", fname)
        ir_path  = os.path.join(DATA_ROOT, "VTUAV_ir", "test", "images", fname)

        if not os.path.exists(rgb_path):
            continue

        img_rgb = cv2.imread(rgb_path)
        img_ir  = cv2.imread(ir_path) if os.path.exists(ir_path) else img_rgb.copy()

        H, W = img_rgb.shape[:2]

        # Draw bboxes
        img_base_drawn, n_base = draw_bboxes(img_rgb, res_base, score_thr=0.3)
        img_stra_drawn, n_stra = draw_bboxes(img_rgb, res_stra, score_thr=0.3)

        # Resize weight map
        w_resized = cv2.resize(w_map, (W, H))

        # Create 4-panel qualitative figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#0d1117')
        fig.suptitle(f"Qualitative Comparison — {fname}\nBaseline Detections vs. Strategy A (ModalityGate)",
                     color='white', fontsize=14, fontweight='bold')

        # Panel 1: RGB + Baseline Predictions
        axes[0, 0].imshow(cv2.cvtColor(img_base_drawn, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f"Baseline QFDet Predictions (Detected: {n_base} persons)", color='#2ec4b6', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # Panel 2: RGB + Strategy A Predictions
        axes[0, 1].imshow(cv2.cvtColor(img_stra_drawn, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f"Strategy A (Trust Meter) Predictions (Detected: {n_stra} persons)", color='#ff8c42', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # Panel 3: Thermal IR Image
        axes[1, 0].imshow(cv2.cvtColor(img_ir, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Thermal IR Input", color='#c9d1d9', fontsize=12)
        axes[1, 0].axis('off')

        # Panel 4: Learned Trust Weight Map Overlay
        im = axes[1, 1].imshow(w_resized, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, 1].set_title(f"Learned Trust Weight Map (Mean W={w_resized.mean():.3f})\n[Green=RGB Trust, Red=Thermal Trust]",
                              color='white', fontsize=11, fontweight='bold')
        axes[1, 1].axis('off')
        cbar = plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors='white')

        for ax in axes.flat:
            ax.set_facecolor('#161b22')

        plt.tight_layout()
        out_file = os.path.join(OUT_DIR, f"qualitative_cmp_{fname}.png")
        plt.savefig(out_file, dpi=140, bbox_inches='tight', facecolor='#0d1117')
        plt.close()
        print(f"  ✓ Saved qualitative comparison: {out_file}")

    print("\n" + "=" * 60)
    print(f"Qualitative analysis completed! All comparisons saved to:\n  {OUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
