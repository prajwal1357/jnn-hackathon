"""
Side-by-Side Qualitative Comparison for Image 03284.jpg:
Comparing QFDet Baseline vs. Strategy C (High-Resolution P2 FPN Level)
"""

import os
import sys
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT       = r"p:\project\hackothon\jnn_shivamogga"
MMDET_ROOT = os.path.join(ROOT, "mmdet-rgbtdroneperson")
DATA_ROOT  = os.path.join(ROOT, "VTUAV_subset")

OUT_DIR_C  = os.path.join(ROOT, "output", "strategy_C_qualitative")
OUT_DIR_A  = os.path.join(ROOT, "output", "strategy_A_qualitative")
for d in [OUT_DIR_C, OUT_DIR_A]:
    os.makedirs(d, exist_ok=True)

if MMDET_ROOT not in sys.path:
    sys.path.insert(0, MMDET_ROOT)

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset

BASE_CFG   = os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_eval_fused.py")
BASE_CKPT  = os.path.join(ROOT, "epoch_11_qfdet_vtuav.pth")

STRATC_CFG = os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_r50_fpn_1x_vtuav_strategy_C.py")
STRATC_CKPT= os.path.join(ROOT, "output", "strategy_C_highres_fpn", "work_dir", "strategy_C_finetuned.pth")

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

def draw_bboxes(img, bboxes, score_thr=0.30, color=(0, 255, 0)):
    out = img.copy()
    count = 0
    for cat_idx, cat_bboxes in enumerate(bboxes):
        for bbox in cat_bboxes:
            score = bbox[4]
            if score >= score_thr:
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                label_text = f"{score:.2f}"
                cv2.putText(out, label_text, (x1, max(y1-4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                count += 1
    return out, count

def main():
    img_id = "03284.jpg"
    rgb_path = os.path.join(DATA_ROOT, "VTUAV_co", "test", "images", img_id)
    ir_path  = os.path.join(DATA_ROOT, "VTUAV_ir", "test", "images", img_id)

    print(f"Generating Strategy C Side-by-Side Comparison for {img_id}...")

    # Load dataset sample
    cfg_c = mmcv.Config.fromfile(STRATC_CFG)
    dataset = build_dataset(cfg_c.data.test)
    for info in dataset.data_infos:
        if not info['filename'].startswith('test/images/'):
            info['filename'] = f"test/images/{info['file_name']}"

    target_idx = None
    for idx, info in enumerate(dataset.data_infos):
        if img_id in info['filename']:
            target_idx = idx
            break

    if target_idx is None:
        target_idx = 0

    item = dataset[target_idx]
    img_input, img_metas = unpack_item(item)

    raw_rgb = cv2.imread(rgb_path)
    raw_rgb_rgb = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB)
    raw_ir = cv2.imread(ir_path)
    raw_ir_rgb = cv2.cvtColor(raw_ir, cv2.COLOR_BGR2RGB)

    # 1. Baseline Inference
    print("  --> Running QFDet Baseline model...")
    cfg_base = mmcv.Config.fromfile(BASE_CFG)
    cfg_base.model.pretrained = None
    model_base = build_detector(cfg_base.model, test_cfg=cfg_base.get('test_cfg')).cuda()
    load_checkpoint(model_base, BASE_CKPT, map_location='cuda')
    model_base.eval()

    with torch.no_grad():
        res_base = model_base.simple_test(img_input, img_metas, rescale=True)
        bboxes_base = res_base[0]

    img_base, count_base = draw_bboxes(raw_rgb_rgb, bboxes_base, score_thr=0.30, color=(220, 20, 60))

    # 2. Strategy C Inference
    print("  --> Running Strategy C (High-Res P2 FPN) model...")
    cfg_c.model.pretrained = None
    model_c = build_detector(cfg_c.model, test_cfg=cfg_c.get('test_cfg')).cuda()
    load_checkpoint(model_c, STRATC_CKPT, map_location='cuda')
    model_c.eval()

    with torch.no_grad():
        res_c = model_c.simple_test(img_input, img_metas, rescale=True)
        bboxes_c = res_c[0]

    img_c, count_c = draw_bboxes(raw_rgb_rgb, bboxes_c, score_thr=0.30, color=(50, 205, 50))

    # 3. Create Side-by-Side Figure (3 Panels: RGB Input, Baseline, Strategy C)
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0d1117')
    fig.suptitle(f"Qualitative Comparison — Sample {img_id}: Baseline vs Strategy C (High-Res P2 FPN)",
                 fontsize=16, fontweight='bold', color='white', y=0.98)

    axes[0].imshow(raw_rgb_rgb)
    axes[0].set_title(f"Input RGB Frame ({img_id})", fontsize=12, fontweight='bold', color='white', pad=8)
    axes[0].axis('off')

    axes[1].imshow(img_base)
    axes[1].set_title(f"QFDet Baseline\n({count_base} Detections @ Thr 0.30)", fontsize=12, fontweight='bold', color='#ef4444', pad=8)
    axes[1].axis('off')

    axes[2].imshow(img_c)
    axes[2].set_title(f"Strategy C (High-Res P2 FPN)\n({count_c} Detections @ Thr 0.30)", fontsize=12, fontweight='bold', color='#22c55e', pad=8)
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save to Strategy C qualitative directory
    out_c_path = os.path.join(OUT_DIR_C, f"qualitative_cmp_{img_id}.png")
    plt.savefig(out_c_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')

    # Overwrite in Strategy A qualitative directory as requested by user
    out_a_path = os.path.join(OUT_DIR_A, f"qualitative_cmp_{img_id}.png")
    plt.savefig(out_a_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')

    plt.close()

    print(f"\nSUCCESS: Generated Strategy C side-by-side comparison for {img_id}:")
    print(f"  Saved: {out_c_path}")
    print(f"  Saved: {out_a_path}")

if __name__ == "__main__":
    main()
