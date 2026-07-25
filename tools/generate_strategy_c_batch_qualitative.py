"""
Batch Qualitative Comparison Generator for Strategy C (High-Resolution P2 FPN Level):
Generates 10 side-by-side comparison images (Baseline vs Strategy C)
Saved in: output/strategy_C_qualitative/
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
os.makedirs(OUT_DIR_C, exist_ok=True)

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
    cfg_c = mmcv.Config.fromfile(STRATC_CFG)
    dataset = build_dataset(cfg_c.data.test)
    for info in dataset.data_infos:
        if not info['filename'].startswith('test/images/'):
            info['filename'] = f"test/images/{info['file_name']}"

    # Select 10 diverse test image indices
    total_imgs = len(dataset)
    indices = np.linspace(0, total_imgs - 1, 10, dtype=int)

    print(f"Loading QFDet Baseline model...")
    cfg_base = mmcv.Config.fromfile(BASE_CFG)
    cfg_base.model.pretrained = None
    model_base = build_detector(cfg_base.model, test_cfg=cfg_base.get('test_cfg')).cuda()
    load_checkpoint(model_base, BASE_CKPT, map_location='cuda')
    model_base.eval()

    print(f"Loading Strategy C model...")
    cfg_c.model.pretrained = None
    model_c = build_detector(cfg_c.model, test_cfg=cfg_c.get('test_cfg')).cuda()
    load_checkpoint(model_c, STRATC_CKPT, map_location='cuda')
    model_c.eval()

    plt.style.use('dark_background')

    for i, idx in enumerate(indices):
        item = dataset[idx]
        img_info = dataset.data_infos[idx]
        img_name = img_info['file_name']

        rgb_path = os.path.join(DATA_ROOT, "VTUAV_co", "test", "images", img_name)
        ir_path  = os.path.join(DATA_ROOT, "VTUAV_ir", "test", "images", img_name)

        if not os.path.exists(rgb_path):
            continue

        raw_rgb = cv2.imread(rgb_path)
        raw_rgb_rgb = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB)

        img_input, img_metas = unpack_item(item)

        with torch.no_grad():
            res_base = model_base.simple_test(img_input, img_metas, rescale=True)
            res_c    = model_c.simple_test(img_input, img_metas, rescale=True)

        img_base, count_base = draw_bboxes(raw_rgb_rgb, res_base[0], score_thr=0.30, color=(220, 20, 60))
        img_c, count_c       = draw_bboxes(raw_rgb_rgb, res_c[0], score_thr=0.30, color=(50, 205, 50))

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0d1117')
        fig.suptitle(f"Qualitative Comparison [{i+1}/10] — Sample {img_name}: Baseline vs Strategy C",
                     fontsize=16, fontweight='bold', color='white', y=0.98)

        axes[0].imshow(raw_rgb_rgb)
        axes[0].set_title(f"Input RGB Frame ({img_name})", fontsize=12, fontweight='bold', color='white', pad=8)
        axes[0].axis('off')

        axes[1].imshow(img_base)
        axes[1].set_title(f"QFDet Baseline\n({count_base} Detections @ Thr 0.30)", fontsize=12, fontweight='bold', color='#ef4444', pad=8)
        axes[1].axis('off')

        axes[2].imshow(img_c)
        axes[2].set_title(f"Strategy C (High-Res P2 FPN)\n({count_c} Detections @ Thr 0.30)", fontsize=12, fontweight='bold', color='#22c55e', pad=8)
        axes[2].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = os.path.join(OUT_DIR_C, f"qualitative_cmp_{img_name}.png")
        plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='#0d1117')
        plt.close()

        print(f"[{i+1}/10] Generated Strategy C qualitative image for {img_name}: {out_path}")

    print(f"\n[✓] Successfully generated 10 Strategy C qualitative images in:\n  {OUT_DIR_C}")

if __name__ == "__main__":
    main()
