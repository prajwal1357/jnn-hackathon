"""
Generate 4-Grid Qualitative Comparison Chart:
Panel 1: QFDet Baseline
Panel 2: Strategy A (ModalityGate)
Panel 3: Strategy A+B (Small-Object Loss)
Panel 4: Strategy A+B+C (P2 High-Res FPN Level)
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
OUT_DIR    = os.path.join(ROOT, "output", "qualitative_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

if MMDET_ROOT not in sys.path:
    sys.path.insert(0, MMDET_ROOT)

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset

CKPTS = {
    "QFDet Baseline": {
        "cfg": os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_eval_fused.py"),
        "ckpt": os.path.join(ROOT, "epoch_11_qfdet_vtuav.pth"),
        "color": (220, 20, 60), # Red
    },
    "Strategy A (ModalityGate)": {
        "cfg": os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_r50_fpn_1x_vtuav_modality_gate.py"),
        "ckpt": os.path.join(ROOT, "output", "strategy_A_modality_gate", "work_dir", "strategy_A_finetuned.pth"),
        "color": (255, 140, 0), # Orange
    },
    "Strategy A+B (Small-Object Loss)": {
        "cfg": os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_r50_fpn_1x_vtuav_strategy_B.py"),
        "ckpt": os.path.join(ROOT, "output", "strategy_B_small_object_loss", "work_dir", "strategy_B_finetuned.pth"),
        "color": (30, 144, 255), # Blue
    },
    "Strategy A+B+C (High-Res P2 FPN)": {
        "cfg": os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_r50_fpn_1x_vtuav_strategy_C.py"),
        "ckpt": os.path.join(ROOT, "output", "strategy_C_highres_fpn", "work_dir", "strategy_C_finetuned.pth"),
        "color": (50, 205, 50), # Green
    }
}

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
    # Build dataset from Strategy C config for test split
    cfg_c = mmcv.Config.fromfile(CKPTS["Strategy A+B+C (High-Res P2 FPN)"]["cfg"])
    test_dataset = build_dataset(cfg_c.data.test)
    for info in test_dataset.data_infos:
        if not info['filename'].startswith('test/images/'):
            info['filename'] = f"test/images/{info['file_name']}"

    target_idx = 0  # test_00024.jpg
    item = test_dataset[target_idx]
    img_info = test_dataset.data_infos[target_idx]
    img_name = img_info['file_name']

    raw_rgb_path = os.path.join(ROOT, "VTUAV_subset", "VTUAV_co", "test", "images", img_name)
    raw_rgb = cv2.imread(raw_rgb_path)
    raw_rgb_rgb = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB)

    panel_imgs = []
    panel_titles = []

    for name, spec in CKPTS.items():
        print(f"  --> Running inference with {name}...")
        cfg = mmcv.Config.fromfile(spec['cfg'])
        cfg.model.pretrained = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')).cuda()
        load_checkpoint(model, spec['ckpt'], map_location='cuda')
        model.eval()

        img_input, img_metas = unpack_item(item)
        with torch.no_grad():
            res = model.simple_test(img_input, img_metas, rescale=True)
            bboxes = res[0]

        drawn_img, count = draw_bboxes(raw_rgb_rgb, bboxes, score_thr=0.30, color=spec['color'])
        panel_imgs.append(drawn_img)
        panel_titles.append(f"{name}\n({count} Pedestrians Detected @ Thr 0.30)")

    # Plot 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#0d1117')
    fig.suptitle(f"Qualitative 4-Stage Ablation Grid — Sample {img_name}",
                 fontsize=18, fontweight='bold', color='white', y=0.98)

    for idx, ax in enumerate(axes.flat):
        ax.imshow(panel_imgs[idx])
        ax.set_title(panel_titles[idx], fontsize=13, fontweight='bold', color='white', pad=8)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUT_DIR, "ablation_4grid_comparison.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()

    print(f"\nSUCCESS: 4-Grid qualitative comparison saved to:\n  {out_path}")

if __name__ == "__main__":
    main()
