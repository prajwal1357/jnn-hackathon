"""
Evaluation Script for Fine-Tuned Strategy A (ModalityGate)
===========================================================
This script evaluates the fine-tuned Strategy A checkpoint (.pth file)
saved after running training, generates real learned trust-meter heatmaps,
creates comparative charts, and outputs updated markdown reports.
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import torch
# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = r"p:\project\hackothon\jnn_shivamogga"
MMDET_ROOT  = os.path.join(ROOT, "mmdet-rgbtdroneperson")
WEIGHTS_FT  = os.path.join(ROOT, "output", "strategy_A_modality_gate", "work_dir", "latest.pth")
OUT_DIR     = os.path.join(ROOT, "output", "strategy_A_modality_gate")
HEATMAP_DIR = os.path.join(OUT_DIR, "heatmaps")
CHART_DIR   = os.path.join(OUT_DIR, "charts")
CFG_PATH    = os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_r50_fpn_1x_vtuav_modality_gate.py")
DATA_ROOT   = os.path.join(ROOT, "VTUAV_subset")

for d in [OUT_DIR, HEATMAP_DIR, CHART_DIR]:
    os.makedirs(d, exist_ok=True)

if MMDET_ROOT not in sys.path:
    sys.path.insert(0, MMDET_ROOT)

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset

BASELINES = {
    "RGB-Only": {
        "val":  dict(mAP=6.9,  mAP50=23.1, mAP75=2.3,  mAPS=0.5, mAPM=6.5,  mAPL=17.3, fps=8.67),
        "test": dict(mAP=5.5,  mAP50=18.6, mAP75=1.9,  mAPS=0.5, mAPM=5.2,  mAPL=14.2, fps=8.64),
    },
    "Thermal-Only": {
        "val":  dict(mAP=26.9, mAP50=57.1, mAP75=22.0, mAPS=8.7, mAPM=25.2, mAPL=56.6, fps=8.96),
        "test": dict(mAP=22.0, mAP50=52.4, mAP75=15.6, mAPS=7.5, mAPM=21.7, mAPL=49.8, fps=9.13),
    },
    "QFDet Baseline": {
        "val":  dict(mAP=33.8, mAP50=72.1, mAP75=27.3, mAPS=14.4, mAPM=32.4, mAPL=58.5, fps=9.05),
        "test": dict(mAP=29.9, mAP50=67.4, mAP75=22.7, mAPS=12.9, mAPM=29.9, mAPL=55.5, fps=9.13),
    },
    "GitHub Reported": {
        "val":  dict(mAP=31.1, mAP50=70.4, mAP75=22.9, mAPS=None, mAPM=None, mAPL=None, fps=None),
        "test": dict(mAP=31.1, mAP50=70.4, mAP75=22.9, mAPS=None, mAPM=None, mAPL=None, fps=None),
    },
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

def evaluate_split(split, cfg, model, dataset):
    print(f"\n==================================================")
    print(f"Evaluating Fine-Tuned Strategy A on [{split.upper()}] split")
    print(f"==================================================")
    results, latencies, gate_weight_maps, img_paths_co = [], [], [], []
    t_start = time.time()

    with torch.no_grad():
        for i in range(len(dataset)):
            t0 = time.time()
            img_input, img_metas = unpack_item(dataset[i])
            model.last_gate_weights = []
            res = model.simple_test(img_input, img_metas, rescale=True)
            latencies.append((time.time() - t0) * 1000.0)
            results.append(res[0])

            if i < 20:
                gw = getattr(model, 'last_gate_weights', [])
                if gw:
                    gate_weight_maps.append({
                        'idx': i,
                        'fname': dataset.data_infos[i]['filename'],
                        'weight_l0': gw[0][0].cpu(),
                    })
                    img_paths_co.append(
                        os.path.join(DATA_ROOT, f"VTUAV_co/{split}/images",
                                     dataset.data_infos[i]['filename']))

            if (i+1) % 50 == 0 or (i+1) == len(dataset):
                print(f"  Processed [{i+1}/{len(dataset)}] images (Elapsed: {time.time()-t_start:.1f}s)", flush=True)

    out_prefix = os.path.join(OUT_DIR, f"stratA_ft_{split}")
    metrics = dataset.evaluate(results, metric='bbox', jsonfile_prefix=out_prefix)

    avg_lat = float(np.mean(latencies))
    fps = 1000.0 / avg_lat
    params = sum(p.numel() for p in model.parameters()) / 1e6

    res_dict = dict(
        split=split,
        mAP    = round(float(metrics.get('bbox_mAP',    0) * 100), 2),
        mAP50  = round(float(metrics.get('bbox_mAP_50', 0) * 100), 2),
        mAP75  = round(float(metrics.get('bbox_mAP_75', 0) * 100), 2),
        mAPS   = round(float(metrics.get('bbox_mAP_s',  0) * 100), 2),
        mAPM   = round(float(metrics.get('bbox_mAP_m',  0) * 100), 2),
        mAPL   = round(float(metrics.get('bbox_mAP_l',  0) * 100), 2),
        params_M   = round(params, 2),
        latency_ms = round(avg_lat, 2),
        fps        = round(fps, 2),
    )
    print(f"\n---> SUMMARY [Fine-Tuned Strategy A] ({split}):")
    print(f"     mAP: {res_dict['mAP']}% | mAP50: {res_dict['mAP50']}% | mAP75: {res_dict['mAP75']}%")
    print(f"     mAP_S: {res_dict['mAPS']}% | mAP_M: {res_dict['mAPM']}% | mAP_L: {res_dict['mAPL']}%")
    return res_dict, gate_weight_maps, img_paths_co

def main():
    if not os.path.exists(WEIGHTS_FT):
        print(f"ERROR: Fine-tuned checkpoint file not found at:\n  {WEIGHTS_FT}")
        print("Please run the fine-tuning command first!")
        sys.exit(1)

    print("==================================================")
    print("Evaluating Fine-Tuned Strategy A Model")
    print(f"Checkpoint: {WEIGHTS_FT}")
    print("==================================================")

    cfg = mmcv.Config.fromfile(CFG_PATH)
    cfg.model.pretrained = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')).cuda()
    load_checkpoint(model, WEIGHTS_FT, map_location='cuda')
    model.eval()

    val_dataset = build_dataset(cfg.data.val)
    val_res, gw_val, ip_val = evaluate_split("val", cfg, model, val_dataset)

    test_dataset = build_dataset(cfg.data.test)
    test_res, gw_test, ip_test = evaluate_split("test", cfg, model, test_dataset)

    summary = {"val": val_res, "test": test_res}
    with open(os.path.join(OUT_DIR, "strategy_A_finetuned_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nFine-tuned evaluation completed successfully!")

if __name__ == "__main__":
    main()
