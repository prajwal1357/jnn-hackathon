"""
Strategy C (High-Resolution P2 Feature Pyramid Level) Evaluation & Report Generator
=====================================================================================
Full evaluation pipeline for Stacked Strategy A + B + C.
Generates:
  1. Full dataset evaluation (Val + Test splits)
  2. 4-Stage Ablation Progression Chart (Baseline -> Strategy A -> Strategy A+B -> Strategy A+B+C)
  3. Visualizations & Efficiency Assessment
  4. Strategy C markdown report: strategy_C_report.md
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
import warnings
warnings.filterwarnings('ignore')

ROOT        = r"p:\project\hackothon\jnn_shivamogga"
MMDET_ROOT  = os.path.join(ROOT, "mmdet-rgbtdroneperson")
WEIGHTS_C   = os.path.join(ROOT, "output", "strategy_C_highres_fpn", "work_dir", "strategy_C_finetuned.pth")
OUT_DIR     = os.path.join(ROOT, "output", "strategy_C_highres_fpn")
CHART_DIR   = os.path.join(OUT_DIR, "charts")
CFG_PATH    = os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_r50_fpn_1x_vtuav_strategy_C.py")
DATA_ROOT   = os.path.join(ROOT, "VTUAV_subset")

for d in [OUT_DIR, CHART_DIR]:
    os.makedirs(d, exist_ok=True)

if MMDET_ROOT not in sys.path:
    sys.path.insert(0, MMDET_ROOT)

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset

BASELINES = {
    "QFDet Baseline": {
        "val":  dict(mAP=33.8, mAP50=72.1, mAP75=27.3, mAPS=14.4, mAPM=32.4, mAPL=58.5, fps=9.05),
        "test": dict(mAP=29.9, mAP50=67.4, mAP75=22.7, mAPS=12.9, mAPM=29.9, mAPL=55.5, fps=9.13),
    },
    "Strategy A (Gate Only)": {
        "val":  dict(mAP=33.3, mAP50=71.8, mAP75=26.6, mAPS=15.0, mAPM=32.3, mAPL=57.6, fps=8.90),
        "test": dict(mAP=29.4, mAP50=67.6, mAP75=22.7, mAPS=12.4, mAPM=29.6, mAPL=55.5, fps=8.70),
    },
    "Strategy A+B (Small Obj Loss)": {
        "val":  dict(mAP=30.7, mAP50=70.2, mAP75=21.8, mAPS=14.2, mAPM=30.1, mAPL=52.6, fps=8.90),
        "test": dict(mAP=26.8, mAP50=65.1, mAP75=18.8, mAPS=12.1, mAPM=27.8, mAPL=48.7, fps=8.70),
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

def evaluate_split(split, cfg, model, dataset):
    print(f"\n==================================================")
    print(f"Evaluating Strategy C (P2 High-Res FPN) on [{split.upper()}] split")
    print(f"==================================================")
    results, latencies = [], []
    t_start = time.time()

    with torch.no_grad():
        for i in range(len(dataset)):
            t0 = time.time()
            img_input, img_metas = unpack_item(dataset[i])
            res = model.simple_test(img_input, img_metas, rescale=True)
            latencies.append((time.time() - t0) * 1000.0)
            results.append(res[0])

            if (i+1) % 50 == 0 or (i+1) == len(dataset):
                print(f"  Processed [{i+1}/{len(dataset)}] images (Elapsed: {time.time()-t_start:.1f}s)", flush=True)

    out_prefix = os.path.join(OUT_DIR, f"stratC_{split}")
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
    print(f"\n---> SUMMARY [Strategy C (High-Res P2 FPN)] ({split}):")
    print(f"     mAP: {res_dict['mAP']}% | mAP50: {res_dict['mAP50']}% | mAP75: {res_dict['mAP75']}% | mAP_S: {res_dict['mAPS']}%")
    return res_dict

def plot_full_ablation_bar(stratC_val, stratC_test):
    """4-stage bar chart showing progression across all baseline and strategy combinations."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0d1117')
    fig.suptitle('Full 4-Stage Ablation Progression: Baseline → Strategy A → Strategy A+B → Strategy A+B+C',
                 color='white', fontsize=14, fontweight='bold')

    models = ["QFDet Baseline", "Strategy A\n(ModalityGate)", "Strategy A+B\n(+ SmallObj Loss)", "Strategy A+B+C\n(+ High-Res P2)"]
    colors = ["#2ec4b6", "#ff8c42", "#a855f7", "#00b4d8"]

    for ax, split, resC in zip(axes, ['val', 'test'], [stratC_val, stratC_test]):
        res0 = BASELINES["QFDet Baseline"][split]
        resA = BASELINES["Strategy A (Gate Only)"][split]
        resB = BASELINES["Strategy A+B (Small Obj Loss)"][split]

        mAP_s_vals = [res0['mAPS'], resA['mAPS'], resB['mAPS'], resC['mAPS']]
        bars = ax.bar(models, mAP_s_vals, color=colors, edgecolor='white', width=0.55, zorder=3)

        for bar, v in zip(bars, mAP_s_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f"{v:.1f}%", ha='center', va='bottom', fontsize=11, color='white', fontweight='bold')

        ax.set_facecolor('#161b22')
        ax.set_title(f"{split.title()} Split — Small Objects (mAP_S)", color='#c9d1d9', fontsize=12)
        ax.set_ylabel("mAP_Small (%)", color='#c9d1d9', fontsize=11)
        ax.set_ylim(0, max(mAP_s_vals) * 1.3)
        ax.tick_params(colors='#c9d1d9')
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, color='#21262d', linestyle='--', alpha=0.6, zorder=0)

    plt.tight_layout()
    out = os.path.join(CHART_DIR, "C_full_ablation_mAPS.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  Saved: {out}")

def write_report(val_res, test_res):
    report_content = f"""# Strategy C — High-Resolution P2 Feature Pyramid Level (Layered on Strategy A+B)

**Strategy ID:** C  
**Strategy Name:** High-Resolution P2 Feature Pyramid Level  
**Layered On Top Of:** Strategy A (ModalityGate) + Strategy B (Small-Object Loss)  
**Status:** ✅ Implemented & Evaluated  

---

## 1. Concept & Motivation

By default, standard FPN begins at stride 8 (P3). Tiny pedestrians (< 16x16 pixels) lose critical spatial details through downsampling. Strategy C adds a high-resolution P2 pyramid level at **stride 4** (96 x 160 resolution), tapping directly into ResNet stage C2 features to preserve fine-grained spatial information for tiny objects.

---

## 2. Full 4-Stage Ablation Table

| Model / Stage | Val mAP | Val mAP50 | Val mAP_S (Small) | Val mAP_M (Med) | Val mAP_L (Large) | Test mAP | Test mAP50 | Test mAP_S | Params | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **QFDet Baseline** | 33.8% | 72.1% | 14.4% | 32.4% | 58.5% | 29.9% | 67.4% | 12.9% | 60.63 M | 9.05 |
| **Strategy A (Gate)** | 33.3% | 71.8% | 15.0% | 32.3% | 57.6% | 29.4% | 67.6% | 12.4% | 60.67 M | 8.90 |
| **Strategy A+B (Loss)** | 30.7% | 70.2% | 14.2% | 30.1% | 52.6% | 26.8% | 65.1% | 12.1% | 60.67 M | 8.90 |
| **Strategy A+B+C (P2 High-Res)** | **{val_res['mAP']:.1f}%** | **{val_res['mAP50']:.1f}%** | **{val_res['mAPS']:.1f}%** | **{val_res['mAPM']:.1f}%** | **{val_res['mAPL']:.1f}%** | **{test_res['mAP']:.1f}%** | **{test_res['mAP50']:.1f}%** | **{test_res['mAPS']:.1f}%** | **{val_res['params_M']:.2f} M** | **{val_res['fps']:.2f}** |
"""

    report_file = os.path.join(OUT_DIR, "strategy_C_report.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"  Saved report: {report_file}")

def main():
    print("=" * 60)
    print("  STRATEGY C EVALUATION (P2 High-Resolution FPN Level)")
    print(f"  Evaluating Checkpoint: {WEIGHTS_C}")
    print("=" * 60)

    cfg = mmcv.Config.fromfile(CFG_PATH)
    cfg.model.pretrained = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')).cuda()
    load_checkpoint(model, WEIGHTS_C, map_location='cuda')
    model.eval()

    val_dataset = build_dataset(cfg.data.val)
    val_res = evaluate_split("val", cfg, model, val_dataset)

    test_dataset = build_dataset(cfg.data.test)
    for info in test_dataset.data_infos:
        if not info['filename'].startswith('test/images/'):
            info['filename'] = f"test/images/{info['file_name']}"
    test_res = evaluate_split("test", cfg, model, test_dataset)

    summary = {"val": val_res, "test": test_res}
    with open(os.path.join(OUT_DIR, "strategy_C_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    plot_full_ablation_bar(val_res, test_res)
    write_report(val_res, test_res)

    print("\n" + "=" * 60)
    print(f"STRATEGY C (HIGH-RES P2 FPN) EVALUATION COMPLETE")
    print(f"  Val:  mAP={val_res['mAP']}% | mAP50={val_res['mAP50']}% | mAP_S={val_res['mAPS']}%")
    print(f"  Test: mAP={test_res['mAP']}% | mAP50={test_res['mAP50']}% | mAP_S={test_res['mAPS']}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
