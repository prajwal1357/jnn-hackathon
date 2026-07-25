"""
Unified Evaluation Runner for RGB-T Drone Pedestrian Detection
Supports evaluating:
  - Baseline (QFDet)
  - Strategy A (ModalityGate)
  - Strategy B (Small-Object Loss)
  - Strategy C (P2 High-Res FPN)

Usage:
  python tools/evaluate.py --strategy C --split test
"""

import os
import sys
import time
import json
import argparse
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MMDET_ROOT = os.path.join(ROOT, "mmdet-rgbtdroneperson")
OUT_DIR    = os.path.join(ROOT, "output", "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

if MMDET_ROOT not in sys.path:
    sys.path.insert(0, MMDET_ROOT)

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset

CONFIGS = {
    "baseline": {
        "cfg": os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_r50_fpn_1x_vtuav.py"),
        "ckpt": os.path.join(ROOT, "epoch_11_qfdet_vtuav.pth")
    },
    "A": {
        "cfg": os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_r50_fpn_1x_vtuav_modality_gate.py"),
        "ckpt": os.path.join(ROOT, "output", "strategy_A_modality_gate", "work_dir", "strategy_A_finetuned.pth")
    },
    "B": {
        "cfg": os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_r50_fpn_1x_vtuav_strategy_B.py"),
        "ckpt": os.path.join(ROOT, "output", "strategy_B_small_object_loss", "work_dir", "strategy_B_finetuned.pth")
    },
    "C": {
        "cfg": os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_r50_fpn_1x_vtuav_strategy_C.py"),
        "ckpt": os.path.join(ROOT, "output", "strategy_C_highres_fpn", "work_dir", "strategy_C_finetuned.pth")
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

def evaluate_strategy(strategy_key, split_name):
    strat = CONFIGS.get(strategy_key.lower(), CONFIGS[strategy_key.upper()])
    cfg = mmcv.Config.fromfile(strat['cfg'])
    cfg.model.pretrained = None
    
    print(f"Loading detector [{strategy_key}] from: {strat['ckpt']}...")
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')).cuda()
    load_checkpoint(model, strat['ckpt'], map_location='cuda')
    model.eval()

    test_cfg = cfg.data.val if split_name == 'val' else cfg.data.test
    dataset = build_dataset(test_cfg)
    if split_name == 'test':
        for info in dataset.data_infos:
            if not info['filename'].startswith('test/images/'):
                info['filename'] = f"test/images/{info['file_name']}"

    results = []
    print(f"Running evaluation on {split_name.upper()} split ({len(dataset)} images)...")
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(dataset)):
            img_input, img_metas = unpack_item(dataset[i])
            res = model.simple_test(img_input, img_metas, rescale=True)
            results.append(res[0])

    out_prefix = os.path.join(OUT_DIR, f"eval_{strategy_key}_{split_name}")
    metrics = dataset.evaluate(results, metric='bbox', jsonfile_prefix=out_prefix)
    
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS: Strategy [{strategy_key}] ({split_name.upper()})")
    print(f"  mAP: {metrics.get('bbox_mAP', 0)*100:.1f}% | mAP50: {metrics.get('bbox_mAP_50', 0)*100:.1f}% | mAP_S: {metrics.get('bbox_mAP_s', 0)*100:.1f}%")
    print(f"  Total Time: {elapsed:.1f}s | Speed: {len(dataset)/elapsed:.2f} FPS")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Evaluate QFDet strategies")
    parser.add_argument("--strategy", type=str, default="C", choices=["baseline", "A", "B", "C"], help="Strategy variant")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split")
    args = parser.parse_args()
    evaluate_strategy(args.strategy, args.split)

if __name__ == "__main__":
    main()
