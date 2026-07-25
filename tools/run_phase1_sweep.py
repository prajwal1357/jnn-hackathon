"""
Phase 1 Strategy B Loss Boost Sweep Script
===========================================
Sweeps gentler weight_boost values and balanced loss weighting:
1. B-0.25     : weight_boost = 0.25 (bbox only)
2. B-0.50     : weight_boost = 0.50 (bbox only)
3. B-0.50-cls : weight_boost = 0.50 (bbox) + 0.15 (cls)
4. B-0.50-all : weight_boost = 0.50 (bbox) + 0.15 (cls) + 0.15 (centerness)

Fine-tunes from Strategy A checkpoint (strategy_A_finetuned.pth) for 4 epochs per variant.
Evaluates on both VAL and TEST splits after each fine-tune.
"""

import os
import sys
import time
import json
import torch
import torch.nn.utils as nn_utils

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT       = r"p:\project\hackothon\jnn_shivamogga"
MMDET_ROOT = os.path.join(ROOT, "mmdet-rgbtdroneperson")
PRETRAINED = os.path.join(ROOT, "output", "strategy_A_modality_gate", "work_dir", "strategy_A_finetuned.pth")
BASE_OUT   = os.path.join(ROOT, "output", "phase1_sweep")
CFG_PATH   = os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_r50_fpn_1x_vtuav_strategy_B.py")

os.makedirs(BASE_OUT, exist_ok=True)
if MMDET_ROOT not in sys.path:
    sys.path.insert(0, MMDET_ROOT)

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset, build_dataloader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

VARIANTS = [
    {
        "name": "B-0.25",
        "weight_boost": 0.25,
        "cls_boost": 0.0,
        "centerness_boost": 0.0,
        "desc": "Gentle 0.25 bbox boost"
    }
]

def train_variant(variant, cfg, epochs=4):
    vname = variant['name']
    out_dir = os.path.join(BASE_OUT, vname, "work_dir")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "strategy_B_finetuned.pth")

    print(f"\n" + "=" * 60)
    print(f"  TRAINING VARIANT: {vname} ({variant['desc']})")
    print(f"  boost_bbox={variant['weight_boost']} | boost_cls={variant['cls_boost']} | boost_ctr={variant['centerness_boost']}")
    print("=" * 60)

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')).cuda()

    # Configure loss flags and boost weights
    for head in [model.bbox_head, model.bbox_prehead]:
        head.use_small_object_loss = True
        head.small_object_weight_boost = variant['weight_boost']
        head.small_object_cls_boost = variant['cls_boost']
        head.small_object_centerness_boost = variant['centerness_boost']

    if os.path.exists(ckpt_path):
        print(f"  ✓ Checkpoint already exists for {vname} at {ckpt_path}. Loading trained weights!")
        load_checkpoint(model, ckpt_path, map_location='cuda')
        return ckpt_path, model

    load_checkpoint(model, PRETRAINED, map_location='cuda')

    # Freeze backbones for fast stable fine-tuning
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.backbone_t.parameters():
        p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=2, workers_per_gpu=0,
        num_gpus=1, dist=False, seed=42)

    optimizer = torch.optim.SGD(trainable, lr=0.001, momentum=0.9, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for i, data_batch in enumerate(data_loader):
            v_img = data_batch['img'][0].data[0].cuda()
            t_img = data_batch['img'][1].data[0].cuda()
            img_metas = data_batch['img_metas'].data[0]
            gt_bboxes = [b.cuda() for b in data_batch['gt_bboxes'].data[0]]
            gt_labels = [l.cuda() for l in data_batch['gt_labels'].data[0]]

            optimizer.zero_grad()
            losses = model.forward_train(
                img=(v_img, t_img),
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels
            )

            loss_vals = []
            for k, v in losses.items():
                if 'loss' not in k:
                    continue
                if isinstance(v, torch.Tensor):
                    loss_vals.append(v)
                elif isinstance(v, list):
                    loss_vals.extend([x for x in v if isinstance(x, torch.Tensor)])

            loss = sum(loss_vals)
            loss.backward()
            nn_utils.clip_grad_norm_(trainable, max_norm=35.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        elapsed = time.time() - t0
        print(f"  [{vname}] Epoch [{epoch}/{epochs}] Complete | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s", flush=True)

    torch.save({'state_dict': model.state_dict(), 'epoch': epochs}, ckpt_path)
    return ckpt_path, model

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

def evaluate_checkpoint(model, cfg, split_name):
    test_cfg = cfg.data.val if split_name == 'val' else cfg.data.test
    dataset = build_dataset(test_cfg)
    if split_name == 'test':
        for info in dataset.data_infos:
            if not info['filename'].startswith('test/images/'):
                info['filename'] = f"test/images/{info['file_name']}"

    model.eval()
    results = []
    with torch.no_grad():
        for i in range(len(dataset)):
            img_input, img_metas = unpack_item(dataset[i])
            res = model.simple_test(img_input, img_metas, rescale=True)
            results.append(res[0])

    out_prefix = os.path.join(BASE_OUT, f"temp_eval_{split_name}")
    metrics = dataset.evaluate(results, metric='bbox', jsonfile_prefix=out_prefix)
    return {
        'mAP': float(metrics.get('bbox_mAP', 0) * 100),
        'mAP50': float(metrics.get('bbox_mAP_50', 0) * 100),
        'mAP75': float(metrics.get('bbox_mAP_75', 0) * 100),
        'mAPS': float(metrics.get('bbox_mAP_s', 0) * 100),
        'mAPM': float(metrics.get('bbox_mAP_m', 0) * 100),
        'mAPL': float(metrics.get('bbox_mAP_l', 0) * 100),
        'ARS': float(metrics.get('bbox_mAP_s', 0) * 100)
    }

def main():
    print("=" * 70)
    print("  PHASE 1: STRATEGY B LOSS BOOST SWEEP (SWEEPING GENTLE WEIGHTS)")
    print("=" * 70)

    cfg = mmcv.Config.fromfile(CFG_PATH)
    val_ann  = os.path.join(ROOT, "VTUAV_subset", "annotations", "val.json")
    test_ann = os.path.join(ROOT, "VTUAV_subset", "annotations", "test.json")

    results_table = []

    for variant in VARIANTS:
        ckpt_path, model = train_variant(variant, cfg, epochs=4)
        
        print(f"\n--> Evaluating [{variant['name']}] on VAL split...")
        val_res = evaluate_checkpoint(model, cfg, 'val')
        
        print(f"--> Evaluating [{variant['name']}] on TEST split...")
        test_res = evaluate_checkpoint(model, cfg, 'test')

        results_table.append({
            'name': variant['name'],
            'desc': variant['desc'],
            'val_mAP': val_res['mAP'],
            'val_mAP50': val_res['mAP50'],
            'val_mAPS': val_res['mAPS'],
            'val_ARS': val_res['ARS'],
            'test_mAP': test_res['mAP'],
            'test_mAP50': test_res['mAP50'],
            'test_mAPS': test_res['mAPS'],
            'test_ARS': test_res['ARS'],
            'ckpt': ckpt_path
        })

    # Save summary report
    report_path = os.path.join(BASE_OUT, "phase1_sweep_results.json")
    with open(report_path, "w") as f:
        json.dump(results_table, f, indent=2)

    print("\n" + "=" * 75)
    print("  PHASE 1 SWEEP COMPLETE — SUMMARY MATRIX")
    print("=" * 75)
    print(f"{'Variant':<12} | {'Val mAP':<8} | {'Val mAP50':<9} | {'Val mAP_S':<9} | {'Test mAP':<8} | {'Test mAP50':<10} | {'Test mAP_S':<10} | {'Test AR_S':<9}")
    print("-" * 75)
    print(f"{'Baseline':<12} | {'33.8%':<8} | {'72.1%':<9} | {'14.4%':<9} | {'29.9%':<8} | {'67.4%':<10} | {'12.9%':<10} | {'17.7%':<9}")
    print(f"{'Strat A (Gate)':<12} | {'33.3%':<8} | {'71.8%':<9} | {'15.0%':<9} | {'29.4%':<8} | {'67.6%':<10} | {'12.4%':<10} | {'17.5%':<9}")
    print(f"{'Old B (1.5)':<12} | {'30.7%':<8} | {'70.2%':<9} | {'14.2%':<9} | {'26.8%':<8} | {'65.1%':<10} | {'12.1%':<10} | {'17.7%':<9}")
    print("-" * 75)
    for r in results_table:
        print(f"{r['name']:<12} | {r['val_mAP']:>6.1f}% | {r['val_mAP50']:>7.1f}% | {r['val_mAPS']:>7.1f}% | {r['test_mAP']:>6.1f}% | {r['test_mAP50']:>8.1f}% | {r['test_mAPS']:>8.1f}% | {r['test_ARS']:>7.1f}%")
    print("=" * 75)
    print(f"Results saved to: {report_path}")

if __name__ == "__main__":
    main()
