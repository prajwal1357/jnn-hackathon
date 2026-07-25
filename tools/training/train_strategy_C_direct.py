"""
Strategy C (High-Resolution P2 Feature Pyramid Level) Fine-Tuning Script
========================================================================
Stacking Strategy C (P2 High-Res FPN Level, Stride 4) ON TOP OF Strategy A + B.
Initializes weights from fine-tuned Strategy B:
  output/strategy_B_small_object_loss/work_dir/strategy_B_finetuned.pth

Fine-tunes the combined model (Strategy A + B + C) for 5 epochs on local GPU.
Saves stacked checkpoint to:
  output/strategy_C_highres_fpn/work_dir/strategy_C_finetuned.pth
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
PRETRAINED = os.path.join(ROOT, "output", "strategy_B_small_object_loss", "work_dir", "strategy_B_finetuned.pth")
OUT_DIR    = os.path.join(ROOT, "output", "strategy_C_highres_fpn", "work_dir")
CFG_PATH   = os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_r50_fpn_1x_vtuav_strategy_C.py")

os.makedirs(OUT_DIR, exist_ok=True)
if MMDET_ROOT not in sys.path:
    sys.path.insert(0, MMDET_ROOT)

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset, build_dataloader

def main():
    print("=" * 60)
    print("  Strategy C Fine-Tuning: High-Resolution P2 FPN Level")
    print("  Stacking A + B + C | 5 Epochs | SGD lr=0.001")
    print("=" * 60)

    cfg = mmcv.Config.fromfile(CFG_PATH)
    cfg.model.pretrained = None

    # 1. Build detector with P2 High-Res FPN Level
    print("\n[1/4] Building Strategy C Detector (P2 Stride 4 FPN + Gate + SmallObj Loss)...")
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')).cuda()

    model.bbox_head.use_small_object_loss = True
    model.bbox_prehead.use_small_object_loss = True

    # 2. Load Strategy B fine-tuned weights (strict=False to allow P2 FPN convs initialization)
    if not os.path.exists(PRETRAINED):
        print(f"WARNING: Strategy B checkpoint not found at {PRETRAINED}. Falling back to Strategy A.")
        PRETRAINED_LOAD = os.path.join(ROOT, "output", "strategy_A_modality_gate", "work_dir", "strategy_A_finetuned.pth")
    else:
        PRETRAINED_LOAD = PRETRAINED

    print(f"[2/4] Loading Strategy B pretrained checkpoint: {PRETRAINED_LOAD}...")
    ckpt = torch.load(PRETRAINED_LOAD, map_location='cuda')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        # Remap lateral convs and fpn convs for start_level=0 shift
        if 'neck.lateral_convs.' in k:
            idx = int(k.split('neck.lateral_convs.')[1].split('.')[0])
            new_k = k.replace(f'neck.lateral_convs.{idx}.', f'neck.lateral_convs.{idx+1}.')
            new_state_dict[new_k] = v
        elif 'neck.fpn_convs.' in k:
            idx = int(k.split('neck.fpn_convs.')[1].split('.')[0])
            new_k = k.replace(f'neck.fpn_convs.{idx}.', f'neck.fpn_convs.{idx+1}.')
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"  ✓ Pretrained weights remapped & loaded cleanly (Missing keys: {len(msg.missing_keys)} - P2 level initialized).")

    # Freeze backbones
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.backbone_t.parameters():
        p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    train_p = sum(p.numel() for p in trainable) / 1e6
    print(f"  ✓ Total params:     {total_p:.2f} M")
    print(f"  ✓ Trainable params: {train_p:.2f} M (backbones frozen)")

    # 3. Build dataset
    print("\n[3/4] Building dataset...")
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=2, workers_per_gpu=0,
        num_gpus=1, dist=False, seed=42)
    print(f"  ✓ Train: {len(dataset)} images, {len(data_loader)} batches/epoch")

    # 4. Training loop
    optimizer = torch.optim.SGD(trainable, lr=0.001, momentum=0.9, weight_decay=1e-4)
    epochs = 5

    print(f"\n[4/4] Starting Strategy C Fine-Tuning for {epochs} epochs...")
    print("-" * 60)

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

            if (i + 1) % 50 == 0 or (i + 1) == len(data_loader):
                print(f"  Epoch [{epoch}/{epochs}] Batch [{i+1}/{len(data_loader)}] Loss: {loss.item():.4f}", flush=True)

        avg = epoch_loss / len(data_loader)
        elapsed = time.time() - t0
        print(f"===> Epoch [{epoch}/{epochs}] Complete | Avg Loss: {avg:.4f} | Time: {elapsed:.1f}s")

        # Save checkpoint
        ckpt_path = os.path.join(OUT_DIR, f"epoch_{epoch}.pth")
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, ckpt_path)

    # Save final Strategy C checkpoint
    final = os.path.join(OUT_DIR, "strategy_C_finetuned.pth")
    torch.save({'state_dict': model.state_dict(), 'epoch': epochs}, final)
    latest = os.path.join(OUT_DIR, "latest.pth")
    torch.save({'state_dict': model.state_dict(), 'epoch': epochs}, latest)

    print("\n" + "=" * 60)
    print(f"SUCCESS: Strategy C fine-tuned checkpoint saved to:")
    print(f"  {final}")
    print(f"  {latest}")
    print("=" * 60)

if __name__ == "__main__":
    main()
