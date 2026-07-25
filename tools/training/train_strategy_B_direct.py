"""
Strategy B (Small-Object-Weighted Loss) Fine-Tuning Script
===========================================================
Continues fine-tuning from Strategy A checkpoint (strategy_A_finetuned.pth).
Applies inverse-area scale loss weighting for small bounding boxes (< 32^2 px, weight_boost=1.5).

Saves checkpoint to:
  output/strategy_B_small_object_loss/work_dir/strategy_B_finetuned.pth
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
OUT_DIR    = os.path.join(ROOT, "output", "strategy_B_small_object_loss", "work_dir")
CFG_PATH   = os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_r50_fpn_1x_vtuav_strategy_B.py")

os.makedirs(OUT_DIR, exist_ok=True)
if MMDET_ROOT not in sys.path:
    sys.path.insert(0, MMDET_ROOT)

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset, build_dataloader

def main():
    print("=" * 60)
    print("  Strategy B Fine-Tuning: Small-Object-Weighted Loss")
    print("  Stacking on Strategy A | 5 Epochs | SGD lr=0.001 | boost=1.5")
    print("=" * 60)

    cfg = mmcv.Config.fromfile(CFG_PATH)
    cfg.model.pretrained = None

    # 1. Build detector with Small-Object-Weighted Loss
    print("\n[1/4] Building Detector with Strategy B Loss Weighting...")
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')).cuda()

    # Enable use_small_object_loss flag on heads
    model.bbox_head.use_small_object_loss = True
    model.bbox_prehead.use_small_object_loss = True

    # 2. Load Strategy A fine-tuned weights
    if not os.path.exists(PRETRAINED):
        print(f"WARNING: Strategy A checkpoint not found at {PRETRAINED}. Falling back to baseline.")
        PRETRAINED_LOAD = os.path.join(ROOT, "epoch_11_qfdet_vtuav.pth")
    else:
        PRETRAINED_LOAD = PRETRAINED

    print(f"[2/4] Loading Strategy A pretrained checkpoint: {PRETRAINED_LOAD}...")
    load_checkpoint(model, PRETRAINED_LOAD, map_location='cuda')

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

    print(f"\n[4/4] Starting Strategy B Fine-Tuning for {epochs} epochs...")
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

            if i == 0 and epoch == 1:
                print(f"  ✓ STEP 5 SANITY CHECK: Batch 1 Loss = {loss.item():.4f} (Small-Object Weighting Active!)", flush=True)

            if (i + 1) % 50 == 0 or (i + 1) == len(data_loader):
                print(f"  Epoch [{epoch}/{epochs}] Batch [{i+1}/{len(data_loader)}] Loss: {loss.item():.4f}", flush=True)

        avg = epoch_loss / len(data_loader)
        elapsed = time.time() - t0
        print(f"===> Epoch [{epoch}/{epochs}] Complete | Avg Loss: {avg:.4f} | Time: {elapsed:.1f}s")

        # Save checkpoint
        ckpt_path = os.path.join(OUT_DIR, f"epoch_{epoch}.pth")
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, ckpt_path)

    # Save final Strategy B checkpoint
    final = os.path.join(OUT_DIR, "strategy_B_finetuned.pth")
    torch.save({'state_dict': model.state_dict(), 'epoch': epochs}, final)
    latest = os.path.join(OUT_DIR, "latest.pth")
    torch.save({'state_dict': model.state_dict(), 'epoch': epochs}, latest)

    print("\n" + "=" * 60)
    print(f"SUCCESS: Strategy B fine-tuned checkpoint saved to:")
    print(f"  {final}")
    print(f"  {latest}")
    print("=" * 60)

if __name__ == "__main__":
    main()
