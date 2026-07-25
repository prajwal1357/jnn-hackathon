"""
Strategy A Fine-Tuning: ModalityGate (Trust Meter)
====================================================
Direct single-GPU training bypassing MMDataParallel.
Unwraps MMDet DataContainer batch structure:
  v_img     = data_batch['img'][0].data[0].cuda()
  t_img     = data_batch['img'][1].data[0].cuda()
  img_metas = data_batch['img_metas'].data[0]
  gt_bboxes = [b.cuda() for b in data_batch['gt_bboxes'].data[0]]
  gt_labels = [l.cuda() for l in data_batch['gt_labels'].data[0]]
"""

import os
import sys
import time
import torch
import torch.nn.utils as nn_utils

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT       = r"p:\project\hackothon\jnn_shivamogga"
MMDET_ROOT = os.path.join(ROOT, "mmdet-rgbtdroneperson")
WEIGHTS    = os.path.join(ROOT, "epoch_11_qfdet_vtuav.pth")
OUT_DIR    = os.path.join(ROOT, "output", "strategy_A_modality_gate", "work_dir")
CFG_PATH   = os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_r50_fpn_1x_vtuav_train_modality_gate.py")

os.makedirs(OUT_DIR, exist_ok=True)
if MMDET_ROOT not in sys.path:
    sys.path.insert(0, MMDET_ROOT)

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset, build_dataloader


def main():
    print("=" * 60)
    print("  Strategy A Fine-Tuning: ModalityGate (Trust Meter)")
    print("  Direct single-GPU | 5 epochs | SGD lr=0.001")
    print("=" * 60)

    cfg = mmcv.Config.fromfile(CFG_PATH)
    cfg.model.pretrained = None

    # 1. Build model
    print("\n[1/4] Building model...")
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg')).cuda()

    # 2. Load pretrained weights
    print("[2/4] Loading pretrained checkpoint...")
    load_checkpoint(model, WEIGHTS, map_location='cuda')

    # Freeze backbones
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.backbone_t.parameters():
        p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    train_p = sum(p.numel() for p in trainable) / 1e6
    print(f"  Total params:     {total_p:.2f} M")
    print(f"  Trainable params: {train_p:.2f} M (backbones frozen)")

    # 3. Build dataset & dataloader
    print("\n[3/4] Building dataset...")
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=2, workers_per_gpu=0,
        num_gpus=1, dist=False, seed=42)
    print(f"  Train: {len(dataset)} images, {len(data_loader)} batches/epoch")

    # 4. Optimizer & training loop
    optimizer = torch.optim.SGD(trainable, lr=0.001, momentum=0.9, weight_decay=1e-4)
    epochs = 5

    print(f"\n[4/4] Starting Fine-Tuning for {epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for i, data_batch in enumerate(data_loader):
            # Unwrap DataContainers cleanly
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

        # Save per-epoch checkpoint
        ckpt_path = os.path.join(OUT_DIR, f"epoch_{epoch}.pth")
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, ckpt_path)

    # Save final checkpoints
    final = os.path.join(OUT_DIR, "strategy_A_finetuned.pth")
    torch.save({'state_dict': model.state_dict(), 'epoch': epochs}, final)
    latest = os.path.join(OUT_DIR, "latest.pth")
    torch.save({'state_dict': model.state_dict(), 'epoch': epochs}, latest)

    print("\n" + "=" * 60)
    print(f"SUCCESS: Fine-tuned Strategy A checkpoint saved to:")
    print(f"  {final}")
    print(f"  {latest}")
    print("=" * 60)

if __name__ == "__main__":
    main()
