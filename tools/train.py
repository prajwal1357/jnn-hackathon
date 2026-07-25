"""
Unified Training Script for RGB-T Drone Pedestrian Detection
Supports fine-tuning:
  - Strategy A (ModalityGate)
  - Strategy B (Small-Object Loss)
  - Strategy C (P2 High-Res FPN)

Usage:
  python tools/train.py --strategy C --epochs 5
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.utils as nn_utils

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MMDET_ROOT = os.path.join(ROOT, "mmdet-rgbtdroneperson")
PRETRAINED = os.path.join(ROOT, "epoch_11_qfdet_vtuav.pth")

if MMDET_ROOT not in sys.path:
    sys.path.insert(0, MMDET_ROOT)

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset, build_dataloader

CONFIGS = {
    "A": {
        "cfg": os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_r50_fpn_1x_vtuav_train_modality_gate.py"),
        "out": os.path.join(ROOT, "output", "strategy_A_modality_gate", "work_dir")
    },
    "B": {
        "cfg": os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_r50_fpn_1x_vtuav_strategy_B.py"),
        "out": os.path.join(ROOT, "output", "strategy_B_small_object_loss", "work_dir")
    },
    "C": {
        "cfg": os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_r50_fpn_1x_vtuav_strategy_C.py"),
        "out": os.path.join(ROOT, "output", "strategy_C_highres_fpn", "work_dir")
    }
}

def train_strategy(strategy_key, epochs=5):
    strat = CONFIGS[strategy_key.upper()]
    out_dir = strat['out']
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n==================================================")
    print(f"  Fine-Tuning Strategy [{strategy_key}] ({epochs} Epochs)")
    print(f"==================================================")

    cfg = mmcv.Config.fromfile(strat['cfg'])
    cfg.model.pretrained = None
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')).cuda()

    load_checkpoint(model, PRETRAINED, map_location='cuda')

    # Freeze backbones for stable fine-tuning
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
        print(f"  Strategy [{strategy_key}] Epoch [{epoch}/{epochs}] Complete | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s", flush=True)

    ckpt_path = os.path.join(out_dir, f"strategy_{strategy_key}_finetuned.pth")
    torch.save({'state_dict': model.state_dict(), 'epoch': epochs}, ckpt_path)
    print(f"\n[✓] Checkpoint saved to: {ckpt_path}")

def main():
    parser = argparse.ArgumentParser(description="Train QFDet strategy")
    parser.add_argument("--strategy", type=str, default="C", choices=["A", "B", "C"], help="Strategy variant")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs")
    args = parser.parse_args()
    train_strategy(args.strategy, args.epochs)

if __name__ == "__main__":
    main()
