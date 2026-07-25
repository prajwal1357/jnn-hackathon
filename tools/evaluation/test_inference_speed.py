import sys
import time
import torch
import mmcv

MMDET_ROOT = r"p:\project\hackothon\jnn_shivamogga\mmdet-rgbtdroneperson"
if MMDET_ROOT not in sys.path:
    sys.path.insert(0, MMDET_ROOT)

from mmdet.models import build_detector
from mmdet.datasets import build_dataset, build_dataloader
from mmcv.runner import load_checkpoint

WEIGHTS_PATH = r"p:\project\hackothon\jnn_shivamogga\epoch_11_qfdet_vtuav.pth"
CONFIG_PATH = r"p:\project\hackothon\jnn_shivamogga\mmdet-rgbtdroneperson\qfdet_configs\qfdet_eval_fused.py"

print("Loading config and building dataset...")
cfg = mmcv.Config.fromfile(CONFIG_PATH)
dataset = build_dataset(cfg.data.val)
loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

print("Building model and loading checkpoint...")
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')).cuda()
_ = load_checkpoint(model, WEIGHTS_PATH, map_location='cuda')
model.eval()

print("Testing 10 dataset iterations timing...")
times = []
with torch.no_grad():
    for i, data in enumerate(loader):
        if i >= 10:
            break
        t0 = time.time()
        res = model(return_loss=False, rescale=True, **data)
        t1 = time.time()
        dt = (t1 - t0) * 1000.0
        times.append(dt)
        print(f"Sample {i+1}/10: {dt:.2f} ms | Detected {len(res[0][0])} pedestrians")

print(f"\nAverage sample latency: {sum(times)/len(times):.2f} ms per image pair")
