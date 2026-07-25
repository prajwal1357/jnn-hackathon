import os
import sys
import time
import json
import torch
import numpy as np

MMDET_ROOT = r"p:\project\hackothon\jnn_shivamogga\mmdet-rgbtdroneperson"
if MMDET_ROOT not in sys.path:
    sys.path.insert(0, MMDET_ROOT)

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.apis import single_gpu_test
from mmcv.parallel import MMDataParallel

WEIGHTS_PATH = r"p:\project\hackothon\jnn_shivamogga\epoch_11_qfdet_vtuav.pth"
OUTPUT_DIR = r"p:\project\hackothon\jnn_shivamogga\output\stage2_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

configs = [
    ("RGB-Only", os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_eval_rgb_only.py")),
    ("Thermal-Only", os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_eval_thermal_only.py")),
    ("Full QFDet Baseline", os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_eval_fused.py")),
]

results_summary = {}

for mode_name, config_path in configs:
    print(f"\n==================================================", flush=True)
    print(f"Evaluating Baseline [{mode_name}] on VAL split", flush=True)
    print(f"==================================================", flush=True)

    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.pretrained = None

    dataset = build_dataset(cfg.data.val)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')).cuda()
    _ = load_checkpoint(model, WEIGHTS_PATH, map_location='cuda')
    model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    weights_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)

    # Measure latency
    dummy_v = torch.randn(1, 3, 512, 640).cuda()
    dummy_t = torch.randn(1, 3, 512, 640).cuda()
    latencies = []
    with torch.no_grad():
        for _ in range(5):
            _ = model.forward_dummy((dummy_v, dummy_t))
        for _ in range(20):
            t0 = time.time()
            _ = model.forward_dummy((dummy_v, dummy_t))
            torch.cuda.synchronize()
            latencies.append((time.time() - t0) * 1000.0)

    avg_latency = float(np.mean(latencies))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    t_start = time.time()
    model_dp = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model_dp, loader)
    eval_time = time.time() - t_start
    pipeline_fps = len(dataset) / eval_time

    out_prefix = os.path.join(OUTPUT_DIR, f"{mode_name.lower().replace(' ', '_')}_val")
    eval_results = dataset.evaluate(outputs, metric='bbox', jsonfile_prefix=out_prefix)

    res_dict = {
        "mode": mode_name,
        "mAP": round(float(eval_results.get('bbox_mAP', 0.0) * 100), 2),
        "mAP50": round(float(eval_results.get('bbox_mAP_50', 0.0) * 100), 2),
        "mAP75": round(float(eval_results.get('bbox_mAP_75', 0.0) * 100), 2),
        "mAPS": round(float(eval_results.get('bbox_mAP_s', 0.0) * 100), 2),
        "mAPM": round(float(eval_results.get('bbox_mAP_m', 0.0) * 100), 2),
        "mAPL": round(float(eval_results.get('bbox_mAP_l', 0.0) * 100), 2),
        "params_M": round(float(total_params), 2),
        "weights_MB": round(float(weights_mb), 2),
        "latency_ms": round(float(avg_latency), 2),
        "fps": round(float(fps), 2),
        "pipeline_fps": round(float(pipeline_fps), 2)
    }

    print(f"\n---> [{mode_name}] RESULTS:", flush=True)
    print(f"     mAP: {res_dict['mAP']}% | mAP50: {res_dict['mAP50']}% | mAP75: {res_dict['mAP75']}%", flush=True)
    print(f"     mAP_S: {res_dict['mAPS']}% | mAP_M: {res_dict['mAPM']}% | mAP_L: {res_dict['mAPL']}%", flush=True)
    print(f"     Params: {res_dict['params_M']}M | Latency: {res_dict['latency_ms']}ms ({res_dict['fps']} FPS)", flush=True)

    results_summary[mode_name] = res_dict

summary_file = os.path.join(OUTPUT_DIR, "stage2_benchmark_summary.json")
with open(summary_file, "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"\n==================================================", flush=True)
print(f"All Stage 2 Benchmark evaluations completed! Results saved to {summary_file}", flush=True)
print(f"==================================================", flush=True)
