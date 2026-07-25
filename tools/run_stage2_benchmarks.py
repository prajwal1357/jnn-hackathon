import os
import sys
import time
import json
import torch
import numpy as np

# Ensure mmdet is in sys.path
MMDET_ROOT = r"p:\project\hackothon\jnn_shivamogga\mmdet-rgbtdroneperson"
if MMDET_ROOT not in sys.path:
    sys.path.insert(0, MMDET_ROOT)

import mmcv
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.models import build_detector
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.apis import single_gpu_test

WEIGHTS_PATH = r"p:\project\hackothon\jnn_shivamogga\epoch_11_qfdet_vtuav.pth"
OUTPUT_DIR = r"p:\project\hackothon\jnn_shivamogga\output\stage2_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

configs = {
    "RGB-Only": os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_eval_rgb_only.py"),
    "Thermal-Only": os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_eval_thermal_only.py"),
    "Full QFDet (Baseline Fused)": os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_eval_fused.py"),
}

def get_model_size_mb(filepath):
    return os.path.getsize(filepath) / (1024 * 1024)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6, trainable_params / 1e6

def evaluate_config_split(config_name, config_path, split="val"):
    print(f"\n==================================================")
    print(f"Evaluating [{config_name}] on [{split.upper()} split]")
    print(f"==================================================")

    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.pretrained = None

    # Build dataset & dataloader
    if split == "val":
        dataset_cfg = cfg.data.val
    else:
        dataset_cfg = cfg.data.test

    dataset = build_dataset(dataset_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False
    )

    # Build model & load weights
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, WEIGHTS_PATH, map_location='cpu')

    model = model.cuda()
    model.eval()

    total_params, trainable_params = count_parameters(model)
    weights_size_mb = get_model_size_mb(WEIGHTS_PATH)

    # Benchmark FLOPs on single dummy input (640x512)
    # Using dummy dual image tensor tuple (1, 3, 512, 640)
    dummy_v = torch.randn(1, 3, 512, 640).cuda()
    dummy_t = torch.randn(1, 3, 512, 640).cuda()
    
    # Warmup GPU
    with torch.no_grad():
        for _ in range(10):
            _ = model.forward_dummy((dummy_v, dummy_t))

    # Benchmark latency / FPS
    num_warmup = 10
    num_runs = 50
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_event.record()
            _ = model.forward_dummy((dummy_v, dummy_t))
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))

    avg_latency_ms = float(np.mean(latencies))
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0

    # Run inference on entire split dataset
    print(f"Running inference over {len(dataset)} image pairs...")
    start_time = time.time()
    outputs = single_gpu_test(model, data_loader, show=False)
    eval_time = time.time() - start_time
    dataset_fps = len(dataset) / eval_time

    # Run COCO evaluation
    out_prefix = os.path.join(OUTPUT_DIR, f"{config_name.lower().replace(' ', '_')}_{split}")
    eval_results = dataset.evaluate(outputs, metric='bbox', jsonfile_prefix=out_prefix)

    # Extract COCO metrics
    map_val = eval_results.get('bbox_mAP', 0.0)
    map50 = eval_results.get('bbox_mAP_50', 0.0)
    map75 = eval_results.get('bbox_mAP_75', 0.0)
    maps = eval_results.get('bbox_mAP_s', 0.0)
    mapm = eval_results.get('bbox_mAP_m', 0.0)
    mapl = eval_results.get('bbox_mAP_l', 0.0)

    res = {
        "config_name": config_name,
        "split": split,
        "mAP": round(float(map_val * 100), 2),
        "mAP50": round(float(map50 * 100), 2),
        "mAP75": round(float(map75 * 100), 2),
        "mAPS": round(float(maps * 100), 2),
        "mAPM": round(float(mapm * 100), 2),
        "mAPL": round(float(mapl * 100), 2),
        "params_M": round(float(total_params), 2),
        "weights_size_MB": round(float(weights_size_mb), 2),
        "latency_ms": round(float(avg_latency_ms), 2),
        "fps_forward": round(float(fps), 2),
        "fps_dataset": round(float(dataset_fps), 2),
        "pred_json": f"{out_prefix}.bbox.json"
    }

    print(f"Results for [{config_name}] ({split}):")
    print(f"  mAP: {res['mAP']} | mAP50: {res['mAP50']} | mAP75: {res['mAP75']}")
    print(f"  mAP_S: {res['mAPS']} | mAP_M: {res['mAPM']} | mAP_L: {res['mAPL']}")
    print(f"  Params: {res['params_M']}M | Size: {res['weights_size_MB']}MB | Latency: {res['latency_ms']}ms ({res['fps_forward']} FPS)")

    return res

if __name__ == "__main__":
    all_results = {}
    for cfg_name, cfg_path in configs.items():
        all_results[cfg_name] = {
            "val": evaluate_config_split(cfg_name, cfg_path, split="val"),
            "test": evaluate_config_split(cfg_name, cfg_path, split="test")
        }

    # Save complete JSON
    out_summary_json = os.path.join(OUTPUT_DIR, "stage2_benchmark_summary.json")
    with open(out_summary_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll Stage 2 benchmark evaluations saved to {out_summary_json}")
