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
from mmdet.datasets import build_dataset

WEIGHTS_PATH = r"p:\project\hackothon\jnn_shivamogga\epoch_11_qfdet_vtuav.pth"
OUTPUT_DIR = r"p:\project\hackothon\jnn_shivamogga\output\stage2_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

configs = {
    "RGB-Only": os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_eval_rgb_only.py"),
    "Thermal-Only": os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_eval_thermal_only.py"),
    "Full QFDet Baseline (Fused)": os.path.join(MMDET_ROOT, "qfdet_configs", "qfdet_eval_fused.py"),
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

    meta_data = item['img_metas'][0].data
    if isinstance(meta_data, dict):
        img_metas = [meta_data]
    else:
        img_metas = meta_data

    return img_input, img_metas

all_results = {}

for mode_name, config_path in configs.items():
    all_results[mode_name] = {}
    for split in ["val", "test"]:
        print(f"\n==================================================", flush=True)
        print(f"Running [{mode_name}] on [{split.upper()}] split", flush=True)
        print(f"==================================================", flush=True)

        cfg = mmcv.Config.fromfile(config_path)
        cfg.model.pretrained = None

        dataset_cfg = cfg.data.val if split == "val" else cfg.data.test
        dataset = build_dataset(dataset_cfg)

        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')).cuda()
        _ = load_checkpoint(model, WEIGHTS_PATH, map_location='cuda')
        model.eval()

        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        weights_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)

        results = []
        start_t = time.time()
        latencies = []
        with torch.no_grad():
            for i in range(len(dataset)):
                t_sample_start = time.time()
                img_input, img_metas = unpack_item(dataset[i])

                res = model.simple_test(img_input, img_metas, rescale=True)
                latencies.append((time.time() - t_sample_start) * 1000.0)
                results.append(res[0])

                if (i + 1) % 50 == 0 or (i + 1) == len(dataset):
                    print(f"  Processed [{i+1}/{len(dataset)}] images (Elapsed: {time.time()-start_t:.1f}s)", flush=True)

        total_eval_time = time.time() - start_t
        eval_fps = len(dataset) / total_eval_time
        avg_latency = float(np.mean(latencies))

        clean_name = mode_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        out_prefix = os.path.join(OUTPUT_DIR, f"{clean_name}_{split}")
        eval_metrics = dataset.evaluate(results, metric='bbox', jsonfile_prefix=out_prefix)

        res_dict = {
            "mode": mode_name,
            "split": split,
            "mAP": round(float(eval_metrics.get('bbox_mAP', 0.0) * 100), 2),
            "mAP50": round(float(eval_metrics.get('bbox_mAP_50', 0.0) * 100), 2),
            "mAP75": round(float(eval_metrics.get('bbox_mAP_75', 0.0) * 100), 2),
            "mAPS": round(float(eval_metrics.get('bbox_mAP_s', 0.0) * 100), 2),
            "mAPM": round(float(eval_metrics.get('bbox_mAP_m', 0.0) * 100), 2),
            "mAPL": round(float(eval_metrics.get('bbox_mAP_l', 0.0) * 100), 2),
            "params_M": round(float(total_params), 2),
            "weights_MB": round(float(weights_mb), 2),
            "latency_ms": round(float(avg_latency), 2),
            "fps_model": round(float(1000.0 / avg_latency if avg_latency > 0 else 0), 2),
            "fps_full_pipeline": round(float(eval_fps), 2),
            "pred_json": f"{out_prefix}.bbox.json"
        }

        print(f"\n---> SUMMARY [{mode_name}] ({split}):", flush=True)
        print(f"     mAP: {res_dict['mAP']}% | mAP50: {res_dict['mAP50']}% | mAP75: {res_dict['mAP75']}%", flush=True)
        print(f"     mAP_S: {res_dict['mAPS']}% | mAP_M: {res_dict['mAPM']}% | mAP_L: {res_dict['mAPL']}%", flush=True)
        print(f"     Params: {res_dict['params_M']}M | Size: {res_dict['weights_MB']}MB | Latency: {res_dict['latency_ms']}ms ({res_dict['fps_model']} FPS)", flush=True)

        all_results[mode_name][split] = res_dict

summary_path = os.path.join(OUTPUT_DIR, "stage2_final_summary.json")
with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n==================================================", flush=True)
print(f"Stage 2 Evaluation Completed! Summary saved to {summary_path}", flush=True)
print(f"==================================================", flush=True)
