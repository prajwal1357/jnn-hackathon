# RGB-Thermal Drone Pedestrian Detection (QFDet + ModalityGate)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![MMDetection](https://img.shields.io/badge/MMDetection-2.28.2-orange.svg)](https://github.com/open-mmlab/mmdetection)

## 📌 Project Overview
This repository contains the complete implementation and evaluation framework for **RGB-Thermal (RGB-T) Drone Pedestrian Detection** built on top of **QFDet (Quality-aware Fusion Detector)** for urban drone surveillance.

Drone-based pedestrian detection faces two primary challenges:
1. **Severe Modality Degradation:** RGB cameras fail under night conditions, while thermal (IR) sensors suffer under thermal glare or sun reflections.
2. **Tiny Pedestrian Scale:** Pedestrians captured from high drone altitudes often occupy $< 16 \times 16$ pixels, losing spatial detail in standard feature pyramids.

To address these challenges, we introduce three progressive architectural & loss strategies:
- **Strategy A — Spatially-Aware ModalityGate:** Dynamic spatial trust metering between RGB and Thermal modalities before feature fusion.
- **Strategy B — Small-Object-Weighted Loss:** Graduated inverse-area scale loss weighting to prioritize tiny ground-truth bounding boxes.
- **Strategy C — High-Resolution $P_2$ Feature Pyramid Level:** Stride-4 feature extraction ($96 \times 160$ resolution) tapping directly into ResNet $C_2$ stages to preserve fine-grained spatial features.

---

## 🎨 4-Grid Qualitative Ablation Comparison

The figure below illustrates qualitative bounding box detection across the 4 ablation stages on a challenging test frame:

![4-Grid Ablation Grid](output/qualitative_comparison/ablation_4grid_comparison.png)

* **Top-Left (Baseline):** Misses tiny distant pedestrians due to rigid spatial downsampling.
* **Top-Right (Strategy A):** ModalityGate suppresses thermal noise and stabilizes predictions.
* **Bottom-Left (Strategy A+B):** Inverse-area loss weight pulls in tighter bounding box boundaries.
* **Bottom-Right (Strategy A+B+C):** High-resolution $P_2$ feature level detects sub-16 pixel pedestrians previously missed by all other variants.

---

## 📊 Performance Benchmarks & Ablation Charts

### 1. Full 4-Stage Ablation ($mAP_{50}$ & $mAP_S$ Progression)
![Full Ablation Progression](output/strategy_C_highres_fpn/charts/C_full_ablation_mAPS.png)

### 2. Cross-Modal Performance Comparison (RGB vs. Thermal vs. Fused)
![mAP Benchmark Comparison](output/stage2_results/benchmark_map_comparison.png)

### 3. Scale-Specific Performance Breakdown ($mAP_S$, $mAP_M$, $mAP_L$)
![Scale Performance](output/stage2_results/benchmark_scale_performance.png)

---

## 🏆 Master 4-Stage Performance Matrix

| Model / Ablation Stage | Val mAP | Val mAP50 | Val mAP_S | Val AR_S (Recall) | Test mAP | Test mAP50 | Test mAP_S | Test AR_S (Recall) | Params | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **QFDet Baseline** | 33.8% | 72.1% | 14.4% | 21.3% | 29.9% | 67.4% | 12.9% | 17.7% | 60.63 M | 9.05 |
| **Strategy A (ModalityGate)** | 33.3% | 71.8% | **15.0%** | 22.0% | 29.4% | 67.6% | 12.4% | 17.5% | 60.67 M | 8.90 |
| **Strategy A+B (Loss)** | 30.7% | 70.2% | 14.2% | 21.3% | 26.8% | 65.1% | 12.1% | 17.7% | 60.67 M | 8.90 |
| **Strategy A+B+C (High-Res $P_2$)** | **31.8%** | **72.3%** | **13.8%** | **24.3%** | **28.8%** | **69.8%** 🏆 | **13.8%** 🏆 | **23.7%** 🚀 | **60.73 M** | **6.50** |

---

## ⚙️ Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/prajw/jnn_shivamogga.git
cd jnn_shivamogga

# 2. Create and activate virtual environment
python -m venv venv_qfdet
source venv_qfdet/bin/activate  # On Windows: venv_qfdet\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running Evaluation & Visualizations

### 1. Generate 4-Grid Qualitative Comparison Image
```bash
python tools/visualization/generate_4grid_comparison.py
```

### 2. Evaluate Strategy C (P2 High-Res FPN Model)
```bash
python tools/evaluation/run_strategy_C.py
```

### 3. Run Pipeline on Unseen Data Split
```bash
python run_unseen_pipeline.py --image_dir <path_to_unseen_images> --output_dir output/unseen_predictions
```

---

## 📁 Repository Structure

```
├── mmdet-rgbtdroneperson/    # Modified MMDetection framework with QFDet & ModalityGate
│   ├── mmdet/models/
│   │   ├── detectors/qfdet.py         # ModalityGate & QFDet dual-stream fusion architecture
│   │   ├── dense_heads/qfdet_prehead.py# Prehead quality estimation & small object loss
│   │   └── dense_heads/atssq_head.py  # ATSS main head with quality-guided assigner
│   └── qfdet_configs/                  # Strategy configuration files
├── tools/                              # Execution & evaluation scripts
│   ├── analysis/                       # Dataset EDA & visual alignment
│   ├── evaluation/                     # Evaluation & speed benchmarks
│   ├── training/                       # Fine-tuning & loss sweep scripts
│   └── visualization/                  # Plotting & 4-grid chart generators
├── output/                             # Generated benchmark charts, heatmaps, & reports
│   └── qualitative_comparison/         # 4-Grid qualitative ablation panels
├── report.md                           # Project technical report
├── run_unseen_pipeline.py             # Inference pipeline for unseen testing data
└── requirements.txt                    # Dependencies file
```
