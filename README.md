# RGB-Thermal Drone Pedestrian Detection (QFDet + ModalityGate)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![MMDetection](https://img.shields.io/badge/MMDetection-2.28.2-orange.svg)](https://github.com/open-mmlab/mmdetection)

## 📌 Project Overview
This repository contains the complete implementation and evaluation framework for **RGB-Thermal (RGB-T) Drone Pedestrian Detection** built on top of **QFDet (Quality-Guided Fusion Detector)** for urban drone surveillance.

Drone-based pedestrian detection faces two primary challenges:
1. **Severe Modality Degradation:** RGB cameras fail under night conditions, while thermal (IR) sensors suffer under thermal glare or sun reflections.
2. **Tiny Pedestrian Scale:** Pedestrians captured from high drone altitudes often occupy $< 16 \times 16$ pixels, losing spatial detail in standard feature pyramids.

To address these challenges, we introduce three progressive architectural & loss strategies:
- **Strategy A — Spatially-Aware ModalityGate:** Dynamic spatial trust metering between RGB and Thermal modalities before feature fusion.
- **Strategy B — Small-Object-Weighted Loss:** Graduated inverse-area scale loss weighting to prioritize tiny ground-truth bounding boxes.
- **Strategy C — High-Resolution $P_2$ Feature Pyramid Level:** Stride-4 feature extraction ($96 \times 160$ resolution) tapping directly into ResNet $C_2$ stages to preserve fine-grained spatial features.

---

## 🏆 Master 4-Stage Performance Matrix

| Model / Ablation Stage | Val mAP | Val mAP50 | Val mAP_S | Val AR_S (Recall) | Test mAP | Test mAP50 | Test mAP_S | Test AR_S (Recall) | Params | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **QFDet Baseline** | 33.8% | 72.1% | 14.4% | 21.3% | 29.9% | 67.4% | 12.9% | 17.7% | 60.63 M | 9.05 |
| **Strategy A (ModalityGate)** | 33.3% | 71.8% | 15.0% | 22.0% | 29.4% | 67.6% | 12.4% | 17.5% | 60.67 M | 8.90 |
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

## 🚀 Running Evaluation & Training

### 1. Evaluate Strategy C (P2 High-Res FPN Model)
```bash
python tools/run_strategy_C.py
```

### 2. Evaluate Strategy A (ModalityGate Model)
```bash
python tools/run_strategy_A.py
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
├── tools/                              # Execution, evaluation, and plotting scripts
├── output/                             # Generated benchmark charts, heatmaps, & reports
├── report.md                           # Project technical report
├── run_unseen_pipeline.py             # Inference pipeline for unseen testing data
└── requirements.txt                    # Dependencies file
```
