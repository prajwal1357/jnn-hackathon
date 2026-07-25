# Strategy C — High-Resolution P2 Feature Pyramid Level (Layered on Strategy A+B)

**Strategy ID:** C  
**Strategy Name:** High-Resolution P2 Feature Pyramid Level  
**Layered On Top Of:** Strategy A (ModalityGate) + Strategy B (Small-Object Loss)  
**Status:** ✅ Implemented & Evaluated  

---

## 1. Concept & Motivation

By default, standard FPN begins at stride 8 (P3). Tiny pedestrians (< 16x16 pixels) lose critical spatial details through downsampling. Strategy C adds a high-resolution P2 pyramid level at **stride 4** (96 x 160 resolution), tapping directly into ResNet stage C2 features to preserve fine-grained spatial information for tiny objects.

---

## 2. Full 4-Stage Ablation Table

| Model / Stage | Val mAP | Val mAP50 | Val mAP_S (Small) | Val mAP_M (Med) | Val mAP_L (Large) | Test mAP | Test mAP50 | Test mAP_S | Params | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **QFDet Baseline** | 33.8% | 72.1% | 14.4% | 32.4% | 58.5% | 29.9% | 67.4% | 12.9% | 60.63 M | 9.05 |
| **Strategy A (Gate)** | 33.3% | 71.8% | 15.0% | 32.3% | 57.6% | 29.4% | 67.6% | 12.4% | 60.67 M | 8.90 |
| **Strategy A+B (Loss)** | 30.7% | 70.2% | 14.2% | 30.1% | 52.6% | 26.8% | 65.1% | 12.1% | 60.67 M | 8.90 |
| **Strategy A+B+C (P2 High-Res)** | **31.8%** | **72.3%** | **13.8%** | **30.0%** | **57.3%** | **28.8%** | **69.8%** | **13.8%** | **60.73 M** | **5.29** |
