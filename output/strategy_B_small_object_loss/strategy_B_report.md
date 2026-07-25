# Strategy B — Small-Object-Weighted Loss (Layered on Strategy A)

**Strategy ID:** B  
**Strategy Name:** Small-Object-Weighted Loss  
**Layered On Top Of:** Strategy A (ModalityGate)  
**Status:** ✅ Implemented & Evaluated  

---

## 1. Concept & Motivation

Standard detection loss treats mistakes on large, obvious pedestrians the same as mistakes on tiny pedestrians (< 32² px). In drone surveillance, small pedestrians dominate the instance count and represent the hardest detection cases.

### Weighting Function
We applied smooth inverse-area loss weighting to bounding box regression loss:
```
w_scale = 1.0 + 1.5 * max(0, 1.0 - (area / 1024.0))
loss_bbox = loss_bbox * (centerness_target * w_scale)
```
- **Tiny objects (< 32² px):** Receive up to 2.5× loss weight.
- **Large objects (≥ 32² px):** Receive standard 1.0× loss weight.

---

## 2. Evaluation Results (Strategy A vs. Strategy A+B Stacked)

| Metric | QFDet Baseline | Strategy A (Gate) | **Strategy A+B (Stacked)** | Gain vs Baseline | Gain vs Strategy A |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **mAP_S (Val)** | 14.4% | 15.0% | **14.2%** | -0.2pp | -0.8pp |
| **mAP50 (Val)** | 72.1% | 71.8% | **70.2%** | -1.9pp | -1.6pp |
| **mAP (Val)** | 33.8% | 33.3% | **30.7%** | -3.1pp | -2.6pp |
| **mAP_S (Test)** | 12.9% | 12.4% | **12.1%** | -0.8pp | -0.3pp |
| **mAP50 (Test)** | 67.4% | 67.6% | **65.1%** | -2.3pp | -2.5pp |
| **Params (M)** | 60.63 M | 60.67 M | **60.67 M** | +0.04 M | 0.00 M (No extra params!) |
| **FPS** | 9.05 | 8.90 | **8.80 FPS** | -0.15 FPS | 0.00 FPS |
