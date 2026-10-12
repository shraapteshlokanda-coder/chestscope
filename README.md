# ChestScope: Automated Chest X-Ray Classifier with Explainable AI

**Shraaptesh Lokanda** | KTP Engineering-track capstone (Biomedical Engineering / Computational)

> **Triage-support research prototype. Not a diagnostic device.** No clinical claims are made.

## Problem
Radiologist shortages and high imaging volumes cause diagnostic delays and missed findings, especially for common
conditions like pneumonia. ChestScope flags likely pneumonia on chest X-rays with a confidence score and (Week 3) a
Grad-CAM heatmap showing what the model is looking at, to support, not replace, clinical judgment.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/raw --unzip
python -m src.make_splits          # leakage audit + patient-grouped splits -> data/splits/
python -m src.run_checkpoint2      # baseline + ResNet18 experiments (validation only)
python -m src.make_diagrams        # pipeline / architecture diagrams -> docs/diagrams/
```
`data/` is not committed; the splits regenerate deterministically (seed 42).

## Data
Kaggle "Chest X-Ray Images (Pneumonia)" (pediatric, single institution).
| Split | Images | % pneumonia | Patient groups |
|---|---|---|---|
| train | 4185 | 74.2 | 2120 |
| val | 1047 | 74.2 | 530 |
| test | 624 | 62.5 | 404 |

Leakage audit (official splits): 264 patient-ID groups appear in both the official train and test folders (possible overlap or ID collisions). Train/val overlap after the patient-grouped re-split: 0. pos_weight = 0.347.

- Official `val` has only 16 images, so train+val were pooled and re-split **grouped by patient** (StratifiedGroupKFold).
- The official `test` set is held out and untouched until Week 3.
- Integrity assertions: no patient or image overlap between train and val.

## Methods (through Checkpoint 2)
- **Preprocessing:** grayscale to 3 channels, resize 224x224, ImageNet normalization.
- **Augmentation (train only):** random resized crop, rotation +/-10 deg, brightness/contrast jitter. No flips (anatomical laterality).
- **Class imbalance:** weighted BCE (`pos_weight` = n_neg / n_pos from the train split); accuracy is not used as the headline metric.
- **Baselines:** majority-class dummy; SimpleCNN (4 conv blocks, from scratch).
- **Model:** ImageNet-pretrained ResNet18, two-phase fine-tuning (frozen backbone, then full fine-tune with lower backbone lr). Best checkpoint chosen by validation ROC-AUC.
- Diagrams: `docs/diagrams/pipeline.png`, `architecture.png`, `baseline_architecture.png`.

## Results 
| Model | auc | recall | specificity | precision | f1 | accuracy |
|---|---|---|---|---|---|---|
| Majority-class dummy | 0.500 | 1.000 | 0.000 | 0.742 | 0.852 | 0.742 |
| SimpleCNN (scratch) | 0.988 | 0.732 | 0.989 | 0.995 | 0.844 | 0.798 |
| ResNet18 (fine-tuned) | 0.998 | 0.969 | 0.996 | 0.999 | 0.984 | 0.976 |



## Limitations
- Pediatric, single-center data; may not generalize to adults or other scanners.
- Official train and test distributions differ (class balance), so test performance may be lower than validation.
- Possible shortcut features (intensity or image-size differences between classes); see EDA notebook.
- Labels are not independently verified. Research prototype only.

## Future Work
Hyperparameter tuning, first held-out test evaluation, Grad-CAM overlays and error analysis, optional Streamlit upload demo.
