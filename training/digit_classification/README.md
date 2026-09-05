# Digit Classification — YOLOv8n-cls

Alternative OCR stage: classify the digit in each cell using a YOLO classification model. Used by the `yolo_digit` pipeline path.

Classes: 0 = empty cell, 1–9 = digit value.

> **Note:** The primary OCR model is `GridOCRNet` in `training/grid_ocr/`. This YOLO-cls model is an alternative that is useful for comparison and as a fallback. End-to-end accuracy is lower (~3/5 vs 4/5 on the test set) because per-cell accuracy (~91.6%) compounds across 30+ clues.

## Dataset

**Source:** `data/grid_ocr/cells/` — 4 774 real 50×50 cell crops labelled by EasyOCR + solver verification. The synthetic/MNIST dataset is deliberately not used here (it's what the XGBoost baseline failed on).

Class imbalance: empty cells outnumber each digit class ~14:1. The prepare script caps class 0 at 2× the mean digit-class count to prevent trivial empty-prediction.

```
training/digit_classification/dataset/
├── 0/   (empty cells)
├── 1/
├── 2/
...
└── 9/
```

## Building the dataset

```bash
uv run python training/digit_classification/prepare_dataset.py
```

This copies and balances crops from `data/grid_ocr/cells/` into the classification dataset layout.

## Training

```bash
uv run python training/digit_classification/train.py
```

Config: `yolov8n-cls.pt`, 120 epochs, imgsz=64, batch=64, patience=30, device=0.

Weights land at: `training/digit_classification/runs/digit_cls/weights/best.pt`

## Results

| Metric | Value |
|--------|-------|
| Top-1 accuracy | **91.6%** |
| Top-5 accuracy | **97.9%** |
| Best epoch | 7 (early-stopped at 37) |

Failure pattern: catastrophic on low-contrast images without per-crop min-max normalisation. With normalisation: 3.8% → 76.9% on the hardest test image.

## Notes

- The model path is resolved by `YoloDigitClassifierConfig` in `src/sudoku_solver/config.py`.
- Future improvement: expose per-cell probabilities so `_recover_with_constraints` can use runner-up predictions (already wired for GridOCRNet).
- Handwriting accuracy is ~57% (the model was trained on printed crops only). The GridOCR mixed-training approach would help here too.
