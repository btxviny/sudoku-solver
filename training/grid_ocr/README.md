# GridOCR — Custom CNN Digit Classifier

Stage 3 (primary): classify the digit in each of the 81 cell patches. A lightweight residual CNN takes a `70×70` grayscale crop and outputs 10 logits (0 = empty, 1–9 = digit).

All 81 cells run in a single batched forward pass: `[81, 1, 70, 70]` → `[81, 10]`.

## Architecture

**GridOCRNet** — 4 residual blocks with stride-2 downsampling, GlobalAvgPool, 2-layer head.

```
Input  [B, 1, 70, 70]
Stem   Conv 3×3 → 32 ch
Block1 ResBlock 32→64,  stride 2  → [B, 64, 35, 35]
Block2 ResBlock 64→128, stride 2  → [B, 128, 17, 17]
Block3 ResBlock 128→256,stride 2  → [B, 256, 8, 8]
Block4 ResBlock 256→256,stride 1  → [B, 256, 8, 8]
Pool   GlobalAvgPool              → [B, 256]
Head   Linear(256→128) → GELU → Dropout(0.4) → Linear(128→10)
```

~1.5M parameters, ~3 MB on disk. Activation: GELU throughout.

## Dataset

Three sources combined:

| Source | Size | Description |
|--------|------|-------------|
| Real printed crops | ~4 774 | 70×70 crops from Roboflow sudoku photos, labelled by EasyOCR + solver verification |
| Font-rendered synthetic | 60 000 | System fonts rendered into sudoku-cell domain |
| MNIST handwritten | 40 000 | MNIST glyphs composited into synthetic cells (MNIST zeros dropped — class 0 = empty, not the digit zero) |

**Location:** `data/grid_ocr/cells/` (real crops, gitignored)

The synthetic and handwritten sets are generated on-the-fly during training by `train_cell_classifier.py`.

## Building the real-crops dataset

```bash
# Extract cell crops from raw sudoku images and label with EasyOCR
uv run python training/grid_ocr/scripts/extract_and_label_cells.py

# (Optional) extract from the Wicht evaluation set
uv run python training/grid_ocr/scripts/extract_wicht_cells.py
```

## Training

```bash
uv run python training/grid_ocr/scripts/train_cell_classifier.py \
    --epochs 50 --batch_size 128 --lr 1e-3 \
    --synthetic_size 60000 --handwritten_size 40000
```

Weights land at: `models/weights/grid_ocr_cnn.pth`  
Checkpoint (best val acc) saved to: `training/grid_ocr/checkpoints/best.pth`

## Results

| Metric | Value |
|--------|-------|
| Accuracy on 3 240 labelled cells (PyTorch) | **87.81%** |
| Accuracy on 3 240 labelled cells (TFLite) | **87.78%** (1 cell difference) |
| Printed digit accuracy (PXL photo) | **100%** |
| Handwritten digit accuracy (PXL photo) | **90.9%** |

## Preprocessing (critical)

The preprocessing must match training exactly — deviating by even one pixel drops accuracy ~20%.

1. **Contrast check** — computed on the 2nd/98th percentile of the image (not min/max, which is fooled by single bright/dark pixels).
2. **Low-contrast images** — min-max stretch applied.
3. **Normal images** — grid-line border removed, digit content vertically re-centred.

The cell patches are sampled as fixed 70×70 windows from the grid image rescaled to 630 px (not per-cell resize), which preserves the scale the model was trained at.

## TFLite export

```bash
uv run python scripts/export_tflite.py
```

Output: `android/app/src/main/assets/gridocr.tflite` — `[81, 1, 70, 70]` → `[81, 10]`  
Softmax is applied in Kotlin after the TFLite call.

## Notes

- The mixed printed+handwritten training set eliminated the original handwriting weakness (47.7% → 90.9%) with no regression on printed digits.
- Constraint recovery in `SudokuPipeline` uses runner-up logits to repair the most ambiguous reads when the raw output doesn't yield a valid puzzle.
- The `yolo_digit` alternative OCR path (`digit_classification/`) scores lower end-to-end (~3/5 vs 4/5) but was trained separately on real crops only (no handwriting augmentation yet).
