# Cell Extraction — YOLOv8n

Stage 2: detect and classify the 81 cells on the rectified (perspective-corrected) grid. Two classes: `filled` (cell with a digit) and `empty` (blank cell).

The 81 bounding boxes are assigned to grid slots via an affine lattice fit. Missing detections are synthesised from the fitted lattice so every slot gets a box regardless of detection confidence.

## Dataset

**Source:** [Roboflow Universe — sudoku-cell-vision v6](https://universe.roboflow.com/pete-mksb1/sudoku-cell-vision/dataset/6) (Pete, workspace: `pete-mksb1`)

```
training/cell_extraction/dataset/
├── data.yaml
├── train/   (125 images)
├── val/     (11 images)
└── test/    (11 images)
```

The dataset is already included in the repository directory (untracked, gitignored).

## Training

```bash
uv run python training/cell_extraction/train.py
```

Config: `yolov8n.pt`, 100 epochs, imgsz=640, batch=16, patience=20, device=0.

Weights land at: `training/cell_extraction/runs/cell_vision_v6/weights/best.pt`

## Results

| Metric | Value |
|--------|-------|
| Precision | **99.98%** |
| Recall | **100.0%** |
| mAP@50 | **99.5%** |
| mAP@50-95 | **89.6%** |
| Model size | ~6 MB |

mAP@50 saturated at **99.5% by epoch 13** — this is an easy binary detection task once the grid is rectified. Early stopping triggered at epoch 81 (best checkpoint ~epoch 61).

## Lattice assignment

Raw YOLO detections are assigned to the 81 grid slots via an affine lattice fit:

1. Initial axis-aligned guess from detection bounding boxes
2. Least-squares refinement: fit `(cx, cy) → (col, row)` over several passes
3. Collisions resolved by distance to the fitted lattice point
4. Missing slots synthesised from the inverse lattice transform using the median detection size

This handles tilted grids and YOLO under-detections (common on ~40% of real photos) robustly. The synthesised cells go through OCR like any other — the digit reader decides if they're empty.

## Notes

- The TFLite export is `cell_vision.tflite`: `[1,3,640,640]` → `[1,6,8400]` (6 = 4 box + 2 class logits).
- `CellLattice.kt` in the Android app implements the same lattice assignment verified byte-identical to Python on 2592 slot assignments.
