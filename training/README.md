# Training

Five models power the pipeline. Each has its own subdirectory with a `train.py` and, where needed, a `prepare_dataset.py` that converts a downloaded dataset into the format the trainer expects.

## Models at a glance

| Directory | Model | Task | Weights land at |
|-----------|-------|------|-----------------|
| `grid_seg/` | YOLOv8n-seg | Locate the grid → segmentation mask | `grid_seg/runs/grid_seg_v1/weights/best.pt` |
| `grid_pose/` | YOLOv8n-pose | Locate the grid → 4 corner keypoints | `grid_pose/runs/grid_pose_v1/weights/best.pt` |
| `cell_extraction/` | YOLOv8n | Detect 81 cells on the rectified grid | `cell_extraction/runs/cell_vision_v6/weights/best.pt` |
| `grid_ocr/` | GridOCRNet (CNN) | Classify digit in each cell (primary OCR) | `models/weights/grid_ocr_cnn.pth` |
| `digit_classification/` | YOLOv8n-cls | Classify digit in each cell (alt OCR) | `digit_classification/runs/digit_cls/weights/best.pt` |

The pipeline uses **grid_seg** (or grid_pose) → **cell_extraction** → **grid_ocr** in that order. `digit_classification` is an alternative OCR path used by `yolo_digit` mode.

## Train everything at once

```bash
uv run python train_all_models.py
```

Train specific steps:

```bash
uv run python train_all_models.py --steps grid_ocr cell_extraction
```

## Train each model individually

See the `README.md` in each subdirectory for dataset download, training commands, and expected results.

| Step | Command |
|------|---------|
| Grid segmentation | `uv run python training/grid_seg/train.py` |
| Grid pose | `uv run python training/grid_pose/prepare_dataset.py && uv run python training/grid_pose/train.py` |
| Cell extraction | `uv run python training/cell_extraction/train.py` |
| GridOCR (primary) | `uv run python training/grid_ocr/scripts/train_cell_classifier.py` |
| Digit classifier (alt) | `uv run python training/digit_classification/prepare_dataset.py && uv run python training/digit_classification/train.py` |

## Data

All datasets are gitignored. Download links are in each module's README.

```
data/
├── segmentation/           # Grid detector datasets (Roboflow)
│   ├── segmentation_dataset/   # YOLOv8 polygon format (grid_seg)
│   └── pose_dataset/           # Converted to pose format (grid_pose)
├── grid_ocr/
│   └── cells/              # 4 774 real 50×50 cell crops, labelled by EasyOCR
└── wicht_sudoku/           # End-to-end evaluation set (held-out)
```

## GPU

All training scripts default to `device=0` (first CUDA GPU). Change to `device="cpu"` for CPU-only training — slower but functional. The project was developed on an RTX 5070 Ti (16 GB).
