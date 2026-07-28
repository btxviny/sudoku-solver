# Sudoku Solver

End-to-end computer vision pipeline that takes a photograph of a sudoku puzzle and returns the completed solution. Three stages: locate the grid, read the digits, solve the puzzle.

## Quick Start

```bash
pip install -e ".[dev]"
python -m sudoku_solver test_images/sudoku.png
```

```python
from sudoku_solver import SudokuPipeline

pipe = SudokuPipeline()
result = pipe.run("test_images/sudoku.png")
pipe.print_result(result)

# result.original_grid  — detected 9×9 puzzle
# result.solved_grid    — completed solution
# result.timing         — per-stage timings
# result.grid_image     — rectified 450×450 grid image
```

## Project Structure

```
├── src/sudoku_solver/          # Main package
│   ├── pipeline.py             # End-to-end orchestrator
│   ├── grid_detector.py        # Mask R-CNN + Hough warp
│   ├── grid_ocr.py             # GridOCRNet digit reader
│   ├── sudoku_solver.py        # OR-Tools CP-SAT solver
│   ├── config.py               # Dataclass configuration
│   └── cli.py                  # CLI entry point
├── training/
│   ├── segmentation/scripts/   # Mask R-CNN training
│   └── grid_ocr/scripts/       # GridOCRNet training
├── data/                       # Datasets (gitignored — see below)
├── models/weights/             # Trained weights (gitignored — see below)
├── test_images/                # Sample images
└── pyproject.toml
```

---

## Pipeline

### Stage 1 — Grid Detection

**Input:** Raw photograph of a sudoku puzzle, any size or angle.  
**Output:** 450×450 pixel image of the grid, perspective-corrected.

A Mask R-CNN (ResNet-50 FPN backbone, pretrained on COCO, fine-tuned on sudoku images) produces an approximate pixel mask of the grid region. Canny edge detection and probabilistic Hough lines then find the four outer borders precisely; their intersections are used to compute a perspective warp to a clean 450×450 square. If Hough fails, the pipeline falls back to the approximate quadrilateral from the mask contour.

**Training dataset:** Roboflow `sudoku-lq9gj` (CC BY 4.0) — 350 train / 100 val images, polygon segmentation masks in YOLOv8 format.

**Training config:** SGD, lr=2×10⁻⁴, momentum=0.9, weight decay=5×10⁻⁴, batch size=2, 10 epochs, early stopping patience=3.

| Epoch | Train loss | Val loss |
|------:|-----------:|---------:|
| 1     | 0.641      | 0.232    |
| 3     | 0.135      | 0.120    |
| 5     | 0.107      | 0.105    |
| 7     | 0.096      | 0.099    |
| 10    | 0.085      | 0.089    |

---

### Stage 2 — Digit Reading (GridOCRNet)

**Input:** 450×450 rectified grid image.  
**Output:** 9×9 matrix of integers (0 = empty, 1–9 = digit) and per-cell confidence scores.

The grid is split into 81 equal 50×50 patches. Each patch is preprocessed before inference: for normal black-on-white images, full-width dark rows (grid border bleed) are erased and the digit content is re-centred vertically; for coloured or low-contrast images, per-patch min-max normalisation remaps digit pixels to black and background to white. All 81 patches run through GridOCRNet in a single batched forward pass.

GridOCRNet is a lightweight CNN: two convolutional blocks (32 then 64 channels, batch norm, GELU), a 128-channel block with global average pooling producing a 2048-dim feature vector, then a two-layer head with dropout mapping to 10 logits (0 = empty, 1–9 = digit).

**Training config:** AdamW, lr=1×10⁻³ with OneCycleLR, weight decay=1×10⁻⁴, cross-entropy with label smoothing=0.05, batch size=128, 50 epochs.

**Augmentation:** Grid-line borders (1–6 px wide, offset 0–8 px from edge, on 1–3 sides, p=0.50), Gaussian noise, Gaussian blur, JPEG compression, ±8° rotation, random erasing, varied fonts and ink colours on synthetic images.

**Per-class results on 4,774 real cell crops:**

| Class    | Support | Precision | Recall | F1    |
|----------|--------:|----------:|-------:|------:|
| empty    | 2,948   | 1.000     | 1.000  | 1.000 |
| 1        |   132   | 0.851     | 0.909  | 0.879 |
| 2        |   221   | 0.960     | 0.982  | 0.971 |
| 3        |   216   | 0.946     | 0.977  | 0.961 |
| 4        |   212   | 0.985     | 0.953  | 0.969 |
| 5        |   223   | 0.977     | 0.969  | 0.973 |
| 6        |   242   | 0.979     | 0.959  | 0.969 |
| 7        |   185   | 0.936     | 0.946  | 0.941 |
| 8        |   212   | 0.967     | 0.958  | 0.962 |
| 9        |   183   | 0.955     | 0.918  | 0.936 |
| **Overall** | **4,774** | **0.983** | **0.983** | **0.983** |

---

### Stage 3 — Solving

**Input:** 9×9 digit matrix.  
**Output:** Completed 9×9 solution grid.

Google OR-Tools CP-SAT applies constraint propagation and backtracking — typically under 10 ms. If the OCR produced an inconsistent grid, the pipeline attempts recovery: it first tries substituting the 2nd or 3rd most probable digit for the least-confident cells (exhaustive 1–3 cell search). If that fails, it zeros out all cells below a confidence threshold (skipped if more than 6 filled cells would be affected) and lets the solver fill them via constraint propagation.

---

## Results

| Image        | Digit accuracy | Result |
|--------------|---------------|--------|
| sudoku.png   | 100 %         | Correct (~0.8 s) |
| sudoku_1.png | 100 %         | Correct (~0.12 s) |
| sudoku_3.png | 32 % (26/81)  | Failed — red coloured digits, out of distribution |
| sudoku_2.png | —             | Failed — Mask R-CNN did not detect the grid |

---

## Reproducing the Models

Model weights are not stored in this repository. Follow the steps below to download the data and retrain each model from scratch.

### Prerequisites

```bash
pip install -e ".[dev]"
```

You will also need a [Roboflow](https://roboflow.com) account and API key to download the datasets.

---

### Model 1 — Mask R-CNN Grid Detector

**Goal:** Train the model that locates and warps the sudoku grid.

#### 1. Download the dataset

Go to [https://universe.roboflow.com/j-w-kydp4/sudoku-lq9gj](https://universe.roboflow.com/j-w-kydp4/sudoku-lq9gj), select version 1, choose **YOLOv8 Segmentation** format, and download to `data/segmentation/segmentation_dataset/`. The expected layout is:

```
data/segmentation/segmentation_dataset/
├── train/
│   ├── images/   # 350 images
│   └── labels/   # polygon .txt files
└── valid/
    ├── images/   # 100 images
    └── labels/
```

Alternatively, use the Roboflow Python SDK:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("j-w-kydp4").project("sudoku-lq9gj")
project.version(1).download("yolov8", location="data/segmentation/segmentation_dataset")
```

#### 2. Train

```bash
python training/segmentation/scripts/train_maskrcnn.py \
    --data_root data/segmentation/segmentation_dataset \
    --num_epochs 10 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --patience 3 \
    --output_dir models/weights
```

Training takes roughly 20–30 minutes on a GPU. The best checkpoint is saved to `models/weights/maskrcnn_sudoku_<timestamp>.pth`. Update `src/sudoku_solver/config.py` with the new filename.

---

### Model 2 — GridOCRNet Digit Classifier

**Goal:** Train the CNN that reads all 81 cell digits in one forward pass.

#### 1. Download the Roboflow cell datasets

Two datasets are used. Download both in **COCO JSON** format.

**Dataset A — sudoku-cell-detector**  
[https://universe.roboflow.com/ninos-workspace-jbsuh/sudoku-cell-detector](https://universe.roboflow.com/ninos-workspace-jbsuh/sudoku-cell-detector)

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("ninos-workspace-jbsuh").project("sudoku-cell-detector")
project.version(1).download("coco", location="data/roboflow/sudoku-cell-detector")
```

**Dataset B — sudoku-cell-vision**  
[https://universe.roboflow.com/pete-mksb1/sudoku-cell-vision](https://universe.roboflow.com/pete-mksb1/sudoku-cell-vision)

```python
project = rf.workspace("pete-mksb1").project("sudoku-cell-vision")
project.version(1).download("coco", location="data/roboflow/sudoku-cell-vision")
```

Each download produces `train/`, `valid/`, and `test/` folders, each containing `_annotations.coco.json` and the images.

#### 2. Extract and label cell crops

This script reads the COCO annotations, crops each annotated cell from the grid images, and saves them under `data/grid_ocr/cells/{digit}/` using the ground-truth digit label. It also requires access to the already-trained Mask R-CNN (or the rectified grid images directly) to pair bounding boxes with digit values.

```bash
python training/grid_ocr/scripts/extract_and_label_cells.py
```

After running, the folder structure should be:

```
data/grid_ocr/cells/
├── 0/    # ~2948 empty cell crops
├── 1/    # ~132  crops
├── 2/    # ~221  crops
...
└── 9/    # ~183  crops
```

#### 3. Train

The training script combines the real cell crops with a 60,000-image synthetic dataset generated on the fly each epoch.

```bash
python training/grid_ocr/scripts/train_cell_classifier.py \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-3 \
    --synthetic_size 60000 \
    --num_workers 4
```

Training takes roughly 1–2 hours on a GPU (50 epochs × ~64k samples). The best checkpoint by validation accuracy is saved to `models/weights/grid_ocr_cnn.pth`.

Expected final metrics: **~98.3% accuracy** on the real cell crop validation set.

---

## Testing

```bash
pytest tests/ -v
```

## Dependencies

Python ≥ 3.10, PyTorch, torchvision, OpenCV, Google OR-Tools, scikit-learn. See `pyproject.toml` for exact versions.
