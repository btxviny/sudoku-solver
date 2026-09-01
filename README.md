# Sudoku Solver

End-to-end computer vision pipeline: photograph a sudoku puzzle, get the solution. Runs as a Python pipeline and as an Android app (Pixel 9 / arm64).

Three stages: locate the grid with YOLOv8n-pose → read digits with GridOCRNet → solve with MRV backtracking.

---

## Quick Start (Python)

```bash
uv sync
uv run python -m sudoku_solver test_images/sudoku.png
```

```python
from sudoku_solver import SudokuPipeline

pipe = SudokuPipeline()
result = pipe.run("test_images/sudoku.png")
pipe.print_result(result)
```

---

## Android App

### Prerequisites

- Android Studio (any recent version) installed on Windows
- WSL2 with this repo at `/home/viny/sudoku-solver`

### Build

**Step 1 — Export models and sync to the Windows project**

Run this once from WSL, and again after any retraining:

```bash
bash scripts/deploy_android.sh
```

This exports all four TFLite models from the best trained weights and copies them — plus the Kotlin source, layouts, and build files — to `C:\Users\csps0\Documents\sudoku-solver`.

**Step 2 — Compile and install**

Open `C:\Users\csps0\Documents\sudoku-solver` in Android Studio, then:

```
Build → Make Project
```

Fix any reported compile errors, then run on the device with the ▶ button.

### Model files in the APK

| File | Input | Output | Size |
|------|-------|--------|------|
| `grid_pose.tflite` | `[1,3,640,640]` | `[1,17,8400]` | ~13 MB |
| `cell_vision.tflite` | `[1,3,640,640]` | `[1,6,8400]` | ~12 MB |
| `gridocr.tflite` | `[81,1,50,50]` | `[81,10]` | ~3 MB |
| `grid_seg.tflite` | `[1,3,640,640]` | `[1,37,8400]` + mask | ~13 MB |

All models use NCHW layout and float32. `grid_seg.tflite` is present as fallback; the app uses `grid_pose.tflite`.

### Accelerator

GPU delegate (LiteRT) where the device supports it, otherwise multi-threaded CPU. NNAPI is deprecated on Android 15; the Tensor TPU is not exposed to third-party apps.

---

## Project Structure

```
├── src/sudoku_solver/          # Python package
│   ├── pipeline.py             # End-to-end orchestrator
│   ├── yolo_grid_detector.py   # YOLOv8n-seg/pose wrapper
│   ├── grid_geometry.py        # Perspective warp, corner ordering
│   ├── grid_ocr.py             # GridOCRNet digit reader
│   ├── config.py               # Dataclass configuration
│   └── cli.py                  # CLI entry point
├── training/
│   ├── grid_pose/              # YOLOv8n-pose grid detector training
│   ├── grid_seg/               # YOLOv8n-seg grid detector training
│   └── cell_extraction/        # YOLOv8n cell detector training
├── scripts/
│   ├── deploy_android.sh       # Export + sync to Windows project (one-liner)
│   ├── export_tflite.py        # Export all models to TFLite
│   ├── verify_tflite_pipeline.py   # TFLite vs PyTorch numeric check
│   ├── verify_kotlin_port.py       # Kotlin solver vs OR-Tools (604 puzzles)
│   ├── verify_kotlin_preprocess.py # Kotlin cell preprocessor byte check
│   ├── verify_kotlin_geometry.py   # Corner ordering + lattice slot check
│   └── compare_grid_detectors.py   # Corner error eval on held-out photos
├── android/                    # Android Studio project (WSL copy)
│   └── app/src/main/java/com/sudokusolver/
│       ├── MainActivity.kt
│       ├── SudokuGridView.kt
│       └── core/
│           ├── SudokuPipeline.kt   # 4-stage pipeline
│           ├── SudokuSolver.kt     # MRV bitmask backtracking
│           ├── ConstraintRecovery.kt
│           ├── YoloDecoder.kt      # Letterbox, NMS, pose decoding
│           ├── GridGeometry.kt     # Perspective warp
│           ├── CellLattice.kt      # Affine lattice slot assignment
│           ├── CellPreprocessor.kt
│           └── Models.kt           # TFLite wrappers
├── data/                       # Datasets (gitignored)
└── pyproject.toml
```

---

## Pipeline

### Stage 1 — Grid Detection

**Models:** YOLOv8n-pose (`grid_pose.tflite`) primary; YOLOv8n-seg (`grid_seg.tflite`) available.

The pose model predicts four corner keypoints directly — no mask prototype decoding needed on device. Corners are canonicalised to TL/TR/BR/BL order by angular sort and a perspective warp produces a square rectified grid.

Hough refinement is disabled for YOLO (the `refine=False` default in `YoloGridDetectorConfig`). Enabling it caused mean corner error to increase from 5.7% to 40.9%.

**Training dataset:** Roboflow `sudoku-lq9gj` (CC BY 4.0), converted to pose format with `training/grid_pose/prepare_dataset.py`.

| Metric | Mask R-CNN (old) | YOLOv8n-seg | YOLOv8n-pose |
|--------|-----------------|-------------|--------------|
| Mean corner error | 5.98% | 3.08% | ~5.7% |
| Model size | 169 MB | 13 MB | 13 MB |
| Inference | slow | 5× faster | 5× faster |

---

### Stage 2 — Cell Detection

**Model:** YOLOv8n (`cell_vision.tflite`), `[1,6,8400]` output (empty / filled classes).

Runs on the rectified grid. Detections are assigned to the 81 grid slots via an affine lattice fit (3×3 normal equations, 6-pass refinement). Missing cells are synthesised from the lattice. Verified: 2592/2592 slot assignments identical to Python, max diff 1.7e-13.

---

### Stage 3 — Digit Reading (GridOCRNet)

All 81 cell patches run in a single batched forward pass (`[81,1,50,50]` → `[81,10]` logits). Softmax is applied in Kotlin after the TFLite call.

Preprocessing matches training exactly (critical — deviating by even one pixel drop accuracy 20%):
- Low-contrast images: min-max stretch
- Normal images: grid-line border removal + vertical re-centring of digit content

Verified: 5103 patches byte-identical to Python.

**Accuracy on 3,240 labelled cells:** PyTorch 87.81%, TFLite 87.78% (one cell difference).

---

### Stage 4 — Solving

MRV (minimum remaining values) bitmask backtracking — 8.9× faster than OR-Tools CP-SAT on the same 604-puzzle set, zero disagreements.

If the digits as read don't yield a valid puzzle, constraint recovery tries runner-up predictions on the least-confident cells (up to 3 cells, 3 substitutions each, then confidence-threshold blanking at 0.80 / 0.70 / 0.60). If the puzzle has more than one solution (too few digits read), it is rejected with an explanatory message rather than returning an arbitrary completion.

---

## Reproducing the Models

Weights are not stored in this repo. Run the export script once weights are trained:

```bash
uv run python scripts/export_tflite.py
```

### Grid detector (YOLOv8n-pose)

```bash
# Prepare pose-format labels from Roboflow seg dataset
uv run python training/grid_pose/prepare_dataset.py

# Train
uv run python training/grid_pose/train.py
```

### Cell detector (YOLOv8n)

```bash
uv run python training/cell_extraction/train.py
```

### GridOCRNet

```bash
uv run python training/grid_ocr/scripts/train_cell_classifier.py \
    --epochs 50 --batch_size 128 --lr 1e-3
```

---

## Verification Scripts

All scripts exit 0 on pass, non-zero on failure.

```bash
uv run python scripts/verify_kotlin_port.py       # 604 puzzles: Kotlin vs OR-Tools
uv run python scripts/verify_kotlin_preprocess.py # 5103 patches: byte identity
uv run python scripts/verify_kotlin_geometry.py   # corner order + lattice slots
uv run python scripts/verify_kotlin_decoder.py    # YOLO decoding vs Ultralytics
uv run python scripts/verify_tflite_pipeline.py   # end-to-end TFLite vs PyTorch
```

---

## Testing

```bash
uv run pytest tests/ -v
```

## Dependencies

Python ≥ 3.10, PyTorch, Ultralytics, OpenCV, scikit-learn. Managed with [uv](https://docs.astral.sh/uv/) — see `pyproject.toml`.

Android: OpenCV 4.10.0 (Maven Central), TensorFlow Lite 2.16.1 + GPU delegate, Kotlin coroutines.
