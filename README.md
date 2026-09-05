# Sudoku Solver

Photograph a sudoku puzzle, get the solution. Runs as a Python web app and as a native Android app.

**Pipeline:** locate grid → detect cells → read digits → solve

---

## Quick start

```bash
uv sync
uv run streamlit run app.py          # web UI
uv run python -m sudoku_solver test_images/wicht_printed_iphone_1024.jpg
```

Or from Python:

```python
from sudoku_solver import SudokuPipeline

pipe = SudokuPipeline()
result = pipe.run("photo.jpg")
pipe.print_result(result)
```

---

## Pipeline

```
Photo
  │
  ▼
┌─────────────────┐
│  Grid Detection │  YOLOv8n-seg → mask → perspective warp
│                 │  Rectified 630×630 grid
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Cell Detection │  YOLOv8n → 81 bounding boxes
│                 │  Affine lattice fit, missing cells synthesised
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Digit OCR     │  CellOCRNet CNN [81,1,70,70] → [81,10] logits
│                 │  98.5% cell accuracy; constraint recovery on failure
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Solver      │  MRV bitmask backtracking
│                 │  8.9× faster than OR-Tools CP-SAT
└─────────────────┘
```

### Digit readers

Two readers are installed and selectable everywhere — the Streamlit sidebar, `--path` on the CLI, and a picker in the Android app. **CellOCR is the default.**

| | GridOCR (1st gen) | CellOCR (2nd gen) |
|---|---|---|
| Network | Residual CNN, 2.42 M params | Squeeze-excitation CNN, 2.01 M params |
| Handwriting corpus | MNIST | EMNIST (MNIST held out) |
| Trained on cleaned patches | no | yes |

### Accuracy (40 held-out real photos, `data/wicht_sudoku/v2_test`)

| Metric | GridOCR | CellOCR |
|---|---|---|
| Grids read perfectly | 13/40 (32.5%) | **38/40 (95.0%)** |
| End-to-end solve | 30/40 (75.0%) | **38/40 (95.0%)** |
| Cell accuracy | 96.08% | **98.52%** |

On `half_mixed_test` (printed clues plus pasted handwriting) CellOCR reads 26/40 grids perfectly against GridOCR's 15/40 and solves 38/40 against 37/40. GridOCR scores higher on that set's *handwritten* cells only because they are the MNIST glyphs it trained on; see [`training/cell_ocr/README.md`](training/cell_ocr/README.md).

Constraint recovery (runner-up logits, confidence thresholding) handles the gap between cell accuracy and end-to-end solve rate.

---

## Android App

### Build

**Step 1** — export TFLite models and sync to Windows:

```bash
bash scripts/deploy_android.sh
```

**Step 2** — open `C:\Users\csps0\Documents\sudoku-solver` in Android Studio → **Build → Make Project** → run on device.

### Models in the APK

| File | Input | Output | Size |
|------|-------|--------|------|
| `grid_seg.tflite` | `[1,3,640,640]` | mask | ~13 MB |
| `grid_pose.tflite` | `[1,3,640,640]` | `[1,17,8400]` | ~13 MB |
| `cell_vision.tflite` | `[1,3,640,640]` | `[1,6,8400]` | ~12 MB |
| `gridocr.tflite` | `[81,1,70,70]` | `[81,10]` | ~3 MB |

GPU delegate (LiteRT) on supported devices; multi-threaded CPU otherwise.

---

## Project structure

```
├── src/sudoku_solver/          # Python package
│   ├── pipeline.py             # End-to-end orchestrator
│   ├── yolo_grid_detector.py   # YOLOv8n-seg/pose wrapper
│   ├── yolo_cell_extractor.py  # YOLOv8n cell detector + lattice assignment
│   ├── grid_geometry.py        # Perspective warp, corner fitting
│   ├── digit_reader.py         # Reading protocol shared by both readers
│   ├── cell_prep.py            # Patch cleanup (ported 1:1 to Kotlin)
│   ├── grid_ocr.py             # GridOCRNet digit reader (1st gen)
│   ├── cell_ocr.py             # CellOCRNet digit reader (2nd gen, default)
│   ├── sudoku_solver.py        # MRV backtracking solver
│   └── config.py               # Dataclass configuration
├── training/                   # One subdirectory per model (see training/README.md)
│   ├── grid_seg/               # YOLOv8n-seg grid detector
│   ├── grid_pose/              # YOLOv8n-pose grid detector
│   ├── cell_extraction/        # YOLOv8n cell detector
│   ├── grid_ocr/               # GridOCRNet CNN digit classifier (1st gen)
│   ├── cell_ocr/               # CellOCRNet SE-CNN digit classifier (2nd gen)
│   └── digit_classification/   # YOLOv8n-cls digit classifier (alt)
├── android/                    # Android Studio project (Kotlin)
│   └── app/src/main/java/com/sudokusolver/core/
│       ├── SudokuPipeline.kt   # 4-stage pipeline
│       ├── SudokuSolver.kt     # MRV bitmask backtracking
│       ├── YoloDecoder.kt      # Letterbox, NMS, pose decoding
│       ├── GridGeometry.kt     # Perspective warp
│       ├── CellLattice.kt      # Affine lattice slot assignment
│       └── Models.kt           # TFLite wrappers
├── scripts/                    # Export, verification, evaluation
├── test_images/                # Sample images (see test_images/README.md)
├── app.py                      # Streamlit web UI
└── train_all_models.py         # Train everything: uv run python train_all_models.py
```

---

## Training

See [`training/README.md`](training/README.md) for dataset downloads and per-model details.

```bash
uv run python train_all_models.py                     # all models
uv run python train_all_models.py --steps grid_ocr    # one model
```

After training, export to TFLite:

```bash
uv run python scripts/export_tflite.py
```

---

## Verification

All scripts exit 0 on pass.

```bash
uv run python scripts/verify_kotlin_port.py       # 604 puzzles: Kotlin vs OR-Tools
uv run python scripts/verify_kotlin_preprocess.py # 5 103 patches: byte identity
uv run python scripts/verify_kotlin_geometry.py   # corner order + lattice slots
uv run python scripts/verify_kotlin_decoder.py    # YOLO decoding vs Ultralytics
uv run python scripts/verify_tflite_pipeline.py   # end-to-end TFLite vs PyTorch
uv run python scripts/verify_tflite.py            # both readers: TFLite vs PyTorch
uv run python scripts/eval_wicht.py data/wicht_sudoku/v2_test   # score every reader
uv run pytest tests/ -v
```

---

## Dependencies

Python ≥ 3.10, PyTorch, Ultralytics, OpenCV, Streamlit. Managed with [uv](https://docs.astral.sh/uv/).

Android: OpenCV 4.10.0, TensorFlow Lite 2.16.1 + GPU delegate, Kotlin coroutines.
