"""Test the YOLOv8n cell-vision model on sudoku images.

Two modes:
  --dataset   Run on the held-out test split and print mAP / cell counts.
  --image     Run on a single image through the full pipeline (digit classify + solve).

Usage:
  uv run python scripts/test_yolo_cell_extractor.py --dataset
  uv run python scripts/test_yolo_cell_extractor.py --image test_images/sudoku.png
  uv run python scripts/test_yolo_cell_extractor.py --image test_images/sudoku.png --visualize
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "training/cell_extraction/runs/cell_vision_v6/weights/best.pt"
DATASET_TEST = ROOT / "training/cell_extraction/dataset/test/images"
sys.path.insert(0, str(ROOT / "src"))


# ── helpers ──────────────────────────────────────────────────────────────────

def load_model():
    from ultralytics import YOLO
    if not WEIGHTS.exists():
        sys.exit(f"Weights not found: {WEIGHTS}")
    return YOLO(str(WEIGHTS))


def boxes_to_grid(detections, image_h: int, image_w: int):
    """Map YOLO detections → 9×9 row-major list of (x1,y1,x2,y2,cls).

    Strategy: cluster box y-centres into 9 rows by equal partitioning after
    sorting, then sort each row by x-centre.  Works well when YOLO finds ~81
    boxes; missing cells stay as None.
    """
    if not detections:
        return [None] * 81

    dets = sorted(detections, key=lambda d: (d[1] + d[3]) / 2)  # sort by cy
    n = len(dets)
    rows_raw = []
    for r in range(9):
        lo = int(round(r * n / 9))
        hi = int(round((r + 1) * n / 9))
        row = sorted(dets[lo:hi], key=lambda d: (d[0] + d[2]) / 2)  # sort by cx
        rows_raw.append(row)

    grid = []
    for row in rows_raw:
        # pad/trim to exactly 9
        row = row[:9]
        row += [None] * (9 - len(row))
        grid.extend(row)
    return grid  # 81 entries


def crop(image: np.ndarray, box) -> np.ndarray | None:
    if box is None:
        return None
    x1, y1, x2, y2, _cls = box
    H, W = image.shape[:2]
    px1, py1 = max(0, int(x1)), max(0, int(y1))
    px2, py2 = min(W, int(x2)), min(H, int(y2))
    return image[py1:py2, px1:px2] if px2 > px1 and py2 > py1 else None


# ── mode 1: dataset evaluation ────────────────────────────────────────────────

def run_dataset(model):
    if not DATASET_TEST.exists():
        sys.exit(f"Test images not found: {DATASET_TEST}")

    print(f"\nRunning YOLO on test set: {DATASET_TEST}")
    results = model.predict(
        source=str(DATASET_TEST),
        conf=0.3,
        iou=0.5,
        verbose=False,
    )

    print(f"\n{'Image':<35} {'Dets':>5} {'Empty':>6} {'Filled':>7}")
    print("─" * 56)
    total_dets, total_empty, total_filled = 0, 0, 0
    for r in results:
        name = Path(r.path).name
        dets = len(r.boxes)
        empty  = int((r.boxes.cls == 0).sum())
        filled = int((r.boxes.cls == 1).sum())
        total_dets   += dets
        total_empty  += empty
        total_filled += filled
        flag = "" if dets == 81 else f"  ← expected 81, got {dets}"
        print(f"  {name:<33} {dets:>5} {empty:>6} {filled:>7}{flag}")

    n = len(results)
    print("─" * 56)
    print(f"  {'Average':<33} {total_dets/n:>5.1f} {total_empty/n:>6.1f} {total_filled/n:>7.1f}")

    # formal val metrics
    print("\nRunning formal validation metrics…")
    metrics = model.val(
        data=str(ROOT / "training/cell_extraction/dataset/data.yaml"),
        split="test",
        verbose=False,
    )
    print(f"\n  mAP@50:      {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:   {metrics.box.mp:.4f}")
    print(f"  Recall:      {metrics.box.mr:.4f}")


# ── mode 2: single image → solve ─────────────────────────────────────────────

def run_image(model, image_path: str, visualize: bool):
    from sudoku_solver.digit_classifier import DigitClassifier
    from sudoku_solver.sudoku_solver import SudokuSolver
    from sudoku_solver.pipeline import SudokuPipeline
    from sudoku_solver.config import DigitClassifierConfig

    img_path = Path(image_path)
    if not img_path.exists():
        sys.exit(f"Image not found: {img_path}")

    raw = cv2.imread(str(img_path))
    if raw is None:
        sys.exit(f"Cannot read image: {img_path}")
    image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]

    # 1. Detect cells
    print(f"\nDetecting cells in {img_path.name}…")
    result = model.predict(str(img_path), conf=0.3, iou=0.5, verbose=False)[0]
    dets = []
    for box, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
        dets.append((*box, int(cls)))  # (x1, y1, x2, y2, cls)
    print(f"  Found {len(dets)} cells  (empty={sum(1 for d in dets if d[4]==0)}, filled={sum(1 for d in dets if d[4]==1)})")

    # 2. Sort into 9×9 grid
    grid_boxes = boxes_to_grid(dets, H, W)

    # 3. Load digit classifier
    cfg = DigitClassifierConfig(model_path=ROOT / "models/weights/xgboost_digit_classifier.model")
    classifier = DigitClassifier(cfg)

    # 4. Classify each filled cell
    puzzle = np.zeros((9, 9), dtype=np.int32)
    filled_crops = []
    filled_positions = []

    for idx, box in enumerate(grid_boxes):
        row, col = divmod(idx, 9)
        if box is None:
            continue
        x1, y1, x2, y2, cls = box
        if cls == 0:  # empty — skip digit classification
            continue
        cell_crop = crop(image, box)
        if cell_crop is None or cell_crop.size == 0:
            continue
        # Preprocess crop to match the format classify() expects:
        # white digit on black 28×28 (same as pipeline's CNN path does)
        preprocessed = SudokuPipeline._preprocess_yolo_crop(cell_crop)
        filled_crops.append(preprocessed)
        filled_positions.append((row, col))

    if filled_crops:
        # classify_grid expects exactly 81 cells; we only have the filled ones,
        # so call classify() per cell instead.
        for (row, col), cell_crop in zip(filled_positions, filled_crops):
            digit, _conf, _empty = classifier.classify(cell_crop)
            puzzle[row, col] = digit if digit != 0 else 0

    print("\nDetected puzzle:")
    SudokuSolver.print_grid(puzzle)

    # 5. Solve
    solver = SudokuSolver()
    try:
        solution, t = solver.solve(puzzle)
        print(f"\nSolved in {t*1000:.1f} ms:")
        SudokuSolver.print_grid(solution)
    except RuntimeError as e:
        print(f"\nCould not solve: {e}")
        print("The grid detection or digit reading may need refinement.")
        solution = None

    # 6. Visualize
    if visualize:
        vis = raw.copy()
        colors = {0: (100, 180, 255), 1: (80, 220, 130)}  # blue=empty, green=filled
        for box in dets:
            x1, y1, x2, y2, cls = box
            color = colors[cls]
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        out_path = img_path.parent / f"{img_path.stem}_yolo_cells.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"\nVisualization saved: {out_path}")
        print("  Blue = empty cell, Green = filled cell")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", action="store_true", help="Evaluate on the held-out test split")
    group.add_argument("--image", metavar="PATH", help="Run full pipeline on a single image")
    ap.add_argument("--visualize", action="store_true", help="Save annotated image (with --image)")
    args = ap.parse_args()

    model = load_model()

    if args.dataset:
        run_dataset(model)
    else:
        run_image(model, args.image, args.visualize)


if __name__ == "__main__":
    main()
