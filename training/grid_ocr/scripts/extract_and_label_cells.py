"""Extract labelled cell crops from the two Roboflow datasets.

For each image:
  - Load COCO annotations (81 bboxes per image, empty/filled labels)
  - Crop each cell, resize to CELL_SIZE × CELL_SIZE, convert to grayscale
  - Empty cells  → label 0  (no OCR needed)
  - Filled cells → run EasyOCR; accept if confidence ≥ CONF_THRESH
  - When the puzzle has enough confident digits, solve it with backtracking
    to fill in any remaining uncertain cells
  - Save as data/grid_ocr/cells/<label>/<hash>.jpg   (0-9 subdirectories)

Usage:
    python training/grid_ocr/scripts/extract_and_label_cells.py
"""
import json
import hashlib
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import easyocr
from tqdm import tqdm

# --------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
from sudoku_solver.sudoku_solver import SudokuSolver

ROBOFLOW_DIR = PROJECT / "data" / "roboflow"
OUT_DIR = PROJECT / "data" / "grid_ocr" / "cells"
CELL_SIZE = 50          # resized cell size (px) — matches 450/9 patch size
CONF_THRESH = 0.80      # EasyOCR confidence threshold
# --------------------------------------------------------------------------

DATASETS = [
    {
        "name": "sudoku-cell-detector",
        "splits": ["train", "valid", "test"],
        "filled_cat": 2,
        "empty_cat":  1,
    },
    {
        "name": "sudoku-cell-vision",
        "splits": ["train", "valid", "test"],
        "filled_cat": 2,
        "empty_cat":  1,
    },
]

solver = SudokuSolver()


def preprocess_for_ocr(cell_bgr: np.ndarray, target: int = 120) -> np.ndarray:
    """Upscale and enhance contrast for better OCR accuracy."""
    upscaled = cv2.resize(cell_bgr, (target, target), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


def save_cell(cell_bgr: np.ndarray, label: int, out_dir: Path) -> Path:
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (CELL_SIZE, CELL_SIZE), interpolation=cv2.INTER_AREA)
    label_dir = out_dir / str(label)
    label_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(resized.tobytes()).hexdigest()[:12]
    out_path = label_dir / f"{h}.jpg"
    cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path


def try_solve(grid: np.ndarray) -> np.ndarray | None:
    """Return solved 9×9 grid or None if puzzle is inconsistent/ambiguous."""
    try:
        solved, _ = solver.solve(grid.copy())
        if solver.is_valid(solved) and (solved > 0).all():
            return solved
    except Exception:
        pass
    return None


def process_image(
    img_bgr: np.ndarray,
    anns: list[dict],
    reader: easyocr.Reader,
    filled_cat: int,
    empty_cat: int,
) -> tuple[np.ndarray | None, list[tuple[np.ndarray, int]]]:
    """
    Returns:
        label_grid  – 9×9 uint8 (0=empty, 1-9=digit) or None if unreliable
        cell_pairs  – list of (cell_bgr_crop, digit_label)
    """
    H, W = img_bgr.shape[:2]
    margin = 2

    # Sort annotations top-left → bottom-right to assign (row, col) indices
    anns_sorted = sorted(anns, key=lambda a: (round(a["bbox"][1] / 10), round(a["bbox"][0] / 10)))

    # Group into 9 rows
    rows: list[list[dict]] = []
    if not anns_sorted:
        return None, []

    row_thresh = (anns_sorted[-1]["bbox"][1] - anns_sorted[0]["bbox"][1]) / 9 * 0.6
    current_row: list[dict] = [anns_sorted[0]]
    for a in anns_sorted[1:]:
        if abs(a["bbox"][1] - current_row[-1]["bbox"][1]) < row_thresh:
            current_row.append(a)
        else:
            rows.append(sorted(current_row, key=lambda x: x["bbox"][0]))
            current_row = [a]
    rows.append(sorted(current_row, key=lambda x: x["bbox"][0]))

    if len(rows) != 9 or any(len(r) != 9 for r in rows):
        return None, []

    grid_confident = np.zeros((9, 9), dtype=np.uint8)
    cell_data: list[tuple[int, int, np.ndarray, int | None, float]] = []  # (r,c,crop,label,conf)

    for r, row in enumerate(rows):
        for c, a in enumerate(row):
            x, y, w, h = a["bbox"]
            x1 = max(0, int(x) - margin)
            y1 = max(0, int(y) - margin)
            x2 = min(W, int(x + w) + margin)
            y2 = min(H, int(y + h) + margin)
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if a["category_id"] == empty_cat:
                cell_data.append((r, c, crop, 0, 1.0))
                grid_confident[r, c] = 0
            else:
                # Filled — OCR
                ocr_img = preprocess_for_ocr(crop)
                results = reader.readtext(ocr_img, allowlist="123456789", detail=1)
                if results:
                    txt, conf = results[0][1].strip(), float(results[0][2])
                    if txt and txt.isdigit() and 1 <= int(txt) <= 9 and conf >= CONF_THRESH:
                        d = int(txt)
                        cell_data.append((r, c, crop, d, conf))
                        grid_confident[r, c] = d
                    else:
                        cell_data.append((r, c, crop, None, conf))
                else:
                    cell_data.append((r, c, crop, None, 0.0))

    # Count uncertain filled cells
    uncertain = [(r, c, crop) for r, c, crop, lbl, _ in cell_data if lbl is None]

    if not uncertain:
        label_grid = grid_confident
    elif len(uncertain) <= 8:
        # Try to solve puzzle to recover uncertain digits
        solved = try_solve(grid_confident.copy())
        if solved is not None:
            label_grid = solved
            # Patch uncertain digits from solved grid
            new_cell_data = []
            for r, c, crop, lbl, conf in cell_data:
                if lbl is None:
                    lbl = int(solved[r, c])
                    conf = 1.0  # trust solver
                new_cell_data.append((r, c, crop, lbl, conf))
            cell_data = new_cell_data
        else:
            return None, []  # Skip image if puzzle can't be solved
    else:
        return None, []  # Too many uncertain cells

    pairs = [(crop, lbl) for _, _, crop, lbl, _ in cell_data if lbl is not None]
    return label_grid, pairs


def main() -> None:
    print("Loading EasyOCR (first run downloads ~100 MB models)…")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    print("Ready.\n")

    total_saved = [0] * 10
    skipped_images = 0

    for ds in DATASETS:
        ds_dir = ROBOFLOW_DIR / ds["name"]
        print(f"=== {ds['name']} ===")

        for split in ds["splits"]:
            ann_path = ds_dir / split / "_annotations.coco.json"
            if not ann_path.exists():
                continue

            with open(ann_path) as f:
                coco = json.load(f)

            img_dir = ds_dir / split
            by_image: dict[int, list] = {}
            for a in coco["annotations"]:
                by_image.setdefault(a["image_id"], []).append(a)

            for img_meta in tqdm(coco["images"], desc=f"{ds['name']}/{split}"):
                img_path = img_dir / img_meta["file_name"]
                if not img_path.exists():
                    continue
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue

                anns = by_image.get(img_meta["id"], [])
                if len(anns) != 81:
                    skipped_images += 1
                    continue

                label_grid, pairs = process_image(
                    img_bgr, anns, reader, ds["filled_cat"], ds["empty_cat"]
                )
                if label_grid is None:
                    skipped_images += 1
                    continue

                for crop, label in pairs:
                    save_cell(crop, label, OUT_DIR)
                    total_saved[label] += 1

    print("\n=== Done ===")
    print(f"Skipped images: {skipped_images}")
    print("Cell counts per digit:")
    for d, n in enumerate(total_saved):
        print(f"  {d}: {n:,}")
    print(f"  Total: {sum(total_saved):,}")


if __name__ == "__main__":
    main()
