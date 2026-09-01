"""Extract labelled cell crops from the Wicht sudoku dataset.

Each image has a .dat sidecar with the ground-truth 9×9 digit grid.
We run the YOLO grid detector to get a rectified grid, split it into
81 uniform cells, and save each one to data/grid_ocr/cells/<label>/.

Only non-test splits are used so the held-out test images stay clean.

Usage:
    uv run python training/grid_ocr/scripts/extract_wicht_cells.py
"""
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))

from sudoku_solver.config import PipelineConfig
from sudoku_solver.yolo_grid_detector import YoloGridDetector

WICHT_DIR = PROJECT / "data" / "wicht_sudoku"
OUT_DIR = PROJECT / "data" / "grid_ocr" / "cells"
CELL_SIZE = 50

# Exclude held-out test sets
SKIP_SPLITS = {"half_mixed_test", "v2_test"}

TRAIN_SPLITS = [
    d for d in WICHT_DIR.iterdir()
    if d.is_dir() and d.name not in SKIP_SPLITS
]


def read_dat(path: Path) -> np.ndarray | None:
    rows = [ln for ln in path.read_text().splitlines() if ln.strip()][-9:]
    if len(rows) != 9:
        return None
    try:
        return np.array([[int(v) for v in r.split()] for r in rows], dtype=np.uint8)
    except ValueError:
        return None


def save_cell(gray_crop: np.ndarray, label: int) -> None:
    resized = cv2.resize(gray_crop, (CELL_SIZE, CELL_SIZE), interpolation=cv2.INTER_AREA)
    label_dir = OUT_DIR / str(label)
    label_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(resized.tobytes()).hexdigest()[:12]
    out_path = label_dir / f"w{h}.jpg"
    cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 90])


def extract_cells(rectified_rgb: np.ndarray, grid: np.ndarray) -> int:
    gray = cv2.cvtColor(rectified_rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    ch, cw = H // 9, W // 9
    saved = 0
    for r in range(9):
        for c in range(9):
            crop = gray[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            label = int(grid[r, c])
            save_cell(crop, label)
            saved += 1
    return saved


def main() -> None:
    cfg = PipelineConfig()
    detector = YoloGridDetector(cfg.yolo_grid_detector)

    total_saved = 0
    total_failed = 0

    for split_dir in sorted(TRAIN_SPLITS):
        images = sorted(p for p in split_dir.glob("*.jpg")
                        if p.with_suffix(".dat").exists())
        if not images:
            continue

        print(f"\n{split_dir.name}: {len(images)} images")
        for img_path in tqdm(images, desc=split_dir.name):
            grid = read_dat(img_path.with_suffix(".dat"))
            if grid is None:
                total_failed += 1
                continue

            raw = cv2.imread(str(img_path))
            if raw is None:
                total_failed += 1
                continue

            try:
                rectified = detector.detect(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
                total_saved += extract_cells(rectified, grid)
            except Exception:
                total_failed += 1

    print(f"\nDone — saved {total_saved:,} cells, failed {total_failed} images")
    print("Counts per label:")
    for label in range(10):
        n = len(list((OUT_DIR / str(label)).glob("*.jpg"))) if (OUT_DIR / str(label)).exists() else 0
        print(f"  {label}: {n:,}")


if __name__ == "__main__":
    main()
