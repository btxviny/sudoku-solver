"""
Score the pipeline against the Wicht & Hennebert ground truth (.dat files).

debug_pipeline.py reports where the pipeline *breaks*; this reports how often it
is *right*, which needs labels. Any directory of imageX.jpg + imageX.dat works.
If a sibling imageX.json is present (written by make_mixed_sudoku.py) accuracy is
also broken down by printed vs handwritten cells.

Usage:
    uv run python scripts/eval_wicht.py data/wicht_sudoku/half_mixed_test
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sudoku_solver.config import PipelineConfig
from sudoku_solver.yolo_grid_detector import YoloGridDetector
from sudoku_solver.grid_ocr import GridOCR


def read_dat(path: Path) -> np.ndarray:
    rows = [ln for ln in path.read_text().splitlines() if ln.strip()][-9:]
    return np.array([[int(v) for v in r.split()] for r in rows], int)


def run(data_dir: Path, limit: int | None) -> None:
    images = sorted(p for p in data_dir.glob("*.jpg") if p.with_suffix(".dat").exists())
    if limit:
        images = images[:limit]
    if not images:
        raise SystemExit(f"No labelled images (imageX.jpg + imageX.dat) in {data_dir}")

    cfg = PipelineConfig()
    detector = YoloGridDetector(cfg.yolo_grid_detector)
    ocr = GridOCR(cfg.grid_ocr)

    n_detect_fail = 0
    perfect = 0
    cell_hit = cell_total = 0
    split = {"printed": [0, 0], "handwritten": [0, 0], "empty": [0, 0]}

    for path in tqdm(images, desc="Scoring"):
        truth = read_dat(path.with_suffix(".dat"))
        raw = cv2.imread(str(path))
        if raw is None:
            n_detect_fail += 1
            continue
        try:
            rectified = detector.detect(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
            pred, _ = ocr.read_with_probs(rectified)
        except Exception:
            n_detect_fail += 1
            continue

        correct = pred == truth
        cell_hit += int(correct.sum())
        cell_total += 81
        perfect += int(correct.all())

        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        for r in range(9):
            for c in range(9):
                if meta is None:
                    kind = "printed" if truth[r, c] else "empty"
                elif [r, c] in meta["handwritten"]:
                    kind = "handwritten"
                elif [r, c] in meta["printed"]:
                    kind = "printed"
                else:
                    kind = "empty"
                split[kind][0] += int(correct[r, c])
                split[kind][1] += 1

    scored = len(images) - n_detect_fail
    print("\n" + "=" * 56)
    print(f"{data_dir}  ({len(images)} images)")
    print("-" * 56)
    print(f"detection/OCR errors : {n_detect_fail}")
    if scored:
        print(f"grids read perfectly : {perfect}/{scored}  ({perfect / scored * 100:.1f}%)")
        print(f"cell accuracy        : {cell_hit}/{cell_total}  ({cell_hit / cell_total * 100:.2f}%)")
        for kind, (hit, tot) in split.items():
            if tot:
                print(f"  {kind:<18} {hit}/{tot}  ({hit / tot * 100:.2f}%)")
    print("=" * 56)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.data_dir, args.limit)


if __name__ == "__main__":
    main()
