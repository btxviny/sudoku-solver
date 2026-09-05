"""Extract real, exactly-labelled cell crops in the inference distribution.

Every crop is produced by the same three steps the pipeline runs at inference
time -- YOLO grid detector, YOLO cell detector, canonical 70 px sampling -- so
what the network trains on is what it will be asked to read.  The previous
corpus was cut on a uniform 9x9 split at 50 px and upscaled at load time, which
is a different distribution from the one the model actually meets.

Labels come from the Wicht `.dat` ground truth, so they are exact rather than
OCR-guessed.

Splits used.  Only photographs whose digits are real ink:

    v2_train             160 phone photos of printed newspaper puzzles
    real_mixed_natural     4 photos with genuine handwriting

`mixed` and `half_mixed_train` are deliberately excluded even though they carry
ground truth.  Their handwriting is pasted MNIST glyphs, and `half_mixed_test`
-- the end-to-end benchmark -- is built by pasting from the same MNIST pool.
Training on them scores the model against its own training glyphs.  Held-out
`v2_test` is excluded for the ordinary reason.

Usage:
    uv run python training/cell_ocr/extract_real_cells.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from sudoku_solver.cell_prep import is_low_contrast, prep_patch   # noqa: E402
from sudoku_solver.config import PipelineConfig                    # noqa: E402
from sudoku_solver.pipeline import SudokuPipeline                  # noqa: E402
from sudoku_solver.yolo_cell_extractor import YoloCellExtractor    # noqa: E402
from sudoku_solver.yolo_grid_detector import YoloGridDetector      # noqa: E402

WICHT = PROJECT / "data" / "wicht_sudoku"
OUT_DIR = PROJECT / "data" / "cell_ocr" / "real"
SPLITS = ("v2_train", "real_mixed_natural")
PATCH = 70


def read_dat(path: Path) -> np.ndarray | None:
    rows = [ln for ln in path.read_text().splitlines() if ln.strip()][-9:]
    if len(rows) != 9:
        return None
    try:
        return np.array([[int(v) for v in r.split()] for r in rows], dtype=np.uint8)
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    args = ap.parse_args()

    cfg = PipelineConfig()
    detector = YoloGridDetector(cfg.yolo_grid_detector)
    extractor = YoloCellExtractor(cfg.yolo_cell_extractor)

    saved = failed = 0
    counts = np.zeros(10, dtype=int)

    for split in args.splits:
        images = sorted(p for p in (WICHT / split).glob("*.jpg")
                        if p.with_suffix(".dat").exists())
        for img_path in tqdm(images, desc=split):
            grid = read_dat(img_path.with_suffix(".dat"))
            raw = cv2.imread(str(img_path))
            if grid is None or raw is None:
                failed += 1
                continue
            try:
                rectified = detector.detect(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
                _crops, boxes_px, _labels = extractor.extract(rectified)
                cells, scaled = SudokuPipeline._canonical_cells(rectified, boxes_px, PATCH)
            except Exception:
                failed += 1
                continue

            low = is_low_contrast(scaled)
            for i, cell in enumerate(cells):
                if cell is None or cell.size == 0:
                    continue
                gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY) if cell.ndim == 3 else cell
                if gray.shape != (PATCH, PATCH):
                    gray = cv2.resize(gray, (PATCH, PATCH), interpolation=cv2.INTER_AREA)
                patch = prep_patch(gray.copy(), low)
                label = int(grid[i // 9, i % 9])
                d = args.out / str(label)
                d.mkdir(parents=True, exist_ok=True)
                # PNG, not JPEG: these patches are already the model's exact
                # input, and a second lossy round-trip would train the model on
                # artefacts inference never produces.
                name = hashlib.md5(patch.tobytes()).hexdigest()[:12]
                cv2.imwrite(str(d / f"{name}.png"), patch)
                counts[label] += 1
                saved += 1

    print(f"\nSaved {saved:,} cells to {args.out}  ({failed} images failed)")
    for label in range(10):
        print(f"  {label}: {counts[label]:,}")


if __name__ == "__main__":
    main()
