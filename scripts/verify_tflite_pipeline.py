"""Compare the whole pipeline running on TFLite models vs the PyTorch originals.

Box-level IoU is the wrong question for this pipeline.  The cell detector is
followed by a lattice fit that re-derives every slot geometrically and
synthesises any cell the detector missed, so a differing box count is routinely
absorbed with no effect on the reading.  What matters is whether the same 81
digits come out.

Holds GridOCR in PyTorch for both runs, so any difference is attributable to the
detectors rather than to the reader (which was verified separately at
3483/3483 patches identical).

Usage:
    uv run python scripts/verify_tflite_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

ASSETS = ROOT / "android/app/src/main/assets"

from sudoku_solver.config import (
    GridOCRConfig,
    YoloCellExtractorConfig,
    YoloGridDetectorConfig,
)
from sudoku_solver.grid_ocr import GridOCR
from sudoku_solver.yolo_cell_extractor import YoloCellExtractor
from sudoku_solver.yolo_grid_detector import YoloGridDetector


def build(use_tflite: bool):
    gd = YoloGridDetectorConfig(mode="seg")
    ce = YoloCellExtractorConfig()
    if use_tflite:
        gd.model_path = ASSETS / "grid_seg.tflite"
        ce.model_path = ASSETS / "cell_vision.tflite"
    return YoloGridDetector(gd), YoloCellExtractor(ce)


def read_grid(detector, extractor, ocr, image: np.ndarray):
    rect = detector.detect(image)
    crops, boxes, labels = extractor.extract(rect)
    PS = ocr.cfg.patch_size
    G = PS * 9
    scaled = cv2.resize(rect, (G, G), interpolation=cv2.INTER_AREA)
    sx, sy = G / rect.shape[1], G / rect.shape[0]
    cells = []
    for i in range(81):
        x1, y1, x2, y2 = boxes[i]
        if x2 <= x1 or y2 <= y1:
            x0, y0 = (i % 9) * PS, (i // 9) * PS
        else:
            cx, cy = (x1 + x2) / 2 * sx, (y1 + y2) / 2 * sy
            x0 = max(0, min(G - PS, int(round(cx - PS / 2))))
            y0 = max(0, min(G - PS, int(round(cy - PS / 2))))
        cells.append(scaled[y0:y0 + PS, x0:x0 + PS])
    grid, _ = ocr.read_cells(cells, contrast_ref=scaled)
    return grid, rect


def main() -> None:
    imgs = sorted(ROOT.glob("test_images/*.png")) + sorted(ROOT.glob("test_images/*.jpg"))
    imgs += sorted(ROOT.glob("data/wicht_sudoku/v2_test/*.jpg"))[:20]
    imgs += sorted(ROOT.glob("data/wicht_sudoku/half_mixed_test/*.jpg"))[:20]

    ocr = GridOCR(GridOCRConfig())
    t_det, t_ext = build(False)
    l_det, l_ext = build(True)

    def read_dat(path):
        rows = [ln for ln in path.read_text().splitlines() if ln.strip()][-9:]
        return np.array([[int(v) for v in r.split()] for r in rows], int)

    n = identical = 0
    cell_same = cell_total = 0
    # Where ground truth exists, the question is not whether the two runtimes
    # agree but whether either is more often right.
    truth_cells = torch_right = lite_right = 0
    warps = []
    per_image_diffs = []

    for path in imgs:
        raw = cv2.imread(str(path))
        if raw is None:
            continue
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        try:
            a, ra = read_grid(t_det, t_ext, ocr, rgb)
            b, rb = read_grid(l_det, l_ext, ocr, rgb)
        except Exception:
            continue
        n += 1
        same = int((a == b).sum())
        cell_same += same
        cell_total += 81
        if same == 81:
            identical += 1
        else:
            per_image_diffs.append((path.name, 81 - same))
        warps.append(float(np.abs(ra.astype(int) - rb.astype(int)).mean()))

        dat = path.with_suffix(".dat")
        if dat.exists():
            truth = read_dat(dat)
            truth_cells += 81
            torch_right += int((a == truth).sum())
            lite_right += int((b == truth).sum())

    print(f"Images compared        : {n}")
    print(f"  identical 81-digit reads : {identical}/{n}  ({100 * identical / max(n,1):.1f} %)")
    print(f"  cells agreeing           : {cell_same}/{cell_total}  "
          f"({100 * cell_same / max(cell_total,1):.2f} %)")
    print(f"  mean |rectified pixel diff| : {np.mean(warps):.2f}")
    if truth_cells:
        print(f"\nAgainst ground truth   : {truth_cells} labelled cells")
        print(f"  PyTorch correct        : {torch_right}/{truth_cells}  "
              f"({100 * torch_right / truth_cells:.2f} %)")
        print(f"  TFLite  correct        : {lite_right}/{truth_cells}  "
              f"({100 * lite_right / truth_cells:.2f} %)")

    if per_image_diffs:
        print("\n  images differing:")
        for name, d in sorted(per_image_diffs, key=lambda x: -x[1])[:10]:
            print(f"    {name:40s} {d} cells")


if __name__ == "__main__":
    main()
