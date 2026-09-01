"""Cross-check GridGeometry.kt and CellLattice.kt against their Python originals.

Transliterates the Kotlin -- including its hand-rolled normal-equations solver,
which replaces NumPy's SVD-based lstsq -- and runs it on real YOLO detections
from the test images, plus randomised quads for the corner ordering.

The lattice fit is the part worth checking: the Kotlin cannot call lstsq, so it
solves the 3x3 normal equations instead. That is a genuinely different numerical
route to the same answer, and this is where it gets confirmed.

Usage:
    uv run python scripts/verify_kotlin_geometry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sudoku_solver.yolo_cell_extractor import YoloCellExtractor
from sudoku_solver.yolo_grid_detector import YoloGridDetector, order_corners


# ── transliteration of GridGeometry.kt ───────────────────────────────────────

def kt_order_corners(pts: np.ndarray) -> np.ndarray:
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    srt = sorted(pts.tolist(), key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
    start, best = 0, float("inf")
    for i, p in enumerate(srt):
        if p[0] + p[1] < best:
            best, start = p[0] + p[1], i
    return np.array([srt[(start + i) % 4] for i in range(4)], dtype=np.float32)


# ── transliteration of CellLattice.kt ────────────────────────────────────────

def kt_solve(a, b):
    n, m = len(a), len(b[0])
    lhs = [list(r) for r in a]
    rhs = [list(r) for r in b]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(lhs[r][col]))
        if abs(lhs[piv][col]) < 1e-12:
            return None
        lhs[col], lhs[piv] = lhs[piv], lhs[col]
        rhs[col], rhs[piv] = rhs[piv], rhs[col]
        d = lhs[col][col]
        lhs[col] = [v / d for v in lhs[col]]
        rhs[col] = [v / d for v in rhs[col]]
        for r in range(n):
            if r == col:
                continue
            f = lhs[r][col]
            if f == 0.0:
                continue
            lhs[r] = [lhs[r][j] - f * lhs[col][j] for j in range(n)]
            rhs[r] = [rhs[r][j] - f * rhs[col][j] for j in range(m)]
    return rhs


def kt_least_squares(design, target):
    cols, outs = len(design[0]), len(target[0])
    ata = [[0.0] * cols for _ in range(cols)]
    atb = [[0.0] * outs for _ in range(cols)]
    for r in range(len(design)):
        for i in range(cols):
            for j in range(cols):
                ata[i][j] += design[r][i] * design[r][j]
            for j in range(outs):
                atb[i][j] += design[r][i] * target[r][j]
    return kt_solve(ata, atb)


def _round_half_away(v: float) -> float:
    """Kotlin's roundToInt rounds .5 away from zero; numpy rounds to even."""
    return float(np.floor(v + 0.5)) if v >= 0 else float(np.ceil(v - 0.5))


def kt_boxes_to_grid(dets: list) -> list:
    grid = [None] * 81
    if not dets:
        return grid
    cxs = [(d[0] + d[2]) / 2.0 for d in dets]
    cys = [(d[1] + d[3]) / 2.0 for d in dets]
    lo_x, hi_x, lo_y, hi_y = min(cxs), max(cxs), min(cys), max(cys)
    px = (hi_x - lo_x) / 8.0 if hi_x - lo_x > 1e-6 else 1.0
    py = (hi_y - lo_y) / 8.0 if hi_y - lo_y > 1e-6 else 1.0
    clamp = lambda v: min(8.0, max(0.0, v))
    ij = [[clamp(_round_half_away((cxs[k] - lo_x) / px)),
           clamp(_round_half_away((cys[k] - lo_y) / py))] for k in range(len(dets))]

    design = [[cxs[k], cys[k], 1.0] for k in range(len(dets))]
    predicted = [list(r) for r in ij]

    if len(dets) >= 20:
        for _ in range(6):
            coef = kt_least_squares(design, ij)
            if coef is None:
                break
            predicted = [[sum(design[i][k] * coef[k][j] for k in range(3))
                          for j in range(2)] for i in range(len(design))]
            changed = False
            for k in range(len(ij)):
                c = clamp(_round_half_away(predicted[k][0]))
                r = clamp(_round_half_away(predicted[k][1]))
                if c != ij[k][0] or r != ij[k][1]:
                    changed = True
                ij[k][0], ij[k][1] = c, r
            if not changed:
                break

    best = [float("inf")] * 81
    for k, d in enumerate(dets):
        col, row = int(ij[k][0]), int(ij[k][1])
        idx = row * 9 + col
        res = abs(predicted[k][0] - ij[k][0]) + abs(predicted[k][1] - ij[k][1])
        if res < best[idx]:
            best[idx] = res
            grid[idx] = d

    kt_fill_missing(grid, dets, ij)
    return grid


def kt_fill_missing(grid, dets, ij) -> None:
    missing = [i for i, g in enumerate(grid) if g is None]
    if not missing or len(dets) < 20:
        return
    design = [[ij[k][0], ij[k][1], 1.0] for k in range(len(ij))]
    centres = [[(d[0] + d[2]) / 2.0, (d[1] + d[3]) / 2.0] for d in dets]
    coef = kt_least_squares(design, centres)
    if coef is None:
        return
    widths = sorted(d[2] - d[0] for d in dets)
    heights = sorted(d[3] - d[1] for d in dets)
    med = lambda s: s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2
    hw, hh = med(widths) / 2.0, med(heights) / 2.0
    for idx in missing:
        row, col = divmod(idx, 9)
        cx = col * coef[0][0] + row * coef[1][0] + coef[2][0]
        cy = col * coef[0][1] + row * coef[1][1] + coef[2][1]
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        grid[idx] = (cx - hw, cy - hh, cx + hw, cy + hh, 1)


# ── checks ───────────────────────────────────────────────────────────────────

def check_corners() -> bool:
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(4000):
        quad = rng.uniform(0, 1000, size=(4, 2))
        # Reject near-degenerate quads: ordering is genuinely ambiguous there.
        if np.linalg.norm(quad[:, None] - quad[None], axis=-1).max() < 50:
            continue
        a = order_corners(quad.astype(np.float32))
        b = kt_order_corners(quad)
        if not np.allclose(a, b, atol=1e-4):
            bad += 1
    print(f"  orderCorners over 4000 random quads: {'match' if not bad else f'{bad} MISMATCH'}")
    return bad == 0


def check_lattice() -> bool:
    imgs = sorted(ROOT.glob("test_images/*.png")) + sorted(ROOT.glob("test_images/*.jpg"))
    imgs += sorted(ROOT.glob("data/wicht_sudoku/v2_test/*.jpg"))[:20]
    extractor = YoloCellExtractor()
    detector = YoloGridDetector()

    n_img = slots_same = slots_total = 0
    max_coord_diff = 0.0
    bad_imgs = 0

    for path in imgs:
        raw = cv2.imread(str(path))
        if raw is None:
            continue
        try:
            rect = detector.detect(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
            result = extractor.model.predict(rect, conf=extractor.cfg.conf,
                                             iou=extractor.cfg.iou, verbose=False)[0]
        except Exception:
            continue
        dets = [(*box, int(c)) for box, c in
                zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist())]
        if not dets:
            continue
        n_img += 1

        ref = YoloCellExtractor._boxes_to_grid(dets)
        mine = kt_boxes_to_grid(dets)

        img_bad = False
        for i in range(81):
            slots_total += 1
            a, b = ref[i], mine[i]
            if (a is None) != (b is None):
                img_bad = True
                continue
            if a is None:
                slots_same += 1
                continue
            d = max(abs(float(a[k]) - float(b[k])) for k in range(4))
            max_coord_diff = max(max_coord_diff, d)
            if d < 1e-6 and int(a[4]) == int(b[4]):
                slots_same += 1
            else:
                img_bad = True
        if img_bad:
            bad_imgs += 1

    print(f"  boxesToGrid over {n_img} real detections:")
    print(f"    slots identical : {slots_same}/{slots_total}")
    print(f"    images differing: {bad_imgs}")
    print(f"    max coord diff  : {max_coord_diff:.2e}")
    return slots_same == slots_total


def main() -> None:
    print("Corner ordering")
    ok = check_corners()
    print("\nLattice assignment")
    ok = check_lattice() and ok
    print("\n" + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
