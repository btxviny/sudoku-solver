"""Cross-check CellPreprocessor.kt against the Python preprocessing it ports.

The Kotlin cannot be compiled here, so this transliterates it -- byte-array row
scans, integer histogram percentiles, the same constants -- and runs it against
`GridOCR._prep_patch` on real cell crops cut from the test images.

This is the port I most wanted checked.  GridOCRNet was trained on exactly these
transforms, so a silent difference here would not crash anything; it would just
quietly cost digits.

Usage:
    uv run python scripts/verify_kotlin_preprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sudoku_solver.grid_ocr import GridOCR

PATCH = 50
DARK, GLYPH = 100, 150
GRID_LINE_FRACTION = 0.75
FLAT_PATCH_SPREAD = 20
CONTRAST_RANGE = 180


# ── transliteration of CellPreprocessor.kt ───────────────────────────────────

def kt_percentiles(gray: np.ndarray, lo_pct=2.0, hi_pct=98.0) -> tuple[int, int]:
    hist = np.bincount(gray.ravel(), minlength=256)
    total = gray.size
    lo_t, hi_t = total * lo_pct / 100.0, total * hi_pct / 100.0
    cum = 0
    lo, hi, lo_set = 0, 255, False
    for v in range(256):
        cum += int(hist[v])
        if not lo_set and cum >= lo_t:
            lo, lo_set = v, True
        if cum >= hi_t:
            hi = v
            break
    return lo, hi


def kt_is_low_contrast(ref: np.ndarray) -> bool:
    g = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY) if ref.ndim == 3 else ref
    lo, hi = kt_percentiles(g)
    return (hi - lo) < CONTRAST_RANGE


def kt_stretch(px: bytearray) -> None:
    mn, mx = 255, 0
    for v in px:
        mn = min(mn, v); mx = max(mx, v)
    if mx - mn > FLAT_PATCH_SPREAD:
        span = float(mx - mn)
        for i in range(len(px)):
            s = (px[i] - mn) / span * 255.0
            px[i] = int(min(255.0, max(0.0, s)))
    else:
        for i in range(len(px)):
            px[i] = 255


def kt_remove_grid_lines(px: bytearray) -> None:
    removed_any = False
    for row in range(PATCH):
        base = row * PATCH
        dark = sum(1 for c in range(PATCH) if px[base + c] < DARK)
        if dark / PATCH > GRID_LINE_FRACTION:
            for c in range(PATCH):
                px[base + c] = 255
            removed_any = True
    if not removed_any:
        return

    top, bottom = -1, -1
    for row in range(PATCH):
        base = row * PATCH
        if any(px[base + c] < GLYPH for c in range(PATCH)):
            if top < 0:
                top = row
            bottom = row
    if top < 0:
        return

    height = bottom - top + 1
    region = bytes(px[top * PATCH:(bottom + 1) * PATCH])
    for i in range(len(px)):
        px[i] = 255
    start = (PATCH - height) // 2
    px[start * PATCH:start * PATCH + len(region)] = region


def kt_prep_patch(patch: np.ndarray, low_contrast: bool) -> np.ndarray:
    px = bytearray(patch.reshape(-1).tolist())
    if low_contrast:
        kt_stretch(px)
    else:
        kt_remove_grid_lines(px)
    return np.frombuffer(bytes(px), dtype=np.uint8).reshape(PATCH, PATCH)


# ── checks ───────────────────────────────────────────────────────────────────

def main() -> None:
    imgs = sorted(ROOT.glob("test_images/*.png")) + sorted(ROOT.glob("test_images/*.jpg"))
    imgs += sorted(ROOT.glob("data/wicht_sudoku/v2_test/*.jpg"))[:25]
    imgs += sorted(ROOT.glob("data/wicht_sudoku/half_mixed_test/*.jpg"))[:25]
    if not imgs:
        raise SystemExit("No test images found")

    n_patch = 0
    mismatched = 0
    max_abs = 0
    contrast_checked = contrast_mismatch = 0

    for path in imgs:
        raw = cv2.imread(str(path))
        if raw is None:
            continue
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        # Uniform 9x9 split of the whole image: the point is to exercise the
        # patch transforms over a wide variety of real content, not to detect.
        scaled = cv2.resize(rgb, (PATCH * 9, PATCH * 9), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(scaled, cv2.COLOR_RGB2GRAY)

        py_low = bool((np.percentile(gray, 98) - np.percentile(gray, 2)) < CONTRAST_RANGE)
        kt_low = kt_is_low_contrast(scaled)
        contrast_checked += 1
        if py_low != kt_low:
            contrast_mismatch += 1
            print(f"  contrast verdict differs on {path.name}: py={py_low} kt={kt_low}")

        for r in range(9):
            for c in range(9):
                cell = gray[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
                ref = GridOCR._prep_patch(cell.copy(), py_low)
                mine = kt_prep_patch(cell.copy(), py_low)
                n_patch += 1
                if not np.array_equal(ref, mine):
                    mismatched += 1
                    max_abs = max(max_abs, int(np.abs(ref.astype(int) - mine.astype(int)).max()))

    print(f"\nPatches compared      : {n_patch}")
    print(f"  byte-identical      : {n_patch - mismatched}")
    print(f"  differing           : {mismatched}")
    if mismatched:
        print(f"  max abs difference  : {max_abs}")
    print(f"Contrast verdicts     : {contrast_checked - contrast_mismatch}/{contrast_checked} agree")

    ok = mismatched == 0 and contrast_mismatch == 0
    print("\n" + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
