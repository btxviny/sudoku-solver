"""Cell-patch preprocessing shared by every digit reader.

Both `GridOCR` (GridOCRNet) and `CellOCR` (CellOCRNet) are trained on patches
produced by exactly these steps, and `CellPreprocessor.kt` is a line-by-line
port of them.  Keeping the code in one place is the only way three
implementations stay in agreement: a patch that differs from the training
distribution by a few pixels does not raise, it just returns confident wrong
digits (measured: cell size drifting from the trained value took digit accuracy
to 20 %).

Change nothing here without re-running `scripts/verify_kotlin_preprocess.py`.
"""
from __future__ import annotations

import cv2
import numpy as np

#: 2nd-to-98th percentile spread below which a grid counts as washed out.
CONTRAST_RANGE = 180
#: A pixel darker than this counts as ink when looking for grid-line rows.
DARK = 100
#: A row more than this fraction dark is a grid line, not a glyph.
GRID_LINE_FRACTION = 0.75
#: A pixel darker than this counts as digit when re-centring.
GLYPH = 150
#: Below this min-max spread a low-contrast patch carries no digit at all.
FLAT_PATCH_SPREAD = 20


def to_gray(img: np.ndarray) -> np.ndarray:
    """Grayscale view of an RGB or already-gray image."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img


def is_low_contrast(reference: np.ndarray) -> bool:
    """Whether `reference` is washed out, judged on percentiles.

    Percentiles rather than raw min/max: a single dark page tab or a specular
    highlight otherwise spans the range and hides the fact that every cell is
    faint.  Judged over the whole grid, so one washed-out cell is never
    normalised into noise on its own -- the gutters and outer frame carry the
    paper tone.
    """
    lo, hi = np.percentile(to_gray(reference), [2, 98])
    return bool((hi - lo) < CONTRAST_RANGE)


def prep_patch(p: np.ndarray, low_contrast: bool) -> np.ndarray:
    """Clean one square grayscale cell patch so it matches training.

    `p` is modified in place as well as returned; pass a copy to keep the
    original.
    """
    PS = p.shape[0]
    if low_contrast:
        # Stretch the tonal range first so grid-line pixels reliably go to
        # near-zero, then fall through to the shared grid-line removal below.
        pf = p.astype(np.float32)
        mn, mx = pf.min(), pf.max()
        if mx - mn > FLAT_PATCH_SPREAD:
            pf = (pf - mn) / (mx - mn) * 255.0
            p = pf.clip(0, 255).astype(np.uint8)
        else:
            return np.full_like(p, 255)   # flat patch -> empty

    # Remove full-width dark rows (horizontal grid lines) and columns
    # (vertical).  Both bleed into neighbouring cells and confuse the reader.
    removed_rows: list[int] = []
    for row_i in range(PS):
        if (p[row_i] < DARK).mean() > GRID_LINE_FRACTION:
            p[row_i] = 255
            removed_rows.append(row_i)
    for col_i in range(PS):
        if (p[:, col_i] < DARK).mean() > GRID_LINE_FRACTION:
            p[:, col_i] = 255

    # Re-centre the digit vertically after border rows are removed: leaving it
    # sitting off-centre is not what the network was trained on.
    if removed_rows:
        digit_rows = [i for i in range(PS) if (p[i] < GLYPH).any()]
        if digit_rows:
            top, bot = digit_rows[0], digit_rows[-1] + 1
            region = p[top:bot, :].copy()
            h = bot - top
            start = (PS - h) // 2
            p = np.full((PS, PS), 255, dtype=np.uint8)
            p[start:start + h, :] = region
    return p
