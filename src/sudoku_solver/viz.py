"""Shared overlay colours and drawing helpers."""

from __future__ import annotations

import cv2
import numpy as np

# One colour per 3×3 box section.
CELL_PALETTE: tuple[tuple[int, int, int], ...] = (
    (100, 200, 255),
    (255, 175, 70),
    (140, 230, 140),
    (255, 110, 110),
    (190, 130, 255),
    (60, 215, 215),
    (255, 205, 80),
    (170, 255, 170),
    (255, 150, 195),
)


def box_color(row: int, col: int) -> tuple[int, int, int]:
    """RGB colour for the 3×3 block containing (row, col)."""
    return CELL_PALETTE[(row // 3) * 3 + (col // 3)]


def draw_cell_boxes(
    image: np.ndarray,
    boxes_px: np.ndarray,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Draw 81 cell boxes. Filled cells use the 3×3 palette; empty cells are gray."""
    vis = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes_px):
        r, c = divmod(i, 9)
        filled = labels is None or int(labels[i]) == 1
        if filled:
            color, thick = box_color(r, c), 2
        else:
            color, thick = (160, 160, 160), 1
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, thick, cv2.LINE_AA)
    return vis
