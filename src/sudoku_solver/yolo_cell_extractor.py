"""YOLO-based cell extractor for sudoku puzzles."""

from __future__ import annotations

import numpy as np

from .config import YoloCellExtractorConfig
from .viz import draw_cell_boxes


class YoloCellExtractor:
    """Locates and classifies (empty/filled) all 81 cells using YOLOv8."""

    def __init__(self, cfg: YoloCellExtractorConfig | None = None):
        from ultralytics import YOLO
        self.cfg = cfg or YoloCellExtractorConfig()
        self.model = YOLO(str(self.cfg.model_path))

    def extract(
        self, image: np.ndarray
    ) -> tuple[list[np.ndarray | None], np.ndarray, np.ndarray]:
        """Run YOLO; return (crops, boxes_px, labels).

        crops   : list[np.ndarray | None], length 81, row-major order
        boxes_px: (81, 4) int32  [x1, y1, x2, y2]
        labels  : (81,) int32    0=empty  1=filled
        """
        H, W = image.shape[:2]
        result = self.model.predict(
            image, conf=self.cfg.conf, iou=self.cfg.iou, verbose=False
        )[0]

        dets = [
            (*box, int(cls))
            for box, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist())
        ]

        grid = self._boxes_to_grid(dets)

        crops: list[np.ndarray | None] = []
        boxes_px = np.zeros((81, 4), dtype=np.int32)
        labels = np.zeros(81, dtype=np.int32)

        for i, box in enumerate(grid):
            if box is None:
                crops.append(None)
                continue
            x1, y1, x2, y2, cls = box
            px1, py1 = max(0, int(x1)), max(0, int(y1))
            px2, py2 = min(W, int(x2)), min(H, int(y2))
            cell = image[py1:py2, px1:px2] if px2 > px1 and py2 > py1 else None
            crops.append(cell)
            boxes_px[i] = [px1, py1, px2, py2]
            labels[i] = int(cls)

        return crops, boxes_px, labels

    @classmethod
    def _boxes_to_grid(cls, dets: list) -> list[tuple | None]:
        """Assign detections to their 9×9 row-major slot; None for missing cells.

        Slots are derived geometrically rather than by splitting the detection
        list into 9 equal-count chunks.  YOLO rarely returns exactly 81 boxes on
        real photos (duplicates and misses are both common), and an equal-count
        split shifts every cell after a miss — scrambling the whole puzzle
        instead of leaving one hole.

        The row and column of a cell are read off an **affine lattice fitted to
        the detections**, not off the bounding box of their centres.  A bounding
        box assumes the grid is perfectly axis-aligned after the warp; any
        residual tilt or page curvature then makes a cell's row depend on its x
        position too.  On a curved photo that pushed the left of row 0 down into
        row 1: four cells collided there and four slots at the top were left
        empty, so those cells never reached OCR at all.

        When two detections still claim one slot, the one sitting closest to the
        lattice point wins — a better test than area, since a duplicate box is
        usually the same size as the real one but less well centred.
        """
        if not dets:
            return [None] * 81

        centres = np.array([[(d[0] + d[2]) / 2, (d[1] + d[3]) / 2] for d in dets],
                           dtype=np.float64)

        # Start from the axis-aligned guess: centres of the outermost cells sit
        # half a cell inside the grid, so their span covers 8 of the 9 pitches.
        lo, hi = centres.min(0), centres.max(0)
        pitch = np.where(hi - lo > 1e-6, (hi - lo) / 8.0, 1.0)
        ij = np.clip(np.round((centres - lo) / pitch), 0, 8)   # [col, row]

        # Refine: fit centre -> (col, row) as an affine map and re-read the
        # indices from it.  Two or three passes are enough to absorb tilt.
        design = np.hstack([centres, np.ones((len(centres), 1))])
        predicted = ij.astype(np.float64)
        if len(dets) >= 20:
            for _ in range(6):
                coef, *_ = np.linalg.lstsq(design, ij, rcond=None)
                predicted = design @ coef
                nxt = np.clip(np.round(predicted), 0, 8)
                if np.array_equal(nxt, ij):
                    break
                ij = nxt

        residual = np.abs(predicted - ij).sum(1)
        grid: list = [None] * 81
        best = np.full(81, np.inf)
        for det, (col, row), res in zip(dets, ij.astype(int), residual):
            idx = row * 9 + col
            if res < best[idx]:
                best[idx] = res
                grid[idx] = det

        cls._fill_missing_from_lattice(grid, dets, ij)
        return grid

    @staticmethod
    def _fill_missing_from_lattice(grid: list, dets: list, ij: np.ndarray) -> None:
        """Place a box on every slot the detector missed, in place.

        YOLO under-detects on a third of real photos — 36 of 90 held-out images
        returned fewer than 81 boxes, one of them short by 44.  A slot with no
        box previously reached the reader as a dead cell -- treated as empty and
        skipped entirely, so those digits were never read.

        Since the cells form a lattice, a missing one can simply be placed: the
        inverse of the (col,row) -> centre fit gives its position, and the median
        detection gives its size.  The cell then goes through OCR like any other
        and the reader decides whether it is empty, rather than the detector
        deciding by omission.  Synthesised cells are marked filled (class 1) for
        exactly that reason.
        """
        missing = [i for i, g in enumerate(grid) if g is None]
        if not missing or len(dets) < 20:
            return

        # Fit (col, row) -> centre, the inverse of the lattice used above.
        boxes = np.array([[d[0], d[1], d[2], d[3]] for d in dets], dtype=np.float64)
        centres = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                            (boxes[:, 1] + boxes[:, 3]) / 2], 1)
        design = np.hstack([ij, np.ones((len(ij), 1))])
        try:
            coef, *_ = np.linalg.lstsq(design, centres, rcond=None)
        except np.linalg.LinAlgError:
            return

        half_w = float(np.median(boxes[:, 2] - boxes[:, 0])) / 2
        half_h = float(np.median(boxes[:, 3] - boxes[:, 1])) / 2
        if not np.isfinite(half_w) or not np.isfinite(half_h):
            return

        for idx in missing:
            row, col = divmod(idx, 9)
            cx, cy = np.array([col, row, 1.0]) @ coef
            if not (np.isfinite(cx) and np.isfinite(cy)):
                continue
            grid[idx] = (cx - half_w, cy - half_h, cx + half_w, cy + half_h, 1)

    @staticmethod
    def seg_overlay(
        image: np.ndarray, boxes_px: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Draw YOLO cell boxes: filled cells colored by 3×3 section, empty cells muted."""
        return draw_cell_boxes(image, boxes_px, labels)
