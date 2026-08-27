"""YOLO-based cell extractor for sudoku puzzles."""

import cv2
import numpy as np

from .config import YoloCellExtractorConfig


class YoloCellExtractor:
    """Locates and classifies (empty/filled) all 81 cells using YOLOv8."""

    def __init__(self, cfg: YoloCellExtractorConfig | None = None):
        from ultralytics import YOLO
        self.cfg = cfg or YoloCellExtractorConfig()
        self.model = YOLO(str(self.cfg.model_path))

    def extract(self, image: np.ndarray):
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

    @staticmethod
    def _boxes_to_grid(dets: list) -> list:
        """Assign detections to their 9×9 row-major slot; None for missing cells.

        Slots are derived geometrically rather than by splitting the detection
        list into 9 equal-count chunks.  YOLO rarely returns exactly 81 boxes on
        real photos (duplicates and misses are both common), and an equal-count
        split shifts every cell after a miss — scrambling the whole puzzle
        instead of leaving one hole.  Here each box is placed by where its
        centre falls in the detected grid's bounding box, so a miss costs
        exactly one cell.  When two boxes claim a slot, the larger one wins.
        """
        if not dets:
            return [None] * 81

        cxs = [(d[0] + d[2]) / 2 for d in dets]
        cys = [(d[1] + d[3]) / 2 for d in dets]
        x_min, x_max = min(cxs), max(cxs)
        y_min, y_max = min(cys), max(cys)

        # Centres of the outermost cells sit half a cell inside the grid, so the
        # centre span covers 8 of the 9 cell widths.
        cell_w = (x_max - x_min) / 8 if x_max > x_min else 1.0
        cell_h = (y_max - y_min) / 8 if y_max > y_min else 1.0

        grid: list = [None] * 81
        for det, cx, cy in zip(dets, cxs, cys):
            c = min(8, max(0, int(round((cx - x_min) / cell_w))))
            r = min(8, max(0, int(round((cy - y_min) / cell_h))))
            idx = r * 9 + c
            prev = grid[idx]
            if prev is None:
                grid[idx] = det
            else:
                area = (det[2] - det[0]) * (det[3] - det[1])
                prev_area = (prev[2] - prev[0]) * (prev[3] - prev[1])
                if area > prev_area:
                    grid[idx] = det
        return grid

    @staticmethod
    def seg_overlay(
        image: np.ndarray, boxes_px: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Draw YOLO cell boxes: filled cells colored by 3×3 section, empty cells muted."""
        vis = image.copy()
        palette = [
            (100, 200, 255), (255, 175, 70),  (140, 230, 140),
            (255, 110, 110), (190, 130, 255), (60,  215, 215),
            (255, 205, 80),  (170, 255, 170), (255, 150, 195),
        ]
        for i, (x1, y1, x2, y2) in enumerate(boxes_px):
            r, c = divmod(i, 9)
            if labels[i] == 1:  # filled
                color = palette[(r // 3) * 3 + (c // 3)]
                thick = 2
            else:              # empty
                color = (160, 160, 160)
                thick = 1
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
        return vis
