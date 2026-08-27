"""YOLO classification-based digit reader for sudoku cells.

Expects a YOLO *classification* model (yolo classify train) with 10 classes:
    class 0 = empty cell
    class 1-9 = digit value

Training example:
    yolo classify train \\
        data=training/digit_classification/dataset \\
        model=yolov8n-cls.pt \\
        epochs=50 imgsz=64 \\
        project=training/digit_classification/runs \\
        name=digit_cls
"""

import numpy as np

from .config import YoloDigitClassifierConfig


class YoloDigitClassifier:
    """Wraps a YOLO classification model for per-cell digit reading."""

    def __init__(self, cfg: YoloDigitClassifierConfig | None = None):
        from ultralytics import YOLO
        self.cfg = cfg or YoloDigitClassifierConfig()
        self.model = YOLO(str(self.cfg.model_path))

    def classify(self, crop: np.ndarray) -> int:
        """Return digit 0-9 for a single cell image (0 = empty)."""
        result = self.model.predict(
            crop, imgsz=self.cfg.imgsz, verbose=False
        )[0]
        return int(result.probs.top1)

    def classify_grid(
        self,
        crops: list[np.ndarray | None],
        yolo_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Classify 81 cell crops; return (9, 9) int32 puzzle array.

        If yolo_labels is supplied (0=empty from cell-detection YOLO),
        empty cells skip inference and are forced to 0.
        """
        puzzle = np.zeros((9, 9), dtype=np.int32)
        batch, positions = [], []

        for i, crop in enumerate(crops):
            if yolo_labels is not None and yolo_labels[i] == 0:
                continue
            if crop is None or crop.size == 0:
                continue
            batch.append(crop)
            positions.append(divmod(i, 9))

        if not batch:
            return puzzle

        results = self.model.predict(
            batch, imgsz=self.cfg.imgsz, verbose=False
        )
        for (r, c), res in zip(positions, results):
            digit = int(res.probs.top1)
            puzzle[r, c] = digit

        return puzzle
