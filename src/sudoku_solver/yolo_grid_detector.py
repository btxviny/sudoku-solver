"""YOLO-based grid detector -- the mobile-friendly replacement for Mask R-CNN.

Same interface as `GridDetector` (`detect`, `detect_debug`), so it drops into
the pipeline in place of the 169 MB Mask R-CNN.  Two backends:

    "pose"  YOLOv8n-pose regresses the four grid corners directly.
    "seg"   YOLOv8n-seg predicts a grid mask, and corners are recovered from it
            exactly as the Mask R-CNN path does.

Either way the quad is then handed to `GridDetector`'s existing Hough
refinement and perspective warp, which are pure OpenCV and backend-agnostic --
so this class changes only *how the grid is located*, never how it is
rectified.  That keeps the comparison against Mask R-CNN honest: any accuracy
difference comes from the detector, not from different post-processing.

The Mask R-CNN detector is left untouched and remains the default.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import YoloGridDetectorConfig
from .grid_detector import GridDetector


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL -- matches the pose training labels."""
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0).astype(np.float32)


class YoloGridDetector:
    """Locate a sudoku grid with YOLO and return the rectified 450x450 crop."""

    def __init__(self, config: YoloGridDetectorConfig | None = None, device: str | None = None):
        from ultralytics import YOLO

        self.cfg = config or YoloGridDetectorConfig()
        if self.cfg.mode not in ("pose", "seg"):
            raise ValueError(f"mode must be 'pose' or 'seg', got {self.cfg.mode!r}")
        if not self.cfg.model_path.exists():
            raise FileNotFoundError(f"YOLO grid weights not found: {self.cfg.model_path}")
        self.model = YOLO(str(self.cfg.model_path))
        self.device = device

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _predict(self, resized: np.ndarray):
        kw = dict(conf=self.cfg.conf, imgsz=self.cfg.imgsz, verbose=False)
        if self.device is not None:
            kw["device"] = self.device
        return self.model.predict(resized, **kw)[0]

    def _corners_pose(self, result, shape: tuple[int, int]) -> np.ndarray:
        kp = result.keypoints
        if kp is None or kp.xy is None or len(kp.xy) == 0:
            raise RuntimeError("No sudoku grid detected (pose head returned no keypoints).")
        # Highest-confidence detection wins; predictions are already sorted by
        # score, but selecting explicitly keeps this independent of that.
        idx = int(np.argmax(result.boxes.conf.cpu().numpy())) if len(result.boxes) else 0
        quad = kp.xy[idx].cpu().numpy().astype(np.float32)
        if quad.shape != (4, 2):
            raise RuntimeError(f"Expected 4 corner keypoints, got {quad.shape[0]}.")
        h, w = shape
        quad[:, 0] = quad[:, 0].clip(0, w - 1)
        quad[:, 1] = quad[:, 1].clip(0, h - 1)
        return order_corners(quad)

    def _mask_seg(self, result, shape: tuple[int, int]) -> np.ndarray:
        masks = result.masks
        if masks is None or len(masks.data) == 0:
            raise RuntimeError("No sudoku grid detected (seg head returned no masks).")
        h, w = shape
        combined = np.zeros((h, w), dtype=np.uint8)
        for m in masks.data.cpu().numpy():
            resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            combined = np.maximum(combined, (resized > 0.5).astype(np.uint8))
        return combined * 255

    @staticmethod
    def _mask_from_quad(quad: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Filled quad as a mask, so the Hough refinement can be reused as-is."""
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
        return mask

    def _run(self, image: np.ndarray):
        """Return (rectified, resized, mask, corners) -- mirrors GridDetector._run."""
        resized = cv2.resize(image, self.cfg.resize_to)
        shape = resized.shape[:2]
        result = self._predict(resized)

        if self.cfg.mode == "pose":
            corners = self._corners_pose(result, shape)
            mask = self._mask_from_quad(corners, shape)
        else:
            mask = self._mask_seg(result, shape)
            corners = GridDetector._corners_from_mask(mask)
            if corners is None:
                contour = GridDetector._mask_contour(mask)
                if contour is None:
                    raise RuntimeError("No contour found from detection mask.")
                corners = GridDetector._find_quad(contour)
            # Both mask-derived quads come out in whatever rotation the contour
            # tracing happened to start at.  `_perspective_warp` re-sorts
            # internally so rectification never noticed, but `corners()` is a
            # public accessor and must agree with the pose path's TL/TR/BR/BL.
            corners = order_corners(corners)

        if self.cfg.refine:
            corners = GridDetector._refine_on_grid_lines(resized, mask, corners)

        rectified = GridDetector._perspective_warp(
            resized, corners, size=self.cfg.output_size
        )
        return rectified, resized, mask, corners

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect the grid and return the perspective-corrected crop (RGB uint8)."""
        return self._run(image)[0]

    def detect_debug(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Like `detect`, plus the mask + corner overlay used by the app."""
        rectified, resized, mask, corners = self._run(image)
        return rectified, GridDetector._seg_overlay(resized, mask, corners)

    def corners(self, image: np.ndarray) -> np.ndarray:
        """The four TL/TR/BR/BL corners, in `resize_to` pixel coordinates."""
        return self._run(image)[3]
