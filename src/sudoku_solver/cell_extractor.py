import cv2
import numpy as np
from .config import CellExtractorConfig


class CellExtractor:
    """Extract and preprocess individual cells from a rectified 9×9 sudoku grid."""

    def __init__(self, config: CellExtractorConfig | None = None):
        self.cfg = config or CellExtractorConfig()

    def extract(self, image: np.ndarray, raw: bool = False) -> list[np.ndarray]:
        """Extract 81 cells from a rectified grid image.

        Args:
            image: Rectified grid image (RGB, after homography).
            raw: If True, skip preprocessing and return raw crops.

        Returns:
            List of 81 cell images in row-major order.
        """
        cells = self._split_grid(image)
        if raw:
            return cells
        return [self._preprocess(c) for c in cells]

    # ------------------------------------------------------------------
    # Grid splitting
    # ------------------------------------------------------------------

    def _split_grid(self, image: np.ndarray) -> list[np.ndarray]:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.copy()
        n = self.cfg.grid_size

        row_spans = self._find_cell_spans(gray, axis="row", n=n, size=h)
        col_spans = self._find_cell_spans(gray, axis="col", n=n, size=w)

        cells = []
        for r in range(n):
            for c in range(n):
                y0, y1 = row_spans[r]
                x0, x1 = col_spans[c]
                cells.append(image[y0:y1, x0:x1])
        return cells

    @staticmethod
    def _find_cell_spans(
        gray: np.ndarray, axis: str, n: int, size: int
    ) -> list[tuple[int, int]]:
        """Return n (start, end) pixel spans for n cells along one axis.

        Uses the mean gray value projected along the axis.  Grid lines are darker
        than cell interiors, so they are local minima.  We invert and find peaks.

        Constraints:
          - n+1 peaks expected (two outer borders + n-1 inner lines).
          - Min separation between peaks = half the expected cell size.
          - If detection fails, fall back to uniform division.
        """
        if axis == "col":
            proj = gray.mean(axis=0).astype(np.float32)
        else:
            proj = gray.mean(axis=1).astype(np.float32)

        # Smooth to suppress intra-cell variation (digit strokes)
        k = max(3, size // n // 6)
        kernel = np.ones(k, np.float32) / k
        proj_s = np.convolve(proj, kernel, mode="same")

        # Invert: dark grid lines become peaks
        inv = proj_s.max() - proj_s

        # Peak detection with min-distance suppression
        min_dist = max(4, size // n // 2)  # half of expected cell size
        threshold = inv.mean() * 1.2

        peaks: list[int] = []
        for i in range(size):
            if inv[i] < threshold:
                continue
            # Local maximum in ±3 neighbourhood
            lo, hi = max(0, i - 3), min(size, i + 4)
            if inv[i] < inv[lo:hi].max() - 1e-6:
                continue
            if peaks and i - peaks[-1] < min_dist:
                # Replace previous if current is higher
                if inv[i] > inv[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)

        # Keep only the n+1 strongest peaks when we have too many.
        if len(peaks) > n + 1:
            peaks = sorted(peaks, key=lambda p: -inv[p])[: n + 1]
            peaks.sort()

        if len(peaks) == n + 1:
            # Each cell spans from just past line i to just before line i+1.
            # Use a small trim (half the average line width estimate) to avoid
            # including grid-line pixels at cell edges.
            avg_gap = (peaks[-1] - peaks[0]) / n
            trim = max(2, int(avg_gap * 0.06))
            spans = []
            for i in range(n):
                start = max(0, peaks[i] + trim)
                end = min(size, peaks[i + 1] - trim)
                if end <= start:
                    end = start + 1
                spans.append((start, end))
            return spans

        # Fallback: uniform division with a small fixed trim.
        # This is robust after a correct homography because cells are nearly equal.
        step = size / n
        trim = max(2, int(step * 0.06))
        return [
            (max(0, int(round(i * step)) + trim), min(size, int(round((i + 1) * step)) - trim))
            for i in range(n)
        ]

    # ------------------------------------------------------------------
    # Per-cell preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(cell: np.ndarray, target_size: tuple[int, int] = (28, 28)) -> np.ndarray:
        """Preprocess a raw cell crop to match the digit-classifier training format.

        Training images are 28×28 white-digit-on-black created from cv2 text
        rendering (synthetic) and MNIST (handwritten).  This function maps a
        sudoku cell (dark ink on white paper) to that representation.

        The warpPerspective step fills out-of-image areas with pure black
        (gray=0), so every cell crop contains at least one black outlier pixel.
        Using raw min/max for contrast stretching would compress the actual
        image range and amplify paper texture into fake digit patterns.
        We therefore use 5th/95th percentile clipping.
        """
        if cell.size == 0:
            return np.zeros((*target_size, 3), dtype=np.uint8)

        gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY) if len(cell.shape) == 3 else cell.copy()
        cell_area = gray.size

        # ------ Empty cell detection: count genuinely dark ink pixels ------
        # Paper is typically gray > 150; printed digit ink is < 120.
        # Otsu fails here because the histogram is near-unimodal for empty
        # cells (paper texture only) and the threshold picks up JPEG noise.
        ink_ratio = float(np.count_nonzero(gray < 120)) / cell_area
        if ink_ratio < 0.015:
            return np.zeros((*target_size, 3), dtype=np.uint8)

        # ------ Build feature image using percentile normalization ------
        # Clamp to [5, 250] to eliminate warp-boundary black pixels (outliers)
        # before computing the normalization range.
        gray_c = np.clip(gray.astype(np.float32), 5.0, 250.0)
        inv = 250.0 - gray_c   # dark ink → high value, paper → low value

        mn = float(np.percentile(inv, 5))
        mx = float(np.percentile(inv, 95))

        # If the 5–95 percentile range is small, the cell is nearly uniform
        # (no real ink) → treat as empty.
        if mx - mn < 25:
            return np.zeros((*target_size, 3), dtype=np.uint8)

        stretched = ((inv - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)

        # ------ Tight-crop around digit content (MNIST-style centering) ------
        _, fg_mask = cv2.threshold(stretched, 40, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(fg_mask)
        if coords is not None and len(coords) > cell_area * 0.02:
            x, y, bw, bh = cv2.boundingRect(coords)
            pad = max(2, min(bw, bh) // 8)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2 = min(stretched.shape[1], x + bw + pad)
            y2 = min(stretched.shape[0], y + bh + pad)
            cropped = stretched[y1:y2, x1:x2]
        else:
            cropped = stretched

        if cropped.size == 0:
            return np.zeros((*target_size, 3), dtype=np.uint8)

        # Light blur to approximate JPEG compression artifacts in training data
        cropped = cv2.GaussianBlur(cropped, (3, 3), 0.5)
        rgb = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
        return cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
