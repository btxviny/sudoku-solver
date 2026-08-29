"""GridOCR: read a 9×9 digit grid from a rectified sudoku image.

Reads digits from a rectified grid, replacing per-cell classification.  The
underlying model (GridOCRNet) is a lightweight CNN trained on real cell
crops (from Roboflow datasets) and synthetic digit images.

The model processes each of the 81 equal-sized cell patches independently
using a shared encoder. Light per-patch cleanup (grid-line bleed, low-contrast
normalisation) is applied so inference matches the training distribution.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GridOCRConfig


class GridOCRNet(nn.Module):
    """Lightweight CNN: CELL_SIZE × CELL_SIZE grayscale → 10 logits.

    Class 0 = empty cell; classes 1–9 = digit.
    """

    def __init__(self, cell_size: int = 50):
        super().__init__()
        self.cell_size = cell_size
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class GridOCR:
    """Inference wrapper: rectified grid image → 9×9 digit matrix.

    `read_cells` is the entry point the pipeline uses: it takes the 81 cells
    YOLO located and classifies each one, batched into a single forward pass.
    `read`/`read_with_probs` take a whole grid image instead and fall back to a
    uniform 9×9 split, for callers that have no cell boxes.
    """

    def __init__(self, config: GridOCRConfig | None = None, device: str | None = None):
        self.cfg = config or GridOCRConfig()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self._load()
        self.model.eval()

    def _load(self) -> GridOCRNet:
        net = GridOCRNet(cell_size=self.cfg.patch_size)
        state = torch.load(
            self.cfg.model_path, map_location=self.device, weights_only=True
        )
        net.load_state_dict(state)
        net.to(self.device)
        return net

    @staticmethod
    def _prep_patch(p: np.ndarray, low_contrast: bool) -> np.ndarray:
        """Clean one cell patch so it matches the training distribution."""
        PS = p.shape[0]
        if low_contrast:
            # Low-contrast mode (coloured/faint digits): skip grid-line removal
            # because the dark-pixel threshold (< 100) can't reliably tell grid
            # lines from digit pixels in a compressed tonal range.  Stretch the
            # patch instead.
            pf = p.astype(np.float32)
            mn, mx = pf.min(), pf.max()
            if mx - mn > 20:
                pf = (pf - mn) / (mx - mn) * 255.0
                p = pf.clip(0, 255).astype(np.uint8)
            else:
                p = np.full_like(p, 255)          # no contrast -> treat as empty
        else:
            # Normal mode (dark ink on white paper): remove full-width grid-line
            # bars.  Rows where > 75 % of pixels are dark (< 100) are border
            # lines, not digit strokes.
            removed: list[int] = []
            for row_i in range(PS):
                if (p[row_i] < 100).mean() > 0.75:
                    p[row_i] = 255
                    removed.append(row_i)
            # Re-centre the digit vertically: with border rows removed from the
            # top or bottom the remaining glyph sits off-centre, which does not
            # match the training distribution.
            if removed:
                digit_rows = [i for i in range(PS) if (p[i] < 150).any()]
                if digit_rows:
                    top, bot = digit_rows[0], digit_rows[-1] + 1
                    region = p[top:bot, :].copy()
                    h = bot - top
                    start = (PS - h) // 2
                    p = np.full((PS, PS), 255, dtype=np.uint8)
                    p[start:start + h, :] = region
        return p

    def read(self, image: np.ndarray) -> np.ndarray:
        """Read digits from a rectified grid image.

        Args:
            image: RGB uint8, shape (H, W, 3).  H and W should be divisible
                   by 9 (the pipeline produces 450×450 by default).

        Returns:
            9×9 uint8 ndarray; 0 = empty, 1–9 = digit.
        """
        grid, _ = self._infer(image)
        return grid

    def read_with_probs(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (9×9 grid, 81×10 probability matrix) for a whole grid image.

        Assumes the grid fills the image, so cells fall on a uniform 9×9 split.
        The pipeline uses `read_cells` instead, feeding the cells YOLO located.
        """
        return self._infer(image)

    def read_cells(
        self,
        crops: list[np.ndarray | None],
        contrast_ref: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read digits from 81 individual cell crops (row-major).

        This is the per-cell entry point: the caller supplies the cells YOLO
        located, and each one is classified on its own.  Contrast is judged
        across all crops together, so one washed-out cell cannot be normalised
        into noise on its own.

        Returns (9x9 grid, 81x10 probabilities); a missing crop reads as empty.
        """
        PS = self.cfg.patch_size
        grays: list[np.ndarray | None] = []
        for c in crops:
            if c is None or c.size == 0:
                grays.append(None)
                continue
            g = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY) if c.ndim == 3 else c
            grays.append(cv2.resize(g, (PS, PS), interpolation=cv2.INTER_AREA))

        # Judge contrast over the whole grid when a reference is supplied:
        # inter-cell gutters and the outer frame carry the paper tone, so the
        # decision is steadier than one taken from the cell interiors alone.
        if contrast_ref is not None:
            ref = (cv2.cvtColor(contrast_ref, cv2.COLOR_RGB2GRAY)
                   if contrast_ref.ndim == 3 else contrast_ref)
            lo, hi = np.percentile(ref, [2, 98])
            low_contrast = (hi - lo) < 180
        else:
            present = [g for g in grays if g is not None]
            low_contrast = False
            if present:
                lo, hi = np.percentile(np.concatenate([g.ravel() for g in present]), [2, 98])
                low_contrast = (hi - lo) < 180

        blank = np.full((PS, PS), 255, dtype=np.uint8)
        patches = [blank if g is None else self._prep_patch(g.copy(), low_contrast)
                   for g in grays]
        return self._classify_patches(patches)

    def _classify_patches(self, patches: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Batch 81 prepared patches through the CNN -> (9x9 grid, 81x10 probs)."""
        t = torch.from_numpy(np.stack(patches)).float().unsqueeze(1) / 255.0
        t = t.to(self.device)
        with torch.no_grad():
            logits = self.model(t)  # (81, 10)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        grid = probs.argmax(axis=1).reshape(9, 9).astype(np.uint8)
        return grid, probs

    def _infer(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        PS = self.cfg.patch_size
        grid_size = PS * 9

        img = cv2.resize(image, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Detect low-contrast grids (e.g. colored digits on cream paper).
        # Normal black-on-white images have a global range > 180; coloured
        # digit images are compressed into a narrower band.  In the
        # low-contrast case we apply per-patch min-max normalisation to map
        # digit pixels to near-0 and background to near-255, matching the
        # training distribution without amplifying noise in good-quality images.
        # Measured on percentiles, not raw min/max: a single dark page tab or
        # specular highlight otherwise spans the range and hides the fact that
        # every cell is washed out.
        _lo, _hi = np.percentile(gray, [2, 98])
        global_range = int(_hi - _lo)
        low_contrast = global_range < 180

        patches = [
            self._prep_patch(gray[r * PS:(r + 1) * PS, c * PS:(c + 1) * PS].copy(),
                             low_contrast)
            for r in range(9) for c in range(9)
        ]

        return self._classify_patches(patches)
