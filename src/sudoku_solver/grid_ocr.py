"""GridOCR: read a 9×9 digit grid from a rectified sudoku image.

Replaces the CellExtractor + DigitClassifier pair in the pipeline.  The
underlying model (GridOCRNet) is a lightweight CNN trained on real cell
crops (from Roboflow datasets) and synthetic digit images.

The model processes each of the 81 equal-sized cell patches independently
using a shared encoder, avoiding all the fragile preprocessing steps that
plagued the previous pipeline (Otsu binarisation, min-max contrast stretch,
Hough peak detection).
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

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

    The model reads all 81 cells in a single batched forward pass.
    Grid lines, border artifacts, and empty-cell noise are handled
    implicitly by the trained CNN — no manual preprocessing required.
    """

    def __init__(self, config: "GridOCRConfig | None" = None):
        self.cfg = config or GridOCRConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        """Return (9×9 grid, 81×10 probability matrix) for constraint recovery."""
        return self._infer(image)

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
        global_range = int(gray.max()) - int(gray.min())
        low_contrast = global_range < 180

        patches = []
        for r in range(9):
            for c in range(9):
                p = gray[r * PS:(r + 1) * PS, c * PS:(c + 1) * PS].copy()
                if low_contrast:
                    # Low-contrast mode (coloured/faint digits): skip grid-line
                    # removal because the dark-pixel threshold (< 100) can't
                    # reliably distinguish grid lines from digit pixels in a
                    # compressed tonal range.  Use per-patch normalisation instead.
                    pf = p.astype(np.float32)
                    mn, mx = pf.min(), pf.max()
                    if mx - mn > 20:
                        pf = (pf - mn) / (mx - mn) * 255.0
                        p = pf.clip(0, 255).astype(np.uint8)
                    else:
                        p = np.full_like(p, 255)  # no contrast → treat as empty
                else:
                    # Normal mode (dark ink on white paper): remove full-width
                    # horizontal grid-line bars.  Rows where >75 % of pixels are
                    # dark (< 100) are border/frame lines, not digit strokes.
                    removed: list[int] = []
                    for row_i in range(PS):
                        if (p[row_i] < 100).mean() > 0.75:
                            p[row_i] = 255
                            removed.append(row_i)
                    # Re-centre digit content vertically: when border rows were
                    # removed from the top/bottom, the remaining digit sits in the
                    # lower/upper fraction of the patch.  Compacting it to the
                    # vertical centre matches the training distribution.
                    if removed:
                        digit_rows = [i for i in range(PS) if (p[i] < 150).any()]
                        if digit_rows:
                            top, bot = digit_rows[0], digit_rows[-1] + 1
                            region = p[top:bot, :].copy()
                            h = bot - top
                            start = (PS - h) // 2
                            p = np.full((PS, PS), 255, dtype=np.uint8)
                            p[start:start + h, :] = region
                patches.append(p)

        t = torch.from_numpy(np.stack(patches)).float().unsqueeze(1) / 255.0
        t = t.to(self.device)

        with torch.no_grad():
            logits = self.model(t)                          # (81, 10)

        import torch.nn.functional as F
        probs = F.softmax(logits, dim=1).cpu().numpy()     # (81, 10)
        grid  = probs.argmax(axis=1).reshape(9, 9).astype(np.uint8)
        return grid, probs
