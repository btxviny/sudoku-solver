"""The reading protocol shared by every per-cell digit model.

`GridOCR` and `CellOCR` differ only in the network behind them.  Everything
else -- how a crop becomes a patch, how contrast is judged, how the 81 patches
are batched into one forward pass -- lives here, so a comparison between the two
readers measures the models and not an accidental difference in plumbing.

Subclasses supply the network; this class supplies the rest.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cell_prep import is_low_contrast, prep_patch


class DigitReader:
    """Base reader: rectified grid or 81 cell crops -> 9x9 digits + probabilities.

    `read_cells` is the entry point the pipeline uses: it takes the 81 cells
    YOLO located and classifies each one, batched into a single forward pass.
    `read`/`read_with_probs` take a whole grid image instead and fall back to a
    uniform 9x9 split, for callers that have no cell boxes.
    """

    def __init__(
        self,
        net: nn.Module,
        model_path: Path,
        patch_size: int,
        device: str | None = None,
    ):
        self.patch_size = patch_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        net.load_state_dict(state)
        net.to(self.device)
        net.eval()
        self.model = net

    #: Patch cleanup lives in `cell_prep` so both readers and the Kotlin port
    #: cannot drift apart; exposed here as a static method for existing callers.
    _prep_patch = staticmethod(prep_patch)

    def read(self, image: np.ndarray) -> np.ndarray:
        """Read digits from a rectified grid image (RGB uint8) -> 9x9 uint8."""
        grid, _ = self._infer(image)
        return grid

    def read_with_probs(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(9x9 grid, 81x10 probabilities) for a whole grid image.

        Assumes the grid fills the image, so cells fall on a uniform 9x9 split.
        The pipeline uses `read_cells` instead, feeding the cells YOLO located.
        """
        return self._infer(image)

    def read_cells(
        self,
        crops: list[np.ndarray | None],
        contrast_ref: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read digits from 81 individual cell crops (row-major).

        Contrast is judged across the whole grid rather than per crop, so one
        washed-out cell cannot be normalised into noise on its own.

        Returns (9x9 grid, 81x10 probabilities); a missing crop reads as empty.
        """
        PS = self.patch_size
        grays: list[np.ndarray | None] = []
        for c in crops:
            if c is None or c.size == 0:
                grays.append(None)
                continue
            g = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY) if c.ndim == 3 else c
            grays.append(cv2.resize(g, (PS, PS), interpolation=cv2.INTER_AREA))

        # The gutters and outer frame carry the paper tone, so a reference over
        # the whole grid gives a steadier verdict than the cell interiors alone.
        if contrast_ref is not None:
            low_contrast = is_low_contrast(contrast_ref)
        else:
            present = [g for g in grays if g is not None]
            low_contrast = False
            if present:
                lo, hi = np.percentile(
                    np.concatenate([g.ravel() for g in present]), [2, 98]
                )
                low_contrast = bool((hi - lo) < 180)

        blank = np.full((PS, PS), 255, dtype=np.uint8)
        patches = [blank if g is None else prep_patch(g.copy(), low_contrast)
                   for g in grays]
        return self._classify_patches(patches)

    def _classify_patches(
        self, patches: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch 81 prepared patches through the net -> (9x9 grid, 81x10 probs)."""
        t = torch.from_numpy(np.stack(patches)).float().unsqueeze(1) / 255.0
        t = t.to(self.device)
        with torch.no_grad():
            logits = self.model(t)  # (81, 10)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        grid = probs.argmax(axis=1).reshape(9, 9).astype(np.uint8)
        return grid, probs

    def _infer(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Uniform 9x9 split of a whole grid image, then classify."""
        PS = self.patch_size
        grid_size = PS * 9
        img = cv2.resize(image, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        low_contrast = is_low_contrast(gray)
        patches = [
            prep_patch(gray[r * PS:(r + 1) * PS, c * PS:(c + 1) * PS].copy(),
                       low_contrast)
            for r in range(9) for c in range(9)
        ]
        return self._classify_patches(patches)
