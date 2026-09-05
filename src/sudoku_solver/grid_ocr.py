"""GridOCR: read a 9×9 digit grid from a rectified sudoku image.

Reads digits from a rectified grid, replacing per-cell classification.  The
underlying model (GridOCRNet) is a lightweight CNN trained on real cell
crops (from Roboflow datasets) and synthetic digit images.

The model processes each of the 81 equal-sized cell patches independently
using a shared encoder. Light per-patch cleanup (grid-line bleed, low-contrast
normalisation) is applied so inference matches the training distribution.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import GridOCRConfig
from .digit_reader import DigitReader


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.skip: nn.Module = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))) + self.skip(x))


class GridOCRNet(nn.Module):
    """Residual CNN: CELL_SIZE × CELL_SIZE grayscale → 10 logits.

    Class 0 = empty cell; classes 1–9 = digit.
    4 residual blocks (32→64→128→256) with stride-2 downsampling + GlobalAvgPool.
    """

    def __init__(self, cell_size: int = 50):
        super().__init__()
        self.cell_size = cell_size
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
        )
        self.layer1 = _ResBlock(32, 64, stride=2)    # 25×25
        self.layer2 = _ResBlock(64, 128, stride=2)   # 12×12
        self.layer3 = _ResBlock(128, 256, stride=2)  # 6×6
        self.layer4 = _ResBlock(256, 256, stride=1)  # 6×6
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(self.pool(x))


class GridOCR(DigitReader):
    """The original digit reader: GridOCRNet behind the shared reading protocol.

    Everything except the network lives in `DigitReader`; see `cell_ocr.CellOCR`
    for the newer model that plugs into the same protocol.
    """

    def __init__(self, config: GridOCRConfig | None = None, device: str | None = None):
        self.cfg = config or GridOCRConfig()
        super().__init__(
            net=GridOCRNet(cell_size=self.cfg.patch_size),
            model_path=self.cfg.model_path,
            patch_size=self.cfg.patch_size,
            device=device,
        )
