"""CellOCR: the second-generation per-cell digit reader.

Same job as `GridOCR` and the same reading protocol (`DigitReader`), a
different network and a different training corpus.  Three things separate
`CellOCRNet` from `GridOCRNet`:

**Squeeze-excitation.**  A cell patch is mostly paper.  Channel gating lets the
network suppress the channels that fired on a grid-line remnant or a page
texture before those features reach the classifier, which is where the old
model's handwriting errors came from -- a faint stroke and a border artefact
excite similar early filters, and plain residual blocks have no way to weigh
one down.

**Average *and* max pooling at the head.**  Global average pooling alone
measures how much of the patch looks like ink, which suits thick printed
glyphs and washes out thin ballpoint strokes.  Concatenating a max-pooled
vector keeps the strongest evidence for a feature regardless of how little of
the cell it covers.

**A wider stem at full resolution.**  A 5x5 stem sees whole stroke junctions
(the crossing in an 8, the closure of a 6) before any downsampling; the old
3x3 stem downsampled immediately after a single 3x3 convolution.

The corpus differs too, and deliberately: handwriting comes from EMNIST, with
MNIST held out entirely, because the mixed evaluation grids in
`data/wicht_sudoku` are built by pasting MNIST glyphs.  A model trained on
MNIST scores against its own training glyphs there.  See
`training/cell_ocr/README.md`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import CellOCRConfig
from .digit_reader import DigitReader


class _SEBlock(nn.Module):
    """Squeeze-excitation: per-channel gating from global context."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.act = nn.SiLU()
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mean over H,W rather than AdaptiveAvgPool2d(1): both export cleanly,
        # but the explicit reduction keeps the converted graph readable.
        s = x.mean(dim=(2, 3), keepdim=True)
        return x * self.gate(self.fc2(self.act(self.fc1(s))))


class _SEResBlock(nn.Module):
    """Pre-activation residual block with squeeze-excitation."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = _SEBlock(out_ch)
        self.act = nn.SiLU()
        self.skip: nn.Module = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.conv1(x)))
        y = self.se(self.bn2(self.conv2(y)))
        return self.act(y + self.skip(x))


class CellOCRNet(nn.Module):
    """SE-residual CNN: cell_size x cell_size grayscale -> 10 logits.

    Class 0 = empty cell; classes 1-9 = digit.  Every operation here converts
    to TFLite without a custom op, which is a hard requirement: this model ships
    to Android through `scripts/export_tflite.py`.

    At the default 70 px:
        stem   70x70x32   (5x5 conv, full resolution)
        s1     35x35x64   (2 blocks)
        s2     18x18x128  (2 blocks)
        s3      9x9x192   (2 blocks)
        head   concat(avgpool, maxpool) = 384 -> 10
    """

    def __init__(self, cell_size: int = 70, width: tuple[int, int, int] = (64, 128, 192)):
        super().__init__()
        self.cell_size = cell_size
        w1, w2, w3 = width
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(),
        )
        self.stage1 = nn.Sequential(_SEResBlock(32, w1, stride=2), _SEResBlock(w1, w1))
        self.stage2 = nn.Sequential(_SEResBlock(w1, w2, stride=2), _SEResBlock(w2, w2))
        self.stage3 = nn.Sequential(_SEResBlock(w2, w3, stride=2), _SEResBlock(w3, w3))
        self.classifier = nn.Sequential(
            nn.Linear(w3 * 2, 192), nn.BatchNorm1d(192), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(192, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage3(self.stage2(self.stage1(self.stem(x))))
        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        return self.classifier(torch.cat([avg, mx], dim=1))


class CellOCR(DigitReader):
    """Inference wrapper for `CellOCRNet`, interchangeable with `GridOCR`."""

    def __init__(self, config: CellOCRConfig | None = None, device: str | None = None):
        self.cfg = config or CellOCRConfig()
        super().__init__(
            net=CellOCRNet(cell_size=self.cfg.patch_size),
            model_path=self.cfg.model_path,
            patch_size=self.cfg.patch_size,
            device=device,
        )
