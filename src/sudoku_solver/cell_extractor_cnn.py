"""CNN model for sudoku cell extraction.

Input:  RGB image (any size) containing a sudoku puzzle.
Output: 81 bounding boxes in row-major order (top-left → bottom-right),
        each as [x1, y1, x2, y2] normalised to [0, 1] relative to the
        input image dimensions.

The extracted crops are meant to be fed directly to a digit classifier.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from .config import CellExtractorCNNConfig


class CellExtractorCNN(nn.Module):
    """MobileNetV3-Small backbone + bbox regression head.

    Outputs 81 normalised [x1, y1, x2, y2] bboxes in row-major order.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool          # AdaptiveAvgPool2d(1) → (B, 576, 1, 1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 1024),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(1024, 81 * 4),
            nn.Sigmoid(),                       # all outputs in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) float32 ImageNet-normalised.
        Returns:
            (B, 81, 4) normalised [x1, y1, x2, y2] bboxes.
        """
        return self.head(self.avgpool(self.features(x))).view(-1, 81, 4)


class CellExtractorCNNInference:
    """Wraps CellExtractorCNN for inference: image in → 81 BGR crops out."""

    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(self, cfg: CellExtractorCNNConfig | None = None):
        self.cfg = cfg or CellExtractorCNNConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load()

    def _load(self) -> CellExtractorCNN:
        m = CellExtractorCNN(pretrained=False)
        m.load_state_dict(
            torch.load(self.cfg.model_path, map_location=self.device, weights_only=True)
        )
        m.to(self.device).eval()
        return m

    def _forward(self, image: np.ndarray):
        """Run model inference; return (crops, boxes_px) where boxes_px is (81, 4) int."""
        from torchvision import transforms

        H, W = image.shape[:2]
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.cfg.input_size, self.cfg.input_size), antialias=True),
            transforms.Normalize(self._MEAN, self._STD),
        ])(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            boxes_norm = self.model(t)[0].cpu().numpy()   # (81, 4) normalised

        crops = []
        boxes_px = np.zeros((81, 4), dtype=np.int32)
        for i, (x1, y1, x2, y2) in enumerate(boxes_norm):
            px1 = max(0, int(round(x1 * W)))
            py1 = max(0, int(round(y1 * H)))
            px2 = min(W, int(round(x2 * W)))
            py2 = min(H, int(round(y2 * H)))
            crops.append(image[py1:py2, px1:px2])
            boxes_px[i] = [px1, py1, px2, py2]
        return crops, boxes_px

    def extract(self, image: np.ndarray) -> list[np.ndarray]:
        """Return 81 cell crops (RGB uint8) in row-major order."""
        crops, _ = self._forward(image)
        return crops

    def extract_with_boxes(self, image: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """Return (81 crops, (81, 4) pixel bounding boxes [x1, y1, x2, y2])."""
        return self._forward(image)

    @staticmethod
    def seg_overlay(image: np.ndarray, boxes_px: np.ndarray) -> np.ndarray:
        """Draw the 81 CNN-predicted cell bounding boxes on image."""
        vis = image.copy()
        # One colour per 3×3 box section for orientation
        palette = [
            (100, 200, 255), (255, 175, 70), (140, 230, 140),
            (255, 110, 110), (190, 130, 255), (60, 215, 215),
            (255, 205, 80),  (170, 255, 170), (255, 150, 195),
        ]
        for i, (x1, y1, x2, y2) in enumerate(boxes_px):
            r, c = divmod(i, 9)
            color = palette[(r // 3) * 3 + (c // 3)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        return vis
