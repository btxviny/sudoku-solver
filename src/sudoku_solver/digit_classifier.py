import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import xgboost as xgb
from .config import DigitClassifierConfig, ImageNetConfig, CellExtractorConfig


class DigitClassifier:
    """Classify digits in preprocessed sudoku cells via ResNet18 features + XGBoost."""

    def __init__(self, config: DigitClassifierConfig | None = None):
        self.cfg = config or DigitClassifierConfig()
        self.cell_cfg = CellExtractorConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet = self._build_feature_extractor()
        self.model = self._load_xgboost()
        imagenet = ImageNetConfig()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet.mean, std=imagenet.std),
        ])

    def _build_feature_extractor(self) -> nn.Module:
        net = models.resnet18(weights="IMAGENET1K_V1")
        net = nn.Sequential(*list(net.children())[:-1])
        net.to(self.device)
        net.eval()
        return net

    def _load_xgboost(self) -> xgb.XGBClassifier:
        clf = xgb.XGBClassifier()
        clf.load_model(self.cfg.model_path)
        return clf

    def classify(self, cell: np.ndarray) -> tuple[int, float, bool]:
        """Classify a single cell.

        Returns:
            (digit, confidence, is_empty)

        digit is 0-9 (0 = empty), confidence in [0, 1], is_empty is bool.
        """
        if self._is_empty_heuristic(cell):
            return 0, 1.0, True

        img_rgb = self._ensure_rgb(cell)
        # Test-time augmentation: average probabilities over small rotations
        # to reduce sensitivity to font style variations.
        all_probs = []
        for angle in (0, -8, 8):
            aug = self._rotate(img_rgb, angle)
            tensor = self.transform(aug).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.resnet(tensor).view(1, -1).cpu().numpy()
            all_probs.append(self.model.predict_proba(feat)[0])
        probs = np.mean(all_probs, axis=0)
        digit = int(np.argmax(probs))
        conf = float(np.max(probs))
        if conf < self.cfg.confidence_threshold:
            return 0, conf, True
        return digit, conf, False

    @staticmethod
    def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def classify_grid(self, cells: list[np.ndarray]) -> np.ndarray:
        """Classify 81 cells and return a 9x9 numpy grid."""
        if len(cells) != 81:
            raise ValueError(f"Expected 81 cells, got {len(cells)}")
        grid = np.zeros((9, 9), dtype=np.uint8)
        for i, cell in enumerate(cells):
            digit, _, _ = self.classify(cell)
            grid[i // 9, i % 9] = digit
        return grid

    @staticmethod
    def _is_empty_heuristic(cell: np.ndarray, threshold: float = 200, ratio: float = 0.02) -> bool:
        # After THRESH_BINARY_INV preprocessing: empty cells have nearly no white pixels;
        # cells with a digit have white pixels where the digit strokes are.
        if len(cell.shape) == 3:
            gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
        else:
            gray = cell
        white_ratio = np.sum(gray > threshold) / gray.size
        return white_ratio < ratio

    @staticmethod
    def _ensure_rgb(cell: np.ndarray) -> np.ndarray:
        if len(cell.shape) == 2:
            return np.repeat(cell[:, :, np.newaxis], 3, axis=2)
        if cell.shape[2] == 1:
            return np.repeat(cell, 3, axis=2)
        return cell