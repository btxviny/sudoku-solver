import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelPaths:
    """Paths to trained model weights."""
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3] / "models" / "weights")
    maskrcnn: Path = field(default_factory=lambda: Path("models/weights/maskrcnn_sudoku_20250913_141632.pth"))
    xgboost: Path = field(default_factory=lambda: Path("models/weights/xgboost_digit_classifier.model"))

    def __post_init__(self):
        self.maskrcnn = Path(self.maskrcnn)
        self.xgboost = Path(self.xgboost)


@dataclass
class GridDetectorConfig:
    """Configuration for the grid detection module."""
    model_path: Path = field(default_factory=lambda: Path("models/weights/maskrcnn_sudoku_20250913_141632.pth"))
    detection_threshold: float = 0.5
    output_size: int = 450
    resize_to: tuple = (1024, 1024)
    contour_epsilon: float = 0.02


@dataclass
class CellExtractorConfig:
    """Configuration for the cell extraction module."""
    grid_size: int = 9
    cell_size: tuple = (28, 28)
    empty_pixel_threshold: float = 200
    empty_ratio_threshold: float = 0.7


@dataclass
class DigitClassifierConfig:
    """Configuration for the digit classification module."""
    model_path: Path = field(default_factory=lambda: Path("models/weights/xgboost_digit_classifier.model"))
    confidence_threshold: float = 0.4
    feature_dim: int = 512
    num_classes: int = 10
    image_size: tuple = (28, 28)


@dataclass
class ImageNetConfig:
    """ImageNet normalization constants."""
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


@dataclass
class GridOCRConfig:
    """Configuration for the GridOCR CNN digit reader."""
    model_path: Path = field(default_factory=lambda: Path("models/weights/grid_ocr_cnn.pth"))
    patch_size: int = 50   # cell size in pixels (grid output_size / 9)

    def __post_init__(self):
        self.model_path = Path(self.model_path)


@dataclass
class PipelineConfig:
    """Top-level configuration for the entire pipeline."""
    grid_detector: GridDetectorConfig = field(default_factory=GridDetectorConfig)
    cell_extractor: CellExtractorConfig = field(default_factory=CellExtractorConfig)
    digit_classifier: DigitClassifierConfig = field(default_factory=DigitClassifierConfig)
    grid_ocr: GridOCRConfig = field(default_factory=GridOCRConfig)
    device: str = "cuda"

    @property
    def effective_device(self):
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self.device


# Default configuration instance
default_config = PipelineConfig()