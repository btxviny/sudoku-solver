"""Configuration for the sudoku solver pipeline.

All model paths are anchored to the project root, so the pipeline works
regardless of the process working directory.  Paths given as absolute are
used as-is; relative paths are resolved against PROJECT_ROOT.
"""

from dataclasses import dataclass, field
from pathlib import Path

# src/sudoku_solver/config.py -> src/sudoku_solver -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"


def resolve(path: Path | str) -> Path:
    """Resolve a config path against the project root unless already absolute."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def latest_maskrcnn() -> Path:
    """Newest `maskrcnn_sudoku_*.pth` in the weights dir.

    Weights are timestamped at training time, so pinning one filename in the
    config goes stale after every retrain.  Falls back to a conventional path
    (which simply won't exist) so callers can report a missing-weights error
    instead of crashing on an empty glob.
    """
    candidates = sorted(WEIGHTS_DIR.glob("maskrcnn_sudoku_*.pth"))
    return candidates[-1] if candidates else WEIGHTS_DIR / "maskrcnn_sudoku.pth"


@dataclass
class GridDetectorConfig:
    """Configuration for the grid detection module."""
    model_path: Path = field(default_factory=latest_maskrcnn)
    detection_threshold: float = 0.5
    output_size: int = 450
    resize_to: tuple = (1024, 1024)
    contour_epsilon: float = 0.02

    def __post_init__(self):
        self.model_path = resolve(self.model_path)


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
    model_path: Path = field(
        default_factory=lambda: WEIGHTS_DIR / "xgboost_digit_classifier.model"
    )
    confidence_threshold: float = 0.4
    feature_dim: int = 512
    num_classes: int = 10
    image_size: tuple = (28, 28)

    def __post_init__(self):
        self.model_path = resolve(self.model_path)


@dataclass
class ImageNetConfig:
    """ImageNet normalization constants."""
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


@dataclass
class CellExtractorCNNConfig:
    """Configuration for the CNN-based cell extractor."""
    model_path: Path = field(
        default_factory=lambda: WEIGHTS_DIR / "cell_extractor_cnn.pth"
    )
    input_size: int = 320   # resize shorter edge to this before inference

    def __post_init__(self):
        self.model_path = resolve(self.model_path)


@dataclass
class GridOCRConfig:
    """Configuration for the GridOCR CNN digit reader."""
    model_path: Path = field(default_factory=lambda: WEIGHTS_DIR / "grid_ocr_cnn.pth")
    patch_size: int = 50   # cell size in pixels (grid output_size / 9)

    def __post_init__(self):
        self.model_path = resolve(self.model_path)


@dataclass
class YoloCellExtractorConfig:
    """Configuration for the YOLO-based cell extractor."""
    model_path: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / "training/cell_extraction/runs/cell_vision_v6/weights/best.pt"
    )
    conf: float = 0.3
    iou: float = 0.5

    def __post_init__(self):
        self.model_path = resolve(self.model_path)


@dataclass
class YoloDigitClassifierConfig:
    """Configuration for the YOLO classification model that reads digit values 0-9.

    Train with:
        yolo classify train data=<digit-dataset> model=yolov8n-cls.pt epochs=50
    Expected classes: 0=empty, 1-9=digits  (10 classes total).
    """
    model_path: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / "training/digit_classification/runs/digit_cls/weights/best.pt"
    )
    imgsz: int = 64

    def __post_init__(self):
        self.model_path = resolve(self.model_path)


@dataclass
class PipelineConfig:
    """Top-level configuration for the entire pipeline."""
    grid_detector: GridDetectorConfig = field(default_factory=GridDetectorConfig)
    cell_extractor: CellExtractorConfig = field(default_factory=CellExtractorConfig)
    cell_extractor_cnn: CellExtractorCNNConfig = field(default_factory=CellExtractorCNNConfig)
    digit_classifier: DigitClassifierConfig = field(default_factory=DigitClassifierConfig)
    grid_ocr: GridOCRConfig = field(default_factory=GridOCRConfig)
    yolo_cell_extractor: YoloCellExtractorConfig = field(default_factory=YoloCellExtractorConfig)
    yolo_digit_classifier: YoloDigitClassifierConfig = field(default_factory=YoloDigitClassifierConfig)
    device: str = "auto"   # "auto" | "cuda" | "cpu"

    @property
    def effective_device(self) -> str:
        """Resolve `device` to a torch device string that actually exists here."""
        import torch
        if self.device == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"


# Default configuration instance
default_config = PipelineConfig()
