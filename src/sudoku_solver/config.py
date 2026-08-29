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
    resize_to: tuple[int, int] = (1024, 1024)
    contour_epsilon: float = 0.02

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
class YoloGridDetectorConfig:
    """Configuration for the YOLO grid detector (Android-friendly Mask R-CNN replacement).

    `mode` selects the backend:
        "seg"   YOLOv8n-seg predicts a grid mask; corners come from the mask,
                exactly as the Mask R-CNN path derives them.
        "pose"  YOLOv8n-pose regresses the four corners directly.  Simpler to
                port (no mask post-processing at all), slightly less accurate.

    `refine` controls the shared Hough edge-snapping step.  It defaults to
    **off** here, unlike the Mask R-CNN path which always applies it: that
    refinement exists to correct Mask R-CNN's habit of under-segmenting the
    bottom edge, and its search band (BAND_OUT = 0.14) is wide enough to
    reach a page edge or table rule.  The YOLO detectors do not have that
    defect, so on their already-accurate quads the wide band is pure risk --
    measured on the 100 held-out images it pushed mean corner error from
    5.7 % to 40.9 %, blowing up roughly a third of images completely.

    Leave `model_path` as None to pick the weights matching `mode`.

    Train with:
        uv run python training/grid_pose/prepare_dataset.py   # seg needs no prep
        uv run python training/grid_pose/train.py
        uv run python training/grid_seg/train.py
    """
    mode: str = "seg"
    model_path: Path | None = None
    conf: float = 0.25
    imgsz: int = 640
    output_size: int = 450
    resize_to: tuple[int, int] = (1024, 1024)
    refine: bool = False

    def __post_init__(self):
        if self.model_path is None:
            run = "grid_seg/runs/grid_seg_v1" if self.mode == "seg" else "grid_pose/runs/grid_pose_v1"
            self.model_path = PROJECT_ROOT / "training" / run / "weights/best.pt"
        self.model_path = resolve(self.model_path)


@dataclass
class PipelineConfig:
    """Top-level configuration for the entire pipeline."""
    grid_detector: GridDetectorConfig = field(default_factory=GridDetectorConfig)
    grid_ocr: GridOCRConfig = field(default_factory=GridOCRConfig)
    yolo_cell_extractor: YoloCellExtractorConfig = field(default_factory=YoloCellExtractorConfig)
    yolo_grid_detector: YoloGridDetectorConfig = field(default_factory=YoloGridDetectorConfig)
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
