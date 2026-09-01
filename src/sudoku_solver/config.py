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


@dataclass
class GridOCRConfig:
    """Configuration for the GridOCR CNN digit reader."""
    model_path: Path = field(default_factory=lambda: WEIGHTS_DIR / "grid_ocr_cnn.pth")
    patch_size: int = 70   # cell size in pixels (grid output_size / 9)

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
    """Configuration for the YOLO grid detector (step 1: locate and rectify).

    `mode` selects the backend:
        "seg"   YOLOv8n-seg predicts a grid mask; corners come from the mask,
                fitted to the mask.
        "pose"  YOLOv8n-pose regresses the four corners directly.  Simpler to
                port (no mask post-processing at all), slightly less accurate.

    `refine` controls the Hough edge-snapping step in `grid_geometry`, and
    defaults to **off**.  That refinement was written for a detector that
    under-segmented the bottom edge, and its search band (BAND_OUT = 0.14) is
    wide enough to reach a page edge or table rule.  These detectors do not have
    that defect, so on their already-accurate quads the wide band is pure risk --
    measured on 100 held-out images it pushed mean corner error from 5.7 % to
    40.9 %, blowing up roughly a third of images completely.

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
    output_size: int = 630
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
