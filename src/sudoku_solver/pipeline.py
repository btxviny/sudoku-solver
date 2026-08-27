"""End-to-end sudoku solver pipeline."""

from dataclasses import dataclass, field
import cv2
import numpy as np
import time

from .config import PipelineConfig
from .cell_extractor import CellExtractor
from .digit_classifier import DigitClassifier
from .sudoku_solver import SudokuSolver


@dataclass(frozen=True)
class PipelinePath:
    """One selectable (segmentation, OCR) combination.

    The UI builds its mode list from `PIPELINE_PATHS` rather than hardcoding
    combinations, so a path can never be offered without its weights or hidden
    while its weights are present.
    """
    key: str
    label: str
    seg: str
    ocr: str
    description: str
    requires: tuple[str, ...]    # SudokuPipeline attributes that must be loaded
    hint: str = ""               # shown when the path is unavailable
    recommended: bool = False


# Ordered best-first: the UI defaults to the first available entry.
PIPELINE_PATHS: tuple[PipelinePath, ...] = (
    PipelinePath(
        key="maskrcnn_gridocr",
        label="Mask R-CNN warp · GridOCR CNN",
        seg="griddetector",
        ocr="grid_ocr",
        description=(
            "Mask R-CNN locates the grid and corrects perspective, then a small CNN "
            "reads all 81 cells in one batched pass. Constraint recovery uses the CNN's "
            "per-cell probabilities to repair ambiguous digits."
        ),
        requires=("detector", "grid_ocr"),
        hint="Needs Mask R-CNN weights + models/weights/grid_ocr_cnn.pth",
        recommended=True,
    ),
    PipelinePath(
        key="yolo_gridocr",
        label="Mask R-CNN warp · YOLO cells · GridOCR CNN",
        seg="maskrcnn_yolo",
        ocr="grid_ocr",
        description=(
            "As above, but YOLOv8n additionally marks each cell as empty or filled on "
            "the rectified grid. The overlay is diagnostic — GridOCR still reads the "
            "digits from the warped image."
        ),
        requires=("detector", "yolo_extractor", "grid_ocr"),
        hint="Needs Mask R-CNN weights + YOLO cell weights + grid_ocr_cnn.pth",
    ),
    PipelinePath(
        key="yolo_yolodigit",
        label="YOLO cells · YOLO digits",
        seg="yolo",
        ocr="yolo_digit",
        description=(
            "YOLOv8n locates all 81 cells directly on the raw photo — no perspective "
            "warp. A second YOLO classification model reads each digit from its crop."
        ),
        requires=("yolo_extractor", "yolo_digit_classifier"),
        hint="Train the YOLO digit classifier: training/digit_classification/train.py",
    ),
    PipelinePath(
        key="maskrcnn_yolo_yolodigit",
        label="Mask R-CNN warp · YOLO cells · YOLO digits",
        seg="maskrcnn_yolo",
        ocr="yolo_digit",
        description=(
            "Mask R-CNN corrects perspective, YOLO locates cells on the clean grid, "
            "and the YOLO classifier reads each digit."
        ),
        requires=("detector", "yolo_extractor", "yolo_digit_classifier"),
        hint="Needs Mask R-CNN weights + a trained YOLO digit classifier.",
    ),
    PipelinePath(
        key="yolo_xgboost",
        label="YOLO cells · XGBoost digits",
        seg="yolo",
        ocr="classifier",
        description=(
            "YOLOv8n locates cells on the raw photo; ResNet18 features + XGBoost read "
            "each filled crop. Legacy path — accurate cell detection, but digit "
            "recognition is unreliable on photos."
        ),
        requires=("yolo_extractor", "classifier"),
    ),
    PipelinePath(
        key="maskrcnn_xgboost",
        label="Mask R-CNN warp · XGBoost digits",
        seg="griddetector",
        ocr="classifier",
        description=(
            "Mask R-CNN warps the grid, cells are split by projection peaks, and "
            "ResNet18 + XGBoost classifies each one. Original pipeline — kept for "
            "comparison; superseded by GridOCR."
        ),
        requires=("detector", "classifier"),
    ),
)


def _try_load(name: str, model_path, build):
    """Build an optional component, returning None (with a note) on any failure.

    Missing weights are the normal case for the untrained paths, so they are
    reported quietly; an exception during construction is reported loudly
    because it means the weights exist but are broken or incompatible.
    """
    if not model_path.exists():
        return None
    try:
        return build()
    except Exception as e:
        print(f"[pipeline] Failed to load {name} from {model_path}: {e}")
        return None


@dataclass
class PipelineResult:
    """Structured result from running the pipeline.

    Fields are None when their step failed; check `errors` for the reason.
    """
    original_grid: np.ndarray | None      # detected puzzle (clues only)
    solved_grid: np.ndarray | None        # complete solution
    timing: dict[str, float] = field(default_factory=dict)
    grid_image: np.ndarray | None = None  # rectified grid image (GridDetector) or raw (YOLO)
    errors: dict[str, str] = field(default_factory=dict)  # step -> error message
    seg_grid_image: np.ndarray | None = None      # segmentation overlay (Step 1)
    seg_cells_image: np.ndarray | None = None     # 9×9 cell mosaic (Step 2)
    recognition_image: np.ndarray | None = None   # seg image + digit overlay (Step 3)


class SudokuPipeline:
    """Orchestrates the full image-to-solution pipeline.

    A run is a (segmentation, OCR) pair; the valid combinations are enumerated
    in `PIPELINE_PATHS`.  Every optional component loads independently, so a
    pipeline built without, say, the YOLO weights still serves every path that
    does not need them — query `available_paths()` before calling `run()`.

    Segmentation modes:
        "griddetector"  Mask R-CNN warp, cells split by projection peaks
        "maskrcnn_yolo" Mask R-CNN warp, then YOLO cell detection on the warp
        "yolo"          YOLO cell detection on the raw image (no warp)

    OCR modes:
        "grid_ocr"      GridOCR CNN over the rectified grid (needs a warp)
        "yolo_digit"    YOLO classification model over each cell crop
        "classifier"    ResNet18 features + XGBoost over each cell crop
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.cfg = config or PipelineConfig()
        self.solver = SudokuSolver()
        device = self.cfg.effective_device

        self.extractor = CellExtractor(self.cfg.cell_extractor)

        self.classifier = _try_load(
            "DigitClassifier", self.cfg.digit_classifier.model_path,
            lambda: DigitClassifier(self.cfg.digit_classifier, device=device),
        )

        def _build_detector():
            from .grid_detector import GridDetector
            return GridDetector(self.cfg.grid_detector, device=device)

        self.detector = _try_load(
            "GridDetector", self.cfg.grid_detector.model_path, _build_detector
        )

        def _build_grid_ocr():
            from .grid_ocr import GridOCR
            return GridOCR(self.cfg.grid_ocr, device=device)

        self.grid_ocr = _try_load(
            "GridOCR", self.cfg.grid_ocr.model_path, _build_grid_ocr
        )

        def _build_yolo():
            from .yolo_cell_extractor import YoloCellExtractor
            return YoloCellExtractor(self.cfg.yolo_cell_extractor)

        self.yolo_extractor = _try_load(
            "YoloCellExtractor", self.cfg.yolo_cell_extractor.model_path, _build_yolo
        )

        def _build_yolo_digit():
            from .yolo_digit_classifier import YoloDigitClassifier
            return YoloDigitClassifier(self.cfg.yolo_digit_classifier)

        self.yolo_digit_classifier = _try_load(
            "YoloDigitClassifier", self.cfg.yolo_digit_classifier.model_path,
            _build_yolo_digit,
        )

        def _build_cell_cnn():
            from .cell_extractor_cnn import CellExtractorCNNInference
            return CellExtractorCNNInference(self.cfg.cell_extractor_cnn)

        self.cell_extractor_cnn = _try_load(
            "CellExtractorCNN", self.cfg.cell_extractor_cnn.model_path, _build_cell_cnn
        )

        ready = [p.key for p in self.available_paths()]
        print(f"[pipeline] device={device} | paths: {', '.join(ready) or 'none'}")

    @staticmethod
    def _make_recognition_image(
        base: np.ndarray,
        puzzle: np.ndarray,
        boxes_px: np.ndarray | None = None,
    ) -> np.ndarray:
        """Overlay recognised digits on base_image.

        If boxes_px is supplied (YOLO mode) digits are centred inside each cell
        box; otherwise they are placed on a uniform 9×9 grid (GridDetector mode).
        """
        img = base.copy()
        if boxes_px is not None:
            for i, (x1, y1, x2, y2) in enumerate(boxes_px):
                r, c = divmod(i, 9)
                digit = int(puzzle[r, c])
                if digit == 0:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue        # cell the detector missed — nothing to label
                cell_h = max(1, y2 - y1)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                scale = cell_h / 45.0
                thick = max(1, int(cell_h / 25))
                pos = (int(cx - cell_h * 0.18), int(cy + cell_h * 0.18))
                cv2.putText(img, str(digit), pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                            (255, 255, 255), thick + 2, cv2.LINE_AA)
                cv2.putText(img, str(digit), pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                            (20, 20, 20), thick, cv2.LINE_AA)
        else:
            h, w = img.shape[:2]
            cell_h, cell_w = h / 9, w / 9
            for r in range(9):
                for c in range(9):
                    digit = int(puzzle[r, c])
                    if digit == 0:
                        continue
                    cx = int((c + 0.5) * cell_w)
                    cy = int((r + 0.5) * cell_h)
                    scale = cell_h / 60
                    thick = max(1, int(cell_h / 30))
                    pos = (cx - int(cell_w * 0.18), cy + int(cell_h * 0.18))
                    cv2.putText(img, str(digit), pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                (255, 255, 255), thick + 3, cv2.LINE_AA)
                    cv2.putText(img, str(digit), pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                (20, 20, 20), thick, cv2.LINE_AA)
        return img

    @staticmethod
    def _make_cell_mosaic(
        cells: list[np.ndarray],
        cell_size: int = 40,
    ) -> np.ndarray:
        """Arrange up to 81 raw cell crops into a 9×9 mosaic image.

        Thin gaps separate cells; thicker gaps separate 3×3 boxes.
        """
        gap = 1
        box_gap = 3
        n = 9
        total = n * cell_size + (n - 1) * gap + (n // 3 - 1) * (box_gap - gap)
        canvas = np.full((total, total, 3), 200, dtype=np.uint8)

        offset = [0] * n
        for i in range(n):
            offset[i] = i * cell_size + i * gap + (i // 3) * (box_gap - gap)

        for idx in range(min(81, len(cells))):
            r, c = divmod(idx, 9)
            cell = cells[idx]
            if cell is None or cell.size == 0:
                continue
            if len(cell.shape) == 2:
                cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2RGB)
            resized = cv2.resize(cell, (cell_size, cell_size))
            y0, x0 = offset[r], offset[c]
            canvas[y0:y0 + cell_size, x0:x0 + cell_size] = resized

        return canvas

    @staticmethod
    def _preprocess_yolo_crop(crop: np.ndarray) -> np.ndarray:
        """Preprocess a detector-extracted cell crop for digit classification.

        Detector bboxes include the surrounding grid border pixels.  A plain
        CellExtractor._preprocess() call inverts those dark border lines to
        bright white, producing a rectangular frame artifact that the classifier
        reads as a digit.

        Fix: check the center 60 % of the crop for ink content (border-free).
        If that region has no ink the cell is empty.  Otherwise trim the outer
        12 % before calling the standard preprocessing.
        """
        h, w = crop.shape[:2]
        cy, cx = int(h * 0.2), int(w * 0.2)
        center = crop[cy: h - cy, cx: w - cx]
        if center.size > 0:
            gray_ctr = cv2.cvtColor(center, cv2.COLOR_RGB2GRAY) if len(center.shape) == 3 else center
            if float(np.count_nonzero(gray_ctr < 120)) / gray_ctr.size < 0.03:
                return np.zeros((28, 28, 3), dtype=np.uint8)  # empty cell

        trim = max(1, int(min(h, w) * 0.12))
        trimmed = crop[trim: h - trim, trim: w - trim]
        return CellExtractor._preprocess(trimmed)

    @staticmethod
    def _is_valid_partial_sudoku(grid: np.ndarray) -> bool:
        """Return True if the grid has no obvious digit conflicts (rows/cols/boxes)."""
        for i in range(9):
            row = grid[i][grid[i] > 0]
            col = grid[:, i][grid[:, i] > 0]
            if len(row) != len(set(row)) or len(col) != len(set(col)):
                return False
        for br in range(3):
            for bc in range(3):
                box = grid[br * 3:(br + 1) * 3, bc * 3:(bc + 1) * 3].flatten()
                box = box[box > 0]
                if len(box) != len(set(box)):
                    return False
        return True

    def path_available(self, path: PipelinePath) -> bool:
        """True when every component this path needs is loaded."""
        return all(getattr(self, attr, None) is not None for attr in path.requires)

    def available_paths(self) -> list[PipelinePath]:
        """Paths from PIPELINE_PATHS whose weights are all present, best first."""
        return [p for p in PIPELINE_PATHS if self.path_available(p)]

    def unavailable_paths(self) -> list[PipelinePath]:
        """Paths that are missing at least one component."""
        return [p for p in PIPELINE_PATHS if not self.path_available(p)]

    def available_seg_modes(self) -> list[str]:
        """Segmentation modes whose required weights are loaded."""
        modes = []
        if self.yolo_extractor is not None:
            modes.append("yolo")
        if self.detector is not None and self.yolo_extractor is not None:
            modes.append("maskrcnn_yolo")
        if self.detector is not None:
            modes.append("griddetector")
        return modes

    def available_ocr_modes(self, seg_mode: str) -> list[str]:
        """OCR modes compatible with the given segmentation mode."""
        modes = []
        if seg_mode in ("griddetector", "maskrcnn_yolo") and self.grid_ocr is not None:
            modes.append("grid_ocr")
        if self.yolo_digit_classifier is not None and seg_mode != "griddetector":
            modes.append("yolo_digit")
        if self.classifier is not None:
            modes.append("classifier")
        return modes

    def run_path(self, image: np.ndarray | str, path: PipelinePath) -> PipelineResult:
        """Run a `PipelinePath` — the mode pair the UI and CLI select from."""
        return self.run(image, seg_mode=path.seg, ocr_mode=path.ocr)

    def run(self, image: np.ndarray | str, seg_mode: str = "griddetector", ocr_mode: str = "grid_ocr") -> PipelineResult:
        """Run the full pipeline on an image path or numpy array.

        Args:
            image:    File path (str) or RGB numpy array.
            seg_mode: "yolo"          — YOLO cell detection on raw image
                      "maskrcnn_yolo" — MaskRCNN warp, then YOLO on warped grid
                      "griddetector"  — MaskRCNN warp, then projection-split cells
            ocr_mode: "grid_ocr"     — GridOCR CNN (seg_mode griddetector/maskrcnn_yolo only)
                      "yolo_digit"   — YOLO classification model (seg_mode yolo/maskrcnn_yolo)
                      "classifier"   — ResNet18 + XGBoost digit classifier

        Never raises for a failed step: the failing stage is recorded in
        `PipelineResult.errors` along with whatever earlier stages produced.
        Only an unreadable `image` argument raises (FileNotFoundError).
        """
        timings: dict[str, float] = {}
        errors: dict[str, str] = {}
        t0 = time.perf_counter()

        if isinstance(image, str):
            raw = cv2.imread(image)
            if raw is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
            image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        ocr_probs = None
        rectified = None
        puzzle = None
        raw_crops: list | None = None
        preprocessed: list | None = None
        yolo_labels: np.ndarray | None = None
        boxes_px: np.ndarray | None = None
        seg_grid_image = None
        seg_cells_image = None
        recognition_image = None

        # ══ Step 1: Segmentation ══════════════════════════════════════════════
        try:
            if seg_mode == "yolo":
                if self.yolo_extractor is None:
                    raise RuntimeError("YOLO weights not found — train or download the model first")
                t = time.perf_counter()
                raw_crops, boxes_px, yolo_labels = self.yolo_extractor.extract(image)
                timings["yolo_cell_extraction"] = time.perf_counter() - t
                rectified = image
                from .yolo_cell_extractor import YoloCellExtractor
                seg_grid_image = YoloCellExtractor.seg_overlay(image, boxes_px, yolo_labels)
                seg_cells_image = self._make_cell_mosaic(raw_crops)
                preprocessed = [
                    np.zeros((28, 28, 3), dtype=np.uint8)
                    if (yolo_labels[i] == 0 or c is None or c.size == 0)
                    else self._preprocess_yolo_crop(c)
                    for i, c in enumerate(raw_crops)
                ]

            elif seg_mode == "maskrcnn_yolo":
                if self.detector is None:
                    raise RuntimeError("GridDetector weights not found")
                if self.yolo_extractor is None:
                    raise RuntimeError("YOLO cell extractor weights not found")
                t = time.perf_counter()
                rectified, seg_grid_image = self.detector.detect_debug(image)
                timings["grid_detection"] = time.perf_counter() - t
                t = time.perf_counter()
                raw_crops, boxes_px, yolo_labels = self.yolo_extractor.extract(rectified)
                timings["yolo_cell_extraction"] = time.perf_counter() - t
                from .yolo_cell_extractor import YoloCellExtractor
                # Step 2 visual: YOLO boxes drawn on the warped grid
                seg_cells_image = YoloCellExtractor.seg_overlay(rectified, boxes_px, yolo_labels)
                preprocessed = [
                    np.zeros((28, 28, 3), dtype=np.uint8)
                    if (yolo_labels[i] == 0 or c is None or c.size == 0)
                    else self._preprocess_yolo_crop(c)
                    for i, c in enumerate(raw_crops)
                ]

            else:  # griddetector
                if self.detector is None:
                    raise RuntimeError("GridDetector weights not found")
                t = time.perf_counter()
                rectified, seg_grid_image = self.detector.detect_debug(image)
                timings["grid_detection"] = time.perf_counter() - t
                seg_cells_image = self._make_cell_mosaic(
                    self.extractor.extract(rectified, raw=True)
                )

        except Exception as e:
            errors["detection"] = str(e)
            timings["total"] = time.perf_counter() - t0
            return PipelineResult(
                original_grid=None, solved_grid=None, timing=timings,
                grid_image=None, errors=errors,
                seg_grid_image=seg_grid_image, seg_cells_image=seg_cells_image,
            )

        # ══ Step 2: OCR ══════════════════════════════════════════════════════
        try:
            if ocr_mode == "yolo_digit":
                if self.yolo_digit_classifier is None:
                    raise RuntimeError(
                        "YOLO digit classifier not trained. "
                        "Run training/digit_classification/train.py first."
                    )
                if raw_crops is None:
                    raise RuntimeError(
                        "The YOLO digit classifier reads per-cell crops, which "
                        "griddetector segmentation does not produce. Use yolo or "
                        "maskrcnn_yolo segmentation."
                    )
                t = time.perf_counter()
                puzzle = self.yolo_digit_classifier.classify_grid(raw_crops, yolo_labels)
                timings["yolo_digit_classification"] = time.perf_counter() - t

            elif ocr_mode == "grid_ocr":
                if self.grid_ocr is None:
                    raise RuntimeError("GridOCR model not available — check model weights")
                if seg_mode == "yolo":
                    raise RuntimeError(
                        "GridOCR requires a perspective-corrected grid. "
                        "Use griddetector or maskrcnn_yolo segmentation."
                    )
                t = time.perf_counter()
                puzzle, ocr_probs = self.grid_ocr.read_with_probs(rectified)
                timings["grid_ocr"] = time.perf_counter() - t

            else:  # classifier (ResNet18 features + XGBoost)
                if self.classifier is None:
                    raise RuntimeError(
                        "XGBoost digit classifier not available — check "
                        "models/weights/xgboost_digit_classifier.model"
                    )
                if preprocessed is not None:
                    t = time.perf_counter()
                    puzzle = self.classifier.classify_grid(preprocessed)
                    timings["digit_classification"] = time.perf_counter() - t
                    flat = puzzle.flatten()
                    for i in range(81):
                        if yolo_labels[i] == 0:
                            flat[i] = 0
                    puzzle = flat.reshape(9, 9)
                else:
                    t = time.perf_counter()
                    cells = self.extractor.extract(rectified)
                    timings["cell_extraction"] = time.perf_counter() - t
                    t = time.perf_counter()
                    puzzle = self.classifier.classify_grid(cells)
                    timings["digit_classification"] = time.perf_counter() - t

            # Build recognition image: digits overlaid on warped grid (or raw for
            # yolo-only).  GridOCR reads a uniform 9×9 split, so its digits are
            # placed on that same uniform grid even when YOLO boxes exist.
            use_boxes = boxes_px if (boxes_px is not None and ocr_mode != "grid_ocr") else None
            recognition_image = self._make_recognition_image(
                rectified, puzzle, use_boxes
            )

        except Exception as e:
            errors["ocr"] = str(e)
            timings["total"] = time.perf_counter() - t0
            return PipelineResult(
                original_grid=None, solved_grid=None, timing=timings,
                grid_image=rectified, errors=errors,
                seg_grid_image=seg_grid_image, seg_cells_image=seg_cells_image,
                recognition_image=recognition_image,
            )

        # ══ Step 3: Solving ══════════════════════════════════════════════════
        try:
            t = time.perf_counter()
            try:
                solution, solve_time = self.solver.solve(puzzle)
            except RuntimeError:
                if ocr_probs is not None:
                    puzzle, solution, solve_time = self._recover_with_constraints(
                        puzzle, ocr_probs
                    )
                else:
                    raise
            timings["solving"] = solve_time
        except Exception as e:
            errors["solving"] = str(e)
            timings["total"] = time.perf_counter() - t0
            return PipelineResult(
                original_grid=puzzle, solved_grid=None, timing=timings,
                grid_image=rectified, errors=errors,
                seg_grid_image=seg_grid_image, seg_cells_image=seg_cells_image,
                recognition_image=recognition_image,
            )

        timings["total"] = time.perf_counter() - t0
        return PipelineResult(
            original_grid=puzzle,
            solved_grid=solution,
            timing=timings,
            grid_image=rectified,
            errors=errors,
            seg_grid_image=seg_grid_image,
            seg_cells_image=seg_cells_image,
            recognition_image=recognition_image,
        )

    def _recover_with_constraints(
        self,
        puzzle: np.ndarray,
        probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Attempt to fix OCR errors by substituting alternative digit predictions.

        Strategy:
          1. Sort the 81 cells by their top-1 confidence (ascending = most uncertain first).
          2. For each cell in that order, try swapping its digit to the next most
             probable alternative, then attempt to solve.
          3. Backtrack / continue until a valid solution is found or candidates exhaust.

        This handles the common case where 1–2 cells are ambiguous (e.g. "5" vs "6"
        at the outer border) while the rest of the puzzle is correctly identified.
        """
        import itertools

        candidates: list[list[int]] = []
        top1_confs: list[float] = []
        flat = puzzle.flatten()
        for i in range(81):
            ranked = sorted(range(10), key=lambda d: -probs[i, d])
            top1_confs.append(float(probs[i, ranked[0]]))
            alts = [d for d in ranked[1:4] if probs[i, d] >= 0.03]
            if flat[i] > 0:
                alts = [d for d in alts if d != 0]
            candidates.append(alts)

        uncertain_idx = [
            i for i in range(81)
            if candidates[i] and top1_confs[i] < 0.60
        ]
        uncertain_idx.sort(key=lambda i: top1_confs[i])

        base = puzzle.flatten().astype(np.uint8)
        t0 = time.perf_counter()
        for n_flip in (1, 2, 3):
            for combo in itertools.combinations(uncertain_idx[:12], n_flip):
                alt_lists = [candidates[i] for i in combo]
                for alt_digits in itertools.product(*alt_lists):
                    candidate = base.copy()
                    for idx, digit in zip(combo, alt_digits):
                        candidate[idx] = digit
                    candidate_grid = candidate.reshape(9, 9)
                    try:
                        sol, st = self.solver.solve(candidate_grid.copy())
                        if sol.min() > 0:
                            return candidate_grid, sol, time.perf_counter() - t0
                    except RuntimeError:
                        pass

        for conf_threshold in (0.80, 0.70, 0.60):
            n_uncertain = sum(1 for i in range(81) if top1_confs[i] < conf_threshold and base[i] > 0)
            if n_uncertain > 6:
                continue
            confident_grid = base.copy()
            for i in range(81):
                if top1_confs[i] < conf_threshold:
                    confident_grid[i] = 0
            try:
                sol, st = self.solver.solve(confident_grid.reshape(9, 9).copy())
                if sol.min() > 0:
                    return confident_grid.reshape(9, 9), sol, time.perf_counter() - t0
            except RuntimeError:
                pass

        solve_time = time.perf_counter() - t0
        raise RuntimeError(
            "No valid solution found after constraint recovery. "
            "The puzzle image may be unclear or the grid detection may have failed."
        )

    @staticmethod
    def print_result(result: PipelineResult) -> None:
        SudokuSolver.print_grid(result.original_grid, "Detected Puzzle")
        SudokuSolver.print_grid(result.solved_grid, "Solution")
        print("\nTiming:")
        for k, v in result.timing.items():
            print(f"  {k}: {v:.4f}s")
