"""End-to-end sudoku solver pipeline."""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import PipelineConfig
from .sudoku_solver import SudokuSolver

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelinePath:
    """One selectable (segmentation, OCR) combination.

    The UI builds its mode list from `PIPELINE_PATHS` rather than hardcoding
    combinations, so a path can never be offered without its weights or hidden
    while its weights are present.
    """
    key: str
    label: str
    ocr: str
    description: str
    requires: tuple[str, ...]    # SudokuPipeline attributes that must be loaded
    warp: str = "maskrcnn"       # "maskrcnn" | "yolo" — which detector rectifies
    hint: str = ""               # shown when the path is unavailable
    recommended: bool = False


# Ordered best-first: the UI defaults to the first available entry.
PIPELINE_PATHS: tuple[PipelinePath, ...] = (
    PipelinePath(
        key="yolo_gridocr",
        label="Mask R-CNN warp · YOLO cells · GridOCR CNN",
        ocr="grid_ocr",
        description=(
            "Mask R-CNN locates the grid and corrects perspective; YOLOv8n locates "
            "the 81 cells on the rectified grid. Each cell is then read by the GridOCR "
            "CNN, and its per-cell probabilities drive constraint recovery when the "
            "first solve attempt fails."
        ),
        requires=("detector", "yolo_extractor", "grid_ocr"),
        hint="Needs Mask R-CNN weights + YOLO cell weights + grid_ocr_cnn.pth",
        recommended=True,
    ),
    PipelinePath(
        key="yolowarp_gridocr",
        label="YOLO grid warp · YOLO cells · GridOCR CNN",
        ocr="grid_ocr",
        description=(
            "Same as the recommended path, but the grid is located by a ~6 MB "
            "YOLOv8n model instead of the 169 MB Mask R-CNN — seg or pose, "
            "selectable in the sidebar — and exports to TFLite for the Android "
            "port. The perspective warp is shared with the Mask R-CNN path, so "
            "only the detector differs."
        ),
        requires=("yolo_grid_detector", "yolo_extractor", "grid_ocr"),
        warp="yolo",
        hint="Needs YOLO grid weights (training/grid_seg or grid_pose) + YOLO cell weights + grid_ocr_cnn.pth",
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
        logger.warning("Failed to load %s from %s: %s", name, model_path, e)
        return None


@dataclass
class PipelineResult:
    """Structured result from running the pipeline.

    Fields are None when their step failed; check `errors` for the reason.
    """
    original_grid: np.ndarray | None      # detected puzzle (clues only)
    solved_grid: np.ndarray | None        # complete solution
    timing: dict[str, float] = field(default_factory=dict)
    grid_image: np.ndarray | None = None  # rectified grid (maskrcnn_yolo) or raw (yolo)
    errors: dict[str, str] = field(default_factory=dict)  # step -> error message
    seg_grid_image: np.ndarray | None = None      # segmentation overlay (Step 1)
    seg_cells_image: np.ndarray | None = None     # 9×9 cell mosaic (Step 2)
    recognition_image: np.ndarray | None = None   # seg image + digit overlay (Step 3)


class SudokuPipeline:
    """Orchestrates the full image-to-solution pipeline.

    Segmentation is fixed: Mask R-CNN corrects perspective, then YOLO locates
    the 81 cells on the rectified grid.  A run selects only the digit reader,
    and the valid choices are enumerated in `PIPELINE_PATHS`.  Every optional
    component loads independently, so a pipeline built without, say, the YOLO
    digit weights still serves every path that does not need them — query
    `available_paths()` before calling `run()`.

    OCR modes:
        "grid_ocr"      GridOCR CNN over each detected cell
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.cfg = config or PipelineConfig()
        self.solver = SudokuSolver()
        device = self.cfg.effective_device

        def _build_detector():
            from .grid_detector import GridDetector
            return GridDetector(self.cfg.grid_detector, device=device)

        self.detector = _try_load(
            "GridDetector", self.cfg.grid_detector.model_path, _build_detector
        )

        def _build_yolo_grid_detector():
            from .yolo_grid_detector import YoloGridDetector
            return YoloGridDetector(self.cfg.yolo_grid_detector, device=device)

        self.yolo_grid_detector = _try_load(
            "YoloGridDetector", self.cfg.yolo_grid_detector.model_path,
            _build_yolo_grid_detector,
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

        ready = [p.key for p in self.available_paths()]
        logger.info("device=%s | paths: %s", device, ", ".join(ready) or "none")

    @staticmethod
    def _make_recognition_image(
        base: np.ndarray,
        puzzle: np.ndarray,
        boxes_px: np.ndarray | None = None,
    ) -> np.ndarray:
        """Overlay recognised digits on base_image.

        If boxes_px is supplied digits are centred inside each cell box;
        otherwise they are placed on a uniform 9×9 grid (GridOCR's own split).
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
    def _canonical_cells(
        rectified: np.ndarray, boxes_px: np.ndarray, patch: int
    ) -> tuple[list[np.ndarray | None], np.ndarray]:
        """Cut one image per cell from the rectified grid, at the reader's scale.

        Each cell is a `patch`x`patch` window centred on its YOLO box, taken
        from the grid rescaled so that one cell measures `patch` px.  Sampling
        at that single canonical scale (rather than resizing each box crop
        individually) matters: GridOCRNet learned 50 px cells cut from a 450 px
        grid, and independently rescaling tight boxes shifts how much of the
        cell the digit fills.  Measured over 90 held-out photos, per-box
        rescaling solved 68 and canonical sampling 74.

        Returns (81 crops row-major, the rescaled grid used for contrast).
        """
        G = patch * 9
        scaled = cv2.resize(rectified, (G, G), interpolation=cv2.INTER_AREA)
        sx = G / rectified.shape[1]
        sy = G / rectified.shape[0]
        cells: list[np.ndarray | None] = []
        for i in range(81):
            x1, y1, x2, y2 = boxes_px[i]
            if x2 <= x1 or y2 <= y1:            # cell the detector missed
                r, c = divmod(i, 9)
                x0, y0 = c * patch, r * patch
            else:
                cx = (x1 + x2) / 2 * sx
                cy = (y1 + y2) / 2 * sy
                x0 = max(0, min(G - patch, int(round(cx - patch / 2))))
                y0 = max(0, min(G - patch, int(round(cy - patch / 2))))
            cells.append(scaled[y0:y0 + patch, x0:x0 + patch])
        return cells, scaled

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

    def path_available(self, path: PipelinePath) -> bool:
        """True when every component this path needs is loaded."""
        return all(getattr(self, attr, None) is not None for attr in path.requires)

    def available_paths(self) -> list[PipelinePath]:
        """Paths from PIPELINE_PATHS whose weights are all present, best first."""
        return [p for p in PIPELINE_PATHS if self.path_available(p)]

    def unavailable_paths(self) -> list[PipelinePath]:
        """Paths that are missing at least one component."""
        return [p for p in PIPELINE_PATHS if not self.path_available(p)]

    def available_ocr_modes(self) -> list[str]:
        """Digit readers whose weights are loaded."""
        return ["grid_ocr"] if self.grid_ocr is not None else []

    def run_path(self, image: np.ndarray | str, path: PipelinePath) -> PipelineResult:
        """Run a `PipelinePath` — the mode pair the UI and CLI select from."""
        return self.run(image, ocr_mode=path.ocr, warp=path.warp)

    def run(
        self,
        image: np.ndarray | str,
        ocr_mode: str = "grid_ocr",
        warp: str = "maskrcnn",
    ) -> PipelineResult:
        """Run the full pipeline on an image path or numpy array.

        Args:
            image:    File path (str) or RGB numpy array.
            ocr_mode: "grid_ocr"     — GridOCR CNN over the cells YOLO located
            warp:     "maskrcnn"     — Mask R-CNN locates the grid (default)
                      "yolo"         — YOLOv8n pose/seg locates the grid

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
        yolo_labels: np.ndarray | None = None
        boxes_px: np.ndarray | None = None
        seg_grid_image = None
        seg_cells_image = None
        recognition_image = None

        # ══ Step 1: Segmentation ══════════════════════════════════════════════
        try:
            if warp == "yolo":
                detector = self.yolo_grid_detector
                if detector is None:
                    raise RuntimeError(
                        "YOLO grid detector weights not found. "
                        "Run training/grid_pose/train.py first."
                    )
            elif warp == "maskrcnn":
                detector = self.detector
                if detector is None:
                    raise RuntimeError("Mask R-CNN grid detector weights not found")
            else:
                raise RuntimeError(
                    f"Unknown warp mode {warp!r}. Use 'maskrcnn' or 'yolo'."
                )
            if self.yolo_extractor is None:
                raise RuntimeError("YOLO cell extractor weights not found")
            t = time.perf_counter()
            # Both detectors expose the same detect_debug contract, so every
            # downstream step is identical regardless of which located the grid.
            rectified, seg_grid_image = detector.detect_debug(image)
            timings["grid_detection"] = time.perf_counter() - t
            t = time.perf_counter()
            _crops, boxes_px, yolo_labels = self.yolo_extractor.extract(rectified)
            timings["yolo_cell_extraction"] = time.perf_counter() - t
            from .yolo_cell_extractor import YoloCellExtractor
            # Step 2 visual: YOLO boxes drawn on the warped grid
            seg_cells_image = YoloCellExtractor.seg_overlay(rectified, boxes_px, yolo_labels)

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
            if ocr_mode != "grid_ocr":
                raise RuntimeError(
                    f"Unknown OCR mode {ocr_mode!r}. Only 'grid_ocr' is supported."
                )
            if self.grid_ocr is None:
                raise RuntimeError("GridOCR model not available — check model weights")
            t = time.perf_counter()
            cells, scaled = self._canonical_cells(
                rectified, boxes_px, self.grid_ocr.cfg.patch_size
            )
            puzzle, ocr_probs = self.grid_ocr.read_cells(cells, contrast_ref=scaled)
            timings["grid_ocr"] = time.perf_counter() - t

            # GridOCR reads the cells YOLO located, so the digits are drawn
            # back into those same boxes.
            recognition_image = self._make_recognition_image(
                rectified, puzzle, boxes_px
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
            try:
                solution, solve_time = self.solver.solve(puzzle)
            except RuntimeError:
                if ocr_probs is not None:
                    puzzle, solution, solve_time = recover_with_constraints(
                        self.solver, puzzle, ocr_probs
                    )
                else:
                    raise
            # A photo is only genuinely solved if the clues we read determine
            # one answer.  When recognition misses too many digits the puzzle is
            # under-determined and the solver returns an arbitrary completion --
            # a confident wrong answer.  Measured over held-out roboflow images,
            # this was 14 of 74 apparent GridOCR "successes".
            t_u = time.perf_counter()
            ambiguous = self.solver.has_other_solution(puzzle, solution)
            timings["uniqueness_check"] = time.perf_counter() - t_u
            if ambiguous:
                raise RuntimeError(
                    f"Digit recognition was too incomplete: the {int((puzzle > 0).sum())} "
                    "clues read have more than one valid solution, so this is not "
                    "necessarily the puzzle in the image."
                )
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
        return recover_with_constraints(self.solver, puzzle, probs)

    @staticmethod
    def print_result(result: PipelineResult) -> None:
        SudokuSolver.print_grid(result.original_grid, "Detected Puzzle")
        SudokuSolver.print_grid(result.solved_grid, "Solution")
        print("\nTiming:")
        for k, v in result.timing.items():
            print(f"  {k}: {v:.4f}s")


def recover_with_constraints(
    solver: SudokuSolver,
    puzzle: np.ndarray,
    probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fix OCR errors by substituting alternative digit predictions.

    Tries 1–3 substitutions on the least-confident cells, then falls back to
    zeroing cells below a confidence threshold so the solver can fill them.
    """
    candidates: list[list[int]] = []
    top1_confs: list[float] = []
    flat = puzzle.flatten()
    for i in range(81):
        ranked = sorted(range(10), key=lambda d, i=i: -probs[i, d])
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
                    sol, _ = solver.solve(candidate_grid.copy())
                    if sol.min() > 0:
                        return candidate_grid, sol, time.perf_counter() - t0
                except RuntimeError:
                    pass

    for conf_threshold in (0.80, 0.70, 0.60):
        n_uncertain = sum(
            1 for i in range(81) if top1_confs[i] < conf_threshold and base[i] > 0
        )
        if n_uncertain > 6:
            continue
        confident_grid = base.copy()
        for i in range(81):
            if top1_confs[i] < conf_threshold:
                confident_grid[i] = 0
        try:
            sol, _ = solver.solve(confident_grid.reshape(9, 9).copy())
            if sol.min() > 0:
                return confident_grid.reshape(9, 9), sol, time.perf_counter() - t0
        except RuntimeError:
            pass

    raise RuntimeError(
        "No valid solution found after constraint recovery. "
        "The puzzle image may be unclear or the grid detection may have failed."
    )
