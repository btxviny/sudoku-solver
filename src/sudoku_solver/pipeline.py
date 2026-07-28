"""End-to-end sudoku solver pipeline."""

from dataclasses import dataclass, field
from typing import Sequence
import cv2
import numpy as np
import time

from .config import PipelineConfig
from .grid_detector import GridDetector
from .cell_extractor import CellExtractor
from .digit_classifier import DigitClassifier
from .sudoku_solver import SudokuSolver

# GridOCR is imported lazily so the pipeline still works when the model
# file doesn't exist yet (before training).
_grid_ocr_available: bool | None = None


def _try_load_grid_ocr(config: "PipelineConfig"):
    """Return a GridOCR instance if the model file exists, else None."""
    global _grid_ocr_available
    from pathlib import Path
    from .config import GridOCRConfig
    model_path = Path(config.grid_ocr.model_path)
    if not model_path.exists():
        if _grid_ocr_available is not False:
            print(f"[pipeline] GridOCR model not found at {model_path}. "
                  "Falling back to CellExtractor+DigitClassifier. "
                  "Run training/grid_ocr/scripts/train_cell_classifier.py to train it.")
        _grid_ocr_available = False
        return None
    try:
        from .grid_ocr import GridOCR
        ocr = GridOCR(config.grid_ocr)
        _grid_ocr_available = True
        return ocr
    except Exception as e:
        print(f"[pipeline] Failed to load GridOCR: {e}. Using fallback.")
        _grid_ocr_available = False
        return None


@dataclass
class PipelineResult:
    """Structured result from running the pipeline."""
    original_grid: np.ndarray
    solved_grid: np.ndarray
    timing: dict[str, float] = field(default_factory=dict)
    grid_image: np.ndarray | None = None


class SudokuPipeline:
    """Orchestrates the full image-to-solution pipeline.

    Digit reading strategy (in order of preference):
      1. GridOCR CNN  – if models/weights/grid_ocr_cnn.pth exists.
         A single lightweight CNN that reads all 81 cells at once.
         Train it with: python training/grid_ocr/scripts/train_cell_classifier.py
      2. CellExtractor + DigitClassifier (ResNet18 + XGBoost) fallback.
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.cfg = config or PipelineConfig()
        self.detector = GridDetector(self.cfg.grid_detector)
        self.solver = SudokuSolver()

        self.grid_ocr = _try_load_grid_ocr(self.cfg)
        if self.grid_ocr is None:
            self.extractor = CellExtractor(self.cfg.cell_extractor)
            self.classifier = DigitClassifier(self.cfg.digit_classifier)
        else:
            self.extractor = None
            self.classifier = None

    def run(self, image: np.ndarray | str) -> PipelineResult:
        """Run the full pipeline on an image path or numpy array.

        Args:
            image: File path (str) or RGB numpy array.

        Returns:
            PipelineResult with detected grid, solved grid, and timing.
        """
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        if isinstance(image, str):
            raw = cv2.imread(image)
            if raw is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
            image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        # Grid detection + perspective warp
        t = time.perf_counter()
        rectified = self.detector.detect(image)
        timings["grid_detection"] = time.perf_counter() - t

        # Digit reading
        t = time.perf_counter()
        if self.grid_ocr is not None:
            puzzle, ocr_probs = self.grid_ocr.read_with_probs(rectified)
            timings["digit_reading"] = time.perf_counter() - t
        else:
            cells = self.extractor.extract(rectified)
            timings["cell_extraction"] = time.perf_counter() - t
            t = time.perf_counter()
            puzzle = self.classifier.classify_grid(cells)
            ocr_probs = None
            timings["digit_classification"] = time.perf_counter() - t

        # Constraint solving (with OCR-confidence-guided recovery when needed)
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

        timings["total"] = time.perf_counter() - t0
        return PipelineResult(
            original_grid=puzzle,
            solved_grid=solution,
            timing=timings,
            grid_image=rectified,
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

        # Build per-cell candidate lists sorted by probability (skip p<0.03)
        candidates: list[list[int]] = []
        top1_confs: list[float] = []
        flat = puzzle.flatten()
        for i in range(81):
            ranked = sorted(range(10), key=lambda d: -probs[i, d])
            top1_confs.append(float(probs[i, ranked[0]]))
            # Keep up to 3 alternatives whose probability is at least 3 %
            # Never substitute 0 (empty) for a cell predicted as a digit —
            # removing a clue can let the solver find a wrong-but-consistent solution.
            alts = [d for d in ranked[1:4] if probs[i, d] >= 0.03]
            if flat[i] > 0:
                alts = [d for d in alts if d != 0]
            candidates.append(alts)

        # Cells that have alternatives and are not highly confident (conf < 0.60)
        uncertain_idx = [
            i for i in range(81)
            if candidates[i] and top1_confs[i] < 0.60
        ]
        # Sort uncertain cells by confidence ascending (most uncertain first)
        uncertain_idx.sort(key=lambda i: top1_confs[i])

        # Try flipping 1, then 2 uncertain cells
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

        # Tier 2: fall back to using only high-confidence cells as clues and
        # letting the solver determine everything else via constraint propagation.
        # Only activates when few cells are uncertain — with too many unknowns
        # the solver may find a valid-but-wrong Sudoku instead of the intended one.
        for conf_threshold in (0.80, 0.70, 0.60):
            n_uncertain = sum(1 for i in range(81) if top1_confs[i] < conf_threshold and base[i] > 0)
            if n_uncertain > 6:
                continue  # too many uncertain filled cells → solution not uniquely constrained
            confident_grid = base.copy()
            for i in range(81):
                if top1_confs[i] < conf_threshold:
                    confident_grid[i] = 0  # mark uncertain cells as unknown
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