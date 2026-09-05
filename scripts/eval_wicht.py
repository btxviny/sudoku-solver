"""
Score the digit readers against the Wicht & Hennebert ground truth (.dat files).

debug_pipeline.py reports where the pipeline *breaks*; this reports how often it
is *right*, which needs labels. Any directory of imageX.jpg + imageX.dat works.
If a sibling imageX.json is present (written by make_mixed_sudoku.py) accuracy is
also broken down by printed vs handwritten cells.

Every reader named with --ocr is scored on the *same* rectified grids and the
same cell crops, in one pass.  That makes the comparison paired: a difference in
the numbers is a difference between the networks, never between two runs of the
grid detector.

Usage:
    uv run python scripts/eval_wicht.py data/wicht_sudoku/half_mixed_test
    uv run python scripts/eval_wicht.py data/wicht_sudoku/v2_test --ocr cell_ocr
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sudoku_solver.config import PipelineConfig
from sudoku_solver.pipeline import SudokuPipeline, recover_with_constraints
from sudoku_solver.sudoku_solver import SudokuSolver
from sudoku_solver.yolo_grid_detector import YoloGridDetector
from sudoku_solver.yolo_cell_extractor import YoloCellExtractor


def read_dat(path: Path) -> np.ndarray:
    rows = [ln for ln in path.read_text().splitlines() if ln.strip()][-9:]
    return np.array([[int(v) for v in r.split()] for r in rows], int)


def build_readers(names: list[str] | None, cfg: PipelineConfig) -> dict:
    """Instantiate the requested readers, skipping any whose weights are absent."""
    from sudoku_solver.cell_ocr import CellOCR
    from sudoku_solver.grid_ocr import GridOCR

    factories = {
        "grid_ocr": (cfg.grid_ocr, GridOCR),
        "cell_ocr": (cfg.cell_ocr, CellOCR),
    }
    wanted = names or list(factories)
    readers = {}
    for name in wanted:
        if name not in factories:
            raise SystemExit(f"Unknown reader {name!r}; choose from {list(factories)}")
        conf, cls = factories[name]
        if not conf.model_path.exists():
            print(f"skipping {name}: no weights at {conf.model_path}")
            continue
        readers[name] = cls(conf)
    if not readers:
        raise SystemExit("No readers available — train one first.")
    return readers


class Score:
    """Running tallies for one reader."""

    def __init__(self) -> None:
        self.perfect = 0
        self.solved = 0
        self.cell_hit = 0
        self.cell_total = 0
        self.split = {"printed": [0, 0], "handwritten": [0, 0], "empty": [0, 0]}

    def add(self, pred: np.ndarray, truth: np.ndarray, meta: dict | None,
            solver: SudokuSolver) -> None:
        correct = pred == truth
        self.cell_hit += int(correct.sum())
        self.cell_total += 81
        self.perfect += int(correct.all())
        for r in range(9):
            for c in range(9):
                if meta is None:
                    kind = "printed" if truth[r, c] else "empty"
                elif [r, c] in meta["handwritten"]:
                    kind = "handwritten"
                elif [r, c] in meta["printed"]:
                    kind = "printed"
                else:
                    kind = "empty"
                self.split[kind][0] += int(correct[r, c])
                self.split[kind][1] += 1

    def report(self, name: str, scored: int) -> None:
        print(f"\n{name}")
        print("-" * 56)
        if not scored:
            print("  nothing scored")
            return
        print(f"  grids read perfectly : {self.perfect}/{scored}  "
              f"({self.perfect / scored * 100:.1f}%)")
        print(f"  grids solved         : {self.solved}/{scored}  "
              f"({self.solved / scored * 100:.1f}%)   [after constraint recovery]")
        print(f"  cell accuracy        : {self.cell_hit}/{self.cell_total}  "
              f"({self.cell_hit / self.cell_total * 100:.2f}%)")
        for kind, (hit, tot) in self.split.items():
            if tot:
                print(f"    {kind:<16} {hit}/{tot}  ({hit / tot * 100:.2f}%)")


def solves(puzzle: np.ndarray, probs: np.ndarray, truth: np.ndarray,
           solver: SudokuSolver) -> bool:
    """Whether this read yields the puzzle's true solution, recovery included.

    The pipeline's own success test: solve the digits as read, fall back to
    constraint recovery, and require the answer to be the one the ground truth
    determines.  A grid that solves to something else is a confident wrong
    answer, and is counted as a failure here.
    """
    try:
        truth_solution, _ = solver.solve(truth.astype(np.uint8).copy())
    except RuntimeError:
        return False
    try:
        solution, _ = solver.solve(puzzle.astype(np.uint8).copy())
    except RuntimeError:
        try:
            _, solution, _ = recover_with_constraints(solver, puzzle, probs)
        except RuntimeError:
            return False
    return bool(np.array_equal(solution, truth_solution))


def run(data_dir: Path, limit: int | None, ocr_names: list[str] | None) -> None:
    images = sorted(p for p in data_dir.glob("*.jpg") if p.with_suffix(".dat").exists())
    if limit:
        images = images[:limit]
    if not images:
        raise SystemExit(f"No labelled images (imageX.jpg + imageX.dat) in {data_dir}")

    cfg = PipelineConfig()
    detector = YoloGridDetector(cfg.yolo_grid_detector)
    readers = build_readers(ocr_names, cfg)
    solver = SudokuSolver()

    # Use the full pipeline path (YOLO cells → read_cells) if available, otherwise
    # fall back to the uniform split so the eval still runs without YOLO cell weights.
    try:
        extractor = YoloCellExtractor(cfg.yolo_cell_extractor)
    except Exception:
        extractor = None

    scores = {name: Score() for name in readers}
    n_detect_fail = 0

    for path in tqdm(images, desc="Scoring"):
        truth = read_dat(path.with_suffix(".dat"))
        raw = cv2.imread(str(path))
        if raw is None:
            n_detect_fail += 1
            continue
        try:
            rectified = detector.detect(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
            boxes_px = None
            if extractor is not None:
                _crops, boxes_px, _ = extractor.extract(rectified)
        except Exception:
            n_detect_fail += 1
            continue

        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None

        for name, reader in readers.items():
            try:
                if boxes_px is not None:
                    cells, scaled = SudokuPipeline._canonical_cells(
                        rectified, boxes_px, reader.patch_size
                    )
                    pred, probs = reader.read_cells(cells, contrast_ref=scaled)
                else:
                    pred, probs = reader.read_with_probs(rectified)
            except Exception:
                continue
            scores[name].add(pred, truth, meta, solver)
            scores[name].solved += int(solves(pred, probs, truth, solver))

    scored = len(images) - n_detect_fail
    print("\n" + "=" * 56)
    print(f"{data_dir}  ({len(images)} images)")
    print(f"grid detection failures : {n_detect_fail}")
    print("=" * 56)
    for name, score in scores.items():
        score.report(name, scored)
    print("=" * 56)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ocr", nargs="*", default=None,
                    help="readers to score (default: every one with weights)")
    args = ap.parse_args()
    run(args.data_dir, args.limit, args.ocr)


if __name__ == "__main__":
    main()
