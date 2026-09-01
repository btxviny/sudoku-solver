"""
Debug the full pipeline on the Roboflow training set.

Runs each stage independently so failures can be pinned to the exact step:
  1. Grid detection  (YOLOv8n seg/pose + warp)
  2. Digit reading   (GridOCR)
  3. Constraint check (detected clues valid before solving)
  4. Solving         (OR-Tools CP-SAT)

Saves annotated diagnostic images to  debug_output/<stage>/<image_name>.jpg
and prints a summary table at the end.

Usage:
    uv run python scripts/debug_pipeline.py [--data-dir PATH] [--limit N]
"""

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sudoku_solver.config import PipelineConfig
from sudoku_solver.yolo_grid_detector import YoloGridDetector
from sudoku_solver.sudoku_solver import SudokuSolver

# ── helpers ──────────────────────────────────────────────────────────────────

class Stage(str, Enum):
    OK = "ok"
    DETECTION = "detection_failure"
    OCR = "ocr_failure"
    INVALID_CLUES = "invalid_clues"
    SOLVE = "solve_failure"


@dataclass
class ImageResult:
    path: Path
    stage: Stage
    error: str = ""
    n_clues: int = 0
    grid_image: np.ndarray | None = field(default=None, repr=False)
    puzzle: np.ndarray | None = field(default=None, repr=False)


def load_digit_reader(cfg: PipelineConfig):
    """Return the GridOCR reader, which this script requires."""
    from pathlib import Path as P
    if not P(cfg.grid_ocr.model_path).exists():
        raise SystemExit(
            f"GridOCR weights not found: {cfg.grid_ocr.model_path}\n"
            "This script reads digits with GridOCR."
        )
    from sudoku_solver.grid_ocr import GridOCR
    return GridOCR(cfg.grid_ocr)


def read_digits(grid_ocr, rectified: np.ndarray):
    return grid_ocr.read_with_probs(rectified)


# ── visualisation ─────────────────────────────────────────────────────────────

def render_puzzle_rgb(puzzle: np.ndarray, highlight_mask: np.ndarray | None = None) -> np.ndarray:
    """Render a 9×9 numpy grid as an RGB image (270×270)."""
    cell = 30
    img = np.full((cell * 9, cell * 9, 3), 255, dtype=np.uint8)

    for r in range(9):
        for c in range(9):
            x0, y0 = c * cell, r * cell
            bg = (255, 200, 200) if (highlight_mask is not None and highlight_mask[r, c]) else (255, 255, 255)
            img[y0:y0+cell, x0:x0+cell] = bg
            val = int(puzzle[r, c])
            if val:
                cv2.putText(img, str(val), (x0 + 7, y0 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)

    for i in range(10):
        thickness = 2 if i % 3 == 0 else 1
        cv2.line(img, (i * cell, 0), (i * cell, cell * 9), (80, 80, 80), thickness)
        cv2.line(img, (0, i * cell), (cell * 9, i * cell), (80, 80, 80), thickness)

    return img


def conflict_mask(puzzle: np.ndarray) -> np.ndarray:
    """Return boolean mask (9×9) of cells involved in constraint violations."""
    bad = np.zeros((9, 9), bool)
    for i in range(9):
        row = puzzle[i]
        for v in set(row[row != 0]):
            if (row == v).sum() > 1:
                bad[i, row == v] = True
    for j in range(9):
        col = puzzle[:, j]
        for v in set(col[col != 0]):
            if (col == v).sum() > 1:
                bad[col == v, j] = True
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = puzzle[br:br+3, bc:bc+3]
            flat = box.flatten()
            for v in set(flat[flat != 0]):
                if (flat == v).sum() > 1:
                    bad[br:br+3, bc:bc+3] |= box == v
    return bad


def save_detection_failure(result: ImageResult, out_dir: Path) -> None:
    orig = cv2.cvtColor(cv2.imread(str(result.path)), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(orig)
    ax.set_title(f"Detection failure\n{result.error[:80]}", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{result.path.stem}.jpg", dpi=80)
    plt.close(fig)


def save_ocr_failure(result: ImageResult, out_dir: Path) -> None:
    panels = []
    if result.grid_image is not None:
        panels.append(("Warped grid", result.grid_image))
    if result.puzzle is not None:
        panels.append(("Detected digits", render_puzzle_rgb(result.puzzle)))

    orig = cv2.cvtColor(cv2.imread(str(result.path)), cv2.COLOR_BGR2RGB)
    panels = [("Original", orig)] + panels

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(f"OCR failure: {result.error[:100]}", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / f"{result.path.stem}.jpg", dpi=80)
    plt.close(fig)


def save_invalid_or_solve_failure(result: ImageResult, out_dir: Path) -> None:
    orig = cv2.cvtColor(cv2.imread(str(result.path)), cv2.COLOR_BGR2RGB)
    bad = conflict_mask(result.puzzle) if result.puzzle is not None else None
    puzzle_img = render_puzzle_rgb(result.puzzle, bad) if result.puzzle is not None else None

    panels = [("Original", orig)]
    if result.grid_image is not None:
        panels.append(("Warped grid", result.grid_image))
    if puzzle_img is not None:
        panels.append(("Detected clues\n(red=conflict)", puzzle_img))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    n_conflicts = int(bad.sum()) if bad is not None else 0
    fig.suptitle(
        f"{result.stage.value} | clues={result.n_clues} conflicts={n_conflicts}\n{result.error[:120]}",
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{result.path.stem}.jpg", dpi=80)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def run(data_dir: Path, limit: int | None, out_root: Path) -> None:
    images = sorted(data_dir.glob("*.jpg")) + sorted(data_dir.glob("*.png"))
    images = [p for p in images if not p.name.startswith("_")]  # skip annotation files
    if limit:
        images = images[:limit]

    print(f"Found {len(images)} images in {data_dir}")

    cfg = PipelineConfig()
    print("Loading models…")
    detector = YoloGridDetector(cfg.yolo_grid_detector)
    grid_ocr = load_digit_reader(cfg)
    solver = SudokuSolver()
    print("Digit reader: GridOCR\n")

    stage_dirs = {s: out_root / s.value for s in Stage}
    for d in stage_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    results: list[ImageResult] = []

    for img_path in tqdm(images, desc="Running pipeline"):
        raw = cv2.imread(str(img_path))
        if raw is None:
            results.append(ImageResult(img_path, Stage.DETECTION, error="cv2.imread returned None"))
            continue
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        res = ImageResult(path=img_path, stage=Stage.OK)

        # ── Stage 1: grid detection ───────────────────────────────────────
        try:
            rectified = detector.detect(image)
            res.grid_image = rectified
        except Exception as e:
            res.stage = Stage.DETECTION
            res.error = str(e)
            results.append(res)
            save_detection_failure(res, stage_dirs[Stage.DETECTION])
            continue

        # ── Stage 2: digit reading ────────────────────────────────────────
        try:
            puzzle, probs = read_digits(grid_ocr, rectified)
            res.puzzle = puzzle
            res.n_clues = int((puzzle != 0).sum())
        except Exception as e:
            res.stage = Stage.OCR
            res.error = str(e)
            results.append(res)
            save_ocr_failure(res, stage_dirs[Stage.OCR])
            continue

        # ── Stage 3: clue validity ────────────────────────────────────────
        if not SudokuSolver.is_valid(puzzle):
            res.stage = Stage.INVALID_CLUES
            bad = conflict_mask(puzzle)
            n_bad = int(bad.sum())
            res.error = f"{n_bad} cells in conflict"
            results.append(res)
            save_invalid_or_solve_failure(res, stage_dirs[Stage.INVALID_CLUES])
            continue

        # ── Stage 4: solving ──────────────────────────────────────────────
        try:
            solution, _ = solver.solve(puzzle)
            if solution.min() == 0:
                raise RuntimeError("Solver returned incomplete grid")
        except Exception as e:
            # Try recovery if we have probabilities
            if probs is not None:
                try:
                    from sudoku_solver.pipeline import SudokuPipeline
                    import time
                    pipe = SudokuPipeline.__new__(SudokuPipeline)
                    pipe.cfg = cfg
                    pipe.solver = solver
                    puzzle2, solution, _ = pipe._recover_with_constraints(puzzle, probs)
                    res.puzzle = puzzle2
                    # recovery succeeded → mark OK
                except Exception as e2:
                    res.stage = Stage.SOLVE
                    res.error = str(e2)
                    results.append(res)
                    save_invalid_or_solve_failure(res, stage_dirs[Stage.SOLVE])
                    continue
            else:
                res.stage = Stage.SOLVE
                res.error = str(e)
                results.append(res)
                save_invalid_or_solve_failure(res, stage_dirs[Stage.SOLVE])
                continue

        results.append(res)

    # ── summary ──────────────────────────────────────────────────────────────
    counts = {s: 0 for s in Stage}
    for r in results:
        counts[r.stage] += 1

    total = len(results)
    print("\n" + "=" * 60)
    print(f"{'STAGE':<22} {'COUNT':>6}  {'%':>6}")
    print("-" * 60)
    print(f"{'ok':<22} {counts[Stage.OK]:>6}  {counts[Stage.OK]/total*100:>5.1f}%")
    print(f"{'detection_failure':<22} {counts[Stage.DETECTION]:>6}  {counts[Stage.DETECTION]/total*100:>5.1f}%")
    print(f"{'ocr_failure':<22} {counts[Stage.OCR]:>6}  {counts[Stage.OCR]/total*100:>5.1f}%")
    print(f"{'invalid_clues':<22} {counts[Stage.INVALID_CLUES]:>6}  {counts[Stage.INVALID_CLUES]/total*100:>5.1f}%")
    print(f"{'solve_failure':<22} {counts[Stage.SOLVE]:>6}  {counts[Stage.SOLVE]/total*100:>5.1f}%")
    print("=" * 60)
    print(f"Total: {total}")

    # Per-failure detail
    failures = [r for r in results if r.stage != Stage.OK]
    if failures:
        print(f"\n{'IMAGE':<50} {'STAGE':<22} {'CLUES':>5}  ERROR")
        print("-" * 120)
        for r in failures:
            print(f"{r.path.name:<50} {r.stage.value:<22} {r.n_clues:>5}  {r.error[:60]}")

    # Save JSON summary
    summary = {
        "total": total,
        "counts": {s.value: counts[s] for s in Stage},
        "failures": [
            {
                "image": r.path.name,
                "stage": r.stage.value,
                "n_clues": r.n_clues,
                "error": r.error,
            }
            for r in failures
        ],
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nDiagnostic images → {out_root}/")
    print(f"JSON summary      → {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Debug pipeline on training images")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data/roboflow/sudoku-cell-vision/train",
        help="Directory of images to test",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap number of images")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "debug_output",
        help="Where to write diagnostic images",
    )
    args = parser.parse_args()
    run(args.data_dir, args.limit, args.out_dir)


if __name__ == "__main__":
    main()
