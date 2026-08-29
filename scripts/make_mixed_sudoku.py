"""
Generate half-filled Sudoku photos mixing printed and handwritten digits.

Source material is the Wicht & Hennebert dataset (data/wicht_sudoku): real phone
photos of newspaper puzzles, each with a .dat ground truth and grid corners in
outlines_sorted.csv. The published "mixed" variant fills *every* empty cell with
an MNIST digit, so no cell is left blank. This script instead fills only a
fraction of the empty cells, producing puzzles that still look like puzzles:
printed clues + a handful of handwritten entries + genuine empty cells.

Output mirrors the dataset layout so the result drops straight into
scripts/debug_pipeline.py:

    <out>/imageX.jpg    photo with pasted handwriting
    <out>/imageX.dat    ground truth (same format as the source .dat)
    <out>/imageX.json    which cells are printed / handwritten

Usage:
    uv run python scripts/make_mixed_sudoku.py --fill 0.35 --seed 0
"""

import argparse
import gzip
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "wicht_sudoku"
MNIST_RAW = ROOT / "training" / "sudoku_digit_classification" / "mnist_data" / "MNIST" / "raw"

INK = (150, 60, 30)  # BGR — blue ballpoint, matching the real handwritten samples


# ── inputs ───────────────────────────────────────────────────────────────────

def load_mnist() -> dict[int, list[np.ndarray]]:
    """Return {digit: [28x28 uint8, ...]} for digits 1-9 from the raw MNIST files."""

    def read(path: Path) -> bytes:
        gz = path.with_suffix(path.suffix + ".gz")
        if path.exists():
            return path.read_bytes()
        if gz.exists():
            return gzip.decompress(gz.read_bytes())
        raise SystemExit(f"MNIST file not found: {path}")

    images = np.frombuffer(read(MNIST_RAW / "train-images-idx3-ubyte"), np.uint8, offset=16)
    images = images.reshape(-1, 28, 28)
    labels = np.frombuffer(read(MNIST_RAW / "train-labels-idx1-ubyte"), np.uint8, offset=8)

    by_digit: dict[int, list[np.ndarray]] = {}
    for d in range(1, 10):
        by_digit[d] = list(images[labels == d][:2000])
    return by_digit


def load_outlines() -> dict[str, np.ndarray]:
    """Return {image stem: 4x2 float32 corners} from outlines_sorted.csv."""
    outlines = {}
    for line in (DATA / "outlines_sorted.csv").read_text().splitlines()[1:]:
        parts = line.strip().split(",")
        if len(parts) != 9:
            continue
        stem = Path(parts[0].strip('"')).stem
        pts = np.array(parts[1:], np.float32).reshape(4, 2)
        outlines[stem] = pts
    return outlines


def read_dat(path: Path) -> tuple[list[str], np.ndarray]:
    """Return (header lines, 9x9 int grid) for a dataset .dat file."""
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    header, rows = lines[:-9], lines[-9:]
    grid = np.array([[int(v) for v in r.split()] for r in rows], int)
    return header, grid


# ── rendering ────────────────────────────────────────────────────────────────

def render_digit(patch: np.ndarray, cell_px: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Scale an MNIST patch into a cell-sized (colour, alpha) pair."""
    size = int(cell_px * rng.uniform(0.55, 0.75))
    glyph = cv2.resize(patch, (size, size), interpolation=cv2.INTER_CUBIC)

    alpha = np.zeros((cell_px, cell_px), np.float32)
    off = (cell_px - size) // 2
    jitter = int(cell_px * 0.06)
    x = np.clip(off + rng.randint(-jitter, jitter), 0, cell_px - size)
    y = np.clip(off + rng.randint(-jitter, jitter), 0, cell_px - size)
    alpha[y:y + size, x:x + size] = glyph.astype(np.float32) / 255.0
    alpha = np.clip(alpha * 1.25, 0, 1)  # MNIST strokes are soft; ink is not

    colour = np.zeros((cell_px, cell_px, 3), np.float32)
    colour[:] = INK
    return colour, alpha


def paste_handwriting(image: np.ndarray, corners: np.ndarray, cells: dict[tuple[int, int], int],
                      mnist: dict[int, list[np.ndarray]], rng: random.Random) -> np.ndarray:
    """Draw digits into the given (row, col) cells, warped into the photo's perspective."""
    cell_px = 64
    side = cell_px * 9
    canvas_pts = np.array([[0, 0], [side, 0], [side, side], [0, side]], np.float32)
    to_image = cv2.getPerspectiveTransform(canvas_pts, corners.astype(np.float32))

    overlay = np.zeros((side, side, 3), np.float32)
    mask = np.zeros((side, side), np.float32)

    for (r, c), value in cells.items():
        patch = rng.choice(mnist[value])
        colour, alpha = render_digit(patch, cell_px, rng)
        y0, x0 = r * cell_px, c * cell_px
        overlay[y0:y0 + cell_px, x0:x0 + cell_px] = colour
        mask[y0:y0 + cell_px, x0:x0 + cell_px] = alpha

    h, w = image.shape[:2]
    warped = cv2.warpPerspective(overlay, to_image, (w, h))
    warped_mask = cv2.warpPerspective(mask, to_image, (w, h))[..., None]
    warped_mask = cv2.GaussianBlur(warped_mask, (3, 3), 0)[..., None]

    out = image.astype(np.float32) * (1 - warped_mask) + warped * warped_mask
    return np.clip(out, 0, 255).astype(np.uint8)


# ── main ─────────────────────────────────────────────────────────────────────

def solve(grid: np.ndarray) -> np.ndarray | None:
    """Fill a puzzle so handwritten digits stay consistent with the printed clues."""
    from sudoku_solver.sudoku_solver import SudokuSolver
    try:
        solved, _ = SudokuSolver().solve(grid.copy())
        return solved
    except Exception:
        return None


def run(src_dirs: list[Path], out_dir: Path, fill: float, seed: int, limit: int | None) -> None:
    import sys
    sys.path.insert(0, str(ROOT / "src"))

    rng = random.Random(seed)
    mnist = load_mnist()
    outlines = load_outlines()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for d in src_dirs for p in d.glob("*.jpg"))
    if limit:
        images = images[:limit]

    written = skipped = 0
    for path in images:
        dat = path.with_suffix(".dat")
        if not dat.exists() or path.stem not in outlines:
            skipped += 1
            continue

        header, clues = read_dat(dat)
        solution = solve(clues)
        if solution is None:
            skipped += 1
            continue

        empty = [(r, c) for r in range(9) for c in range(9) if clues[r, c] == 0]
        n = int(round(len(empty) * fill))
        chosen = rng.sample(empty, n)
        handwritten = {(r, c): int(solution[r, c]) for r, c in chosen}

        image = cv2.imread(str(path))
        if image is None:
            skipped += 1
            continue
        out_img = paste_handwriting(image, outlines[path.stem], handwritten, mnist, rng)

        grid = clues.copy()
        for (r, c), v in handwritten.items():
            grid[r, c] = v

        cv2.imwrite(str(out_dir / f"{path.stem}.jpg"), out_img)
        (out_dir / f"{path.stem}.dat").write_text(
            "\n".join(header + [" ".join(str(v) for v in row) for row in grid]) + "\n"
        )
        (out_dir / f"{path.stem}.json").write_text(json.dumps({
            "printed": [[r, c] for r in range(9) for c in range(9) if clues[r, c]],
            "handwritten": [[r, c] for r, c in chosen],
        }))
        written += 1

    print(f"Wrote {written} images to {out_dir} ({skipped} skipped)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", nargs="+", type=Path,
                    default=[DATA / "v2_test", DATA / "v2_train"],
                    help="source directories of imageX.jpg + imageX.dat")
    ap.add_argument("--out", type=Path, default=DATA / "half_mixed")
    ap.add_argument("--fill", type=float, default=0.35,
                    help="fraction of empty cells to fill with handwriting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.src, args.out, args.fill, args.seed, args.limit)


if __name__ == "__main__":
    main()
