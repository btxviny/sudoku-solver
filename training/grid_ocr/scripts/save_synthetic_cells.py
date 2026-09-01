"""Render synthetic (PIL fonts) and MNIST handwritten cells to disk.

Saves generated cells to data/grid_ocr/cells/<label>/ alongside the real
crops, using prefixed filenames (s<hash>.jpg for synthetic, h<hash>.jpg for
handwritten) so they never collide with real cells (no prefix / w prefix).

Usage:
    uv run python training/grid_ocr/scripts/save_synthetic_cells.py
    uv run python training/grid_ocr/scripts/save_synthetic_cells.py \
        --synthetic 8000 --handwritten 6000
"""
import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))

# Re-use the rendering helpers from the training script
_train = PROJECT / "training" / "grid_ocr" / "scripts" / "train_cell_classifier.py"
import importlib.util
spec = importlib.util.spec_from_file_location("tcc", _train)
tcc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tcc)

OUT_DIR = PROJECT / "data" / "grid_ocr" / "cells"
CELL_SIZE = tcc.CELL_SIZE


def save(img: np.ndarray, label: int, prefix: str) -> None:
    label_dir = OUT_DIR / str(label)
    label_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(img.tobytes()).hexdigest()[:12]
    path = label_dir / f"{prefix}{h}.jpg"
    if not path.exists():
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def generate_synthetic(n_per_class: int) -> None:
    print(f"Generating {n_per_class} synthetic cells per class (0–9)…")
    for label in range(10):
        for _ in tqdm(range(n_per_class), desc=f"synth class {label}", leave=False):
            img = tcc._render_digit(label, CELL_SIZE)
            save(img, label, "s")
    print("  Synthetic done.")


def generate_handwritten(n_per_class: int) -> None:
    print(f"Generating {n_per_class} MNIST handwritten cells per class (1–9)…")
    mnist_imgs, mnist_labels = tcc._load_mnist("train")
    by_class: dict[int, list[np.ndarray]] = {}
    for img, lbl in zip(mnist_imgs, mnist_labels):
        by_class.setdefault(int(lbl), []).append(img)

    import random
    for label in range(1, 10):
        pool = by_class.get(label, [])
        if not pool:
            print(f"  No MNIST samples for class {label}, skipping")
            continue
        for _ in tqdm(range(n_per_class), desc=f"hand  class {label}", leave=False):
            glyph = random.choice(pool)
            img = tcc._render_handwritten(glyph, CELL_SIZE)
            save(img, label, "h")
    print("  Handwritten done.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--synthetic", type=int, default=8_000,
                    help="Synthetic cells to generate per class (default 8000)")
    ap.add_argument("--handwritten", type=int, default=6_000,
                    help="Handwritten cells to generate per class 1-9 (default 6000)")
    args = ap.parse_args()

    generate_synthetic(args.synthetic)
    generate_handwritten(args.handwritten)

    print("\nFinal counts per label:")
    total = 0
    for label in range(10):
        n = len(list((OUT_DIR / str(label)).glob("*.jpg")))
        total += n
        print(f"  {label}: {n:,}")
    print(f"  Total: {total:,}")


if __name__ == "__main__":
    main()
