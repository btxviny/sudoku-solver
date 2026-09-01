"""Build the OCR cell dataset from scratch — no real crops, no EasyOCR noise.

Sources (both guaranteed-correct labels):
  handwritten  All 54 k MNIST train images (digits 1-9) rendered into the
               sudoku-cell domain via _render_handwritten.
  synthetic    Equal count per class rendered with PIL system fonts via
               _render_digit.
  empty (0)    Synthetic only; count = mean digit-class size so the model
               sees a balanced distribution.

Wipes data/grid_ocr/cells/ before writing so there are no stale files.

Usage:
    uv run python training/grid_ocr/scripts/build_clean_dataset.py
"""
import hashlib
import importlib.util
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))

# Load rendering helpers from the training script
_train = PROJECT / "training" / "grid_ocr" / "scripts" / "train_cell_classifier.py"
spec = importlib.util.spec_from_file_location("tcc", _train)
tcc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tcc)

OUT_DIR = PROJECT / "data" / "grid_ocr" / "cells"
CELL_SIZE = tcc.CELL_SIZE


def save(img: np.ndarray, label: int, prefix: str) -> None:
    label_dir = OUT_DIR / str(label)
    label_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(img.tobytes()).hexdigest()[:12]
    cv2.imwrite(
        str(label_dir / f"{prefix}{h}.jpg"),
        img,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )


def main() -> None:
    # ── Wipe existing dataset ──────────────────────────────────────────────
    if OUT_DIR.exists():
        print(f"Removing {OUT_DIR} …")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    print("Cleared.\n")

    # ── MNIST handwritten cells ────────────────────────────────────────────
    print("Loading MNIST train split …")
    mnist_imgs, mnist_labels = tcc._load_mnist("train")
    by_class: dict[int, list[np.ndarray]] = {}
    for img, lbl in zip(mnist_imgs, mnist_labels):
        by_class.setdefault(int(lbl), []).append(img)

    counts: dict[int, int] = {}
    print("Rendering MNIST handwritten cells …")
    for label in range(1, 10):
        glyphs = by_class.get(label, [])
        for glyph in tqdm(glyphs, desc=f"hand {label}", leave=False):
            save(tcc._render_handwritten(glyph, CELL_SIZE), label, "h")
        counts[label] = len(glyphs)
        print(f"  class {label}: {len(glyphs):,} handwritten cells")

    # ── Synthetic PIL cells (same count per class as MNIST) ────────────────
    print("\nRendering synthetic (PIL fonts) cells …")
    for label in range(1, 10):
        n = counts[label]
        for _ in tqdm(range(n), desc=f"synth {label}", leave=False):
            save(tcc._render_digit(label, CELL_SIZE), label, "s")
        print(f"  class {label}: {n:,} synthetic cells")

    # ── Empty cells (class 0) — mean digit-class size ──────────────────────
    mean_count = int(sum(counts.values()) / len(counts))
    print(f"\nRendering {mean_count:,} empty cells (class 0) …")
    for _ in tqdm(range(mean_count), desc="empty 0", leave=False):
        save(tcc._render_digit(0, CELL_SIZE), 0, "s")

    # ── Report ─────────────────────────────────────────────────────────────
    print("\nFinal dataset:")
    total = 0
    for label in range(10):
        n = len(list((OUT_DIR / str(label)).glob("*.jpg")))
        total += n
        src = "hand+synth" if label > 0 else "synth only"
        print(f"  {label}: {n:,}  ({src})")
    print(f"  Total: {total:,}")


if __name__ == "__main__":
    main()
