"""Build a YOLO-classification dataset of sudoku cell crops.

Source: data/grid_ocr/cells/<label>/<hash>.jpg — real 50x50 cell crops taken
from the Roboflow sudoku datasets and labelled by EasyOCR with solver
verification (see training/grid_ocr/scripts/extract_and_label_cells.py).

These are the same crops GridOCR learned from, and GridOCR is the only digit
reader that currently works end-to-end.  The 120k-image synthetic/MNIST set in
data/digit_classification/digits is deliberately NOT used: it is what the
ResNet18+XGBoost classifier was trained on, and that classifier fails on every
real photo in test_images/.

Output layout (what `yolo classify train` expects):

    training/digit_classification/dataset/
        train/0/*.jpg ... train/9/*.jpg
        val/0/*.jpg   ... val/9/*.jpg

Usage:
    uv run python training/digit_classification/prepare_dataset.py
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT / "data" / "grid_ocr" / "cells"
OUTPUT_DIR = Path(__file__).resolve().parent / "dataset"

# Empty cells outnumber every digit class ~14:1.  Left unchecked the model can
# score ~62% by answering "empty" for everything, and the loss barely rewards
# the digits we actually care about.  Cap class 0 at a small multiple of the
# mean digit-class size instead of discarding it entirely: the classifier still
# needs to recognise an empty cell, because the cell detector's empty/filled
# call is not always right.
EMPTY_CLASS = "0"
EMPTY_RATIO = 2.0     # keep at most 2x the mean digit-class count


def collect(source_dir: Path) -> dict[str, list[Path]]:
    """Map class name -> image paths, for the 10 class directories."""
    per_class: dict[str, list[Path]] = {}
    for label in sorted(source_dir.iterdir()):
        if not label.is_dir():
            continue
        images = sorted(
            p for p in label.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if images:
            per_class[label.name] = images
    return per_class


def cap_empty_class(per_class: dict[str, list[Path]], rng: random.Random) -> None:
    """Downsample the empty class in place to EMPTY_RATIO x mean digit count."""
    digit_counts = [len(v) for k, v in per_class.items() if k != EMPTY_CLASS]
    if not digit_counts or EMPTY_CLASS not in per_class:
        return
    mean_digits = sum(digit_counts) / len(digit_counts)
    cap = int(mean_digits * EMPTY_RATIO)
    empties = per_class[EMPTY_CLASS]
    if len(empties) > cap:
        per_class[EMPTY_CLASS] = rng.sample(empties, cap)


def build(source_dir: Path, output_dir: Path, val_frac: float, seed: int) -> None:
    if not source_dir.exists():
        raise SystemExit(
            f"Source dataset not found: {source_dir}\n"
            "Run training/grid_ocr/scripts/extract_and_label_cells.py first."
        )

    rng = random.Random(seed)
    per_class = collect(source_dir)
    if not per_class:
        raise SystemExit(f"No class directories with images under {source_dir}")

    cap_empty_class(per_class, rng)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for label, images in sorted(per_class.items()):
        images = list(images)
        rng.shuffle(images)
        # At least one validation image per class, and never the whole class.
        n_val = min(len(images) - 1, max(1, round(len(images) * val_frac)))
        splits = {"val": images[:n_val], "train": images[n_val:]}
        for split, paths in splits.items():
            dest = output_dir / split / label
            dest.mkdir(parents=True, exist_ok=True)
            for p in paths:
                shutil.copy2(p, dest / p.name)
            counts[split][label] = len(paths)

    print(f"Dataset written to {output_dir}\n")
    header = f"{'class':>6} {'train':>8} {'val':>6}"
    print(header)
    print("-" * len(header))
    for label in sorted(per_class):
        print(f"{label:>6} {counts['train'].get(label, 0):>8} {counts['val'].get(label, 0):>6}")
    print("-" * len(header))
    print(f"{'total':>6} {sum(counts['train'].values()):>8} {sum(counts['val'].values()):>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(args.source, args.output, args.val_frac, args.seed)


if __name__ == "__main__":
    main()
