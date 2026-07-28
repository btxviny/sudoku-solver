#!/usr/bin/env python3
"""Train all sudoku solver models. Run from the repository root:

    uv run python train_all_models.py [--steps grid_ocr segmentation]
"""

import os
import sys
import subprocess
import argparse


def _run(script: str, *extra_args: str) -> bool:
    try:
        subprocess.run([sys.executable, script, *extra_args], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAIL: {script} — {e}")
        return False


def train_grid_ocr(steps: list[str]) -> bool:
    if "all" not in steps and "grid_ocr" not in steps:
        return True
    print("=" * 60)
    print("STEP 1: TRAINING GRIDOCRNET DIGIT CLASSIFIER")
    print("=" * 60)
    return _run(
        "training/grid_ocr/scripts/train_cell_classifier.py",
        "--epochs", "50",
        "--batch_size", "128",
        "--lr", "1e-3",
        "--synthetic_size", "60000",
    )


def train_segmentation(steps: list[str]) -> bool:
    if "all" not in steps and "segmentation" not in steps:
        return True
    print("=" * 60)
    print("STEP 2: TRAINING MASK R-CNN GRID DETECTOR")
    print("=" * 60)
    return _run(
        "training/segmentation/scripts/train_maskrcnn.py",
        "--data_root", "data/segmentation/segmentation_dataset",
        "--output_dir", "models/weights",
        "--num_epochs", "10",
        "--batch_size", "2",
    )


def main():
    parser = argparse.ArgumentParser(description="Train sudoku solver models")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["grid_ocr", "segmentation", "all"],
        default=["all"],
        help="Which models to train (default: all)",
    )
    args = parser.parse_args()

    print("SUDOKU SOLVER — MODEL TRAINING")
    print(f"Working directory: {os.getcwd()}")
    print(f"Steps: {args.steps}\n")

    tasks = [
        ("GridOCRNet digit classifier", train_grid_ocr),
        ("Mask R-CNN grid detector", train_segmentation),
    ]

    passed = sum(fn(args.steps) for _, fn in tasks)
    total = len(tasks)

    print("=" * 60)
    print(f"Results: {passed}/{total} successful")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
