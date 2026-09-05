#!/usr/bin/env python3
"""Train all sudoku solver models. Run from the repository root:

    uv run python train_all_models.py
    uv run python train_all_models.py --steps grid_ocr cell_extraction

Steps (run in this order):
    grid_ocr          GridOCRNet CNN digit classifier, 1st gen (PyTorch, ~9 MB)
    cell_ocr          CellOCRNet SE-CNN digit classifier, 2nd gen (PyTorch, ~8 MB)
    cell_extraction   YOLOv8n cell detector - empty vs filled (YOLO, ~6 MB)
    grid_seg          YOLOv8n-seg grid locator (YOLO, ~13 MB)
    grid_pose         YOLOv8n-pose grid locator - 4 corners (YOLO, ~13 MB)
    digit_cls         YOLOv8n-cls digit classifier (YOLO, alternative OCR)

Each step prints its own progress. Weights land under training/<step>/runs/.
"""

import os
import sys
import subprocess
import argparse

ALL_STEPS = ["grid_ocr", "cell_ocr", "cell_extraction", "grid_seg", "grid_pose", "digit_cls"]


def _run(script: str, *extra_args: str) -> bool:
    try:
        subprocess.run([sys.executable, script, *extra_args], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAIL: {script} — {e}")
        return False


def train_grid_ocr(steps: list[str]) -> bool:
    if "grid_ocr" not in steps:
        return True
    print("=" * 60)
    print("STEP: GridOCRNet digit classifier")
    print("  Output: models/weights/grid_ocr_cnn.pth")
    print("=" * 60)
    return _run(
        "training/grid_ocr/scripts/train_cell_classifier.py",
        "--epochs", "50",
        "--batch_size", "128",
        "--lr", "1e-3",
        "--synthetic_size", "60000",
        "--handwritten_size", "40000",
    )


def train_cell_ocr(steps: list[str]) -> bool:
    if "cell_ocr" not in steps:
        return True
    print("=" * 60)
    print("STEP: CellOCRNet digit classifier (2nd generation)")
    print("  Extracting real cells first...")
    print("  Output: models/weights/cell_ocr_cnn.pth")
    print("=" * 60)
    # The real crops are cut by the grid and cell detectors, so this step needs
    # their weights to exist -- which is why it runs after them in ALL_STEPS
    # order only when the whole set is trained from nothing.
    if not _run("training/cell_ocr/extract_real_cells.py"):
        return False
    return _run("training/cell_ocr/train.py", "--epochs", "40")


def train_cell_extraction(steps: list[str]) -> bool:
    if "cell_extraction" not in steps:
        return True
    print("=" * 60)
    print("STEP: YOLOv8n cell extractor (empty/filled)")
    print("  Output: training/cell_extraction/runs/cell_vision_v6/weights/best.pt")
    print("=" * 60)
    return _run("training/cell_extraction/train.py")


def train_grid_seg(steps: list[str]) -> bool:
    if "grid_seg" not in steps:
        return True
    print("=" * 60)
    print("STEP: YOLOv8n-seg grid locator")
    print("  Output: training/grid_seg/runs/grid_seg_v1/weights/best.pt")
    print("=" * 60)
    return _run("training/grid_seg/train.py")


def train_grid_pose(steps: list[str]) -> bool:
    if "grid_pose" not in steps:
        return True
    print("=" * 60)
    print("STEP: YOLOv8n-pose grid locator (4 corner keypoints)")
    print("  Preparing dataset first...")
    print("  Output: training/grid_pose/runs/grid_pose_v1/weights/best.pt")
    print("=" * 60)
    if not _run("training/grid_pose/prepare_dataset.py"):
        return False
    return _run("training/grid_pose/train.py")


def train_digit_cls(steps: list[str]) -> bool:
    if "digit_cls" not in steps:
        return True
    print("=" * 60)
    print("STEP: YOLOv8n-cls digit classifier (alternative OCR)")
    print("  Preparing dataset first...")
    print("  Output: training/digit_classification/runs/digit_cls/weights/best.pt")
    print("=" * 60)
    if not _run("training/digit_classification/prepare_dataset.py"):
        return False
    return _run("training/digit_classification/train.py")


def main():
    parser = argparse.ArgumentParser(
        description="Train sudoku solver models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available steps: {', '.join(ALL_STEPS)}",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=ALL_STEPS,
        help="Which models to train (default: all)",
    )
    args = parser.parse_args()

    print("SUDOKU SOLVER — MODEL TRAINING")
    print(f"Working directory: {os.getcwd()}")
    print(f"Steps: {args.steps}\n")

    step_map = {
        "grid_ocr": ("GridOCRNet digit classifier", train_grid_ocr),
        "cell_ocr": ("CellOCRNet digit classifier", train_cell_ocr),
        "cell_extraction": ("YOLOv8n cell extractor", train_cell_extraction),
        "grid_seg": ("YOLOv8n-seg grid locator", train_grid_seg),
        "grid_pose": ("YOLOv8n-pose grid locator", train_grid_pose),
        "digit_cls": ("YOLOv8n-cls digit classifier", train_digit_cls),
    }

    results = []
    for step in args.steps:
        label, fn = step_map[step]
        ok = fn(args.steps)
        results.append((label, ok))

    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    print(f"Results: {passed}/{len(results)} successful")
    for label, ok in results:
        print(f"  {'OK' if ok else 'FAIL':4s}  {label}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
