#!/usr/bin/env python3
"""
Master training script for all sudoku solver models.
Run from the repository root:  python train_all_models.py
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def _run(script: str, *extra_args: str) -> bool:
    """Run a training script and return success."""
    try:
        subprocess.run(
            [sys.executable, script, *extra_args],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAIL: {script} — {e}")
        return False


def create_hybrid_dataset(steps: list[str]) -> bool:
    if "all" not in steps and "dataset" not in steps:
        return True
    print("=" * 60)
    print("STEP 1: CREATING HYBRID DATASET (Synthetic + MNIST)")
    print("=" * 60)
    labels = Path("data/digit_classification/digits/labels.txt")
    if labels.exists():
        print("Dataset already exists, skipping.")
        return True
    return _run("training/digit_classification/scripts/create_mnsit_printed_dataset.py")


def extract_resnet_features(steps: list[str]) -> bool:
    if "all" not in steps and "features" not in steps:
        return True
    print("=" * 60)
    print("STEP 2: EXTRACTING RESNET18 FEATURES")
    print("=" * 60)
    return _run("training/digit_classification/scripts/create_resnet_feature_dataset.py")


def train_digit_classifier(steps: list[str]) -> bool:
    if "all" not in steps and "digit_classifier" not in steps:
        return True
    print("=" * 60)
    print("STEP 3: TRAINING DIGIT CLASSIFICATION (XGBoost)")
    print("=" * 60)
    return _run("training/digit_classification/scripts/train_xgboost_classifier.py")


def train_segmentation(steps: list[str]) -> bool:
    if "all" not in steps and "segmentation" not in steps:
        return True
    print("=" * 60)
    print("STEP 4: TRAINING SEGMENTATION (Mask R-CNN)")
    print("=" * 60)
    return _run(
        "training/segmentation/scripts/train_maskrcnn.py",
        "--data_root", "data/segmentation/segmentation_dataset",
        "--output_dir", "models/weights",
        "--num_epochs", "10",
        "--batch_size", "2",
    )


def main():
    parser = argparse.ArgumentParser(description="Train all sudoku solver models")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["dataset", "features", "digit_classifier", "segmentation", "all"],
        default=["all"],
        help="Which steps to run",
    )
    args = parser.parse_args()

    print("SUDOKU SOLVER MODEL TRAINING")
    print(f"Working directory: {os.getcwd()}")
    print(f"Steps: {args.steps}")

    tasks = [
        ("Dataset", create_hybrid_dataset),
        ("ResNet features", extract_resnet_features),
        ("XGBoost classifier", train_digit_classifier),
        ("Mask R-CNN segmentation", train_segmentation),
    ]

    passed = 0
    total = 0
    for name, fn in tasks:
        total += 1
        if fn(args.steps):
            passed += 1
            print(f"OK: {name}\n")
        else:
            print(f"FAIL: {name}\n")

    print("=" * 60)
    print(f"Results: {passed}/{total} successful")
    if passed == total:
        print("All training completed!")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()