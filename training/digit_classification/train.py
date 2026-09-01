"""Train the YOLO classification model that reads sudoku digit values.

Classes: 0 = empty cell, 1-9 = digit value.

The trained weights land at the path `YoloDigitClassifierConfig` expects:
    training/digit_classification/runs/digit_cls/weights/best.pt
Once that file exists, the YOLO-digit
pipeline paths become available in the CLI and the Streamlit UI.

Usage:
    uv run python training/digit_classification/prepare_dataset.py
    uv run python training/digit_classification/train.py
"""

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
RUNS = HERE / "runs"
RUN_NAME = "digit_cls"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=DATASET)
    parser.add_argument("--model", default="yolov8n-cls.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=64,
                        help="Must match YoloDigitClassifierConfig.imgsz")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu")
    parser.add_argument("--name", default=RUN_NAME)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Dataset not found: {args.data}\n"
            "Run: uv run python training/digit_classification/prepare_dataset.py"
        )

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=str(RUNS),
        name=args.name,
        exist_ok=True,
        # Cell crops arrive from the detector with varying amounts of grid
        # border, scale and lighting, so translate/scale/erasing augmentation
        # matters more than colour.  Flips are disabled outright: a mirrored
        # digit is a different glyph (or nonsense), and rotation is kept small
        # because 6 and 9 are the same shape 180 degrees apart.
        degrees=8.0,
        translate=0.12,
        scale=0.25,
        shear=4.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.35,
        erasing=0.2,
        mosaic=0.0,
    )

    best = RUNS / args.name / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    print("Validating best checkpoint…")
    metrics = YOLO(str(best)).val(data=str(args.data), imgsz=args.imgsz, split="val")
    print(f"top-1 accuracy: {metrics.top1:.4f}")
    print(f"top-5 accuracy: {metrics.top5:.4f}")


if __name__ == "__main__":
    main()
