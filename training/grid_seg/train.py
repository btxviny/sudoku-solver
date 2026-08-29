"""Train YOLOv8n-seg on the sudoku grid masks.

The segmentation counterpart to grid_pose: it predicts a grid mask rather than
four corners, so the existing `_corners_from_mask` + `_refine_on_grid_lines`
Hough snapping in GridDetector still applies.  Slower and more code to port
than the pose model, but it keeps the mask-then-snap behaviour that the current
Mask R-CNN pipeline depends on.

Trains directly on data/segmentation/segmentation_dataset -- already YOLOv8
polygon format, no conversion needed.
"""

import json
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = (
    Path(__file__).resolve().parents[2]
    / "data/segmentation/segmentation_dataset/data.yaml"
)
OUTPUT_DIR = Path(__file__).parent / "runs"
RUN_NAME = "grid_seg_v1"

EPOCHS = 300
IMG_SIZE = 640
BATCH = 16
MODEL = "yolov8n-seg.pt"


def main():
    model = YOLO(MODEL)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        device=0,
        workers=8,
        patience=50,
        degrees=10.0,
        scale=0.5,
        translate=0.1,
        shear=3.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        close_mosaic=25,
        save=True,
        plots=True,
        val=True,
        verbose=True,
    )

    d = results.results_dict
    metrics = {
        "mask_mAP50": float(d.get("metrics/mAP50(M)", 0)),
        "mask_mAP50-95": float(d.get("metrics/mAP50-95(M)", 0)),
        "box_mAP50": float(d.get("metrics/mAP50(B)", 0)),
        "box_mAP50-95": float(d.get("metrics/mAP50-95(B)", 0)),
    }

    out = OUTPUT_DIR / RUN_NAME / "final_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))

    print("\n=== Final Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nBest weights: {OUTPUT_DIR / RUN_NAME / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
