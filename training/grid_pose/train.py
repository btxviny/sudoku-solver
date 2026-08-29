"""Train YOLOv8n-pose to predict the 4 sudoku grid corners (TL, TR, BR, BL).

Replaces the 169 MB Mask R-CNN grid detector with a ~6 MB model that exports
cleanly to TFLite/LiteRT for the Android port.  Run prepare_dataset.py first.
"""

import json
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(__file__).resolve().parents[2] / "data/segmentation/pose_dataset/data.yaml"
OUTPUT_DIR = Path(__file__).parent / "runs"
RUN_NAME = "grid_pose_v1"

EPOCHS = 300
IMG_SIZE = 640
BATCH = 16
MODEL = "yolov8n-pose.pt"


def main():
    if not DATA_YAML.exists():
        raise SystemExit(
            f"{DATA_YAML} missing -- run training/grid_pose/prepare_dataset.py first"
        )

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
        # Corner regression is far more sensitive to localisation than the
        # empty/filled detector is, so the keypoint loss carries most of the
        # weight and rotation augmentation stays modest -- the model only ever
        # sees roughly upright photos, and large rotations make the canonical
        # TL/TR/BR/BL assignment ambiguous.
        pose=16.0,
        degrees=10.0,
        scale=0.5,
        translate=0.1,
        shear=3.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        close_mosaic=25,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        save=True,
        plots=True,
        val=True,
        verbose=True,
    )

    d = results.results_dict
    metrics = {
        "pose_mAP50": float(d.get("metrics/mAP50(P)", 0)),
        "pose_mAP50-95": float(d.get("metrics/mAP50-95(P)", 0)),
        "box_mAP50": float(d.get("metrics/mAP50(B)", 0)),
        "box_mAP50-95": float(d.get("metrics/mAP50-95(B)", 0)),
        "precision(P)": float(d.get("metrics/precision(P)", 0)),
        "recall(P)": float(d.get("metrics/recall(P)", 0)),
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
