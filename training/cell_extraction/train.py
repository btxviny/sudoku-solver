"""Train YOLOv8n on sudoku-cell-vision v6 (empty/filled detection)."""

import json
from pathlib import Path

from ultralytics import YOLO

DATASET_DIR = Path(__file__).parent / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
OUTPUT_DIR = Path(__file__).parent / "runs"

EPOCHS = 100
IMG_SIZE = 640
BATCH = 16
MODEL = "yolov8n.pt"


def main():
    model = YOLO(MODEL)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=str(OUTPUT_DIR),
        name="cell_vision_v6",
        exist_ok=True,
        device=0,
        workers=4,
        patience=20,
        save=True,
        plots=True,
        val=True,
        verbose=True,
    )

    metrics = {
        "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
        "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
    }

    metrics_path = OUTPUT_DIR / "cell_vision_v6" / "final_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("\n=== Final Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nBest weights: {OUTPUT_DIR}/cell_vision_v6/weights/best.pt")


if __name__ == "__main__":
    main()
