# Grid Pose — YOLOv8n-pose

Stage 1 (alternative): detect the sudoku grid and regress the four corner keypoints (TL, TR, BR, BL) directly. No mask post-processing needed — the corners drive the perspective warp directly.

This is simpler to port to mobile (no mask prototype decoding) and is the model embedded in the Android app (`grid_pose.tflite`).

## Dataset

**Source:** [Roboflow Universe — sudoku-lq9gj](https://universe.roboflow.com/sudoku-lq9gj) (CC BY 4.0)

The dataset is originally in segmentation format. `prepare_dataset.py` converts the polygon masks to pose-format keypoints.

```
data/segmentation/pose_dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/        # pose keypoints: x y visibility for each corner
└── valid/
    ├── images/
    └── labels/
```

## Training

```bash
# Convert segmentation labels to pose format (run once)
uv run python training/grid_pose/prepare_dataset.py

# Train
uv run python training/grid_pose/train.py
```

Config: `yolov8n-pose.pt`, 300 epochs, imgsz=640, batch=16, patience=50, device=0.

Key augmentation choices:
- `pose=16.0` — high keypoint loss weight (corner regression needs precision)
- `degrees=10.0` — small rotation only; large rotations make TL/TR/BR/BL assignment ambiguous
- `flipud=0.0` — no vertical flip for the same reason

Weights land at: `training/grid_pose/runs/grid_pose_v1/weights/best.pt`

## Results

| Metric | Value |
|--------|-------|
| Pose mAP@50 | ~99% |
| Mean corner error (90 held-out images) | **~5.7%** |
| Model size | ~13 MB |
| TFLite export | `grid_pose.tflite` — `[1,3,640,640]` → `[1,17,8400]` |

## Comparison with grid_seg

| Model | Mean corner error | Notes |
|-------|-------------------|-------|
| YOLOv8n-seg | 3.08% | Requires mask post-processing |
| YOLOv8n-pose | ~5.7% | Direct corner regression, simpler port |

The seg model is more accurate; the pose model is simpler to deploy. Both are exported to TFLite for the Android app; `grid_pose.tflite` is the one actually used.

## Notes

- The output tensor `[1,17,8400]` encodes 17 values per anchor: 5 box params + 12 keypoint params (4 corners × 3: x, y, visibility). `YoloDecoder.kt` and `yolo_grid_detector.py` handle the decoding.
- Canonical corner order: TL → TR → BR → BL (clockwise from top-left). The angular sort in `grid_geometry.order_corners` enforces this regardless of which YOLO outputs which corner.
