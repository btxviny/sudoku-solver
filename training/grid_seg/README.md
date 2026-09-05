# Grid Segmentation — YOLOv8n-seg

Stage 1 (primary): detect the sudoku grid in the input photo and produce a polygon mask. The mask is then used by `grid_geometry.py` to compute a perspective warp that rectifies the grid to a square.

## Dataset

**Source:** [Roboflow Universe — sudoku-lq9gj](https://universe.roboflow.com/sudoku-lq9gj) (CC BY 4.0)

Already in YOLOv8 polygon format — no conversion script needed.

```
data/segmentation/segmentation_dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/        # polygon masks
└── valid/
    ├── images/
    └── labels/
```

Download the dataset from Roboflow and place it at `data/segmentation/segmentation_dataset/`.

## Training

```bash
uv run python training/grid_seg/train.py
```

Config: `yolov8n-seg.pt`, 300 epochs, imgsz=640, batch=16, patience=50, device=0.

Weights land at: `training/grid_seg/runs/grid_seg_v1/weights/best.pt`

## Results

| Metric | Value |
|--------|-------|
| Box mAP@50 | ~99% |
| Mask mAP@50 | ~99% |
| Mean corner error (90 held-out images) | **3.08%** |
| Model size | ~13 MB |
| Inference (640×640) | ~5 ms on GPU |

## How the mask becomes corners

`grid_geometry._corners_from_mask` fits one line per side to the mask contour (Huber regression, ends trimmed to avoid rounded corners), then `_refine_on_grid_lines` snaps each edge onto the real grid border found in a band around the mask edge. This two-step process outperforms `approxPolyDP` alone by a large margin:

| Corner method | Solved (90 images) |
|---|---|
| approxPolyDP on mask | 35/90 |
| Line fit on mask | 50/90 |
| **Line fit + Hough refinement** | **77/90** |

## Notes

- `refine=False` is the default in `YoloGridDetectorConfig`. Enabling Hough refinement on these accurate quads pushes mean corner error from 5.7% to 40.9%.
- The `grid_pose` model is an alternative that predicts corners directly without mask post-processing — simpler to port to mobile but slightly less accurate.
