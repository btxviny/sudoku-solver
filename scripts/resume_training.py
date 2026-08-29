"""Resume an interrupted Ultralytics run from its `last.pt` checkpoint.

Usage:
    uv run python scripts/resume_training.py training/grid_pose/runs/grid_pose_v1
"""
import sys
from pathlib import Path

from ultralytics import YOLO

run = Path(sys.argv[1]).resolve()
ckpt = run / "weights" / "last.pt"
if not ckpt.exists():
    raise SystemExit(f"No checkpoint at {ckpt}")
YOLO(str(ckpt)).train(resume=True)
