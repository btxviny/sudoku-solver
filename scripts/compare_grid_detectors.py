"""Compare grid detectors against the Mask R-CNN baseline.

Two evaluations, because they answer different questions:

  corners  Corner-localisation error on the held-out split of the Roboflow
           segmentation set, which carries ground-truth quads.  Measures the
           detector in isolation.

  wicht    End-to-end digit accuracy on the labelled Wicht photos, with GridOCR
           held fixed.  Measures whether corner error actually costs digits --
           a detector can be a few pixels worse and still rectify well enough
           for the OCR, or be slightly better and still clip a row.

Corner error is reported as a percentage of the grid's own mean side length,
so large and small grids contribute comparably.

Usage:
    uv run python scripts/compare_grid_detectors.py corners
    uv run python scripts/compare_grid_detectors.py wicht --data-dir data/wicht_sudoku/v2_test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sudoku_solver.config import (
    GridDetectorConfig,
    GridOCRConfig,
    YoloGridDetectorConfig,
)
from sudoku_solver.grid_detector import GridDetector
from sudoku_solver.grid_ocr import GridOCR
from sudoku_solver.yolo_grid_detector import YoloGridDetector, order_corners

RESIZE_TO = (1024, 1024)
POSE_VAL = ROOT / "data/segmentation/pose_dataset/valid"

# A corner is "hit" if it lands within this fraction of the grid's side length.
# 2% of a 450 px grid is ~9 px -- under a fifth of a cell, so the warp still
# puts every digit safely inside its cell.
HIT_TOL = 0.02


# ---------------------------------------------------------------------------
# Detector construction
# ---------------------------------------------------------------------------

def build_detector(name: str, refine: bool = True):
    """Build one of the named detectors, or None if its weights are missing."""
    try:
        if name == "maskrcnn":
            cfg = GridDetectorConfig()
            cfg.resize_to = RESIZE_TO
            return GridDetector(cfg)
        if name in ("pose", "seg"):
            cfg = YoloGridDetectorConfig(mode=name, refine=refine)
            if name == "seg":
                cfg.model_path = ROOT / "training/grid_seg/runs/grid_seg_v1/weights/best.pt"
            cfg.resize_to = RESIZE_TO
            return YoloGridDetector(cfg)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  skipping {name}: {e}")
        return None
    raise ValueError(f"unknown detector {name!r}")


def detector_corners(det, image: np.ndarray) -> np.ndarray:
    """Corners in RESIZE_TO pixel space, TL/TR/BR/BL, for either detector type."""
    if isinstance(det, YoloGridDetector):
        return det.corners(image)
    return order_corners(det._run(image)[3])


# ---------------------------------------------------------------------------
# Eval A -- corner localisation
# ---------------------------------------------------------------------------

def load_truth(label_path: Path) -> np.ndarray | None:
    """Ground-truth quad from a pose label, in RESIZE_TO pixel coordinates."""
    lines = [ln for ln in label_path.read_text().splitlines() if ln.strip()]
    if not lines:
        return None
    vals = np.array(lines[0].split()[1:], dtype=np.float64)
    quad = vals[4:].reshape(-1, 3)[:, :2]
    return (quad * np.array(RESIZE_TO, dtype=np.float64)).astype(np.float32)


def side_length(quad: np.ndarray) -> float:
    return float(np.mean([np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]))


def eval_corners(names: list[str], refine: bool) -> dict:
    if not POSE_VAL.exists():
        raise SystemExit(
            f"{POSE_VAL} missing -- run training/grid_pose/prepare_dataset.py first"
        )
    labels = sorted((POSE_VAL / "labels").glob("*.txt"))
    print(f"Corner eval: {len(labels)} held-out images\n")

    results = {}
    for name in names:
        det = build_detector(name, refine=refine)
        if det is None:
            continue

        errs, per_image, failures, latencies = [], [], 0, []
        for lbl in tqdm(labels, desc=f"{name:9s}"):
            truth = load_truth(lbl)
            img_path = next((POSE_VAL / "images").glob(lbl.stem + ".*"), None)
            if truth is None or img_path is None:
                continue
            raw = cv2.imread(str(img_path))
            if raw is None:
                failures += 1
                continue
            image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            try:
                t0 = time.perf_counter()
                pred = detector_corners(det, image)
                latencies.append((time.perf_counter() - t0) * 1000)
            except Exception:
                failures += 1
                continue

            s = side_length(truth)
            d = np.linalg.norm(pred - truth, axis=1) / s      # per-corner, normalised
            errs.extend(d.tolist())
            per_image.append(d.max())                          # worst corner

        if not per_image:
            continue
        errs_a, worst = np.array(errs), np.array(per_image)
        results[name] = {
            "n": len(per_image),
            "failures": failures,
            "mean_corner_err_pct": float(errs_a.mean() * 100),
            "median_corner_err_pct": float(np.median(errs_a) * 100),
            "p95_corner_err_pct": float(np.percentile(errs_a, 95) * 100),
            "all4_within_tol_pct": float((worst <= HIT_TOL).mean() * 100),
            "worst_corner_p95_pct": float(np.percentile(worst, 95) * 100),
            "median_latency_ms": float(np.median(latencies)) if latencies else None,
        }
        del det

    return results


# ---------------------------------------------------------------------------
# Eval B -- end-to-end digit accuracy
# ---------------------------------------------------------------------------

def read_dat(path: Path) -> np.ndarray:
    rows = [ln for ln in path.read_text().splitlines() if ln.strip()][-9:]
    return np.array([[int(v) for v in r.split()] for r in rows], int)


def eval_wicht(names: list[str], data_dirs: list[Path], refine: bool) -> dict:
    images = []
    for d in data_dirs:
        images += sorted(p for p in d.glob("*.jpg") if p.with_suffix(".dat").exists())
    if not images:
        raise SystemExit(f"No labelled images in {[str(d) for d in data_dirs]}")
    print(f"End-to-end eval: {len(images)} labelled photos\n")

    ocr = GridOCR(GridOCRConfig())
    results = {}
    for name in names:
        det = build_detector(name, refine=refine)
        if det is None:
            continue

        detect_fail = perfect = cell_hit = cell_total = 0
        for path in tqdm(images, desc=f"{name:9s}"):
            truth = read_dat(path.with_suffix(".dat"))
            raw = cv2.imread(str(path))
            if raw is None:
                detect_fail += 1
                continue
            try:
                rect = det.detect(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
                pred, _ = ocr.read_with_probs(rect)
            except Exception:
                detect_fail += 1
                continue
            ok = pred == truth
            cell_hit += int(ok.sum())
            cell_total += 81
            perfect += int(ok.all())

        results[name] = {
            "n": len(images),
            "detect_failures": detect_fail,
            "cell_accuracy_pct": 100 * cell_hit / cell_total if cell_total else 0.0,
            "perfect_grids": perfect,
            "perfect_grid_pct": 100 * perfect / len(images),
        }
        del det

    return results


# ---------------------------------------------------------------------------

def report(title: str, results: dict, keys: list[tuple[str, str]]) -> None:
    print(f"\n=== {title} ===")
    w = max(len(k) for k in results) if results else 8
    head = f"{'detector':<{w}}  " + "  ".join(f"{lab:>{max(len(lab), 8)}}" for _, lab in keys)
    print(head)
    print("-" * len(head))
    for name, r in results.items():
        cells = []
        for key, lab in keys:
            v = r.get(key)
            width = max(len(lab), 8)
            cells.append(f"{'-':>{width}}" if v is None else
                         (f"{v:>{width}.2f}" if isinstance(v, float) else f"{v:>{width}}"))
        print(f"{name:<{w}}  " + "  ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval", choices=["corners", "wicht", "both"])
    ap.add_argument("--detectors", default="maskrcnn,pose,seg")
    ap.add_argument("--data-dir", type=Path, action="append", default=None,
                    help="Wicht dirs (repeatable); defaults to v2_test + half_mixed_test")
    ap.add_argument("--no-refine", action="store_true",
                    help="Disable Hough edge-snapping for the YOLO detectors")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/grid_detector_comparison.json")
    args = ap.parse_args()

    names = [n.strip() for n in args.detectors.split(",") if n.strip()]
    dirs = args.data_dir or [
        ROOT / "data/wicht_sudoku/v2_test",
        ROOT / "data/wicht_sudoku/half_mixed_test",
    ]

    out = {}
    if args.eval in ("corners", "both"):
        out["corners"] = eval_corners(names, refine=not args.no_refine)
        report("Corner localisation (% of grid side length)", out["corners"], [
            ("mean_corner_err_pct", "mean"),
            ("median_corner_err_pct", "median"),
            ("p95_corner_err_pct", "p95"),
            ("all4_within_tol_pct", f"all4<{HIT_TOL:.0%}"),
            ("failures", "fails"),
            ("median_latency_ms", "ms"),
        ])
    if args.eval in ("wicht", "both"):
        out["wicht"] = eval_wicht(names, dirs, refine=not args.no_refine)
        report("End-to-end with GridOCR held fixed", out["wicht"], [
            ("cell_accuracy_pct", "cell %"),
            ("perfect_grid_pct", "grid %"),
            ("perfect_grids", "perfect"),
            ("detect_failures", "fails"),
        ])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
