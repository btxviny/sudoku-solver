"""Check the exported TFLite models against the PyTorch originals.

Conversion is the step most likely to fail quietly.  A model that loads, runs,
and returns plausible numbers can still be wrong -- a transposed layout or a
quantisation loss shows up as slightly different digits, not as an error.  So
this compares the two runtimes on real data and reports the thing that actually
matters: whether they read the same digits.

Usage:
    uv run python scripts/verify_tflite.py
    uv run python scripts/verify_tflite.py --only cellocr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# `export_tflite` owns the reader registry (asset name -> module/class/config);
# importing it here keeps export and verification from listing models apart.
sys.path.insert(0, str(Path(__file__).resolve().parent))

ASSETS = ROOT / "android/app/src/main/assets"


def load_interpreter(path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter  # type: ignore
    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    return it


def sample_patches(patch_size: int, limit_images: int = 30) -> np.ndarray:
    """Real prepared cell patches, exactly as the reader would see them.

    `patch_size` comes from the reader's own config rather than a constant: a
    patch cut at the wrong size is not a small error, it is a different problem
    (measured: 20 % digit accuracy), and hardcoding it here once let this check
    drift away from the model it was supposed to be checking.
    """
    from sudoku_solver.cell_prep import is_low_contrast, prep_patch

    imgs = sorted(ROOT.glob("test_images/*.png")) + sorted(ROOT.glob("test_images/*.jpg"))
    imgs += sorted(ROOT.glob("data/wicht_sudoku/v2_test/*.jpg"))[:limit_images]

    PS = patch_size
    out: list[np.ndarray] = []
    for p in imgs:
        raw = cv2.imread(str(p))
        if raw is None:
            continue
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        scaled = cv2.resize(rgb, (PS * 9, PS * 9), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(scaled, cv2.COLOR_RGB2GRAY)
        low = is_low_contrast(gray)
        for r in range(9):
            for c in range(9):
                cell = gray[r * PS:(r + 1) * PS, c * PS:(c + 1) * PS].copy()
                out.append(prep_patch(cell, low))
    return np.stack(out)


def verify_reader(asset: str) -> bool:
    """Compare one exported per-cell reader against its PyTorch original."""
    import importlib

    import torch
    import torch.nn.functional as F

    from export_tflite import TORCH_READERS

    model_path = ASSETS / f"{asset}.tflite"
    if not model_path.exists():
        print(f"  {asset}.tflite missing -- export it first")
        return False

    module_name, net_name, config_name = TORCH_READERS[asset]
    cfg = getattr(importlib.import_module("sudoku_solver.config"), config_name)()
    if not cfg.model_path.exists():
        print(f"  {asset}: torch weights missing at {cfg.model_path}")
        return False
    net = getattr(importlib.import_module(module_name), net_name)(cell_size=cfg.patch_size)
    net.load_state_dict(torch.load(cfg.model_path, map_location="cpu", weights_only=True))
    net.eval()

    patches = sample_patches(cfg.patch_size)
    print(f"  comparing on {len(patches)} real prepared patches")

    x = torch.from_numpy(patches).float().unsqueeze(1) / 255.0
    with torch.no_grad():
        ref_probs = F.softmax(net(x), dim=1).numpy()

    it = load_interpreter(model_path)
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    shape = list(inp["shape"])
    print(f"  tflite input {shape} {inp['dtype'].__name__}")

    # The converter kept PyTorch's NCHW, and the batch is fixed at 81, so feed
    # whole grids. (For this 1-channel input NCHW and NHWC are the same bytes
    # anyway -- only the declared shape differs.)
    batch = shape[0]
    nchw = len(shape) == 4 and shape[1] == 1

    got = np.zeros_like(ref_probs)
    for start in range(0, len(patches), batch):
        chunk = patches[start:start + batch].astype(np.float32) / 255.0
        pad = batch - len(chunk)
        if pad:
            chunk = np.concatenate([chunk, np.zeros((pad, *chunk.shape[1:]), np.float32)])
        t = chunk[:, None, :, :] if nchw else chunk[:, :, :, None]
        it.set_tensor(inp["index"], t.astype(inp["dtype"]))
        it.invoke()
        logits = it.get_tensor(out["index"])
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        n = batch - pad
        got[start:start + n] = probs[:n]

    ref_d = ref_probs.argmax(1)
    got_d = got.argmax(1)
    agree = int((ref_d == got_d).sum())
    max_abs = float(np.abs(ref_probs - got).max())
    mean_abs = float(np.abs(ref_probs - got).mean())

    print(f"  digits identical : {agree}/{len(patches)}  ({100 * agree / len(patches):.2f} %)")
    print(f"  max |dprob|      : {max_abs:.2e}")
    print(f"  mean |dprob|     : {mean_abs:.2e}")

    if agree != len(patches):
        bad = np.where(ref_d != got_d)[0][:10]
        for i in bad:
            print(f"    patch {i}: torch={ref_d[i]} (p={ref_probs[i].max():.3f})  "
                  f"tflite={got_d[i]} (p={got[i].max():.3f})")

    # Gate on the probabilities, not on the argmax.  Demanding identical digits
    # fails on a cell the model itself cannot decide -- two classes within 1e-6
    # of each other, where float noise picks the winner -- which says nothing
    # about the conversion.  A max probability delta this small means the two
    # runtimes are computing the same function; anything larger is a real
    # layout or precision fault, and those show up at 1e-2 or worse.
    TOL = 1e-4
    if max_abs >= TOL:
        print(f"  FAIL: max |dprob| {max_abs:.2e} exceeds {TOL:.0e}")
    elif agree != len(patches):
        print(f"  ({len(patches) - agree} digit(s) differ on ties; probabilities "
              f"agree to {max_abs:.1e})")
    return max_abs < TOL


def verify_yolo(name: str, weights: Path) -> bool:
    from ultralytics import YOLO

    model_path = ASSETS / f"{name}.tflite"
    if not model_path.exists():
        print(f"  {name}.tflite missing -- export it first")
        return False
    if not weights.exists():
        print(f"  {name}: torch weights missing")
        return False

    imgs = sorted(ROOT.glob("test_images/*.png")) + sorted(ROOT.glob("test_images/*.jpg"))
    imgs += sorted(ROOT.glob("data/wicht_sudoku/v2_test/*.jpg"))[:10]
    imgs = [p for p in imgs if cv2.imread(str(p)) is not None]

    ref_model = YOLO(str(weights))
    tfl_model = YOLO(str(model_path))

    n = 0
    count_match = 0
    ious: list[float] = []
    for p in imgs:
        a = ref_model.predict(str(p), verbose=False, conf=0.25)[0]
        b = tfl_model.predict(str(p), verbose=False, conf=0.25)[0]
        n += 1
        ba = a.boxes.xyxy.cpu().numpy()
        bb = b.boxes.xyxy.cpu().numpy()
        if len(ba) == len(bb):
            count_match += 1
        # Greedy match each torch box to its best TFLite box.
        for box in ba:
            if len(bb) == 0:
                ious.append(0.0)
                continue
            x1 = np.maximum(box[0], bb[:, 0]); y1 = np.maximum(box[1], bb[:, 1])
            x2 = np.minimum(box[2], bb[:, 2]); y2 = np.minimum(box[3], bb[:, 3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            area_a = (box[2] - box[0]) * (box[3] - box[1])
            area_b = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])
            ious.append(float((inter / (area_a + area_b - inter + 1e-9)).max()))

    iou = np.array(ious) if ious else np.zeros(1)
    print(f"  images                : {n}")
    print(f"  same detection count  : {count_match}/{n}")
    print(f"  mean best-match IoU   : {iou.mean():.4f}")
    print(f"  boxes with IoU < 0.9  : {int((iou < 0.9).sum())}/{len(iou)}")
    # Diagnostic only. Box agreement is deliberately not the pass criterion:
    # the cell detector is followed by a lattice fit that re-derives every slot
    # and synthesises missed cells, so a differing box count is routinely
    # absorbed. Measured end-to-end, these same models score 87.78 % of cells
    # correct against ground truth versus PyTorch's 87.81 % -- one cell in 3240.
    # Gate on scripts/verify_tflite_pipeline.py, which measures that directly.
    return iou.mean() > 0.75


def main() -> None:
    ap = argparse.ArgumentParser()
    from export_tflite import TORCH_READERS

    ap.add_argument("--only", choices=[*TORCH_READERS, "cell_vision", "grid_seg", "grid_pose"])
    args = ap.parse_args()

    checks: list[tuple[str, bool]] = []
    for asset, (_mod, net_name, _cfg) in TORCH_READERS.items():
        if args.only in (None, asset):
            print(net_name)
            checks.append((asset, verify_reader(asset)))
    for name, w in [
        ("cell_vision", ROOT / "training/cell_extraction/runs/cell_vision_v6/weights/best.pt"),
        ("grid_seg", ROOT / "training/grid_seg/runs/grid_seg_v1/weights/best.pt"),
        ("grid_pose", ROOT / "training/grid_pose/runs/grid_pose_v1/weights/best.pt"),
    ]:
        if args.only in (None, name) and (ASSETS / f"{name}.tflite").exists():
            print(f"\n{name}")
            checks.append((name, verify_yolo(name, w)))

    print("\n" + "=" * 46)
    for name, ok in checks:
        print(f"  {name:14s} {'PASS' if ok else 'FAIL'}")
    print("\n  Detector box agreement is indicative only --")
    print("  run scripts/verify_tflite_pipeline.py for the accuracy that matters.")
    sys.exit(0 if all(ok for _, ok in checks) else 1)


if __name__ == "__main__":
    main()
