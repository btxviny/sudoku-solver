"""Cross-check YoloDecoder.kt against Ultralytics' own postprocessing.

Unlike the other ports, this one has no Python original -- Ultralytics did the
letterboxing, box decoding, NMS and coordinate mapping internally, and on
Android all of it is ours.  So it is checked against the thing it replaces:
feed the same TFLite model, decode with the transliterated Kotlin, and compare
the resulting boxes and corners with what Ultralytics returns for the same image.

Usage:
    uv run python scripts/verify_kotlin_decoder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

ASSETS = ROOT / "android/app/src/main/assets"
INPUT_SIZE = 640
PAD_VALUE = 114


# ── transliteration of YoloDecoder.kt ────────────────────────────────────────

def kt_letterbox(image: np.ndarray):
    h, w = image.shape[:2]
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (nw, nh))
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), PAD_VALUE, np.uint8)
    pad_x, pad_y = (INPUT_SIZE - nw) // 2, (INPUT_SIZE - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, (scale, pad_x, pad_y)


def kt_to_input_nchw(canvas: np.ndarray) -> np.ndarray:
    n = INPUT_SIZE * INPUT_SIZE
    flat = canvas.reshape(-1, 3)
    out = np.empty(n * 3, np.float32)
    out[:n] = flat[:, 0] / 255.0
    out[n:2 * n] = flat[:, 1] / 255.0
    out[2 * n:] = flat[:, 2] / 255.0
    return out


def kt_unpad(x, y, lb):
    scale, pad_x, pad_y = lb
    return (x * INPUT_SIZE - pad_x) / scale, (y * INPUT_SIZE - pad_y) / scale


def kt_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter)


def kt_nms(boxes, iou_thresh):
    order = sorted(boxes, key=lambda b: -b[4])
    dropped = [False] * len(order)
    keep = []
    for i in range(len(order)):
        if dropped[i]:
            continue
        keep.append(order[i])
        for j in range(i + 1, len(order)):
            if not dropped[j] and kt_iou(order[i], order[j]) > iou_thresh:
                dropped[j] = True
    return keep


def kt_decode_detections(out: np.ndarray, channels: int, lb, conf=0.25, iou=0.5):
    anchors = out.size // channels
    n_cls = channels - 4
    kept = []
    scores = out[4 * anchors:(4 + n_cls) * anchors].reshape(n_cls, anchors)
    best_cls = scores.argmax(0)
    best_score = scores.max(0)
    for a in np.where(best_score >= conf)[0]:
        cx, cy = out[a], out[anchors + a]
        bw, bh = out[2 * anchors + a], out[3 * anchors + a]
        x1, y1 = kt_unpad(cx - bw / 2, cy - bh / 2, lb)
        x2, y2 = kt_unpad(cx + bw / 2, cy + bh / 2, lb)
        kept.append((x1, y1, x2, y2, float(best_score[a]), int(best_cls[a])))
    return kt_nms(kept, iou)


def kt_decode_pose(out: np.ndarray, lb, w, h, conf=0.25):
    channels = 17
    anchors = out.size // channels
    cls = out[4 * anchors:5 * anchors]
    best = int(cls.argmax())
    if cls[best] < conf:
        return None
    pts = []
    for k in range(4):
        kx = out[(5 + k * 3) * anchors + best]
        ky = out[(6 + k * 3) * anchors + best]
        x, y = kt_unpad(kx, ky, lb)
        pts.append((min(max(x, 0.0), w - 1), min(max(y, 0.0), h - 1)))
    from sudoku_solver.yolo_grid_detector import order_corners
    return order_corners(np.array(pts, np.float32))


# ── harness ──────────────────────────────────────────────────────────────────

def run_tflite(path: Path, x: np.ndarray) -> list[np.ndarray]:
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    it.set_tensor(inp["index"], x.reshape(inp["shape"]).astype(inp["dtype"]))
    it.invoke()
    return [it.get_tensor(o["index"]) for o in it.get_output_details()]


def match_boxes(mine, theirs) -> tuple[int, float]:
    """Greedily pair each Ultralytics box with its best decoded counterpart."""
    if len(theirs) == 0:
        return 0, 1.0
    worst = 0.0
    matched = 0
    for t in theirs:
        best = 0.0
        for m in mine:
            best = max(best, kt_iou(m, (t[0], t[1], t[2], t[3])))
        if best > 0.9:
            matched += 1
        worst = max(worst, 1.0 - best)
    return matched, worst


def main() -> None:
    from ultralytics import YOLO

    imgs = sorted(ROOT.glob("test_images/*.png")) + sorted(ROOT.glob("test_images/*.jpg"))
    imgs += sorted(ROOT.glob("data/wicht_sudoku/v2_test/*.jpg"))[:8]
    imgs = [p for p in imgs if cv2.imread(str(p)) is not None]

    ok = True

    # ── detection head ──
    print("cell_vision: decoded boxes vs Ultralytics")
    ref = YOLO(str(ASSETS / "cell_vision.tflite"))
    tot_ref = tot_matched = 0
    worst_all = 0.0
    for p in imgs:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        canvas, lb = kt_letterbox(img)
        out = run_tflite(ASSETS / "cell_vision.tflite", kt_to_input_nchw(canvas))[0]
        mine = kt_decode_detections(out.reshape(-1), 6, lb, conf=0.25, iou=0.5)
        theirs = ref.predict(str(p), verbose=False, conf=0.25, iou=0.5)[0].boxes.xyxy.cpu().numpy()
        m, worst = match_boxes(mine, theirs)
        tot_ref += len(theirs)
        tot_matched += m
        worst_all = max(worst_all, worst)
    print(f"  boxes matched at IoU>0.9 : {tot_matched}/{tot_ref}")
    print(f"  worst unmatched (1-IoU)  : {worst_all:.4f}")
    if tot_ref == 0 or tot_matched < tot_ref * 0.98:
        ok = False

    # ── pose head ──
    print("\ngrid_pose: decoded corners vs Ultralytics keypoints")
    refp = YOLO(str(ASSETS / "grid_pose.tflite"))
    from sudoku_solver.yolo_grid_detector import order_corners
    max_err = 0.0
    n = 0
    for p in imgs:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        canvas, lb = kt_letterbox(img)
        out = run_tflite(ASSETS / "grid_pose.tflite", kt_to_input_nchw(canvas))[0]
        mine = kt_decode_pose(out.reshape(-1), lb, img.shape[1], img.shape[0])
        r = refp.predict(str(p), verbose=False, conf=0.25)[0]
        if mine is None or r.keypoints is None or len(r.keypoints.xy) == 0:
            continue
        theirs = order_corners(r.keypoints.xy[0].cpu().numpy())
        err = float(np.abs(np.asarray(mine) - theirs).max())
        max_err = max(max_err, err)
        n += 1
    print(f"  images compared          : {n}")
    print(f"  max corner difference px : {max_err:.4f}")
    if n == 0 or max_err > 1.0:
        ok = False

    print("\n" + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
