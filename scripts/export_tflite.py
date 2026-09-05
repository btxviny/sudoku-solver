"""Export every pipeline model to TFLite for the Android port.

Three models, two toolchains:

  grid_seg / cell_vision  Ultralytics exports YOLO directly.
  GridOCRNet / CellOCRNet Plain PyTorch CNNs, converted straight from the
                          nn.Module by litert_torch (the converter Ultralytics
                          uses underneath).

Layout is the thing to watch.  PyTorch is NCHW; TFLite is NHWC.  The converter
transposes the weights, but the *caller* must feed NHWC, and getting that wrong
produces confident wrong answers rather than an error -- so this script records
the exported input shape for each model and `verify_tflite.py` checks numerics
against PyTorch rather than trusting the conversion.

Usage:
    uv run python scripts/export_tflite.py            # all
    uv run python scripts/export_tflite.py --only cellocr
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "android/app/src/main/assets"

MODELS = {
    "grid_seg": ROOT / "training/grid_seg/runs/grid_seg_v1/weights/best.pt",
    "grid_pose": ROOT / "training/grid_pose/runs/grid_pose_v1/weights/best.pt",
    "cell_vision": ROOT / "training/cell_extraction/runs/cell_vision_v6/weights/best.pt",
}


def export_yolo(name: str, weights: Path, imgsz: int, half: bool) -> Path | None:
    from ultralytics import YOLO

    if not weights.exists():
        print(f"  {name}: weights missing at {weights} -- skipped")
        return None
    print(f"  {name}: exporting from {weights.name} (imgsz={imgsz}, half={half})")
    produced = YOLO(str(weights)).export(format="tflite", imgsz=imgsz, half=half)
    src = Path(produced)
    dst = OUT / f"{name}.tflite"
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  {name}: -> {dst.relative_to(ROOT)}  ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


#: The per-cell digit readers, each a plain PyTorch CNN.  Keyed by the asset
#: name the Android side loads.
TORCH_READERS = {
    "gridocr": ("sudoku_solver.grid_ocr", "GridOCRNet", "GridOCRConfig"),
    "cellocr": ("sudoku_solver.cell_ocr", "CellOCRNet", "CellOCRConfig"),
}


def export_reader(name: str) -> Path | None:
    """Export one per-cell digit reader: torch -> litert -> assets/<name>.tflite."""
    import importlib

    import torch

    module_name, net_name, config_name = TORCH_READERS[name]
    config_cls = getattr(importlib.import_module("sudoku_solver.config"), config_name)
    net_cls = getattr(importlib.import_module(module_name), net_name)

    cfg = config_cls()
    if not cfg.model_path.exists():
        print(f"  {name}: weights missing at {cfg.model_path} -- skipped")
        return None

    net = net_cls(cell_size=cfg.patch_size)
    net.load_state_dict(torch.load(cfg.model_path, map_location="cpu", weights_only=True))
    net.eval()

    PS = cfg.patch_size
    # Same converter Ultralytics used for the YOLO models, applied directly to
    # the nn.Module. Going via ONNX needs onnxscript for torch 2.13's exporter
    # and adds a hop that can only lose fidelity, so it is skipped entirely.
    import litert_torch

    # Fixed batch of 81: the pipeline always reads a whole grid, and one
    # invocation over all cells is far cheaper on device than 81 of them.
    BATCH = 81
    print(f"  {name}: torch -> litert ({BATCH},1,{PS},{PS})")
    converted = litert_torch.convert(net, (torch.zeros(BATCH, 1, PS, PS),))

    work = ROOT / f"experiments/export_{name}"
    work.mkdir(parents=True, exist_ok=True)
    tmp = work / f"{name}.tflite"
    converted.export(str(tmp))

    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{name}.tflite"
    shutil.copy2(tmp, dst)
    print(f"  {name}: -> {dst.relative_to(ROOT)}  ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def describe(path: Path) -> dict:
    """Record the exported signature so the Kotlin side can be written against it."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter  # type: ignore
    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    return {
        "file": path.name,
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "inputs": [
            {"name": d["name"], "shape": [int(x) for x in d["shape"]],
             "dtype": str(d["dtype"].__name__)}
            for d in it.get_input_details()
        ],
        "outputs": [
            {"name": d["name"], "shape": [int(x) for x in d["shape"]],
             "dtype": str(d["dtype"].__name__)}
            for d in it.get_output_details()
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=[*MODELS, *TORCH_READERS], default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="Export float16 where supported")
    args = ap.parse_args()

    produced: list[Path] = []
    print("Exporting to TFLite\n")

    for name, weights in MODELS.items():
        if args.only and args.only != name:
            continue
        p = export_yolo(name, weights, args.imgsz, args.half)
        if p:
            produced.append(p)

    for name in TORCH_READERS:
        if args.only and args.only != name:
            continue
        p = export_reader(name)
        if p:
            produced.append(p)

    if not produced:
        raise SystemExit("Nothing exported")

    # Describe every model present, not just the ones exported this run: the
    # manifest is what the Android side reads for input shapes, so a partial
    # re-export must not silently drop the others from it.
    print("\nSignatures")
    manifest = []
    for p in sorted(OUT.glob("*.tflite")):
        try:
            info = describe(p)
        except Exception as e:
            print(f"  {p.name}: could not inspect ({e})")
            continue
        manifest.append(info)
        print(f"  {info['file']:22s} {info['size_mb']:6.2f} MB")
        for d in info["inputs"]:
            print(f"      in  {d['name']:28s} {d['shape']} {d['dtype']}")
        for d in info["outputs"]:
            print(f"      out {d['name']:28s} {d['shape']} {d['dtype']}")

    man = OUT / "models.json"
    man.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {man.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
