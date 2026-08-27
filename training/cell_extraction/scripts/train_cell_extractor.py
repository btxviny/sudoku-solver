"""Train CellExtractorCNN on the sudoku-vicfl dataset.

Dataset: sudoku-vicfl – 1000 images, 81 per-cell bboxes with digit labels
         (class names '0'–'8'; all are valid cell bboxes, grid fills the frame).

Model:   MobileNetV3-Small → 81×4 normalised [x1,y1,x2,y2] bbox regression
Output:  models/weights/cell_extractor_cnn.pth

Usage:
    uv run python training/cell_extraction/scripts/train_cell_extractor.py
    uv run python training/cell_extraction/scripts/train_cell_extractor.py \\
        --epochs 100 --batch_size 16 --patience 15
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))

from sudoku_solver.cell_extractor_cnn import CellExtractorCNN

DATA_ROOT_VICFL = PROJECT / "data" / "roboflow" / "sudoku-vicfl"
CKPT_DIR = PROJECT / "training" / "cell_extraction" / "checkpoints"
OUT_MODEL = PROJECT / "models" / "weights" / "cell_extractor_cnn.pth"

CELL_CAT_NAME = "cell"
INPUT_SIZE = 320                    # resize to this square for training

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CellExtractorDataset(Dataset):
    """COCO-annotated sudoku images → (image_tensor, boxes_tensor).

    boxes: (81, 4) float32 normalised [x1, y1, x2, y2] in [0, 1],
           sorted row-major (top-left → bottom-right).
    """

    def __init__(self, split_dir: Path, augment: bool = False):
        self.split_dir = split_dir
        self.augment = augment

        ann_path = split_dir / "_annotations.coco.json"
        with open(ann_path) as f:
            coco = json.load(f)

        cell_cat_id = next(
            c["id"] for c in coco["categories"] if c["name"] == CELL_CAT_NAME
        )

        # Build image_id → annotations index
        anns_by_img: dict[int, list] = {}
        for a in coco["annotations"]:
            if a["category_id"] == cell_cat_id:
                anns_by_img.setdefault(a["image_id"], []).append(a)

        self.samples: list[tuple[Path, np.ndarray]] = []
        for img_meta in coco["images"]:
            img_id = img_meta["id"]
            cells = anns_by_img.get(img_id, [])
            if len(cells) != 81:
                continue                    # skip anomalies (162-cell / 82-cell images)

            W, H = img_meta["width"], img_meta["height"]
            boxes = self._sort_cells(cells, W, H)   # (81, 4) normalised
            img_path = split_dir / img_meta["file_name"]
            self.samples.append((img_path, boxes))

        # transforms.ColorJitter / RandomGrayscale operate on PIL images;
        # ToTensor() must follow them.
        self._base_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True),
            transforms.Normalize(_MEAN, _STD),
        ])
        self._aug_tf = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True),
            transforms.Normalize(_MEAN, _STD),
        ])  # input to aug_tf must be a PIL Image

    @staticmethod
    def _sort_cells(cells: list, W: int, H: int) -> np.ndarray:
        """Sort 81 COCO cells into row-major order; return (81, 4) normalised [x1,y1,x2,y2]."""
        entries = []
        for a in cells:
            x, y, w, h = a["bbox"]
            cx, cy = x + w / 2, y + h / 2
            entries.append((cy, cx, x / W, y / H, (x + w) / W, (y + h) / H))

        entries.sort(key=lambda e: (e[0], e[1]))   # sort by cy then cx
        return np.array([[x1, y1, x2, y2] for _, _, x1, y1, x2, y2 in entries],
                        dtype=np.float32)           # (81, 4)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, boxes = self.samples[idx]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            img_bgr = np.zeros((640, 640, 3), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        tf = self._aug_tf if self.augment else self._base_tf
        # PIL is required by ColorJitter / RandomGrayscale; base_tf also accepts it
        pil_img = Image.fromarray(img_rgb)
        tensor = tf(pil_img)
        return tensor, torch.from_numpy(boxes)


class RowDerivedDataset(Dataset):
    """Derive 81 cell bboxes from 9 row bboxes (numbers-in-matrix dataset).

    Each image has row_1…row_9 COCO annotations covering the full grid width.
    Cells are derived by dividing each row into 9 equal columns, giving a
    (81, 4) normalised [x1, y1, x2, y2] target in row-major order.
    """

    def __init__(self, split_dir: Path, augment: bool = False):
        self.split_dir = split_dir
        self.augment = augment

        ann_path = split_dir / "_annotations.coco.json"
        with open(ann_path) as f:
            coco = json.load(f)

        # row_N categories sorted by row index
        row_cats: dict[int, int] = {}   # category_id → row_index (0-based)
        for c in coco["categories"]:
            if c["name"].startswith("row_"):
                row_idx = int(c["name"].split("_")[1]) - 1   # row_1 → 0
                row_cats[c["id"]] = row_idx

        anns_by_img: dict[int, list] = {}
        for a in coco["annotations"]:
            if a["category_id"] in row_cats:
                anns_by_img.setdefault(a["image_id"], []).append(a)

        self.samples: list[tuple[Path, np.ndarray]] = []
        for img_meta in coco["images"]:
            img_id = img_meta["id"]
            rows = anns_by_img.get(img_id, [])
            if len(rows) != 9:
                continue

            W, H = img_meta["width"], img_meta["height"]
            boxes = self._derive_cells(rows, row_cats, W, H)
            img_path = split_dir / img_meta["file_name"]
            self.samples.append((img_path, boxes))

        self._base_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True),
            transforms.Normalize(_MEAN, _STD),
        ])
        self._aug_tf = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True),
            transforms.Normalize(_MEAN, _STD),
        ])

    @staticmethod
    def _derive_cells(
        rows: list, row_cats: dict[int, int], W: int, H: int
    ) -> np.ndarray:
        """Divide each row into 9 equal columns → (81, 4) normalised [x1,y1,x2,y2]."""
        sorted_rows = sorted(rows, key=lambda a: row_cats[a["category_id"]])
        cells = []
        for a in sorted_rows:
            rx, ry, rw, rh = a["bbox"]
            cell_w = rw / 9
            for j in range(9):
                x1 = (rx + j * cell_w) / W
                y1 = ry / H
                x2 = (rx + (j + 1) * cell_w) / W
                y2 = (ry + rh) / H
                cells.append([x1, y1, x2, y2])
        return np.array(cells, dtype=np.float32)   # (81, 4)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, boxes = self.samples[idx]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            img_bgr = np.zeros((640, 640, 3), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tf = self._aug_tf if self.augment else self._base_tf
        return tf(Image.fromarray(img_rgb)), torch.from_numpy(boxes)


class SudokuVicflDataset(Dataset):
    """sudoku-vicfl: 1000 images, 81 per-cell bboxes with digit labels.

    Category names are digit strings ('0'–'8'); the supercategory 'sudoku'
    (id=0) has no annotations and is excluded.  All other annotations are
    cell bboxes, sorted into row-major order.
    """

    def __init__(self, split_dir: Path, augment: bool = False):
        self.split_dir = split_dir
        self.augment = augment

        ann_path = split_dir / "_annotations.coco.json"
        with open(ann_path) as f:
            coco = json.load(f)

        # Accept any category whose name is a digit string ('0'–'8')
        cell_cat_ids = {
            c["id"] for c in coco["categories"] if c["name"].isdigit()
        }

        anns_by_img: dict[int, list] = {}
        for a in coco["annotations"]:
            if a["category_id"] in cell_cat_ids:
                anns_by_img.setdefault(a["image_id"], []).append(a)

        self.samples: list[tuple[Path, np.ndarray]] = []
        for img_meta in coco["images"]:
            img_id = img_meta["id"]
            cells = anns_by_img.get(img_id, [])
            if len(cells) != 81:
                continue

            W, H = img_meta["width"], img_meta["height"]
            boxes = CellExtractorDataset._sort_cells(cells, W, H)
            img_path = split_dir / img_meta["file_name"]
            self.samples.append((img_path, boxes))

        self._base_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True),
            transforms.Normalize(_MEAN, _STD),
        ])
        self._aug_tf = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True),
            transforms.Normalize(_MEAN, _STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, boxes = self.samples[idx]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            img_bgr = np.zeros((640, 640, 3), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tf = self._aug_tf if self.augment else self._base_tf
        return tf(Image.fromarray(img_rgb)), torch.from_numpy(boxes)


# ---------------------------------------------------------------------------
# Losses & metrics
# ---------------------------------------------------------------------------

def smooth_l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """SmoothL1 over all 81×4 coordinates."""
    return F.smooth_l1_loss(pred, gt, beta=0.01)


def giou_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Generalised IoU loss averaged over all cells and batch.

    Args:
        pred, gt: (B, 81, 4) [x1, y1, x2, y2] normalised.
    """
    px1, py1, px2, py2 = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
    gx1, gy1, gx2, gy2 = gt[..., 0], gt[..., 1], gt[..., 2], gt[..., 3]

    # Intersection
    ix1 = torch.max(px1, gx1)
    iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2)
    iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    # Union
    area_p = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    area_g = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
    union = area_p + area_g - inter + 1e-7

    iou = inter / union

    # Enclosing box
    ex1 = torch.min(px1, gx1)
    ey1 = torch.min(py1, gy1)
    ex2 = torch.max(px2, gx2)
    ey2 = torch.max(py2, gy2)
    enclosing = (ex2 - ex1).clamp(0) * (ey2 - ey1).clamp(0) + 1e-7

    giou = iou - (enclosing - union) / enclosing
    return (1 - giou).mean()


def combined_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return smooth_l1_loss(pred, gt) + 0.5 * giou_loss(pred, gt)


@torch.no_grad()
def mean_iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Average IoU over all cells and batch items."""
    px1, py1, px2, py2 = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
    gx1, gy1, gx2, gy2 = gt[..., 0], gt[..., 1], gt[..., 2], gt[..., 3]

    ix1 = torch.max(px1, gx1)
    iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2)
    iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    area_p = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    area_g = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
    union = area_p + area_g - inter + 1e-7

    return (inter / union).mean().item()


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def step(self, val_loss: float, epoch: int) -> bool:
        """Returns True if training should stop."""
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: CellExtractorCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Run one epoch; returns (avg_loss, avg_miou)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_iou = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for imgs, gt_boxes in loader:
            imgs = imgs.to(device)
            gt_boxes = gt_boxes.to(device)

            pred = model(imgs)
            loss = combined_loss(pred, gt_boxes)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_loss += loss.item()
            total_iou += mean_iou(pred, gt_boxes)

    n = len(loader)
    return total_loss / n, total_iou / n


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    history: list[dict],
    early_stop: EarlyStopping,
    val_iou_at_best: float,
    elapsed: float,
    model_path: Path,
) -> None:
    print("\n" + "=" * 72)
    print("  CELL EXTRACTOR CNN — TRAINING REPORT")
    print("=" * 72)

    # Per-epoch table
    hdr = f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val mIoU':>9}  {'LR':>10}"
    print(hdr)
    print("-" * 72)
    for row in history:
        marker = " ←" if row["epoch"] == early_stop.best_epoch else ""
        print(
            f"{row['epoch']:>6}  {row['train_loss']:>10.5f}  {row['val_loss']:>10.5f}"
            f"  {row['val_miou']:>8.4f}  {row['lr']:>10.2e}{marker}"
        )

    print("-" * 72)

    # Summary
    best = next(r for r in history if r["epoch"] == early_stop.best_epoch)
    reason = (
        f"plateau — no improvement for {early_stop.patience} epochs"
        if early_stop.counter >= early_stop.patience
        else "reached max epochs"
    )
    print(f"\n  Stopped:       {reason}")
    print(f"  Total epochs:  {history[-1]['epoch']}")
    print(f"  Best epoch:    {early_stop.best_epoch}  (val loss {best['val_loss']:.5f})")
    print(f"  Best val mIoU: {val_iou_at_best:.4f}  ({val_iou_at_best * 100:.2f} %)")
    print(f"  Training time: {elapsed / 60:.1f} min")
    print(f"  Saved to:      {model_path}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    epochs: int = 80,
    batch_size: int = 16,
    lr: float = 3e-4,
    patience: int = 15,
    num_workers: int = 4,
    weight_decay: float = 1e-4,
) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:     {device}")
    print(f"Data root:  {DATA_ROOT_VICFL}")
    print(f"Output:     {OUT_MODEL}")
    print()

    # ---- Datasets ----
    train_ds = SudokuVicflDataset(DATA_ROOT_VICFL / "train", augment=True)
    val_ds   = SudokuVicflDataset(DATA_ROOT_VICFL / "valid", augment=False)
    test_ds  = SudokuVicflDataset(DATA_ROOT_VICFL / "test",  augment=False)

    print(f"Train:  {len(train_ds)} images")
    print(f"Valid:  {len(val_ds)} images")
    print(f"Test:   {len(test_ds)} images")
    print()

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    # ---- Model ----
    model = CellExtractorCNN(pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model:      CellExtractorCNN  ({n_params:,} trainable params)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )
    early_stop = EarlyStopping(patience=patience, min_delta=1e-4)

    print(f"Optimizer:  AdamW  lr={lr}  wd={weight_decay}")
    print(f"Scheduler:  ReduceLROnPlateau  factor=0.5  patience=5")
    print(f"Early stop: patience={patience}  min_delta=1e-4")
    print(f"Max epochs: {epochs}")
    print()

    history: list[dict] = []
    best_val_loss = float("inf")
    val_iou_at_best = 0.0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_iou = run_epoch(model, train_loader, optimizer, device)
        val_loss, val_iou = run_epoch(model, val_loader, None, device)

        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_miou": val_iou,
            "lr": current_lr,
        })

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            val_iou_at_best = val_iou
            torch.save(model.state_dict(), CKPT_DIR / "best.pth")

        marker = " ✓" if improved else ""
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train={train_loss:.5f}  val={val_loss:.5f}  "
            f"mIoU={val_iou:.4f}  lr={current_lr:.2e}{marker}"
        )

        if early_stop.step(val_loss, epoch):
            print(f"\n→ Early stopping: no improvement for {patience} epochs.")
            break

    elapsed = time.time() - t_start

    # ---- Restore best checkpoint ----
    best_ckpt = CKPT_DIR / "best.pth"
    if best_ckpt.exists():
        model.load_state_dict(
            torch.load(best_ckpt, map_location=device, weights_only=True)
        )

    # ---- Test set evaluation ----
    test_loss, test_iou = run_epoch(model, test_loader, None, device)
    print(f"\nTest set:  loss={test_loss:.5f}  mIoU={test_iou:.4f}")

    # ---- Save final model ----
    torch.save(model.state_dict(), OUT_MODEL)

    # ---- Report ----
    print_report(history, early_stop, val_iou_at_best, elapsed, OUT_MODEL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CellExtractorCNN")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()
    main(**vars(args))
