"""Train a lightweight CNN digit classifier on real + synthetic cell images.

The trained model is saved to models/weights/grid_ocr_cnn.pth and is used
by src/sudoku_solver/grid_ocr.py for end-to-end grid reading.

Usage:
    python training/grid_ocr/scripts/train_cell_classifier.py [--epochs 40]
"""
import argparse
import random
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))

CELL_SIZE = 50          # must match extract_and_label_cells.py and GridOCR.patch_size
REAL_CELLS_DIR = PROJECT / "data" / "grid_ocr" / "cells"
OUT_MODEL = PROJECT / "models" / "weights" / "grid_ocr_cnn.pth"
CKPT_DIR = PROJECT / "training" / "grid_ocr" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GridOCRNet(nn.Module):
    """Lightweight CNN: CELL_SIZE × CELL_SIZE grayscale → 10 logits (0=empty, 1-9=digit)."""

    def __init__(self, cell_size: int = CELL_SIZE):
        super().__init__()
        self.cell_size = cell_size
        self.features = nn.Sequential(
            # Block 1  cell_size → cell_size/2
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2),
            # Block 2  /2 → /4
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2),
            # Block 3  /4 → /8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(4),   # fixed 4×4 regardless of cell_size
            nn.Flatten(),              # 128 × 16 = 2048
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def read_grid(self, grid_img: np.ndarray, device: torch.device) -> np.ndarray:
        """Read a single rectified grid image → 9×9 uint8 array.

        Args:
            grid_img: (H, W, 3) RGB uint8. H and W must both be divisible by 9.
        """
        H, W = grid_img.shape[:2]
        PS_h, PS_w = H // 9, W // 9
        gray = cv2.cvtColor(grid_img, cv2.COLOR_RGB2GRAY)
        patches = []
        for r in range(9):
            for c in range(9):
                patch = gray[r * PS_h:(r + 1) * PS_h, c * PS_w:(c + 1) * PS_w]
                patch = cv2.resize(patch, (self.cell_size, self.cell_size),
                                   interpolation=cv2.INTER_AREA)
                patches.append(patch)
        t = torch.from_numpy(np.stack(patches)).float().unsqueeze(1) / 255.0
        t = t.to(device)
        with torch.no_grad():
            logits = self(t)  # (81, 10)
        return logits.argmax(dim=1).cpu().numpy().reshape(9, 9).astype(np.uint8)


# ---------------------------------------------------------------------------
# Real-cell dataset
# ---------------------------------------------------------------------------

class RealCellDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.samples: list[tuple[Path, int]] = []
        self.transform = transform
        for label in range(10):
            label_dir = root / str(label)
            if not label_dir.exists():
                continue
            for p in label_dir.glob("*.jpg"):
                self.samples.append((p, label))
        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            img = np.zeros((CELL_SIZE, CELL_SIZE), dtype=np.uint8)
        img = cv2.resize(img, (CELL_SIZE, CELL_SIZE), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        if self.transform:
            t = self.transform(t)
        return t, label


# ---------------------------------------------------------------------------
# Synthetic dataset  (generated online — unlimited variety)
# ---------------------------------------------------------------------------

_CV_FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
]


def _render_digit(digit: int, size: int) -> np.ndarray:
    """Render a single digit (0=empty white square) on a white background."""
    bg_val = random.randint(220, 255)
    canvas = np.ones((size * 3, size * 3), dtype=np.uint8) * bg_val

    if digit > 0:
        font = random.choice(_CV_FONTS)
        scale = random.uniform(2.0, 3.5) * (size / 50)
        thick = random.randint(1, 3)
        color = random.randint(0, 40)
        txt = str(digit)
        (tw, th), _ = cv2.getTextSize(txt, font, scale, thick)
        tx = (canvas.shape[1] - tw) // 2 + random.randint(-3, 3)
        ty = (canvas.shape[0] + th) // 2 + random.randint(-3, 3)
        cv2.putText(canvas, txt, (tx, ty), font, scale, color, thick, cv2.LINE_AA)

    # Random light grid lines on empty cells sometimes
    if digit == 0 and random.random() < 0.2:
        lc = random.randint(160, 210)
        for _ in range(random.randint(1, 3)):
            x = random.randint(0, canvas.shape[1])
            cv2.line(canvas, (x, 0), (x, canvas.shape[0]), lc, 1)
        for _ in range(random.randint(1, 3)):
            y = random.randint(0, canvas.shape[0])
            cv2.line(canvas, (0, y), (canvas.shape[1], y), lc, 1)

    # Random rotation
    angle = random.uniform(-8, 8)
    M = cv2.getRotationMatrix2D(
        (canvas.shape[1] / 2, canvas.shape[0] / 2), angle, 1.0
    )
    canvas = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]),
                             borderValue=bg_val)

    # Crop to centre region (simulate trim)
    s = canvas.shape[0] // 3
    canvas = canvas[s:2 * s, s:2 * s]
    canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)

    # --- Simulate grid-line bleed-in ---
    # Real sudoku cells are bordered by printed lines: outer borders can be
    # 4-6 px thick, 3×3 box boundaries 2-4 px, inner lines 1-2 px.
    # The border may appear at a random offset from the cell edge (e.g. when
    # the warp alignment places the outer frame a few px inside the patch).
    # Training with this augmentation makes the model robust to these
    # artifacts regardless of where the border falls in the patch.
    if random.random() < 0.50:
        n_sides = random.choices([1, 2, 3], weights=[5, 3, 1])[0]
        sides = random.sample(['top', 'bottom', 'left', 'right'], k=n_sides)
        for side in sides:
            w = random.randint(1, 6)          # line width in pixels
            offset = random.randint(0, 8)     # border may start offset from the edge
            color = random.randint(0, 30)      # dark, not necessarily pure black
            if side == 'top':
                canvas[offset:offset + w, :] = color
            elif side == 'bottom':
                end = size - offset
                canvas[max(0, end - w):end, :] = color
            elif side == 'left':
                canvas[:, offset:offset + w] = color
            else:
                end = size - offset
                canvas[:, max(0, end - w):end] = color

    # Augmentation
    if random.random() < 0.5:
        sigma = random.uniform(2, 10)
        noise = np.random.normal(0, sigma, canvas.shape).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.35:
        k = random.choice([3, 5])
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)
    if random.random() < 0.25:
        q = random.randint(50, 85)
        _, enc = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, q])
        canvas = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

    return canvas


class SyntheticCellDataset(Dataset):
    def __init__(self, size: int = 60_000, cell_size: int = CELL_SIZE):
        self.size = size
        self.cell_size = cell_size
        # Balanced: equal samples per class
        self.labels = [i % 10 for i in range(size)]
        random.shuffle(self.labels)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        label = self.labels[idx]
        img = _render_digit(label, self.cell_size)
        t = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        return t, label


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    synthetic_size: int = 60_000,
    num_workers: int = 4,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Augmentation for real cells
    real_aug = transforms.Compose([
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
    ])

    real_ds = RealCellDataset(REAL_CELLS_DIR, transform=real_aug)
    synth_ds = SyntheticCellDataset(size=synthetic_size)

    if len(real_ds) == 0:
        print("WARNING: No real cell data found. Run extract_and_label_cells.py first.")
        print("Training on synthetic data only.")
        train_ds = synth_ds
    else:
        print(f"Real cells: {len(real_ds):,}   Synthetic: {len(synth_ds):,}")
        train_ds = ConcatDataset([real_ds, synth_ds])

    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )

    val_ds = RealCellDataset(REAL_CELLS_DIR)  # re-use real data as quick sanity val
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2) \
        if len(val_ds) > 0 else None

    model = GridOCRNet(cell_size=CELL_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(loader),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            imgs = imgs.to(device)
            labels = torch.tensor(labels).to(device) if not isinstance(labels, torch.Tensor) \
                else labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100
        msg = f"Epoch {epoch:3d}: loss={total_loss / len(loader):.4f}  train_acc={train_acc:.2f}%"

        if val_loader is not None:
            model.eval()
            vcorrect = vtotal = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    vcorrect += (model(imgs).argmax(1) == labels).sum().item()
                    vtotal += labels.size(0)
            val_acc = vcorrect / vtotal * 100
            msg += f"  val_acc={val_acc:.2f}%"
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), CKPT_DIR / "best.pth")
        else:
            if train_acc > best_val_acc:
                best_val_acc = train_acc
                torch.save(model.state_dict(), CKPT_DIR / "best.pth")

        print(msg)

    # Save final model
    best_ckpt = CKPT_DIR / "best.pth"
    if best_ckpt.exists():
        import shutil
        shutil.copy(best_ckpt, OUT_MODEL)
        print(f"\nBest model saved to {OUT_MODEL}")
    else:
        torch.save(model.state_dict(), OUT_MODEL)
        print(f"\nFinal model saved to {OUT_MODEL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--synthetic_size", type=int, default=60_000)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    train(**vars(args))
