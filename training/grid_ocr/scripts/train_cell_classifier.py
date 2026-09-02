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

CELL_SIZE = 70          # must match extract_wicht_cells.py and GridOCRConfig.patch_size
REAL_CELLS_DIR = PROJECT / "data" / "grid_ocr" / "cells"
OUT_MODEL = PROJECT / "models" / "weights" / "grid_ocr_cnn.pth"
CKPT_DIR = PROJECT / "training" / "grid_ocr" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.skip: nn.Module = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))) + self.skip(x))


class GridOCRNet(nn.Module):
    """Residual CNN: CELL_SIZE × CELL_SIZE grayscale → 10 logits (0=empty, 1-9=digit).

    4 residual blocks (32→64→128→256) with stride-2 downsampling + GlobalAvgPool.
    """

    def __init__(self, cell_size: int = CELL_SIZE):
        super().__init__()
        self.cell_size = cell_size
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
        )
        self.layer1 = _ResBlock(32, 64, stride=2)    # 25×25
        self.layer2 = _ResBlock(64, 128, stride=2)   # 12×12
        self.layer3 = _ResBlock(128, 256, stride=2)  # 6×6
        self.layer4 = _ResBlock(256, 256, stride=1)  # 6×6
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(self.pool(x))

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
    def __init__(self, root: Path, transform=None, split: str = "all", val_frac: float = 0.2, seed: int = 42):
        """Load real cell crops.

        split='all'   → all images (legacy behaviour)
        split='train' → 80 % per class, stratified
        split='val'   → 20 % per class, stratified
        """
        self.samples: list[tuple[Path, int]] = []
        self.transform = transform
        rng = random.Random(seed)
        for label in range(10):
            label_dir = root / str(label)
            if not label_dir.exists():
                continue
            paths = sorted(label_dir.glob("*.jpg"))
            if split == "all":
                chosen = paths
            else:
                rng2 = random.Random(seed + label)
                shuffled = list(paths)
                rng2.shuffle(shuffled)
                n_val = max(1, int(len(shuffled) * val_frac))
                chosen = shuffled[n_val:] if split == "train" else shuffled[:n_val]
            for p in chosen:
                self.samples.append((p, label))
        rng.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            img = np.full((CELL_SIZE, CELL_SIZE), 255, dtype=np.uint8)
        img = cv2.resize(img, (CELL_SIZE, CELL_SIZE), interpolation=cv2.INTER_AREA)

        if self.transform:
            # Brightness / contrast jitter on numpy before tensor conversion
            if random.random() < 0.6:
                alpha = random.uniform(0.7, 1.35)   # contrast
                beta = random.randint(-25, 25)       # brightness
                img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
            # Random Gaussian blur
            if random.random() < 0.3:
                k = random.choice([3, 5])
                img = cv2.GaussianBlur(img, (k, k), 0)
            # JPEG compression artefacts
            if random.random() < 0.2:
                q = random.randint(50, 85)
                _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
                img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

        t = torch.from_numpy(img).float().unsqueeze(0) / 255.0

        if self.transform:
            # Tensor-space augmentations
            t = transforms.functional.affine(
                t,
                angle=random.uniform(-8, 8),
                translate=[int(CELL_SIZE * random.uniform(-0.08, 0.08)),
                           int(CELL_SIZE * random.uniform(-0.08, 0.08))],
                scale=random.uniform(0.88, 1.12),
                shear=random.uniform(-4, 4),
                fill=1.0,
            )
            t = transforms.RandomErasing(p=0.2, scale=(0.02, 0.10))(t)

        return t, label


# ---------------------------------------------------------------------------
# Synthetic dataset  (generated online — unlimited variety)
# ---------------------------------------------------------------------------

def _collect_system_fonts() -> list[str]:
    """Return unique OTF/TTF paths that PIL can load successfully."""
    import subprocess
    try:
        raw = subprocess.check_output(["fc-list"], text=True)
    except Exception:
        return []
    seen: set[str] = set()
    paths: list[str] = []
    for line in raw.splitlines():
        p = line.split(":")[0].strip()
        if p.endswith((".otf", ".ttf")) and p not in seen:
            seen.add(p)
            paths.append(p)
    return paths


_SYSTEM_FONTS: list[str] = _collect_system_fonts()


def _render_digit(digit: int, size: int) -> np.ndarray:
    """Render a single digit (0=empty white square) on a white background.

    Uses PIL with a random system font so the model sees dozens of typefaces
    (serif, sans-serif, mono, narrow) rather than just 5 OpenCV faces.
    """
    from PIL import Image, ImageDraw, ImageFont

    bg_val = random.randint(220, 255)
    canvas = np.ones((size * 3, size * 3), dtype=np.uint8) * bg_val

    if digit > 0:
        color = random.randint(0, 40)
        target_h = int(size * random.uniform(0.55, 0.80))
        font_size = max(8, int(target_h * 1.3))  # PIL font size > glyph height

        font: ImageFont.ImageFont | ImageFont.FreeTypeFont
        if _SYSTEM_FONTS and random.random() < 0.92:
            font_path = random.choice(_SYSTEM_FONTS)
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        txt = str(digit)
        pil_img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_img)

        # Measure actual bounding box so we can centre precisely.
        try:
            bbox = draw.textbbox((0, 0), txt, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = font_size, font_size

        cx = canvas.shape[1] // 2
        cy = canvas.shape[0] // 2
        tx = cx - tw // 2 + random.randint(-3, 3)
        ty = cy - th // 2 + random.randint(-3, 3)
        draw.text((tx, ty), txt, fill=int(color), font=font)
        canvas = np.array(pil_img)

    # Random light grid lines on empty cells sometimes
    if digit == 0 and random.random() < 0.2:
        lc = random.randint(160, 210)
        for _ in range(random.randint(1, 3)):
            x = random.randint(0, canvas.shape[1])
            cv2.line(canvas, (x, 0), (x, canvas.shape[0]), lc, 1)
        for _ in range(random.randint(1, 3)):
            y = random.randint(0, canvas.shape[0])
            cv2.line(canvas, (0, y), (canvas.shape[1], y), lc, 1)

    return _finish_cell(canvas, size, bg_val)


def _finish_cell(canvas: np.ndarray, size: int, bg_val: int) -> np.ndarray:
    """Rotate, crop, and apply the cell-domain artifacts (borders, noise, blur).

    Shared by the font-rendered and handwritten sources so both land in the
    same distribution as real extracted cells.
    """
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


MNIST_RAW = PROJECT / "training" / "sudoku_digit_classification" / "mnist_data" / "MNIST" / "raw"
MNIST_JPG_LABELS = PROJECT / "training" / "data" / "digit_classification" / "digits" / "labels.txt"


def _load_mnist(split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST images (uint8, N×28×28) and labels (1-9, zeros dropped).

    Tries IDX binary files first; falls back to the pre-extracted JPGs in
    training/data/digit_classification/digits/ when the IDX files are absent.
    The JPG fallback uses all 60k samples regardless of `split`.
    """
    if MNIST_RAW.exists():
        def _read(path: Path, kind: str) -> np.ndarray:
            raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            if kind == "images":
                n, rows, cols = (int.from_bytes(raw[i:i + 4], "big") for i in (4, 8, 12))
                return raw[16:].reshape(n, rows, cols)
            n = int.from_bytes(raw[4:8], "big")
            return raw[8:8 + n]

        stems = {"train": ["train"], "test": ["t10k"], "all": ["train", "t10k"]}[split]
        imgs = np.concatenate([
            _read(MNIST_RAW / f"{st}-images-idx3-ubyte", "images") for st in stems
        ])
        labels = np.concatenate([
            _read(MNIST_RAW / f"{st}-labels-idx1-ubyte", "labels") for st in stems
        ])
        keep = labels > 0
        return imgs[keep], labels[keep]

    # Fallback: load from pre-extracted JPGs (mnist_*.jpg entries in labels.txt).
    imgs_list, labels_list = [], []
    img_dir = MNIST_JPG_LABELS.parent / "images"
    for line in MNIST_JPG_LABELS.read_text().splitlines():
        if not line.strip():
            continue
        rel_path, label_str = line.rsplit(",", 1)
        label = int(label_str)
        if label == 0:  # digit zero has no class in sudoku OCR
            continue
        fname = Path(rel_path).name
        if not fname.startswith("mnist_"):
            continue
        img_path = img_dir / fname
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        imgs_list.append(img)
        labels_list.append(label)

    return np.stack(imgs_list), np.array(labels_list, dtype=np.uint8)


def _render_handwritten(glyph: np.ndarray, size: int) -> np.ndarray:
    """Render one MNIST glyph as a sudoku cell: dark ink on light paper.

    MNIST is white-on-black and tightly cropped; real cells are dark ink on
    paper with the digit occupying roughly 55-80 % of the cell height.  Ballpoint
    strokes are also lighter and thinner than print, so the ink tone is drawn
    from a wider, lighter range than the font renderer uses.
    """
    bg_val = random.randint(215, 255)
    canvas = np.ones((size * 3, size * 3), dtype=np.uint8) * bg_val

    ink = 255 - glyph                       # -> dark digit on white
    ys, xs = np.where(ink < 200)
    if len(ys) == 0:
        return _finish_cell(canvas, size, bg_val)
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    target_h = int(size * random.uniform(0.55, 0.82))
    scale = target_h / ink.shape[0]
    target_w = max(1, int(ink.shape[1] * scale * random.uniform(0.9, 1.1)))
    ink = cv2.resize(ink, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Biro is lighter than print: lift the darkest stroke value.
    ink_floor = random.randint(20, 90)
    ink = np.clip(ink.astype(np.float32), ink_floor, 255).astype(np.uint8)

    # Vary stroke weight, biased towards thinning.  MNIST is written with a
    # thick marker; ballpoint on paper is much finer.  That matters for closed
    # glyphs above all: a thick looped "2" fills its loop in and reads as a "9"
    # (observed on a real photo, where every looped 2 was misread).  Dilating
    # the light background thins the dark stroke; eroding thickens it.
    k = random.choice([0, 0, 1, 1, 1, 2])
    if k:
        kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
        if random.random() < 0.75:
            ink = cv2.dilate(ink, kernel)     # thinner stroke (biro)
        else:
            ink = cv2.erode(ink, kernel)      # bolder stroke

    # Paste into the centre third (that is the region _finish_cell keeps).
    cy = canvas.shape[0] // 2 + random.randint(-4, 4)
    cx = canvas.shape[1] // 2 + random.randint(-4, 4)
    y0 = cy - target_h // 2
    x0 = cx - target_w // 2
    region = canvas[y0:y0 + target_h, x0:x0 + target_w]
    canvas[y0:y0 + target_h, x0:x0 + target_w] = np.minimum(region, ink)

    return _finish_cell(canvas, size, bg_val)


class HandwrittenCellDataset(Dataset):
    """MNIST digits 1-9 rendered into the sudoku-cell domain."""

    def __init__(self, size: int = 40_000, cell_size: int = CELL_SIZE):
        self.size = size
        self.cell_size = cell_size
        self.images, self.labels_src = _load_mnist("train")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        j = random.randrange(len(self.images))
        img = _render_handwritten(self.images[j], self.cell_size)
        t = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        return t, int(self.labels_src[j])


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
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    synthetic_size: int = 80_000,
    handwritten_size: int = 60_000,
    num_workers: int = 4,
    out_model: Path = OUT_MODEL,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    real_train_ds = RealCellDataset(REAL_CELLS_DIR, transform=True, split="train")
    real_val_ds = RealCellDataset(REAL_CELLS_DIR, split="val")
    synth_ds = SyntheticCellDataset(size=synthetic_size)
    hand_ds = HandwrittenCellDataset(size=handwritten_size) if handwritten_size else None

    if len(real_train_ds) == 0:
        print("WARNING: No real cell data found. Run extract_and_label_cells.py first.")
        print("Training on synthetic data only.")
        train_ds = synth_ds
    else:
        parts = [real_train_ds, synth_ds]
        if hand_ds is not None:
            parts.append(hand_ds)
        print(f"Real train: {len(real_train_ds):,}   Real val: {len(real_val_ds):,}   "
              f"Synthetic: {len(synth_ds):,}   Handwritten: {len(hand_ds) if hand_ds else 0:,}")
        train_ds = ConcatDataset(parts)

    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(real_val_ds, batch_size=256, shuffle=False, num_workers=2) \
        if len(real_val_ds) > 0 else None

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
        shutil.copy(best_ckpt, out_model)
        print(f"\nBest model saved to {out_model}")
    else:
        torch.save(model.state_dict(), out_model)
        print(f"\nFinal model saved to {out_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--synthetic_size", type=int, default=60_000)
    parser.add_argument("--handwritten_size", type=int, default=40_000,
                        help="MNIST-derived handwritten cells per epoch (0 disables)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_model", type=Path, default=OUT_MODEL)
    args = parser.parse_args()
    train(**vars(args))
