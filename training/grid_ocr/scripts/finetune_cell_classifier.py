"""Fine-tune GridOCRNet on the existing cell dataset.

Addresses two failure modes found in the confusion matrix:
  1. Digits called empty (class 0) -- fixed by class weights + faint-digit aug
  2. Digit-to-digit confusions (9→7, 6→4, 2→7) -- fixed by hard-negative aug

Fine-tuning strategy:
  Phase 1 (epochs 1-8):   Freeze stem + layer1 + layer2; train head only.
                           Low LR so classifier adapts without disturbing features.
  Phase 2 (epochs 9-20):  Unfreeze all; full fine-tune at slightly higher LR.

The classifier gains a LayerNorm between Linear(256→128) and GELU; the old
checkpoint's missing LayerNorm weights are silently skipped (strict=False) and
left at their identity initialisation, so the model starts from the same
effective function and diverges only as training proceeds.

Usage:
    uv run python training/grid_ocr/scripts/finetune_cell_classifier.py
    uv run python training/grid_ocr/scripts/finetune_cell_classifier.py --epochs 20 --lr 1e-4
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))

# Re-use everything from the training script except the training loop.
sys.path.insert(0, str(PROJECT / "training" / "grid_ocr" / "scripts"))
from train_cell_classifier import (
    GridOCRNet,
    RealCellDataset,
    SyntheticCellDataset,
    HandwrittenCellDataset,
    CELL_SIZE,
    REAL_CELLS_DIR,
    OUT_MODEL,
    CKPT_DIR,
)

CHECKPOINT = CKPT_DIR / "best.pth"


# ---------------------------------------------------------------------------
# Augmented dataset wrapper: adds faint-digit simulation
# ---------------------------------------------------------------------------

class FaintAugDataset(torch.utils.data.Dataset):
    """Wraps any (tensor, label) dataset and randomly fades digit cells.

    With probability `faint_prob`, pixels are pushed toward white so the model
    learns to distinguish a very light digit stroke from a blank cell — the
    dominant failure mode in the confusion matrix (digit → empty).

    Only applied to digit classes (label > 0); empty cells are left as-is.
    """

    def __init__(self, dataset, faint_prob: float = 0.15):
        self.dataset = dataset
        self.faint_prob = faint_prob

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if label > 0 and random.random() < self.faint_prob:
            # Scale darkness towards white: fade = 1 means fully white, 0 = no change.
            # A fade of 0.5-0.8 produces realistically faint strokes.
            fade = random.uniform(0.50, 0.80)
            # img is [1, H, W] float in [0, 1]; 1.0 = white, 0.0 = black
            # Darkness from white: darkness = 1 - img
            img = 1.0 - (1.0 - img) * (1.0 - fade)
        return img, label


# ---------------------------------------------------------------------------
# Fine-tune loop
# ---------------------------------------------------------------------------

def finetune(
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-4,
    phase2_start: int = 9,
    synthetic_size: int = 30_000,
    handwritten_size: int = 20_000,
    num_workers: int = 4,
    faint_prob: float = 0.15,
    out_model: Path = OUT_MODEL,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = GridOCRNet(cell_size=CELL_SIZE).to(device)

    if CHECKPOINT.exists():
        raw = torch.load(CHECKPOINT, map_location=device)
        # The old classifier had no LayerNorm, so the final Linear lived at
        # index 4; the new head inserts LayerNorm at index 2, shifting the
        # final Linear to index 5.  Remap so the learned output projection is
        # preserved rather than re-initialised.
        remapped = {}
        for k, v in raw.items():
            if k == "classifier.4.weight":
                remapped["classifier.5.weight"] = v
            elif k == "classifier.4.bias":
                remapped["classifier.5.bias"] = v
            else:
                remapped[k] = v
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        fresh = [k for k in missing if "LayerNorm" not in k and "layer_norm" not in k]
        if fresh:
            print(f"Randomly initialised (not in checkpoint): {fresh}")
        print(f"Loaded weights from {CHECKPOINT}  "
              f"(LayerNorm initialised fresh, output projection remapped)")
    else:
        print(f"WARNING: checkpoint not found at {CHECKPOINT}, starting from scratch")

    # ── Datasets ────────────────────────────────────────────────────────────
    real_train = RealCellDataset(REAL_CELLS_DIR, transform=True, split="train")
    real_val   = RealCellDataset(REAL_CELLS_DIR, split="val")
    synth      = SyntheticCellDataset(size=synthetic_size)
    hand       = HandwrittenCellDataset(size=handwritten_size) if handwritten_size else None

    parts = [FaintAugDataset(real_train, faint_prob=faint_prob), synth]
    if hand is not None:
        parts.append(FaintAugDataset(hand, faint_prob=faint_prob))

    train_ds = ConcatDataset(parts)
    print(
        f"Real train: {len(real_train):,}   Real val: {len(real_val):,}   "
        f"Synthetic: {len(synth):,}   Handwritten: {len(hand) if hand else 0:,}"
    )

    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(real_val, batch_size=256, shuffle=False, num_workers=2) \
        if len(real_val) > 0 else None

    # ── Loss: down-weight class 0 to reduce digit→empty false predictions ──
    # The confusion matrix shows ~10-14% of digits mis-predicted as empty.
    # Giving class 0 weight 0.4 means those errors cost the same as a 40 %
    # confidence digit-to-digit error, pushing the model to commit to digits.
    class_weights = torch.tensor(
        [0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)

    # ── Phase 1: freeze backbone, train head only ───────────────────────────
    frozen_modules = [model.stem, model.layer1, model.layer2]
    for m in frozen_modules:
        for p in m.parameters():
            p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr * 0.5,  # half-LR for phase 1
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):

        # ── Phase 2 transition ──────────────────────────────────────────────
        if epoch == phase2_start:
            print(f"\nEpoch {epoch}: unfreezing all layers (phase 2)")
            for m in frozen_modules:
                for p in m.parameters():
                    p.requires_grad_(True)
            # Re-create optimizer so newly unfrozen params are included
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - phase2_start + 1
            )

        model.train()
        total_loss = correct = total = 0
        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            imgs = imgs.to(device)
            labels = (labels.to(device) if isinstance(labels, torch.Tensor)
                      else torch.tensor(labels).to(device))
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_acc = correct / total * 100
        msg = f"Epoch {epoch:3d}: loss={total_loss/len(loader):.4f}  train={train_acc:.2f}%"

        if val_loader is not None:
            model.eval()
            vc = vt = 0
            # Per-class breakdown for key classes
            class_correct = [0] * 10
            class_total   = [0] * 10
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    preds = model(imgs).argmax(1)
                    vc += (preds == labels).sum().item()
                    vt += labels.size(0)
                    for gt, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                        class_total[gt] += 1
                        if gt == pred:
                            class_correct[gt] += 1

            val_acc = vc / vt * 100
            # Show recall for most confused classes: empty(0), 9, 6
            recalls = {
                c: (class_correct[c] / class_total[c] * 100 if class_total[c] else 0)
                for c in [0, 6, 9]
            }
            msg += (f"  val={val_acc:.2f}%"
                    f"  ∅={recalls[0]:.0f}%  6={recalls[6]:.0f}%  9={recalls[9]:.0f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), CKPT_DIR / "best_ft.pth")
                msg += "  ✓ saved"

        print(msg)

    # ── Save final ──────────────────────────────────────────────────────────
    best = CKPT_DIR / "best_ft.pth"
    src  = best if best.exists() else None
    if src:
        import shutil
        shutil.copy(src, out_model)
        # Also overwrite the checkpoint used by future finetunes
        shutil.copy(src, CHECKPOINT)
        print(f"\nBest fine-tuned model → {out_model}")
    else:
        torch.save(model.state_dict(), out_model)
        print(f"\nFinal model → {out_model}")

    print(f"Best val acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--batch_size",      type=int,   default=128)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--phase2_start",    type=int,   default=9,
                        help="Epoch at which to unfreeze all layers")
    parser.add_argument("--synthetic_size",  type=int,   default=30_000)
    parser.add_argument("--handwritten_size",type=int,   default=20_000)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--faint_prob",      type=float, default=0.15,
                        help="Probability of applying faint-digit augmentation (digits only)")
    parser.add_argument("--out_model",       type=Path,  default=OUT_MODEL)
    args = parser.parse_args()
    finetune(**vars(args))
