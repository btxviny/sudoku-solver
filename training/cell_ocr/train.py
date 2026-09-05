"""Train CellOCRNet from scratch on real, typed and handwritten cells.

    uv run python training/cell_ocr/extract_real_cells.py   # once
    uv run python training/cell_ocr/train.py --epochs 30

Weights land at `models/weights/cell_ocr_cnn.pth`, where `CellOCRConfig`
expects them; the best checkpoint (with its EMA weights and optimizer state) is
kept in `training/cell_ocr/checkpoints/best.pth`.

Three validation sets are reported every epoch and they answer different
questions:

    real     held-out photograph cells        -- does it read real paper?
    synth    typed + EMNIST-test handwriting  -- does it generalise in-domain?
    mnist    MNIST glyphs, never trained on   -- does it read *unseen* hands?

Checkpoint selection deliberately ignores `mnist`.  The end-to-end benchmark
(`data/wicht_sudoku/half_mixed_test`) is built by pasting MNIST glyphs, so
selecting on MNIST would tune the model against the benchmark through the back
door.  It is reported, not optimised.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sudoku_solver.cell_ocr import CellOCRNet          # noqa: E402
from sudoku_solver.config import CellOCRConfig         # noqa: E402
import data as D                                       # noqa: E402

CKPT_DIR = PROJECT / "training" / "cell_ocr" / "checkpoints"


class EMA:
    """Exponential moving average of the weights.

    Cheap, and it matters here: the synthetic half of the corpus is regenerated
    every epoch, so the raw weights bounce around with whatever that epoch's
    glyphs happened to be.  The averaged weights are what gets exported.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                s.copy_(v.detach().float())     # BN num_batches_tracked

    def state_dict(self, like: nn.Module) -> dict:
        ref = like.state_dict()
        return {k: self.shadow[k].to(ref[k].dtype) for k in ref}


def build_loaders(args) -> tuple[DataLoader, dict[str, DataLoader], D.SyntheticCells]:
    real_train = D.RealCells(split="train", augment=True)
    if len(real_train) == 0:
        raise SystemExit(
            "No real cells found. Run:\n"
            "  uv run python training/cell_ocr/extract_real_cells.py"
        )
    synth_train = D.SyntheticCells(
        length=args.synthetic, handwritten_frac=args.handwritten_frac,
        seed=args.seed, glyphs=D.load_emnist("train"),
    )
    # Real photographs are the scarce, precious half: 13 k cells against an
    # unlimited synthesiser.  Repeating them keeps their share of each batch at
    # roughly a third instead of 10 %, which is what decides whether the model
    # learns paper texture or font rendering.
    train = ConcatDataset([real_train] * args.real_repeat + [synth_train])

    loader = DataLoader(
        train, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers else None,
    )

    val_synth = D.SyntheticCells(
        length=8192, handwritten_frac=args.handwritten_frac, seed=999_001,
        glyphs=D.load_emnist("test"),        # unseen writers
    )
    val_mnist = D.SyntheticCells(
        length=4096, handwritten_frac=1.0, seed=999_002,
        glyphs=D.load_mnist_holdout(),       # never trained on; reported only
    )
    vals = {
        "real": DataLoader(D.RealCells(split="val", augment=False),
                           batch_size=512, num_workers=max(2, args.workers // 2)),
        "synth": DataLoader(val_synth, batch_size=512, num_workers=max(2, args.workers // 2)),
        "mnist": DataLoader(val_mnist, batch_size=512, num_workers=max(2, args.workers // 2)),
    }
    return loader, vals, synth_train


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device) -> tuple[float, float]:
    """(accuracy over all cells, accuracy over the digit classes 1-9)."""
    model.eval()
    correct = total = digit_correct = digit_total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        hit = pred == y
        correct += int(hit.sum())
        total += y.numel()
        mask = y > 0
        digit_correct += int(hit[mask].sum())
        digit_total += int(mask.sum())
    return correct / max(1, total), digit_correct / max(1, digit_total)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--synthetic", type=int, default=90_000,
                    help="synthetic cells drawn per epoch")
    ap.add_argument("--real-repeat", type=int, default=4)
    ap.add_argument("--handwritten-frac", type=float, default=0.5)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = args.out or CellOCRConfig().model_path
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    loader, vals, synth = build_loaders(args)
    print(f"device={device}  train={len(loader.dataset):,} cells/epoch  "
          f"steps={len(loader):,}")

    model = CellOCRNet(cell_size=D.CELL_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CellOCRNet: {n_params / 1e6:.2f} M parameters")

    # No weight decay on norms and biases: decaying a BatchNorm scale fights the
    # normalisation rather than regularising anything.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    ema = EMA(model, decay=args.ema_decay)

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return args.lr * (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + math.cos(math.pi * t))

    best = -1.0
    step = 0
    ema_model = CellOCRNet(cell_size=D.CELL_SIZE).to(device)

    for epoch in range(args.epochs):
        synth.set_epoch(epoch)      # fresh glyphs, fresh artefacts
        model.train()
        t0 = time.time()
        running = correct = seen = 0
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False)
        for x, y in bar:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ema.update(model)
            step += 1

            running += float(loss.detach()) * y.numel()
            correct += int((logits.argmax(1) == y).sum())
            seen += y.numel()
            if step % 50 == 0:
                bar.set_postfix(loss=f"{running / seen:.4f}", acc=f"{correct / seen:.4f}")

        ema_model.load_state_dict(ema.state_dict(model))
        scores = {name: evaluate(ema_model, dl, device) for name, dl in vals.items()}
        # Selection ignores `mnist` on purpose -- see the module docstring.
        selection = (scores["real"][0] + scores["synth"][0]) / 2

        print(
            f"epoch {epoch + 1:2d}/{args.epochs}  "
            f"loss {running / seen:.4f}  train {correct / seen:.4f}  |  "
            + "  ".join(f"{n} {a:.4f}/{d:.4f}" for n, (a, d) in scores.items())
            + f"  |  sel {selection:.4f}  {time.time() - t0:.0f}s"
        )

        if selection > best:
            best = selection
            torch.save(ema_model.state_dict(), out_path)
            torch.save(
                {"epoch": epoch, "ema": ema_model.state_dict(),
                 "raw": model.state_dict(), "opt": opt.state_dict(),
                 "scores": scores, "selection": selection, "args": vars(args)},
                CKPT_DIR / "best.pth",
            )
            print(f"           saved -> {out_path.relative_to(PROJECT)}")

    print(f"\nBest selection score {best:.4f}; weights at {out_path}")
    print("Next:  uv run python scripts/eval_wicht.py data/wicht_sudoku/half_mixed_test --ocr cell_ocr")


if __name__ == "__main__":
    main()
