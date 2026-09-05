"""Training corpus for CellOCRNet: real crops, typed digits, handwriting.

Three sources, combined per batch:

    real           70 px patches cut from Wicht photographs by the real
                   pipeline (`extract_real_cells.py`), labelled from .dat
    typed          system fonts rendered into the cell domain, generated online
    handwritten    EMNIST glyphs composited into the cell domain, online

**MNIST is not used anywhere here, and that is the point.**  The mixed
evaluation grids in `data/wicht_sudoku` are built by pasting MNIST glyphs, so a
model trained on MNIST is scored on its own training glyphs.  Handwriting comes
from EMNIST-digits (240 000 samples, NIST SD-19 writers) instead, which leaves
MNIST free to serve as a genuinely held-out handwriting test set.

Every sample -- real, typed or handwritten -- leaves this module having passed
through `cell_prep.prep_patch`, the same cleanup the pipeline applies at
inference.  The previous model trained on raw crops and was then asked to read
cleaned ones; closing that gap costs nothing and removes a whole class of
train/serve skew.
"""
from __future__ import annotations

import random
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PROJECT / "src"))
from sudoku_solver.cell_prep import prep_patch   # noqa: E402

CELL_SIZE = 70
REAL_DIR = PROJECT / "data" / "cell_ocr" / "real"
EMNIST_DIR = PROJECT / "data" / "handwritten" / "gzip"
MNIST_JPG = PROJECT / "training" / "data" / "digit_classification" / "digits"


# ── glyph sources ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def system_fonts() -> tuple[str, ...]:
    """Loadable OTF/TTF paths, so the model sees many typefaces, not five."""
    try:
        raw = subprocess.check_output(["fc-list"], text=True)
    except Exception:
        return ()
    seen: dict[str, None] = {}
    for line in raw.splitlines():
        p = line.split(":")[0].strip()
        if p.endswith((".otf", ".ttf")):
            seen.setdefault(p, None)
    return tuple(seen)


def _read_idx(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    magic = int.from_bytes(raw[:4].tobytes(), "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(raw[4 + 4 * i:8 + 4 * i].tobytes(), "big") for i in range(ndim)]
    return raw[4 + 4 * ndim:].reshape(dims)


@lru_cache(maxsize=2)
def load_emnist(split: str = "train") -> tuple[np.ndarray, ...]:
    """EMNIST-digits glyphs grouped by label, index 0 unused (empty is not a digit).

    EMNIST ships its images transposed relative to MNIST; they are corrected
    here, so a glyph comes back upright, white-on-black, 28x28.
    """
    stem = f"emnist-digits-{split}"
    imgs = _read_idx(EMNIST_DIR / f"{stem}-images-idx3-ubyte")
    labels = _read_idx(EMNIST_DIR / f"{stem}-labels-idx1-ubyte")
    imgs = np.transpose(imgs, (0, 2, 1))          # EMNIST is column-major
    return tuple(imgs[labels == d] if d > 0 else np.empty((0, 28, 28), np.uint8)
                 for d in range(10))


@lru_cache(maxsize=1)
def load_mnist_holdout() -> tuple[np.ndarray, ...]:
    """MNIST glyphs, used only to *validate* handwriting -- never to train.

    Reads the pre-extracted JPGs in `training/data/digit_classification/digits`.
    """
    by_digit: list[list[np.ndarray]] = [[] for _ in range(10)]
    labels_file = MNIST_JPG / "labels.txt"
    if not labels_file.exists():
        return tuple(np.empty((0, 28, 28), np.uint8) for _ in range(10))
    for line in labels_file.read_text().splitlines():
        if not line.strip():
            continue
        rel, label_str = line.rsplit(",", 1)
        label = int(label_str)
        name = Path(rel).name
        if label == 0 or not name.startswith("mnist_"):
            continue
        img = cv2.imread(str(MNIST_JPG / "images" / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        by_digit[label].append(cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA))
    return tuple(np.stack(v) if v else np.empty((0, 28, 28), np.uint8) for v in by_digit)


# ── cell-domain rendering ────────────────────────────────────────────────────

def _paper(size: int, rng: random.Random) -> tuple[np.ndarray, int]:
    """A blank 3x-oversized cell of paper, plus its base tone."""
    bg = rng.randint(215, 255)
    return np.full((size * 3, size * 3), bg, dtype=np.uint8), bg


def render_typed(digit: int, size: int, rng: random.Random) -> np.ndarray:
    """A printed digit (or an empty cell for 0) on paper, before cell artefacts."""
    from PIL import Image, ImageDraw, ImageFont

    canvas, bg = _paper(size, rng)
    if digit > 0:
        fonts = system_fonts()
        target_h = int(size * rng.uniform(0.50, 0.82))
        font_size = max(8, int(target_h * 1.3))
        try:
            font = (ImageFont.truetype(rng.choice(fonts), font_size)
                    if fonts and rng.random() < 0.95 else ImageFont.load_default())
        except Exception:
            font = ImageFont.load_default()

        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)
        txt = str(digit)
        try:
            bbox = draw.textbbox((0, 0), txt, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            ox, oy = bbox[0], bbox[1]
        except Exception:
            tw, th, ox, oy = font_size, font_size, 0, 0
        cx, cy = canvas.shape[1] // 2, canvas.shape[0] // 2
        draw.text((cx - tw // 2 - ox + rng.randint(-3, 3),
                   cy - th // 2 - oy + rng.randint(-3, 3)),
                  txt, fill=rng.randint(0, 45), font=font)
        canvas = np.array(pil)

        # Newspaper print is not solid: some glyphs come out bolded by ink
        # spread, others broken up by a coarse screen.
        if rng.random() < 0.25:
            k = np.ones((2, 2), np.uint8)
            canvas = (cv2.erode(canvas, k) if rng.random() < 0.5
                      else cv2.dilate(canvas, k))
    return _finish_cell(canvas, size, bg, rng)


def render_handwritten(glyph: np.ndarray, size: int, rng: random.Random) -> np.ndarray:
    """One 28x28 white-on-black glyph rendered as ink on paper.

    Stroke width is varied in both directions.  Dataset glyphs are drawn with a
    thick marker at 28 px; a ballpoint entry in a newspaper sudoku is thinner
    than anything in EMNIST, and thinning is what the old model never saw.
    """
    canvas, bg = _paper(size, rng)
    ink = 255 - glyph                                  # dark digit on white
    ys, xs = np.where(ink < 200)
    if len(ys) == 0:
        return _finish_cell(canvas, size, bg, rng)

    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    target_h = int(size * rng.uniform(0.45, 0.80))
    scale = target_h / ink.shape[0]
    glyph_img = cv2.resize(
        ink, (max(1, int(ink.shape[1] * scale)), target_h), interpolation=cv2.INTER_AREA
    )

    if rng.random() < 0.55:                            # thin the stroke
        k = np.ones((rng.choice([2, 3]),) * 2, np.uint8)
        glyph_img = cv2.dilate(glyph_img, k)           # dilate on the light image
    elif rng.random() < 0.3:                           # or thicken it
        glyph_img = cv2.erode(glyph_img, np.ones((2, 2), np.uint8))

    if rng.random() < 0.4:                             # slant, as in cursive
        shear = rng.uniform(-0.25, 0.25)
        h, w = glyph_img.shape
        M = np.float32([[1, shear, -shear * h / 2], [0, 1, 0]])
        glyph_img = cv2.warpAffine(glyph_img, M, (w, h), borderValue=255)

    # Ballpoint is lighter and bluer than print; in grayscale that is a tone
    # floor well above black.
    ink_tone = rng.randint(20, 110)
    alpha = np.clip((255.0 - glyph_img) / 255.0 * rng.uniform(0.85, 1.0), 0, 1)
    h, w = glyph_img.shape
    y0 = canvas.shape[0] // 2 - h // 2 + rng.randint(-3, 3)
    x0 = canvas.shape[1] // 2 - w // 2 + rng.randint(-3, 3)
    y0, x0 = max(0, y0), max(0, x0)
    region = canvas[y0:y0 + h, x0:x0 + w].astype(np.float32)
    a = alpha[:region.shape[0], :region.shape[1]]
    canvas[y0:y0 + h, x0:x0 + w] = (region * (1 - a) + ink_tone * a).astype(np.uint8)
    return _finish_cell(canvas, size, bg, rng)


def _neighbour_bleed(canvas: np.ndarray, size: int, rng: random.Random) -> None:
    """Intrude a sliver of an adjacent cell's digit from one edge, in place.

    Canonical sampling centres a fixed window on each detected cell, so when the
    detector's box is a few pixels off the window catches the edge of the digit
    next door.  An empty cell that contains half a stroke is the single most
    common false positive, and no previous training set contained one.
    """
    from PIL import Image, ImageDraw, ImageFont

    fonts = system_fonts()
    if not fonts:
        return
    strip = np.full_like(canvas, 255)
    pil = Image.fromarray(strip)
    try:
        font = ImageFont.truetype(rng.choice(fonts), int(size * 1.1))
    except Exception:
        return
    ImageDraw.Draw(pil).text(
        (strip.shape[1] // 2, strip.shape[0] // 2), str(rng.randint(1, 9)),
        fill=rng.randint(0, 60), font=font, anchor="mm",
    )
    strip = np.array(pil)
    depth = rng.randint(int(size * 0.10), int(size * 0.35))
    side = rng.choice(["top", "bottom", "left", "right"])
    if side == "top":
        canvas[:depth, :] = np.minimum(canvas[:depth, :], strip[-depth:, :])
    elif side == "bottom":
        canvas[-depth:, :] = np.minimum(canvas[-depth:, :], strip[:depth, :])
    elif side == "left":
        canvas[:, :depth] = np.minimum(canvas[:, :depth], strip[:, -depth:])
    else:
        canvas[:, -depth:] = np.minimum(canvas[:, -depth:], strip[:, :depth])


def _finish_cell(canvas: np.ndarray, size: int, bg: int, rng: random.Random) -> np.ndarray:
    """Rotate, crop to one cell, then apply the artefacts a photo really has."""
    angle = rng.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((canvas.shape[1] / 2, canvas.shape[0] / 2), angle, 1.0)
    canvas = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]), borderValue=bg)

    s = canvas.shape[0] // 3
    canvas = canvas[s:2 * s, s:2 * s]
    canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)

    # Printed grid lines, at a random offset: the warp rarely lands the frame
    # exactly on the patch edge.
    if rng.random() < 0.55:
        for side in rng.sample(["top", "bottom", "left", "right"],
                               k=rng.choices([1, 2, 3], weights=[5, 3, 1])[0]):
            w = rng.randint(1, 6)
            off = rng.randint(0, 8)
            colour = rng.randint(0, 30)
            if side == "top":
                canvas[off:off + w, :] = colour
            elif side == "bottom":
                canvas[max(0, size - off - w):size - off, :] = colour
            elif side == "left":
                canvas[:, off:off + w] = colour
            else:
                canvas[:, max(0, size - off - w):size - off] = colour

    if rng.random() < 0.30:
        _neighbour_bleed(canvas, size, rng)

    # Uneven lighting: a photograph of paper is never lit flat, and a soft
    # gradient across the cell is what a shadow or a page curl looks like at
    # this scale.
    if rng.random() < 0.45:
        gy, gx = np.mgrid[0:size, 0:size].astype(np.float32) / size
        theta = rng.uniform(0, 2 * np.pi)
        ramp = gx * np.cos(theta) + gy * np.sin(theta)
        canvas = np.clip(canvas.astype(np.float32) +
                         (ramp - 0.5) * rng.uniform(-70, 70), 0, 255).astype(np.uint8)

    if rng.random() < 0.5:
        noise = np.random.normal(0, rng.uniform(2, 11), canvas.shape).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        canvas = cv2.GaussianBlur(canvas, (rng.choice([3, 5]),) * 2, 0)
    if rng.random() < 0.20:                             # camera shake
        k = rng.choice([5, 7])
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        Mr = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
        canvas = cv2.filter2D(canvas, -1, cv2.warpAffine(kernel, Mr, (k, k)))
    if rng.random() < 0.25:
        q = rng.randint(45, 85)
        _, enc = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, q])
        canvas = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return canvas


def to_model_input(patch: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply the inference-time cleanup, sometimes down the low-contrast branch.

    `prep_patch` behaves differently on a washed-out grid, so the model must see
    both of its outputs.  The compression is applied first, exactly as a dim
    photograph would, and then the branch it triggers is taken.
    """
    if rng.random() < 0.18:
        lo = rng.randint(40, 110)
        hi = lo + rng.randint(40, 120)
        patch = (lo + patch.astype(np.float32) / 255.0 * (hi - lo)).astype(np.uint8)
        return prep_patch(patch, True)
    return prep_patch(patch, False)


# ── datasets ─────────────────────────────────────────────────────────────────

class SyntheticCells(Dataset):
    """Endless typed/handwritten cells, generated fresh every epoch.

    `length` is the samples drawn per epoch; a given index is *not* stable
    across epochs, which is deliberate -- the model should never see the same
    synthetic cell twice.
    """

    def __init__(
        self,
        length: int,
        handwritten_frac: float = 0.5,
        empty_weight: float = 1.6,
        seed: int = 0,
        glyphs: tuple[np.ndarray, ...] | None = None,
    ):
        self.length = length
        self.handwritten_frac = handwritten_frac
        self.seed = seed
        self.glyphs = glyphs if glyphs is not None else load_emnist("train")
        # Empty cells are the majority class in a real puzzle; over-weighting
        # class 0 a little keeps the empty/filled boundary sharp without letting
        # "answer empty" become a winning strategy.
        w = [empty_weight] + [1.0] * 9
        self.class_p = np.array(w) / sum(w)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        rng = random.Random((self.seed, self.epoch, idx).__hash__())
        label = int(np.searchsorted(np.cumsum(self.class_p), rng.random()))
        label = min(label, 9)
        if label > 0 and rng.random() < self.handwritten_frac and len(self.glyphs[label]):
            pool = self.glyphs[label]
            patch = render_handwritten(pool[rng.randrange(len(pool))], CELL_SIZE, rng)
        else:
            patch = render_typed(label, CELL_SIZE, rng)
        patch = to_model_input(patch, rng)
        return torch.from_numpy(patch).float().unsqueeze(0) / 255.0, label


class RealCells(Dataset):
    """Extracted photograph cells, split per class so both sides stay balanced."""

    def __init__(
        self,
        root: Path = REAL_DIR,
        split: str = "train",
        val_frac: float = 0.15,
        augment: bool = True,
        seed: int = 42,
    ):
        self.samples: list[tuple[Path, int]] = []
        self.augment = augment
        for label in range(10):
            paths = sorted((root / str(label)).glob("*.png")) if (root / str(label)).exists() else []
            rng = random.Random(seed + label)
            rng.shuffle(paths)
            n_val = max(1, int(len(paths) * val_frac)) if paths else 0
            chosen = paths[n_val:] if split == "train" else paths[:n_val]
            self.samples += [(p, label) for p in chosen]
        random.Random(seed).shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.full((CELL_SIZE, CELL_SIZE), 255, np.uint8)
        if img.shape != (CELL_SIZE, CELL_SIZE):
            img = cv2.resize(img, (CELL_SIZE, CELL_SIZE), interpolation=cv2.INTER_AREA)

        if self.augment:
            rng = random.Random()
            # These crops are already cleaned patches, so augmentation here is
            # photometric and small-geometric only -- re-running the cell
            # artefacts would double-apply what is baked in.
            if rng.random() < 0.6:
                a, b = rng.uniform(0.75, 1.3), rng.randint(-25, 25)
                img = np.clip(a * img.astype(np.float32) + b, 0, 255).astype(np.uint8)
            if rng.random() < 0.3:
                img = cv2.GaussianBlur(img, (rng.choice([3, 5]),) * 2, 0)
            if rng.random() < 0.5:
                M = cv2.getRotationMatrix2D(
                    (CELL_SIZE / 2, CELL_SIZE / 2), rng.uniform(-7, 7),
                    rng.uniform(0.9, 1.1),
                )
                M[:, 2] += [rng.uniform(-4, 4), rng.uniform(-4, 4)]
                img = cv2.warpAffine(img, M, (CELL_SIZE, CELL_SIZE), borderValue=255)
        return torch.from_numpy(img).float().unsqueeze(0) / 255.0, label
