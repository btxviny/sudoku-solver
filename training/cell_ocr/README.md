# CellOCR — second-generation per-cell digit reader

Stage 3, alternative to [`grid_ocr`](../grid_ocr/README.md): classify the digit in
each of the 81 cell patches.  `CellOCRNet` takes a `70×70` grayscale crop and
returns 10 logits (0 = empty, 1–9 = digit).  All 81 cells go through in one
batched forward pass: `[81, 1, 70, 70]` → `[81, 10]`.

It is trained from scratch — no initialisation from GridOCRNet, no shared
weights — and both readers stay installed side by side so they can be compared
on the same images.

## Why a second model

GridOCRNet's weakness was handwriting, and two things in its setup made that
hard to see, let alone fix:

1. **Its handwriting benchmark was its own training data.**  The mixed grids in
   `data/wicht_sudoku/*mixed*` are built by `scripts/make_mixed_sudoku.py`
   pasting MNIST glyphs into real photos — and GridOCRNet trained on MNIST.
   Its reported handwriting accuracy was measured against glyphs it had already
   seen.
2. **It trained on raw crops and was served cleaned ones.**  At inference the
   pipeline runs `cell_prep.prep_patch` over every cell (grid-line removal,
   re-centring, low-contrast stretch).  The old training set skipped that step,
   so the model met a shifted distribution at serving time.

CellOCR fixes both, which is most of why it reads handwriting better — the
architecture is the smaller half of the story.

## Architecture

**CellOCRNet** — 3 stages of 2 squeeze-excitation residual blocks, dual-pooled
head.  ~2.0 M parameters, ~8 MB on disk.  Every op converts to TFLite without a
custom kernel, which is a hard requirement: this ships to Android.

```
Input  [B, 1, 70, 70]
Stem   Conv 5×5 → 32 ch                  full resolution
S1     2 × SE-ResBlock 32→64,   stride 2  → [B, 64, 35, 35]
S2     2 × SE-ResBlock 64→128,  stride 2  → [B, 128, 18, 18]
S3     2 × SE-ResBlock 128→192, stride 2  → [B, 192, 9, 9]
Head   concat(GlobalAvgPool, GlobalMaxPool) = 384
       Linear(384→192) → BN → SiLU → Dropout(0.3) → Linear(192→10)
```

Three differences from GridOCRNet, each aimed at a specific failure:

| Change | What it is for |
|---|---|
| Squeeze-excitation in every block | A cell is mostly paper. Channel gating suppresses features that fired on a grid-line remnant or page texture before they reach the classifier — the old model confused faint strokes with border artefacts. |
| Avg **and** max pooling at the head | Average pooling measures how much of the patch looks like ink, which favours thick print and washes out thin ballpoint. Max pooling keeps the strongest evidence regardless of coverage. |
| 5×5 stem at full resolution | Sees whole stroke junctions (the crossing of an 8, the closure of a 6) before any downsampling. |

## Dataset

| Source | Per epoch | Description |
|---|---|---|
| Real photo cells | 11 158 × 4 | 70 px patches cut from Wicht photos by the real pipeline, labelled from `.dat` |
| Typed synthetic | ~45 000 | 214 system fonts rendered into the cell domain, generated online |
| Handwritten synthetic | ~45 000 | EMNIST-digits glyphs composited into the cell domain, generated online |

The synthetic halves are regenerated every epoch, so no synthetic cell is ever
seen twice.  Real cells are repeated 4× to hold their share of each batch near a
third — they are the scarce half, and they are what teaches paper texture rather
than font rendering.

**MNIST is not in the training set.**  It is held out precisely because the
end-to-end benchmark is built from it; that makes `data/wicht_sudoku/half_mixed_test`
an honest measure of unseen handwriting, and it is why the numbers below can be
compared with GridOCRNet's at all.

### Building the real crops

```bash
uv run python training/cell_ocr/extract_real_cells.py
```

13 284 cells from 164 photographs (`v2_train`, `real_mixed_natural`), each one
produced by the same YOLO detect → YOLO cells → canonical 70 px sampling →
`prep_patch` chain that runs at inference.  `mixed`, `half_mixed_train` and the
held-out test splits are all excluded.

### Handwriting source

EMNIST-digits (240 000 glyphs, NIST SD-19 writers), unpacked to
`data/handwritten/gzip/`:

```bash
curl -L -o /tmp/emnist.zip https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip
unzip -j /tmp/emnist.zip 'gzip/emnist-digits-*' -d data/handwritten/gzip && gunzip data/handwritten/gzip/*.gz
```

### Cell-domain augmentation

Beyond the usual rotate/blur/noise/JPEG, the synthesiser models three things
the previous generator did not:

- **Stroke width in both directions.** Dataset glyphs are drawn thick at 28 px;
  a real ballpoint entry is thinner than anything in EMNIST. Thinning is what
  the old model never saw.
- **Neighbour bleed.** Canonical sampling centres a fixed window on each cell,
  so a slightly-off detection catches the edge of the digit next door. An empty
  cell holding half a stroke is the most common false positive, and no earlier
  training set contained one.
- **Uneven illumination.** A soft gradient across the cell, which is what a
  shadow or a page curl looks like at this scale.

Every sample then passes through `cell_prep.prep_patch`, sometimes down its
low-contrast branch, so the model trains on exactly what it is served.

## Training

```bash
uv run python training/cell_ocr/train.py --epochs 40
```

AdamW (lr 2e-3, cosine, 2 warm-up epochs), label smoothing 0.05, bf16 autocast,
gradient clipping, and an EMA of the weights — the EMA is what gets exported,
because the synthetic half is redrawn every epoch and the raw weights track
whichever glyphs that epoch happened to produce.

Weights land at `models/weights/cell_ocr_cnn.pth`; the best checkpoint (EMA, raw
and optimizer state) at `training/cell_ocr/checkpoints/best.pth`.

Three validation sets are reported each epoch: `real` (held-out photo cells),
`synth` (typed + EMNIST **test** writers) and `mnist` (held out entirely).
**Checkpoint selection uses only `real` and `synth`.**  Selecting on `mnist`
would tune the model against the end-to-end benchmark through the back door.

## Evaluation

```bash
uv run python scripts/eval_wicht.py data/wicht_sudoku/half_mixed_test
uv run python scripts/eval_wicht.py data/wicht_sudoku/v2_test
```

Both readers are scored on the *same* rectified grids and the same cell crops in
one pass, so any difference is between the networks and not between two runs of
the grid detector.

### Results

**`v2_test` — 40 held-out photographs of printed newspaper puzzles**

| | GridOCR | CellOCR |
|---|---|---|
| Grids read perfectly | 13/40 (32.5 %) | **38/40 (95.0 %)** |
| Grids solved (with recovery) | 30/40 (75.0 %) | **38/40 (95.0 %)** |
| Cell accuracy | 96.08 % | **98.52 %** |
| — printed cells | 95.07 % | **96.11 %** |
| — empty cells | 96.64 % | **99.86 %** |

**`half_mixed_test` — 40 grids, printed clues plus pasted MNIST handwriting**

| | GridOCR | CellOCR |
|---|---|---|
| Grids read perfectly | 15/40 (37.5 %) | **26/40 (65.0 %)** |
| Grids solved (with recovery) | 37/40 (92.5 %) | **38/40 (95.0 %)** |
| Cell accuracy | 96.88 % | **97.99 %** |
| — printed cells | 95.24 % | **96.11 %** |
| — handwritten cells | 99.45 %† | 97.66 % |
| — empty cells | 96.90 % | **99.78 %** |

† **Not a like-for-like number.**  Those handwritten cells *are* MNIST glyphs and
GridOCRNet trained on MNIST, so 99.45 % is a training-set score.  CellOCR's
97.66 % is on handwriting it has never seen — as is the 97.54 % digit accuracy
on the `mnist` validation split during training.  Read the two columns as
"memorised" versus "generalised", not as a 1.8-point loss.

The empty-cell column is where the largest real gain is, and it traces directly
to the neighbour-bleed augmentation: 96.64 % → 99.86 % means roughly 70 spurious
digits per 40 grids disappearing, and a spurious digit is what most often makes
an otherwise-correct grid unsolvable.

Validation at the selected checkpoint (epoch 36 of 40): real 97.20 % cells /
92.64 % digits, synth 98.89 % / 98.71 %, held-out MNIST 97.88 % / 97.54 %.

## TFLite export

```bash
uv run python scripts/export_tflite.py --only cellocr
uv run python scripts/verify_tflite.py --only cellocr
```

Output: `android/app/src/main/assets/cellocr.tflite` — `[81, 1, 70, 70]` →
`[81, 10]`, softmax applied in Kotlin.  The Android picker (`DigitModel`) selects
between the two readers at runtime; both are fed by the same `CellPreprocessor`,
so switching compares the networks and nothing else.
