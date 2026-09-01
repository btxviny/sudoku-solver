# Sudoku Cell Vision — Training Notes

## Pipeline (current)

Three stages, all YOLO-located, no Mask R-CNN anywhere:

1. **Grid detection + warp** — `YoloGridDetector` (YOLOv8n-seg by default, pose
   selectable). Corners are fitted to the predicted mask and the crop is
   rectified by `grid_geometry`.
2. **Cell segmentation** — `YoloCellExtractor` locates the 81 cells on the
   rectified grid, with an affine-lattice assignment and in-fill so every slot
   has a box.
3. **OCR per cell** — `SudokuPipeline._canonical_cells` cuts one image per cell
   and `GridOCR.read_cells` classifies each.

### Mask R-CNN removed

The 169 MB `maskrcnn_resnet50_fpn` detector was deleted along with
`grid_detector.py`, `GridDetectorConfig`, `latest_maskrcnn()`, the `--maskrcnn`
and `--threshold` CLI flags, the detection-threshold slider, the `warp` selector
in `PipelinePath`, and `training/segmentation/`.

It was not a trade-off. Measured on the same 90 held-out photos immediately
before removal:

| detector | cell-extractor | numbers-in-matrix | total | ms |
|---|---|---|---|---|
| Mask R-CNN warp | 44/45 | 31/45 | 75/90 | ~125 |
| **YOLOv8n warp** | **44/45** | **33/45** | **77/90** | **~59** |

The YOLO detector is more accurate *and* 2.4x faster, at 6 MB against 169 MB.
After removal the benchmark is 44/45 + 33/45 = **77/90**, identical to the YOLO
path's score before it — nothing else changed.

**`grid_geometry.py`** is new: the corner-fitting, Hough refinement, perspective
warp and overlay used to live as static methods on `GridDetector`, so the YOLO
detector had to import the Mask R-CNN class just to reach them. They are pure
OpenCV and know nothing about what produced the mask, so they now live in their
own module. The extraction was verified bit-identical against the originals on
9 images across all six helpers before `grid_detector.py` was deleted.

The Mask R-CNN checkpoint itself is still on disk and untracked by git — it is
not loaded by anything, and deleting it is a one-liner:

```bash
rm models/weights/maskrcnn_sudoku_*.pth models/weights/maskrcnn_sudoku_*_history.json
```

---

## Dataset

- **Source:** [Roboflow Universe — sudoku-cell-vision v6](https://universe.roboflow.com/pete-mksb1/sudoku-cell-vision/dataset/6)
- **Author:** Pete (workspace: `pete-mksb1`)
- **Task:** Object detection — locating and classifying cells in a 9×9 Sudoku grid
- **Classes:** `filled` (cell with a digit), `empty` (blank cell)
- **Split:** 125 train · 11 val · 11 test (147 images total)
- **Role in pipeline:** Step 2 of 3 — locate grid → **classify cells** → read digits

## Environment

- GPU: NVIDIA GeForce RTX 5070 Ti (16 GB)
- Python env: `uv` venv
- Ultralytics: 8.4.131 · PyTorch: 2.13.0+cu130

## Training

```bash
cd training/cell_extraction
uv run python train.py
```

Config: `yolov8n.pt`, 100 epochs, imgsz=640, batch=16, patience=20.  
Stopped at epoch 81 (early stopping; best weights from epoch ~61).

---

## Results

| Metric        | Value     |
|---------------|-----------|
| Precision     | **99.98%** |
| Recall        | **100.0%** |
| mAP@50        | **99.5%**  |
| mAP@50–95     | **89.6%**  |

Best weights: `training/cell_extraction/runs/cell_vision_v6/weights/best.pt`

Interactive report (charts + observations): https://claude.ai/code/artifact/1bbdeff6-514f-4cc0-92e7-213409b8aa6c

### Convergence story

mAP@50 hit **99.5%** at epoch 13 — effectively saturated in the first 16% of training.
The remaining 68 epochs only improved mAP@50-95 from 0.871 → 0.896 (stricter IoU thresholds).
Recall reached 100% at epoch 13 and never dropped. Precision stabilized at ~99.98%.

Classification loss (empty vs filled) collapsed from 3.84 → 0.27, confirming this is a trivially easy binary distinction once the grid is located.

### Key observations

- **Zero false negatives** on val from epoch 13 onward
- **Early stopping** (patience=20) triggered at epoch 81; best model around epoch 61 by mAP@50-95
- **YOLOv8n** (nano, ~3.2M params) is well-suited for CPU inference in the pipeline
- Val box loss plateaued at ~0.47–0.49 from epoch 60 onward — model is not overfitting

### mAP@50 progression (selected epochs)

| Epoch | mAP@50 | mAP@50-95 | Note |
|-------|--------|-----------|------|
| 1     | 6.1%   | 5.0%      | Warmup |
| 5     | 44.1%  | 33.0%     | Rapid climb |
| 8     | 97.2%  | 81.1%     | Near-convergence |
| 13    | **99.5%** | 87.1% | **mAP@50 saturates** |
| 23    | 99.5%  | 89.2%     | mAP@50-95 still improving |
| 44    | 99.5%  | 89.8%     | Near-peak strict mAP |
| 61    | 99.5%  | **89.8%** | **Best mAP@50-95 checkpoint** |
| 81    | 99.5%  | 88.6%     | Early stop |

---

## Next steps

- ~~Integrate `best.pt` into the pipeline as step 2 cell classifier~~ — done
  (`yolo` / `maskrcnn_yolo` segmentation modes)
- Evaluate on real sudoku photos (beyond the Roboflow dataset)
- Consider whether the nano model is sufficient or if yolov8s would help on harder cases

---

## Pipeline evaluation and path selection

### The `test_images/` set is not a benchmark

Checked by comparing detected grids: `sudoku.png`, `sudoku_1.png` and
`sudoku_yolo_cells.jpg` are **the same puzzle** (100 % clue agreement — the
canonical Wikipedia grid), and `sudoku_yolo_cells.jpg` is a debug artifact
emitted by `scripts/test_yolo_cell_extractor.py --visualize`, not a photo.
`PXL_20260827_184330968.jpg` is an **already-worked** grid — 80 of 81 cells
filled, mostly by hand — so solving it is trivial even read perfectly, and it
tests OCR rather than the solver. So the five files contain **two distinct
puzzles**, and any "N/5" figure is dominated by one
easy puzzle counted three times. Earlier notes quoting 4/5 were measuring that.

### Held-out evaluation

`sudoku-cell-vision` and `sudoku-cell-detector` were used to build the GridOCR
training set, so they cannot be used for evaluation. Measured on genuinely
held-out real photos (45 images each):

| Path | cell-extractor | numbers-in-matrix | ms |
|---|---|---|---|
| **`yolo_gridocr`** | **98 %** (44/45) | **73 %** (33/45) | 134 |
| `maskrcnn_yolo_yolodigit` | 96 % (43/45) | 38 % (17/45) | 125 |
| `yolo_yolodigit` (no warp) — removed | 0 % | 33 % | 19 |
| `yolo_xgboost` — removed | 0 % | 0 % | 173 |

`sudoku-vicfl` (synthetic) scores 0 % on every path with 17/45 detection
failures; it is not representative and is not used for evaluation.

### Decisions

**Kept.** `yolo_gridocr` is the recommended default — **77/90** held out against
`maskrcnn_yolo_yolodigit`'s 60/90.
`maskrcnn_yolo_yolodigit` is kept as the alternative: it is markedly better on
handwritten digits (57 % vs GridOCR's 14 %), so it is the one to reach for on a
part-worked puzzle.

**Discarded.** `yolo_xgboost` (0 % on both held-out sets) — removing it also
retired `digit_classifier.py`, `DigitClassifierConfig`, `ImageNetConfig`, the
ResNet18 feature extractor and the `xgboost` dependency. `yolo_yolodigit`, the
no-warp variant, produced invalid clues on 45/45 cell-extractor images: the
perspective warp is doing essential work, so the raw-image segmentation branch
was removed too. With one segmentation strategy left (Mask R-CNN warp → YOLO
cells), `seg_mode` was collapsed out of `run()`, which now takes only `ocr_mode`.

### Why `PXL_20260827_184330968.jpg` failed — and what it exposed

Traced stage by stage. Everything before digit recognition is correct:

| Stage | Result |
|---|---|
| Mask R-CNN grid detection | 1 instance, score 0.999 — correctly ignores the second, partial grid at the top of the page |
| YOLO cell extraction | 80 filled + 1 empty, 0 missing, blank placed at `[3,4]` — exactly the one blank cell on the paper |
| Digit recognition | **the only failing stage** |

The photo is an already-worked puzzle: 36 printed clues, 44 handwritten in blue
biro, one blank. Splitting accuracy by ink type (labels read off the photo by
eye) separated two genuinely different failures.

**`yolo_digit` — handwriting.** 94 % on printed digits, **57 %** on handwritten.
The readers learn from `data/grid_ocr/cells/`, printed Roboflow puzzles labelled
by EasyOCR, with no handwriting at all. Confusions are handwriting-shaped:
`2→9`, `1→4`, `4→5`.

**`grid_ocr` — misalignment, not handwriting.** It scored only 28 % on *printed*
digits, which handwriting cannot explain. GridOCR sliced the warped image into a
rigid uniform 9×9 grid, assuming the puzzle exactly fills the warp. It does not:
the perspective warp only approximates the corners, and the error accumulates —
measured drift was **21 px on a 50 px cell** by the last row. Patches held the
bottom of one digit and the top of the next, plus grid lines.

Fixed by aligning the patches to YOLO's detected cell boxes
(`GridOCR._patch_origins`, `read_with_probs(image, boxes_px)`) instead of a
uniform split. GridOCR already ran after YOLO, so the boxes were free:

| | before | after |
|---|---|---|
| PXL digit accuracy | 20 % | **71 %** |
| `sudoku-cell-extractor` solved | 27/45 | **43/45** |
| `numbers-in-matrix` solved | 14/45 | **24/45** |

This makes `yolo_gridocr` the best path again (67/90 vs 56/90 held out), so it
is the recommended default once more. The ranking had flipped only because the
misalignment bug was suppressing it.

**Second bug, same shape.** Both readers decided "is this grid low-contrast?"
from the raw min/max of the image. One dark page tab and one specular highlight
span the range, so the correction never fired on the washed-out photos that
needed it. Both now measure the range on 2nd/98th percentiles. For
`yolo_digit` this was worth 51 % → 74 % on this photo and 82 % → 89 % on
`sudoku-cell-extractor`; for `grid_ocr` it was marginal (20 % → 24 %) and the
alignment fix did the real work.

The image still does not solve — a worked puzzle needs near-perfect reading of
44 handwritten digits — but it was the most informative image in the repo, and
handwriting turned out to be only half the story.

### Cells that never reached OCR

Reported symptom: on the PXL photo some cells were not being read at all. Two
separate causes, both in `YoloCellExtractor._boxes_to_grid`.

**1. Row assignment assumed an axis-aligned grid.** Rows and columns were read
off the bounding box of the detection centres, i.e. `round((cy - y_min) / pitch)`.
That is only valid if the warp leaves the grid perfectly square. The PXL page is
curved, so the left of row 0 sits lower than its right: those four cells were
assigned to row 1, collided with the cells already there, and slots (0,0)-(0,3)
were left empty. YOLO had detected 86 boxes; four cells still reached the reader
as dead cells.

Fixed by fitting an **affine lattice** to the detections (initial axis-aligned
guess, then least-squares `centre -> (col,row)` refined a few passes) and
resolving collisions by distance to the lattice point rather than by box area.
Across the 90 held-out photos this cut empty slots on well-detected images to
**5**, from a much larger number.

**2. YOLO under-detects on a third of real photos.** 36 of 90 held-out images
returned fewer than 81 boxes — the worst short by 44. A slot with no box was a
dead cell: `yolo_digit` was told it was empty and skipped it, so those digits
were never read at all, and `grid_ocr` fell back to a guessed uniform position.

Since the cells form a lattice, a missing one can simply be placed:
`_fill_missing_from_lattice` inverts the fit to get its centre and uses the
median detection for its size. The cell then goes through OCR like any other and
the **reader** decides whether it is empty, instead of the detector deciding by
omission. Synthesised cells are marked filled for that reason.

| | cells with no box | on images |
|---|---|---|
| bounding-box assignment | 147 | 43/90 |
| fitted lattice | 131 | 41/90 |
| fitted lattice + in-fill | **0** | **0/90** |

Effect on the PXL photo: `grid_ocr` digit accuracy 90 % → **96 %**, `yolo_digit`
72 % → **79 %**, with no cell skipped. Held out, `yolo_gridocr` went 77 → **79/90**
and `maskrcnn_yolo_yolodigit` 60 → **61/90**.

### Pipeline shape: detect → warp → YOLO cells → OCR per cell

The intended architecture is three stages, and the third one was not honoured:
GridOCR never received individual cells. It took the whole rectified grid and
sliced it internally (`_patch_origins`), which is why the uniform-split
misalignment bug was possible at all.

The pipeline now cuts the cells and hands them to the reader:

- `SudokuPipeline._canonical_cells(rectified, boxes_px, patch)` returns 81 crops,
  one per YOLO box.
- `GridOCR.read_cells(crops, contrast_ref=...)` classifies each cell.
  `read`/`read_with_probs` remain for callers with no boxes and fall back to a
  uniform split. `_patch_origins` and the `boxes_px` plumbing are gone.

**Scale is the subtlety.** Cutting each YOLO box and resizing it to 50 px
independently *loses* accuracy — 68/90, against 77/90 — because GridOCRNet
learned 50 px cells cut from a 450 px grid, and tight boxes rescaled on their own
change how much of the cell the digit fills. Sampling every cell as a fixed
50 px window from the grid rescaled to 450 px keeps one canonical scale:

| per-cell variant | cell-extractor | numbers-in-matrix | total |
|---|---|---|---|
| YOLO box resized per cell | 39/45 | 29/45 | 68/90 |
| uniform size (median box) | 42/45 | 30/45 | 72/90 |
| ×1.15 of median box | 43/45 | 30/45 | 73/90 |
| **canonical-scale window** | **44/45** | **33/45** | **77/90** |

The canonical variant matches the previous slice-based numbers exactly, because
it samples the same pixels — so the architecture is now the requested one at no
cost. (An intermediate measurement of 74/90 was an artefact of a bench that
skipped constraint recovery, not of the method.)

### Warping from the segmentation mask, not a bounding box

The corners driving the perspective warp were still not coming from the mask.
`_corners_from_hough` used the mask only as an edge gate and a sanity check;
the corners themselves were the intersections of the *outermost* detected
horizontal and vertical lines — an extremal bounding quadrilateral. Overlaid on
a skewed photo it sat visibly outside the mask, swallowing page margin.

Replaced with a genuinely mask-driven construction:

1. `_corners_from_mask` fits **one line per side** to the mask contour (points
   assigned to sides via `minAreaRect`, ends trimmed to avoid rounded corners,
   Huber fit). This beats `approxPolyDP`, which was too crude — mask-only
   corners with approxPolyDP scored 31/45 and 4/45; with line fitting, 41/45
   and 9/45.
2. `_refine_on_grid_lines` then snaps each edge onto the real grid border found
   **within a band around that mask edge** (`BAND_OUT = 0.14`, `BAND_IN = 0.04`,
   orientation within 18°). Selection is by segment length with a mild outward
   preference — never "outermost in the image" — so a page edge or table rule
   away from the mask cannot win.

The refinement step is needed because the mask is a reliable *locator* but not a
reliable *boundary*: on `numbers-in-matrix` this Mask R-CNN under-segments the
bottom edge, cutting off the last row of digits (masks cover only 75-88 % of the
grid). Mask-only warping therefore collapses to 9/45 there. Letting the mask
choose which border to find, and the image supply its exact position, fixes both.

| corner source | cell-extractor | numbers-in-matrix | total |
|---|---|---|---|
| Hough outermost extremes (old) | 43/45 | 27/45 | 70/90 |
| mask only (approxPolyDP) | 31/45 | 4/45 | 35/90 |
| mask only (line fit) | 41/45 | 9/45 | 50/90 |
| **mask line fit + border refinement** | **44/45** | **33/45** | **77/90** |

Both paths improved: `yolo_gridocr` 70 -> 77/90, `maskrcnn_yolo_yolodigit`
55 -> 60/90. `_corners_from_hough`, `_quad_matches_mask` and `MASK_DILATION`
were removed.

### Retraining GridOCR on printed + handwritten

The remaining handwriting weakness was addressed by adding a handwritten source
to `training/grid_ocr/scripts/train_cell_classifier.py`: MNIST glyphs rendered
into the sudoku-cell domain (`HandwrittenCellDataset`, `_render_handwritten`).
The domain simulation — rotation, crop, grid-line bleed, noise, blur, JPEG — was
factored into `_finish_cell` so the font-rendered and handwritten sources go
through exactly the same pipeline.

Two details that matter:

- **MNIST zeros are dropped.** Class 0 here means *empty cell*; a sudoku never
  contains a 0 glyph, so keeping them would poison the empty class.
- **Handwriting is added to, not substituted for, the real printed crops.**
  Training mix is 4.7k real cells + 60k font-rendered + 40k handwritten. The
  earlier XGBoost failure came from training on synthetic data *alone*.

Trained 30 epochs (`--handwritten_size 40000`), evaluated against the previous
checkpoint before adopting:

| | old | mixed |
|---|---|---|
| PXL printed digits | 100 % | 100 % |
| PXL handwritten digits | 47.7 % | **90.9 %** |
| `sudoku-cell-extractor` solved | 43/45 | 43/45 |
| `numbers-in-matrix` solved | 24/45 | **27/45** |

No regression anywhere, so the model was adopted. `yolo_gridocr` now solves
**70/90** held out (78 %), against `maskrcnn_yolo_yolodigit`'s 55/90.

On the PXL photo, digit accuracy went **20 % → 95 %** across this investigation
(alignment fix, then the retrain). It still does not *solve*: it is an
already-worked grid where 80 cells are filled, so the 4 remaining misreads
conflict. A normal unsolved puzzle has far fewer digits to get right.

**Next:** `yolo_digit` is still printed-only (57 % on handwritten) and would
benefit from the same treatment — its training set is built by
`training/digit_classification/prepare_dataset.py` from the same printed crops.

### Improved: silent wrong answers

The pipeline reported success on under-determined puzzles. When recognition
misses digits, the remaining grid has many valid completions and the solver
returns an arbitrary one — a confident wrong answer. On held-out data this was
**14 of 74 apparent GridOCR successes**, some reading as few as 7 clues.

`SudokuSolver.has_other_solution` now re-solves with the first solution
forbidden; if a second exists the run is reported as a failure rather than a
solve. This is stricter than a clue-count threshold: on cell-extractor it
removed 10 "solves" that had ≥ 17 clues but were still ambiguous. All rates
above are post-guard and are therefore lower, and honest.

## Digit classification (YOLOv8n-cls)

Trained to fill the gap that left `yolo_yolodigit` unavailable — the config and
UI referenced `training/digit_classification/`, which had never been written.

- **Data:** `data/grid_ocr/cells/` — 4774 real 50×50 cell crops labelled by
  EasyOCR + solver verification. The 120k synthetic/MNIST set in
  `data/digit_classification/digits` is deliberately unused: it is what the
  failing XGBoost classifier was trained on.
- **Balance:** empties outnumber each digit ~14:1, so class 0 is capped at 2×
  the mean digit-class count → 1897 train / 334 val.
- **Result:** 91.6 % top-1, 97.9 % top-5 on held-out cells (best at epoch 7,
  early-stopped at 37).

```bash
uv run python training/digit_classification/prepare_dataset.py
uv run python training/digit_classification/train.py
```

Per-cell accuracy compounds: ~30 clues at 91.6 % rarely yields a clean grid, so
end-to-end it lands at 3/5 rather than GridOCR's 4/5. Failures are **not**
diffuse — the model is 100 % correct on 3 of 4 reference images and was
catastrophic on the fourth (`sudoku_3.png`, 3.8 % of filled cells) because that
grid is low-contrast (grey range 124 vs 255). Porting GridOCR's per-crop min-max
normalisation to `YoloDigitClassifier` lifted that image to 76.9 %.

**Next:** expose per-cell probabilities from `YoloDigitClassifier` so
`_recover_with_constraints` can repair its ambiguous reads — that machinery is
already wired for GridOCR and is the cheapest remaining win.

---

## Grid detection: mask vs bounding box

`_corners_from_hough` previously reduced the Mask R-CNN mask to a bounding box
padded 15 %, then ran Hough over that whole rectangle — so page text, table
rules and paper edges all competed with the real border. It now gates Canny
edges by the **segmentation mask itself**, and falls back to a quadrilateral
fitted to the mask outline when Hough fails or its quad does not cover roughly
the masked area (`_quad_matches_mask`).

The dilation applied before gating matters a great deal, because Mask R-CNN
under-segments the outer border. Solve counts over `test_images/`:

| Mask dilation (× grid extent) | Solved |
|---|---|
| no gating (full image) | 3/5 |
| 0.03 | 1/5 |
| 0.06 | 2/5 |
| 0.10 | 3/5 |
| **0.15** (`MASK_DILATION`) | **4/5** |
| 0.25 | 4/5 |

Too tight a gate clips the border line's edge, Hough rejects the fragments as
too short, and interior grid lines become the outermost lines — cropping the
outer row and column. Gating off entirely is also worse than 0.15, so the mask
is doing real work; it just has to be given room.
