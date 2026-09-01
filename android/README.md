# Sudoku Solver — Android

Kotlin port of the Python pipeline. Photo in, solved grid out, entirely on device.

## Opening this on WSL

Android Studio for **Windows** reaching into WSL over `\\wsl.localhost\...` warns
that the folder has "restricted write permissions". The Unix permissions are
fine; Windows simply treats the 9p share as a restricted network location, and
the IDE loses file watching and gets very slow Gradle builds.

Run **Android Studio for Linux inside WSL** instead. WSLg gives it a display, the
project stays on ext4 with native permissions, and Gradle is far faster than over
9p:

```bash
sudo apt install -y libxrender1 libxtst6 libxi6 libfreetype6 fontconfig
# download the Linux .tar.gz from developer.android.com/studio, then:
tar -xf android-studio-*-linux.tar.gz -C ~/
~/android-studio/bin/studio.sh
```

Android Studio bundles its own JDK, so no separate JVM install is needed.

Open **`android/`** as the project root -- not the repository root, which is a
Python project and has no Gradle build.

The alternative, if you would rather keep Windows Android Studio, is to copy this
folder onto the Windows filesystem (`C:\Users\<you>\AndroidStudioProjects\`).
That works, but leaves two copies to keep in sync.

## Gradle wrapper

`gradle/wrapper/gradle-wrapper.properties` pins Gradle 8.9, but `gradlew` and
`gradle-wrapper.jar` are not checked in -- the jar is a binary this repo has no
way to generate. Android Studio creates both on first sync. From a shell with
Gradle installed, `gradle wrapper` does the same.

## Dependencies

Nothing to install by hand. OpenCV has published official Android artifacts to
Maven Central since 4.9.0, so `org.opencv:opencv` resolves like any other
dependency -- the SDK-import-as-a-module dance older guides describe is no
longer necessary.

## The launcher icon

Already generated. An adaptive icon (`mipmap-anydpi-v26`) backed by two vectors,
plus raster fallbacks at all five densities and a 512px Play Store image.

To change it, edit the constants in `scripts/make_launcher_icon.py` and re-run:

```bash
uv run python scripts/make_launcher_icon.py
```

Then update `res/drawable/ic_launcher_foreground.xml` to match — the vector and
the script draw the same design and are kept in step by hand.

The `monochrome` layer is a separate drawable holding only the grid lines:
themed icons use the alpha channel and recolour it, so the filled cells would
otherwise flatten into a solid block.

## Layout

```
core/                     pure logic, no Android dependencies
  SudokuSolver.kt         backtracking solver + uniqueness check
  ConstraintRecovery.kt   re-read the doubtful cells when a puzzle won't solve
  CellPreprocessor.kt     cell sampling and cleanup — port this exactly
  GridGeometry.kt         corner ordering and perspective warp
  CellLattice.kt          YOLO detections -> 9x9 slots, with missing cells filled
  YoloDecoder.kt          letterboxing, box/pose decoding, NMS
  Models.kt               LiteRT wrappers
  SudokuPipeline.kt       the whole recipe
MainActivity.kt           camera / gallery, runs the pipeline off the main thread
SudokuGridView.kt         draws clues and solved digits differently
assets/                   the four exported models + models.json
```

`core/` deliberately has no Android imports beyond `Context` for asset loading,
so it can be unit-tested on the JVM.

## Verifying against Python

The logic was cross-checked against the Python implementation before it was
written into Kotlin. From the repository root:

```bash
uv run python scripts/verify_kotlin_port.py        # solver: 604 puzzles
uv run python scripts/verify_kotlin_preprocess.py  # preprocessing: 5103 patches
uv run python scripts/verify_kotlin_geometry.py    # geometry + lattice
uv run python scripts/verify_kotlin_decoder.py     # YOLO decoding vs Ultralytics
uv run python scripts/verify_tflite_pipeline.py    # TFLite vs PyTorch end-to-end
```

These transliterate the Kotlin back into Python and diff it against the real
implementation, so they catch logic drift — but **not** Kotlin compile errors.

## Re-exporting models

After retraining:

```bash
uv run python scripts/export_tflite.py
```

Writes to `app/src/main/assets/` and refreshes `models.json`, which records each
model's input shape. Check it after any re-export: the models take **NCHW**
(`[1, 3, 640, 640]`, `[81, 1, 50, 50]`), not NHWC. A wrong layout does not throw,
it just returns confident nonsense.

## Notes

- **Grid detection uses the pose model**, not seg. Seg measured slightly better
  on corner error but needs mask-prototype decoding on device — a 32-channel
  matmul, sigmoid, crop, upsample and contour trace — for the same four numbers.
  `grid_seg.tflite` is shipped anyway if you want to try it.
- **No NPU.** Tensor's TPU is not exposed to third-party apps and NNAPI is
  deprecated as of Android 15, so this uses the GPU delegate with a CPU
  fallback. At ~7M parameters total it is not the bottleneck.
- **Do not port `refine_on_grid_lines`.** It corrects a Mask R-CNN artefact the
  YOLO detector does not have, and on YOLO quads it made corner error five times
  worse.
