#!/usr/bin/env bash
# One-shot: export TFLite models from best weights, sync everything to the
# Windows Android project.
#
#   bash scripts/deploy_android.sh
#
# After this, open Android Studio (C:\Users\csps0\Documents\sudoku-solver) and
# press Build → Make Project.

set -euo pipefail

WIN_ROOT="/mnt/c/Users/csps0/Documents/sudoku-solver"
WSL_ANDROID="$(cd "$(dirname "$0")/.." && pwd)/android"

echo "=== 1/3  Export TFLite models ==="
cd "$(dirname "$0")/.."
uv run python scripts/export_tflite.py

echo ""
echo "=== 2/3  Sync assets (tflite + models.json) ==="
rsync -av --checksum \
    "$WSL_ANDROID/app/src/main/assets/" \
    "$WIN_ROOT/app/src/main/assets/"

echo ""
echo "=== 3/3  Sync Kotlin source + build files ==="
rsync -av --checksum \
    "$WSL_ANDROID/app/src/main/java/" \
    "$WIN_ROOT/app/src/main/java/"

rsync -av --checksum \
    "$WSL_ANDROID/app/src/main/res/" \
    "$WIN_ROOT/app/src/main/res/"

rsync -av --checksum \
    "$WSL_ANDROID/app/src/main/AndroidManifest.xml" \
    "$WIN_ROOT/app/src/main/AndroidManifest.xml"

rsync -av --checksum \
    "$WSL_ANDROID/app/build.gradle.kts" \
    "$WIN_ROOT/app/build.gradle.kts"

rsync -av --checksum \
    "$WSL_ANDROID/settings.gradle.kts" \
    "$WIN_ROOT/settings.gradle.kts"

echo ""
echo "Done. Open Android Studio and press Build → Make Project."
