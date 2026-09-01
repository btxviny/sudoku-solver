"""Generate the Android launcher PNGs from the same design as the vector icon.

minSdk is 26, so adaptive icons in mipmap-anydpi-v26 cover every device the app
actually runs on.  These rasters exist for the places Android still reaches for
a plain bitmap -- some launchers' recents/shortcut paths, and Play Store
tooling -- and for parity with what Image Asset Studio would have produced.

Kept as a script rather than checked-in art so the icon can be changed in one
place: edit the constants, re-run, and every density regenerates.

Usage:
    uv run python scripts/make_launcher_icon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "android/app/src/main/res"

BACKGROUND = (27, 58, 107, 255)      # matches @color/ic_launcher_background
LINE = (255, 255, 255, 255)
# An opaque mid-blue rather than translucent white. PIL's draw methods replace
# pixels instead of compositing, so a translucent fill came out brighter than
# the grid lines and inverted the icon's read. Opaque keeps the hierarchy
# explicit: white lines brightest, filled cells mid, background darkest.
FILL = (86, 133, 200, 255)

# Launcher icon edge in px per density bucket.
DENSITIES = {
    "mdpi": 48,
    "hdpi": 72,
    "xhdpi": 96,
    "xxhdpi": 144,
    "xxxhdpi": 192,
}

# Supersample, then downscale: circles and thin strokes alias badly at 48 px.
SS = 8

# Grid geometry as fractions of the icon edge, matching the vector's 30..78
# of a 108 viewport.
GRID_START = 26 / 108
GRID_END = 82 / 108
LINE_WIDTH = 2.5 / 108
FILLED_CELLS = [(0, 0), (1, 1), (2, 2)]


def draw_icon(edge: int, round_icon: bool) -> Image.Image:
    """Render one icon.

    Content is drawn opaque on a full-bleed square and the silhouette is applied
    afterwards as an alpha mask.  Drawing the rounded shape directly onto a
    transparent canvas instead leaves colour in the fully-transparent corners,
    which LANCZOS then smears back into the visible edge as fringing.
    """
    size = edge * SS
    base = Image.new("RGB", (size, size), BACKGROUND[:3])
    d = ImageDraw.Draw(base)

    x0, x1 = GRID_START * size, GRID_END * size
    cell = (x1 - x0) / 3.0
    width = max(1, int(LINE_WIDTH * size))

    for col, row in FILLED_CELLS:
        half = width / 2.0
        d.rectangle(
            [x0 + col * cell + half, x0 + row * cell + half,
             x0 + (col + 1) * cell - half, x0 + (row + 1) * cell - half],
            fill=FILL[:3],
        )

    for i in range(4):
        at = x0 + i * cell
        d.line([x0, at, x1, at], fill=LINE[:3], width=width)
        d.line([at, x0, at, x1], fill=LINE[:3], width=width)

    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    if round_icon:
        md.ellipse([0, 0, size - 1, size - 1], fill=255)
    else:
        md.rounded_rectangle([0, 0, size - 1, size - 1], radius=int(size * 0.22), fill=255)

    out = base.convert("RGBA")
    out.putalpha(mask)
    return out.resize((edge, edge), Image.LANCZOS)


def main() -> None:
    written = []
    for bucket, edge in DENSITIES.items():
        out = RES / f"mipmap-{bucket}"
        out.mkdir(parents=True, exist_ok=True)
        for name, is_round in (("ic_launcher", False), ("ic_launcher_round", True)):
            path = out / f"{name}.png"
            draw_icon(edge, is_round).save(path, "PNG", optimize=True)
            written.append(path)

    # A 512px icon is what the Play Console asks for; harmless to keep alongside.
    store = ROOT / "android/ic_launcher-playstore.png"
    draw_icon(512, False).save(store, "PNG", optimize=True)
    written.append(store)

    for p in written:
        print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size / 1024:.1f} kB)")


if __name__ == "__main__":
    main()
