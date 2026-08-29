"""Streamlit UI for the sudoku solver pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

try:
    from sudoku_solver.config import (
        GridDetectorConfig,
        PipelineConfig,
        YoloCellExtractorConfig,
        YoloGridDetectorConfig,
    )
    from sudoku_solver.pipeline import PipelineResult, SudokuPipeline
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from sudoku_solver.config import (
        GridDetectorConfig,
        PipelineConfig,
        YoloCellExtractorConfig,
        YoloGridDetectorConfig,
    )
    from sudoku_solver.pipeline import PipelineResult, SudokuPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="Sudoku Solver", page_icon="🧩", layout="wide")


def source_version() -> str:
    """Fingerprint of the package source and the model weights.

    Passed into `load_pipeline` so its cache key changes whenever the code or a
    checkpoint changes.  Without it a running app keeps serving the pipeline it
    built at startup: `st.cache_resource` holds the object, and Python will not
    re-import a module it has already loaded, so edits appear to have no effect.
    That is genuinely confusing to debug — it looked like a recognition bug when
    the code on disk was already fixed.
    """
    import hashlib

    root = Path(__file__).resolve().parent
    paths = sorted((root / "src" / "sudoku_solver").glob("*.py"))
    paths += sorted((root / "models" / "weights").glob("*"))
    # The YOLO checkpoints live under training/, not models/weights/, so they
    # have to be fingerprinted explicitly or a retrain would be invisible here.
    paths += sorted((root / "training").glob("*/runs/*/weights/best.pt"))
    digest = hashlib.sha256()
    for p in paths:
        try:
            digest.update(f"{p.name}:{p.stat().st_mtime_ns}".encode())
        except OSError:
            continue
    return digest.hexdigest()[:16]


@st.cache_resource(show_spinner="Loading models…")
def load_pipeline(
    device: str,
    detection_threshold: float,
    output_size: int,
    yolo_conf: float,
    yolo_grid_mode: str,
    version: str,
) -> SudokuPipeline:
    """Build a pipeline for one configuration.

    Streamlit caches on the argument tuple, so changing any control in the
    sidebar rebuilds exactly once and reuses the result afterwards.  `version`
    is the source/weights fingerprint, so edits invalidate the cache too.
    """
    cfg = PipelineConfig(device=device)
    cfg.grid_detector = GridDetectorConfig(
        model_path=cfg.grid_detector.model_path,
        detection_threshold=detection_threshold,
        output_size=output_size,
    )
    # GridOCR's patch size is NOT derived from the warp size: GridOCRNet was
    # trained on 50 px patches and its grid-line cleanup is tuned for them, so
    # deriving it here silently wrecked the model whenever the slider moved
    # (measured on a real photo: 20 % digit accuracy at 450 px, 1.2 % at 900 px).
    # Pinned at the trained value, GridOCR resizes internally and is invariant.
    cfg.yolo_cell_extractor = YoloCellExtractorConfig(
        model_path=cfg.yolo_cell_extractor.model_path,
        conf=yolo_conf,
    )
    # model_path is left to the config so it tracks `mode`; the Hough refinement
    # stays off because it is tuned for Mask R-CNN's under-segmented masks and
    # measurably degrades the YOLO quads (see YoloGridDetectorConfig).
    cfg.yolo_grid_detector = YoloGridDetectorConfig(
        mode=yolo_grid_mode,
        output_size=output_size,
    )
    return SudokuPipeline(cfg)


# ── rendering helpers ─────────────────────────────────────────────────────────

def render_grid(
    original: np.ndarray,
    solved: np.ndarray,
    cell_px: int = 38,
    clue_color: str = "inherit",
    new_color: str = "#cc0000",
) -> str:
    rows = [
        f"<table style='border-collapse:collapse;font-family:monospace;"
        f"font-size:{max(13, cell_px - 8)}px;margin:auto;'>"
    ]
    for r in range(9):
        rows.append("<tr>")
        for c in range(9):
            val = solved[r, c]
            is_clue = original[r, c] != 0
            bt = "3px solid #444" if r % 3 == 0 else "1px solid #bbb"
            bl = "3px solid #444" if c % 3 == 0 else "1px solid #bbb"
            bb = "3px solid #444" if r == 8 else "none"
            br = "3px solid #444" if c == 8 else "none"
            color = clue_color if is_clue else new_color
            weight = "bold" if is_clue else "normal"
            style = (
                f"width:{cell_px}px;height:{cell_px}px;"
                f"text-align:center;vertical-align:middle;"
                f"border-top:{bt};border-left:{bl};"
                f"border-bottom:{bb};border-right:{br};"
                f"color:{color};font-weight:{weight};"
            )
            rows.append(f"<td style='{style}'>{val if val != 0 else ''}</td>")
        rows.append("</tr>")
    rows.append("</table>")
    return "".join(rows)


# ── step UI helpers ───────────────────────────────────────────────────────────

def step_label(n: int, title: str) -> None:
    st.markdown(
        f"<p style='font-size:10px;font-weight:700;letter-spacing:.12em;"
        f"text-transform:uppercase;color:#888;margin:0 0 6px'>"
        f"{n} &nbsp;·&nbsp; {title}</p>",
        unsafe_allow_html=True,
    )


def badge_ok(msg: str) -> None:
    st.markdown(
        f"<p style='font-size:11.5px;color:#1a9e5c;margin:5px 0 0'>✓ {msg}</p>",
        unsafe_allow_html=True,
    )


def badge_err(msg: str) -> None:
    st.markdown(
        f"<p style='font-size:11.5px;color:#d63031;margin:5px 0 0'>✗ {msg}</p>",
        unsafe_allow_html=True,
    )


def badge_skip() -> None:
    st.markdown(
        "<p style='font-size:11.5px;color:#aaa;margin:5px 0 0'>— skipped</p>",
        unsafe_allow_html=True,
    )


def format_timing(timing: dict[str, float]) -> str:
    lines = []
    for k, v in timing.items():
        if k == "total":
            continue
        lines.append(f"**{k.replace('_', ' ').title()}:** {v * 1000:.1f} ms")
    lines.append(f"**Total:** {timing.get('total', 0) * 1000:.1f} ms")
    return "  \n".join(lines)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")

    device = st.selectbox(
        "Device", ["auto", "cuda", "cpu"], index=0,
        help="`auto` uses the GPU when one is available.",
    )
    detection_threshold = st.slider(
        "Grid detection threshold", 0.05, 0.95, 0.50, 0.05,
        help="Mask R-CNN score a region must beat to count as a grid. "
             "Lower this if detection fails on a dim or cluttered photo.",
    )
    output_size = st.select_slider(
        "Rectified grid size (px)", options=[450, 630, 900], value=450,
        help="Size of the perspective-corrected grid. Affects how much detail "
             "the cell crops keep; measured to make little difference on real "
             "photos, so leave it at 450 unless a very high-resolution image "
             "reads badly.",
    )
    yolo_conf = st.slider(
        "YOLO cell confidence", 0.05, 0.90, 0.30, 0.05,
        help="Confidence floor for YOLO cell detections. Affects the YOLO paths only.",
    )
    yolo_grid_mode = st.radio(
        "YOLO grid backend", ["seg", "pose"], index=0, horizontal=True,
        help="Which YOLO model locates the grid on the 'YOLO grid warp' paths. "
             "`seg` predicts a mask and derives corners from it; `pose` regresses "
             "the four corners directly. Both are ~6 MB against Mask R-CNN's 169 MB.",
    )

    pipeline = load_pipeline(
        device, detection_threshold, output_size, yolo_conf, yolo_grid_mode,
        source_version(),
    )

    st.divider()
    st.header("Pipeline")

    available = pipeline.available_paths()
    unavailable = pipeline.unavailable_paths()

    if not available:
        st.error("No complete pipeline available — check model weights.")
        st.stop()

    def _label(p) -> str:
        return f"{p.label}  ★" if p.recommended else p.label

    selected = st.radio(
        "Mode",
        options=available,
        format_func=_label,
        label_visibility="collapsed",
    )

    st.caption("★ recommended · what this does")
    st.info(selected.description)

    if unavailable:
        with st.expander(f"Unavailable ({len(unavailable)})", expanded=False):
            for p in unavailable:
                st.markdown(f"**{p.label}**")
                if p.hint:
                    st.caption(p.hint)


# ── main ──────────────────────────────────────────────────────────────────────

st.title("🧩 Sudoku Solver")
st.write("Upload a photo of a sudoku puzzle and press **Solve** to get the solution.")

uploaded = st.file_uploader(
    "Upload sudoku image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    with st.expander("Uploaded image", expanded=False):
        st.image(image, use_container_width=True)

    if st.button("Solve", type="primary", use_container_width=True):
        with st.spinner("Running pipeline…"):
            img_array = np.array(image)
            try:
                result: PipelineResult = pipeline.run_path(img_array, selected)
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.exception(e)
                st.stop()

        detection_failed = "detection" in result.errors
        ocr_failed = "ocr" in result.errors
        solving_failed = "solving" in result.errors

        if not result.errors:
            st.success("Puzzle solved!")
        elif detection_failed:
            st.error("Detection failed — could not locate the grid. "
                     "Try lowering the grid detection threshold in the sidebar.")
        elif ocr_failed:
            st.error("Digit recognition failed.")
        elif solving_failed:
            st.error("Could not find a valid solution — the clues below were misread. "
                     "Try another pipeline mode, or a sharper, straighter photo.")

        col1, col2, col3, col4 = st.columns(4)

        # ── Step 1: Grid / cell detection ────────────────────────────────
        with col1:
            step_label(1, "Grid detection")
            if detection_failed:
                st.warning("Detection failed.")
                badge_err(result.errors["detection"])
            elif result.seg_grid_image is not None:
                st.image(result.seg_grid_image, use_container_width=True)
                badge_ok("Mask R-CNN mask + corners")
            else:
                badge_skip()

        # ── Step 2: Cell extraction ───────────────────────────────────────
        with col2:
            step_label(2, "Cell extraction")
            if detection_failed:
                badge_skip()
            elif result.seg_cells_image is not None:
                st.image(result.seg_cells_image, use_container_width=True)
                badge_ok("YOLO cells on the rectified grid")
            else:
                badge_skip()

        # ── Step 3: Recognition ───────────────────────────────────────────
        with col3:
            step_label(3, "Recognition")
            if detection_failed:
                badge_skip()
            elif ocr_failed:
                badge_err(result.errors["ocr"])
            elif result.recognition_image is not None and result.original_grid is not None:
                st.image(result.recognition_image, use_container_width=True)
                n_clues = int((result.original_grid > 0).sum())
                badge_ok(f"{n_clues} clues  ·  GridOCR CNN")
            else:
                badge_skip()

        # ── Step 4: Solution ──────────────────────────────────────────────
        with col4:
            step_label(4, "Solution")
            if detection_failed or ocr_failed:
                badge_skip()
            elif solving_failed:
                if result.original_grid is not None:
                    st.html(render_grid(
                        result.original_grid, result.original_grid,
                        clue_color="#0066cc",
                    ))
                    st.caption("Clues only — solver failed")
                badge_err(result.errors["solving"])
            elif result.solved_grid is not None and result.original_grid is not None:
                st.caption("Blue = clues · Red = solved")
                st.html(render_grid(
                    result.original_grid, result.solved_grid,
                    clue_color="#0066cc", new_color="#cc0000",
                ))
                badge_ok("Solved")
            else:
                badge_skip()

        if result.timing:
            with st.expander("Timing", expanded=False):
                st.markdown(format_timing(result.timing))
