"""Streamlit UI for the sudoku solver pipeline."""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sudoku_solver.config import PipelineConfig, GridDetectorConfig, YoloCellExtractorConfig
from sudoku_solver.pipeline import PIPELINE_PATHS, SudokuPipeline, PipelineResult

st.set_page_config(page_title="Sudoku Solver", page_icon="🧩", layout="wide")


@st.cache_resource(show_spinner="Loading models…")
def load_pipeline(
    device: str,
    detection_threshold: float,
    output_size: int,
    yolo_conf: float,
) -> SudokuPipeline:
    """Build a pipeline for one configuration.

    Streamlit caches on the argument tuple, so changing any control in the
    sidebar rebuilds exactly once and reuses the result afterwards.
    """
    cfg = PipelineConfig(device=device)
    cfg.grid_detector = GridDetectorConfig(
        model_path=cfg.grid_detector.model_path,
        detection_threshold=detection_threshold,
        output_size=output_size,
    )
    cfg.grid_ocr.patch_size = output_size // 9
    cfg.yolo_cell_extractor = YoloCellExtractorConfig(
        model_path=cfg.yolo_cell_extractor.model_path,
        conf=yolo_conf,
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
            color  = clue_color if is_clue else new_color
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
        "Rectified grid size (px)", options=[360, 450, 540, 630], value=450,
        help="Size of the perspective-corrected grid. GridOCR reads "
             "one ninth of this per cell.",
    )
    yolo_conf = st.slider(
        "YOLO cell confidence", 0.05, 0.90, 0.30, 0.05,
        help="Confidence floor for YOLO cell detections. Affects the YOLO paths only.",
    )

    pipeline = load_pipeline(device, detection_threshold, output_size, yolo_conf)

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
        ocr_failed       = "ocr"       in result.errors
        solving_failed   = "solving"   in result.errors

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

        seg_is_yolo          = selected.seg == "yolo"
        seg_is_maskrcnn_yolo = selected.seg == "maskrcnn_yolo"

        # ── Step 1: Grid / cell detection ────────────────────────────────
        with col1:
            step_label(1, "Grid detection")
            if detection_failed:
                st.warning("Detection failed.")
                badge_err(result.errors["detection"])
            elif result.seg_grid_image is not None:
                st.image(result.seg_grid_image, use_container_width=True)
                if seg_is_yolo:
                    badge_ok("YOLO cells — filled (color) · empty (gray)")
                else:
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
                if seg_is_maskrcnn_yolo:
                    badge_ok("YOLO cells on the rectified grid")
                elif seg_is_yolo:
                    badge_ok("81-cell mosaic from YOLO crops")
                else:
                    badge_ok("81-cell mosaic from projection split")
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
                ocr_tag = {
                    "yolo_digit": "YOLO digit classifier",
                    "classifier": "ResNet18 + XGBoost",
                    "grid_ocr":   "GridOCR CNN",
                }.get(selected.ocr, "")
                badge_ok(f"{n_clues} clues  ·  {ocr_tag}")
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
