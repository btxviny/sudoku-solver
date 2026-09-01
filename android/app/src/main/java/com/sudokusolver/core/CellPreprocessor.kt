package com.sudokusolver.core

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Everything that happens to a rectified grid image before GridOCR sees it.
 *
 * This file is the one to port exactly and change reluctantly.  GridOCRNet was
 * trained on cells produced by precisely these steps, so any drift here is
 * indistinguishable to the network from a different problem: in the Python
 * pipeline, letting the cell size vary from the trained 50 px dropped digit
 * accuracy to 20 %.  The constants below are measured, not chosen.
 */
object CellPreprocessor {

    /** Cell edge in pixels. GridOCRNet was trained at this size; do not derive it. */
    const val PATCH = 50

    /** Grid edge implied by [PATCH]. */
    const val GRID = PATCH * 9

    /** Below this 2nd-to-98th percentile spread the image is "low contrast". */
    private const val CONTRAST_RANGE = 180

    /** A pixel darker than this counts as ink when finding grid-line rows. */
    private const val DARK = 100

    /** A row with more than this fraction of dark pixels is a grid line, not a glyph. */
    private const val GRID_LINE_FRACTION = 0.75

    /** A pixel darker than this counts as part of the digit when re-centring. */
    private const val GLYPH = 150

    /** Below this min-max spread a low-contrast patch carries no digit at all. */
    private const val FLAT_PATCH_SPREAD = 20

    /**
     * Sample the 81 cells at one canonical scale.
     *
     * Every cell is cut from the *whole grid* rescaled so a cell measures
     * [PATCH], rather than resizing each detected box on its own.  That is not
     * a shortcut -- it is the accuracy-relevant choice.  GridOCRNet learned
     * 50 px cells cut from a 450 px grid, and rescaling tight boxes
     * individually changes how much of the cell the digit fills.  Measured over
     * 90 held-out photos: per-box rescaling solved 68, canonical sampling 74.
     *
     * @param rectified RGB or grayscale rectified grid.
     * @param boxesPx   81 boxes as [x1, y1, x2, y2] in [rectified]'s pixels,
     *                  row-major. A degenerate box means the detector missed
     *                  that cell, and its lattice position is used instead.
     * @return the 81 cell patches, plus the rescaled grid used to judge contrast.
     */
    fun canonicalCells(rectified: Mat, boxesPx: Array<IntArray>): Pair<List<Mat>, Mat> {
        require(boxesPx.size == 81) { "Expected 81 boxes, got ${boxesPx.size}" }

        val scaled = Mat()
        Imgproc.resize(rectified, scaled, Size(GRID.toDouble(), GRID.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)

        val sx = GRID.toDouble() / rectified.cols()
        val sy = GRID.toDouble() / rectified.rows()

        val cells = ArrayList<Mat>(81)
        for (i in 0 until 81) {
            val (x1, y1, x2, y2) = boxesPx[i]
            val x0: Int
            val y0: Int
            if (x2 <= x1 || y2 <= y1) {
                x0 = (i % 9) * PATCH
                y0 = (i / 9) * PATCH
            } else {
                val cx = (x1 + x2) / 2.0 * sx
                val cy = (y1 + y2) / 2.0 * sy
                x0 = max(0, min(GRID - PATCH, (cx - PATCH / 2.0).roundToInt()))
                y0 = max(0, min(GRID - PATCH, (cy - PATCH / 2.0).roundToInt()))
            }
            cells.add(Mat(scaled, Rect(x0, y0, PATCH, PATCH)))
        }
        return Pair(cells, scaled)
    }

    /**
     * Whether the grid is washed out, judged over the whole image.
     *
     * Percentiles, not raw min/max: a single dark page tab or a specular
     * highlight otherwise spans the range and hides the fact that every cell is
     * faint.  Judging over the whole grid rather than per cell also keeps one
     * washed-out cell from being normalised into noise on its own -- the
     * gutters and outer frame carry the paper tone.
     */
    fun isLowContrast(reference: Mat): Boolean {
        val gray = toGray(reference)
        val (lo, hi) = percentiles(gray, 2.0, 98.0)
        return (hi - lo) < CONTRAST_RANGE
    }

    /**
     * Clean one [PATCH] x [PATCH] grayscale cell so it matches the training
     * distribution.  Returns a new Mat; [patch] is not modified.
     */
    fun prepPatch(patch: Mat, lowContrast: Boolean): Mat {
        require(patch.rows() == PATCH && patch.cols() == PATCH) {
            "Expected ${PATCH}x$PATCH patch, got ${patch.cols()}x${patch.rows()}"
        }
        val px = ByteArray(PATCH * PATCH)
        toGray(patch).get(0, 0, px)

        if (lowContrast) stretch(px) else removeGridLines(px)

        val out = Mat(PATCH, PATCH, CvType.CV_8UC1)
        out.put(0, 0, px)
        return out
    }

    /**
     * Low-contrast mode: stretch the patch to the full range.
     *
     * Grid-line removal is deliberately skipped here -- with the tones
     * compressed, the dark-pixel threshold cannot tell a grid line from a digit
     * stroke, so removing "lines" would eat the digit.
     */
    private fun stretch(px: ByteArray) {
        var mn = 255
        var mx = 0
        for (b in px) {
            val v = b.toInt() and 0xFF
            if (v < mn) mn = v
            if (v > mx) mx = v
        }
        if (mx - mn > FLAT_PATCH_SPREAD) {
            val span = (mx - mn).toFloat()
            for (i in px.indices) {
                val v = (px[i].toInt() and 0xFF)
                val s = ((v - mn) / span * 255f).coerceIn(0f, 255f)
                px[i] = s.toInt().toByte()
            }
        } else {
            // No contrast at all: nothing here to read.
            px.fill(255.toByte())
        }
    }

    /**
     * Normal mode: strip full-width grid-line bars, then re-centre the glyph.
     *
     * Re-centring is not cosmetic.  Removing border rows from the top or bottom
     * leaves the remaining digit sitting off-centre, which is not what the
     * network was trained on.
     */
    private fun removeGridLines(px: ByteArray) {
        var removedAny = false
        for (row in 0 until PATCH) {
            var dark = 0
            val base = row * PATCH
            for (col in 0 until PATCH) {
                if ((px[base + col].toInt() and 0xFF) < DARK) dark++
            }
            if (dark.toDouble() / PATCH > GRID_LINE_FRACTION) {
                java.util.Arrays.fill(px, base, base + PATCH, 255.toByte())
                removedAny = true
            }
        }
        if (!removedAny) return

        var top = -1
        var bottom = -1
        for (row in 0 until PATCH) {
            val base = row * PATCH
            var hasGlyph = false
            for (col in 0 until PATCH) {
                if ((px[base + col].toInt() and 0xFF) < GLYPH) { hasGlyph = true; break }
            }
            if (hasGlyph) {
                if (top < 0) top = row
                bottom = row
            }
        }
        if (top < 0) return

        val height = bottom - top + 1
        val region = px.copyOfRange(top * PATCH, (bottom + 1) * PATCH)
        px.fill(255.toByte())
        val start = (PATCH - height) / 2
        region.copyInto(px, start * PATCH)
    }

    /**
     * Pack prepared patches into the tensor GridOCR expects.
     *
     * The exported model takes **(81, 1, 50, 50)** -- a fixed batch of one whole
     * grid, in PyTorch's NCHW order, which the LiteRT converter preserved rather
     * than transposing to NHWC.  Verify against `assets/models.json` after any
     * re-export rather than assuming: layout mistakes do not throw, they just
     * return confident wrong digits.
     *
     * For this single-channel input the distinction is moot in memory -- with
     * C = 1, NCHW and NHWC describe the same byte order -- so this flat array is
     * correct either way.  It is emphatically *not* moot for the YOLO models,
     * whose (1, 3, 640, 640) input wants three separate colour planes rather
     * than interleaved RGB pixels.
     */
    fun toModelInput(patches: List<Mat>): FloatArray {
        require(patches.size == 81) { "Expected 81 patches, got ${patches.size}" }
        val out = FloatArray(81 * PATCH * PATCH)
        val buf = ByteArray(PATCH * PATCH)
        for (i in patches.indices) {
            patches[i].get(0, 0, buf)
            val base = i * PATCH * PATCH
            for (j in buf.indices) {
                out[base + j] = (buf[j].toInt() and 0xFF) / 255f
            }
        }
        return out
    }

    /** A blank (all-white) patch, used where the detector found no cell. */
    fun blankPatch(): Mat {
        val m = Mat(PATCH, PATCH, CvType.CV_8UC1)
        m.setTo(org.opencv.core.Scalar(255.0))
        return m
    }

    private fun toGray(m: Mat): Mat {
        if (m.channels() == 1) return m
        val g = Mat()
        Imgproc.cvtColor(m, g, Imgproc.COLOR_RGB2GRAY)
        return g
    }

    /**
     * [lowPct] and [highPct] percentiles of an 8-bit grayscale Mat.
     *
     * Histogram-based, so results are whole numbers where NumPy's `percentile`
     * would interpolate.  Both callers only compare the spread against a
     * threshold of 180, so the sub-integer difference cannot change a verdict.
     */
    private fun percentiles(gray: Mat, lowPct: Double, highPct: Double): Pair<Int, Int> {
        val hist = IntArray(256)
        val row = ByteArray(gray.cols())
        for (r in 0 until gray.rows()) {
            gray.get(r, 0, row)
            for (b in row) hist[b.toInt() and 0xFF]++
        }
        val total = gray.rows().toLong() * gray.cols()
        if (total == 0L) return Pair(0, 0)

        val loTarget = (total * lowPct / 100.0)
        val hiTarget = (total * highPct / 100.0)
        var cum = 0L
        var lo = 0
        var hi = 255
        var loSet = false
        for (v in 0..255) {
            cum += hist[v]
            if (!loSet && cum >= loTarget) { lo = v; loSet = true }
            if (cum >= hiTarget) { hi = v; break }
        }
        return Pair(lo, hi)
    }
}
