package com.sudokusolver.core

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** One YOLO cell detection in rectified-grid pixels. `cls`: 0 empty, 1 filled. */
data class Detection(
    val x1: Double,
    val y1: Double,
    val x2: Double,
    val y2: Double,
    val cls: Int,
) {
    val cx: Double get() = (x1 + x2) / 2.0
    val cy: Double get() = (y1 + y2) / 2.0
}

/**
 * Assigning YOLO's cell detections to their 9x9 slots.
 *
 * The naive approach -- sort the boxes and cut the list into nine rows of nine
 * -- fails on real photos, because YOLO rarely returns exactly 81 boxes.  One
 * missing detection shifts every cell after it, scrambling the whole puzzle
 * instead of leaving a single hole.
 *
 * So slots are derived geometrically, from an affine lattice fitted to the
 * detections.  Fitting matters rather than just bounding the centres: a bounding
 * box assumes the grid is perfectly axis-aligned after warping, and any residual
 * tilt then makes a cell's row depend on its x position.  On one curved photo
 * that pushed the left of row 0 down into row 1 -- four cells collided and four
 * top slots were left empty, so those digits never reached OCR.
 */
object CellLattice {

    /** Below this many detections the lattice fit is not trustworthy. */
    private const val MIN_DETECTIONS_FOR_FIT = 20

    private const val REFINE_PASSES = 6

    /**
     * Map detections onto 81 row-major slots, synthesising any the detector missed.
     *
     * @return 81 entries; an entry is null only when the fit was impossible.
     */
    fun boxesToGrid(dets: List<Detection>): Array<Detection?> {
        val grid = arrayOfNulls<Detection>(81)
        if (dets.isEmpty()) return grid

        // Axis-aligned first guess. The outermost cells' centres sit half a cell
        // inside the grid, so their span covers 8 of the 9 pitches, not 9.
        var loX = Double.MAX_VALUE; var loY = Double.MAX_VALUE
        var hiX = -Double.MAX_VALUE; var hiY = -Double.MAX_VALUE
        for (d in dets) {
            loX = min(loX, d.cx); hiX = max(hiX, d.cx)
            loY = min(loY, d.cy); hiY = max(hiY, d.cy)
        }
        val pitchX = if (hiX - loX > 1e-6) (hiX - loX) / 8.0 else 1.0
        val pitchY = if (hiY - loY > 1e-6) (hiY - loY) / 8.0 else 1.0

        // ij[k] = [col, row]
        val ij = Array(dets.size) { k ->
            doubleArrayOf(
                clamp08(((dets[k].cx - loX) / pitchX).roundToInt().toDouble()),
                clamp08(((dets[k].cy - loY) / pitchY).roundToInt().toDouble()),
            )
        }

        // Refine: fit centre -> (col, row) affinely and re-read the indices from
        // the fit. A couple of passes absorb tilt and mild page curvature.
        val design = Array(dets.size) { doubleArrayOf(dets[it].cx, dets[it].cy, 1.0) }
        var predicted = Array(ij.size) { ij[it].copyOf() }

        if (dets.size >= MIN_DETECTIONS_FOR_FIT) {
            for (pass in 0 until REFINE_PASSES) {
                val coef = leastSquares(design, ij) ?: break
                predicted = multiply(design, coef)
                var changed = false
                for (k in ij.indices) {
                    val c = clamp08(predicted[k][0].roundToInt().toDouble())
                    val r = clamp08(predicted[k][1].roundToInt().toDouble())
                    if (c != ij[k][0] || r != ij[k][1]) changed = true
                    ij[k][0] = c
                    ij[k][1] = r
                }
                // Converged: the indices the fit predicts are the ones it was
                // given, so another pass would change nothing.
                if (!changed) break
            }
        }

        // Where two detections claim a slot, keep the one nearest the lattice
        // point. A duplicate box is usually the same size as the real one but
        // less well centred, so distance beats area as a tie-break.
        val best = DoubleArray(81) { Double.MAX_VALUE }
        for (k in dets.indices) {
            val col = ij[k][0].toInt()
            val row = ij[k][1].toInt()
            val idx = row * 9 + col
            val residual = abs(predicted[k][0] - ij[k][0]) + abs(predicted[k][1] - ij[k][1])
            if (residual < best[idx]) {
                best[idx] = residual
                grid[idx] = dets[k]
            }
        }

        fillMissingFromLattice(grid, dets, ij)
        return grid
    }

    /**
     * Place a box on every slot the detector missed.
     *
     * YOLO under-detects on about a third of real photos -- 36 of 90 held-out
     * images returned fewer than 81 boxes, one short by 44.  A slot with no box
     * reaches the reader as a dead cell: treated as empty and skipped, so those
     * digits are never read at all.
     *
     * Since the cells form a lattice, a missing one can simply be placed: invert
     * the (col,row) -> centre fit for its position, and take the median
     * detection for its size.  The cell then goes through OCR like any other and
     * the *reader* decides whether it is empty -- rather than the detector
     * deciding by omission.  Synthesised cells are marked filled for that reason.
     */
    private fun fillMissingFromLattice(
        grid: Array<Detection?>,
        dets: List<Detection>,
        ij: Array<DoubleArray>,
    ) {
        val missing = (0 until 81).filter { grid[it] == null }
        if (missing.isEmpty() || dets.size < MIN_DETECTIONS_FOR_FIT) return

        // The inverse fit: (col, row) -> centre.
        val design = Array(ij.size) { doubleArrayOf(ij[it][0], ij[it][1], 1.0) }
        val centres = Array(dets.size) { doubleArrayOf(dets[it].cx, dets[it].cy) }
        val coef = leastSquares(design, centres) ?: return

        val halfW = median(dets.map { it.x2 - it.x1 }) / 2.0
        val halfH = median(dets.map { it.y2 - it.y1 }) / 2.0
        if (!halfW.isFinite() || !halfH.isFinite()) return

        for (idx in missing) {
            val row = idx / 9
            val col = idx % 9
            val cx = col * coef[0][0] + row * coef[1][0] + coef[2][0]
            val cy = col * coef[0][1] + row * coef[1][1] + coef[2][1]
            if (!cx.isFinite() || !cy.isFinite()) continue
            grid[idx] = Detection(cx - halfW, cy - halfH, cx + halfW, cy + halfH, 1)
        }
    }

    private fun clamp08(v: Double): Double = min(8.0, max(0.0, v))

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return Double.NaN
        val s = values.sorted()
        val m = s.size / 2
        return if (s.size % 2 == 1) s[m] else (s[m - 1] + s[m]) / 2.0
    }

    private fun multiply(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> =
        Array(a.size) { i ->
            DoubleArray(b[0].size) { j ->
                var sum = 0.0
                for (k in b.indices) sum += a[i][k] * b[k][j]
                sum
            }
        }

    /**
     * Least-squares solution of `design * X = target`, via the normal equations.
     *
     * NumPy's `lstsq` uses an SVD, which also copes with a rank-deficient
     * design.  The normal equations are less forgiving, so this returns null
     * when the 3x3 system is singular -- which happens only if the detections
     * are collinear, and the caller then keeps its previous estimate rather than
     * fitting garbage.
     */
    private fun leastSquares(
        design: Array<DoubleArray>,
        target: Array<DoubleArray>,
    ): Array<DoubleArray>? {
        val cols = design[0].size          // 3
        val outs = target[0].size          // 2

        val ata = Array(cols) { DoubleArray(cols) }
        val atb = Array(cols) { DoubleArray(outs) }
        for (r in design.indices) {
            for (i in 0 until cols) {
                for (j in 0 until cols) ata[i][j] += design[r][i] * design[r][j]
                for (j in 0 until outs) atb[i][j] += design[r][i] * target[r][j]
            }
        }
        return solve(ata, atb)
    }

    /** Gaussian elimination with partial pivoting. Null if singular. */
    private fun solve(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray>? {
        val n = a.size
        val m = b[0].size
        val lhs = Array(n) { a[it].copyOf() }
        val rhs = Array(n) { b[it].copyOf() }

        for (col in 0 until n) {
            var pivot = col
            for (r in col + 1 until n) {
                if (abs(lhs[r][col]) > abs(lhs[pivot][col])) pivot = r
            }
            if (abs(lhs[pivot][col]) < 1e-12) return null
            val tl = lhs[col]; lhs[col] = lhs[pivot]; lhs[pivot] = tl
            val tr = rhs[col]; rhs[col] = rhs[pivot]; rhs[pivot] = tr

            val d = lhs[col][col]
            for (j in 0 until n) lhs[col][j] /= d
            for (j in 0 until m) rhs[col][j] /= d

            for (r in 0 until n) {
                if (r == col) continue
                val f = lhs[r][col]
                if (f == 0.0) continue
                for (j in 0 until n) lhs[r][j] -= f * lhs[col][j]
                for (j in 0 until m) rhs[r][j] -= f * rhs[col][j]
            }
        }
        return rhs
    }
}
