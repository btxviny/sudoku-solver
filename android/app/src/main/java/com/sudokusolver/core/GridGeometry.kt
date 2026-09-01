package com.sudokusolver.core

import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.atan2

/**
 * Turning four detected corners into a square, upright grid image.
 *
 * Deliberately absent: the Hough edge-snapping step (`_refine_on_grid_lines`)
 * from the Python detector.  That exists to correct Mask R-CNN's habit of
 * under-segmenting the bottom edge, and its search band is wide enough to reach
 * a page edge or table rule.  The YOLO detector does not have that defect, and
 * applying the refinement to its already-accurate quads made things much worse
 * -- mean corner error went from 5.7 % to 40.9 % of a grid side, with about a
 * third of images failing outright.  Do not port it.
 */
object GridGeometry {

    /**
     * Order four corners as top-left, top-right, bottom-right, bottom-left.
     *
     * Sorting by angle about the centroid fixes the winding; rotating so the
     * smallest x+y comes first fixes where the cycle starts.  This agrees with
     * the sum/difference rule used elsewhere on roughly upright grids but,
     * unlike it, stays consistent on strongly rotated ones where two corners
     * can share an extremum.
     *
     * This is the same ordering the pose model's keypoints were trained on, so
     * pose output can go straight into [perspectiveWarp].
     *
     * It deliberately differs from the Python `perspective_warp`, which picks
     * corners by smallest and largest x+y and x-y.  The two agree on every
     * orientation measured except within a degree or two of 45, where the sum
     * rule degenerates -- two corners tie on x+y and the choice between them is
     * arbitrary.  At that angle they disagree on 59 % of quads, and this rule is
     * the well-defined one.  A phone held diagonally is not an exotic case, so
     * the difference is worth keeping.
     */
    fun orderCorners(points: Array<DoubleArray>): Array<DoubleArray> {
        require(points.size == 4) { "Expected 4 corners, got ${points.size}" }

        val cx = points.sumOf { it[0] } / 4.0
        val cy = points.sumOf { it[1] } / 4.0

        // atan2 with y increasing downward makes rising angle run clockwise.
        val sorted = points.sortedBy { atan2(it[1] - cy, it[0] - cx) }

        var start = 0
        var best = Double.MAX_VALUE
        for (i in sorted.indices) {
            val s = sorted[i][0] + sorted[i][1]
            if (s < best) { best = s; start = i }
        }
        return Array(4) { sorted[(start + it) % 4].copyOf() }
    }

    /**
     * Warp the quadrilateral [quad] onto a [size] x [size] square.
     *
     * The corners are re-sorted here rather than trusted, so this is safe to
     * call with a quad from any source.
     */
    fun perspectiveWarp(image: Mat, quad: Array<DoubleArray>, size: Int): Mat {
        val rect = orderCorners(quad)

        // 2 % outward expansion absorbs the border line's half-width, so the
        // outer grid lines land inside the output instead of exactly on its edge.
        val cx = rect.sumOf { it[0] } / 4.0
        val cy = rect.sumOf { it[1] } / 4.0
        val expanded = Array(4) {
            doubleArrayOf(
                cx + 1.02 * (rect[it][0] - cx),
                cy + 1.02 * (rect[it][1] - cy),
            )
        }

        val src = MatOfPoint2f(
            Point(expanded[0][0], expanded[0][1]),
            Point(expanded[1][0], expanded[1][1]),
            Point(expanded[2][0], expanded[2][1]),
            Point(expanded[3][0], expanded[3][1]),
        )
        val edge = (size - 1).toDouble()
        val dst = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(edge, 0.0),
            Point(edge, edge),
            Point(0.0, edge),
        )

        val h = Imgproc.getPerspectiveTransform(src, dst)
        val out = Mat()
        Imgproc.warpPerspective(image, out, h, Size(size.toDouble(), size.toDouble()))
        return out
    }
}
