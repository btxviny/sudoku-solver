package com.sudokusolver.core

import android.content.Context
import org.opencv.core.Mat
import java.io.Closeable

/**
 * The whole photo-to-solution recipe, mirroring the Python `SudokuPipeline.run`.
 *
 * Every stage is allowed to fail without taking the others down: the result
 * carries whatever was produced plus the reason it stopped, so the UI can show
 * the rectified grid even when the digits could not be read.
 */
class SudokuPipeline(
    context: Context,
    /**
     * Which digit reader to load.  Only one is held at a time -- each is about
     * 10 MB of mapped weights -- so switching readers means building a new
     * pipeline and closing the old one.
     */
    val digitModel: DigitModel = DigitModel.CELL_OCR,
) : Closeable {

    private val cornerDetector = GridCornerDetector(context)
    private val cellDetector = CellDetector(context)
    private val digitReader = DigitReader(context, digitModel)

    /** Rectified grid edge in pixels. 9 x [CellPreprocessor.PATCH]. */
    private val warpSize = CellPreprocessor.GRID

    enum class Stage { GRID_DETECTION, CELL_DETECTION, DIGIT_READING, SOLVING }

    class Result(
        /** Digits as read; null if reading never happened. */
        val puzzle: IntArray?,
        /** Completed grid; null unless solving succeeded. */
        val solution: IntArray?,
        /** Perspective-corrected grid, for display. */
        val rectified: Mat?,
        val failedAt: Stage?,
        val message: String?,
        val timings: Map<Stage, Long>,
    ) {
        val solved: Boolean get() = solution != null
    }

    fun solve(image: Mat): Result {
        val timings = LinkedHashMap<Stage, Long>()
        var rectified: Mat? = null
        var puzzle: IntArray? = null

        fun fail(stage: Stage, msg: String) =
            Result(puzzle, null, rectified, stage, msg, timings)

        // ── Stage 1: locate the grid and straighten it ──────────────────────
        var t = System.currentTimeMillis()
        val corners = try {
            cornerDetector.detect(image)
        } catch (e: Exception) {
            return fail(Stage.GRID_DETECTION, e.message ?: "Grid detection failed")
        } ?: return fail(Stage.GRID_DETECTION, "No sudoku grid found in the photo.")

        rectified = GridGeometry.perspectiveWarp(image, corners, warpSize)
        timings[Stage.GRID_DETECTION] = System.currentTimeMillis() - t

        // ── Stage 2: find the 81 cells on the rectified grid ────────────────
        t = System.currentTimeMillis()
        val slots = try {
            CellLattice.boxesToGrid(cellDetector.detect(rectified))
        } catch (e: Exception) {
            return fail(Stage.CELL_DETECTION, e.message ?: "Cell detection failed")
        }
        timings[Stage.CELL_DETECTION] = System.currentTimeMillis() - t

        // ── Stage 3: read every cell ────────────────────────────────────────
        t = System.currentTimeMillis()
        val probs: Array<FloatArray>
        try {
            val boxes = Array(81) { i ->
                val d = slots[i]
                // A null slot means even the lattice could not place the cell.
                // Zeroes mark it degenerate, and canonicalCells falls back to
                // the cell's nominal grid position.
                if (d == null) intArrayOf(0, 0, 0, 0)
                else intArrayOf(d.x1.toInt(), d.y1.toInt(), d.x2.toInt(), d.y2.toInt())
            }
            val (cells, scaled) = CellPreprocessor.canonicalCells(rectified, boxes)
            val lowContrast = CellPreprocessor.isLowContrast(scaled)
            val patches = cells.map { CellPreprocessor.prepPatch(it, lowContrast) }
            val read = digitReader.read(patches)
            puzzle = read.first
            probs = read.second
        } catch (e: Exception) {
            return fail(Stage.DIGIT_READING, e.message ?: "Digit reading failed")
        }
        timings[Stage.DIGIT_READING] = System.currentTimeMillis() - t

        // ── Stage 4: solve, recovering from misreads if need be ─────────────
        t = System.currentTimeMillis()
        var clues = puzzle
        val solution = try {
            SudokuSolver.solve(clues)
        } catch (e: SudokuException) {
            // The digits as read do not solve. Try the network's runner-up
            // predictions on the cells it was least sure about.
            try {
                val recovered = ConstraintRecovery.recover(clues, probs)
                clues = recovered.puzzle
                puzzle = recovered.puzzle
                recovered.solution
            } catch (e2: SudokuException) {
                timings[Stage.SOLVING] = System.currentTimeMillis() - t
                return fail(Stage.SOLVING, e2.message ?: "Could not solve")
            }
        }

        // A photo is only genuinely solved when its clues determine one answer.
        // Too few recognised digits leaves the puzzle under-determined, and the
        // solver then returns an arbitrary completion -- a confident wrong
        // answer, which is worse than admitting failure.
        if (SudokuSolver.hasOtherSolution(clues)) {
            timings[Stage.SOLVING] = System.currentTimeMillis() - t
            return Result(
                puzzle, null, rectified, Stage.SOLVING,
                "Too few digits were read clearly - the puzzle has more than one " +
                    "solution. Try a sharper or better-lit photo.",
                timings,
            )
        }
        timings[Stage.SOLVING] = System.currentTimeMillis() - t

        return Result(puzzle, solution, rectified, null, null, timings)
    }

    override fun close() {
        cornerDetector.close()
        cellDetector.close()
        digitReader.close()
    }
}
