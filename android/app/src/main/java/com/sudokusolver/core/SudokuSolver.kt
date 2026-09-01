package com.sudokusolver.core

/** Raised when a grid's clues already conflict, or admit no completion. */
class SudokuException(message: String) : Exception(message)

/**
 * Bitmask backtracking sudoku solver.
 *
 * Replaces the OR-Tools CP-SAT solver used by the Python pipeline, which has no
 * Android build.  A 9x9 sudoku is far below what CP-SAT is for, so a plain
 * search is not a compromise here -- it is the right size of tool.
 *
 * Grids are flat [IntArray] of 81 entries in row-major order, 0 meaning empty,
 * matching the `puzzle.flatten()` layout the Python pipeline passes around.
 *
 * Two details matter for correctness downstream:
 *
 *  - The search is driven by *minimum remaining values*: it always expands the
 *    empty cell with the fewest legal digits, and abandons a branch the moment
 *    any empty cell has none.  Scan-order backtracking is fine for one solve,
 *    but constraint recovery runs several thousand solves per photo, so the
 *    pruning is what keeps that interactive.
 *
 *  - [countSolutions] can enumerate past the first answer, which is how
 *    uniqueness is checked.  The Python version had to forbid its known
 *    solution with an explicit CP-SAT clause and re-solve, because CP-SAT does
 *    not enumerate; counting to two is the same question asked directly.
 */
object SudokuSolver {

    const val N = 9
    const val CELLS = 81

    /** Bits 1..9; bit d set means digit d. Bit 0 is unused so digits index directly. */
    private const val ALL = 0x3FE

    private fun boxOf(index: Int): Int = (index / 27) * 3 + (index % N) / 3

    /**
     * Solve [grid], returning a completed copy.
     *
     * @throws SudokuException if the clues conflict or no completion exists.
     */
    fun solve(grid: IntArray): IntArray {
        require(grid.size == CELLS) { "Expected 81 cells, got ${grid.size}" }
        if (!isValid(grid)) {
            throw SudokuException("Puzzle clues conflict (duplicate in a row, column, or box).")
        }
        val work = grid.copyOf()
        val search = Search(work, limit = 1)
        search.run()
        val first = search.first
            ?: throw SudokuException("No valid solution found. The puzzle may be unsolvable.")
        return first
    }

    /** Solve, or return null instead of throwing. For hot loops that expect failures. */
    fun solveOrNull(grid: IntArray): IntArray? =
        if (!isValid(grid)) null
        else Search(grid.copyOf(), limit = 1).also { it.run() }.first

    /**
     * Number of distinct completions of [grid], counted no further than [limit].
     *
     * Stopping early matters: a badly-read photo can leave a grid with millions
     * of completions, and the only thing the caller ever needs to know is
     * whether there is more than one.
     */
    fun countSolutions(grid: IntArray, limit: Int = 2): Int {
        require(grid.size == CELLS) { "Expected 81 cells, got ${grid.size}" }
        if (!isValid(grid)) return 0
        return Search(grid.copyOf(), limit).also { it.run() }.count
    }

    /**
     * True if [grid] admits more than one completion.
     *
     * A photo is only genuinely solved when its clues determine one answer.
     * When recognition misses enough digits the remainder is under-determined
     * and any solver returns an arbitrary completion -- a confident wrong
     * answer.  Measured on held-out Roboflow images, this was 14 of 74 apparent
     * GridOCR "successes", so this check is load-bearing, not a nicety.
     */
    fun hasOtherSolution(grid: IntArray): Boolean = countSolutions(grid, limit = 2) >= 2

    /** True if a partial or complete grid contains no duplicate clue. */
    fun isValid(grid: IntArray): Boolean {
        val rowMask = IntArray(N)
        val colMask = IntArray(N)
        val boxMask = IntArray(N)
        for (i in 0 until CELLS) {
            val v = grid[i]
            if (v == 0) continue
            if (v < 1 || v > 9) return false
            val bit = 1 shl v
            val r = i / N
            val c = i % N
            val b = boxOf(i)
            if (rowMask[r] and bit != 0) return false
            if (colMask[c] and bit != 0) return false
            if (boxMask[b] and bit != 0) return false
            rowMask[r] = rowMask[r] or bit
            colMask[c] = colMask[c] or bit
            boxMask[b] = boxMask[b] or bit
        }
        return true
    }

    /**
     * One depth-first enumeration over [grid], mutated in place and left in an
     * arbitrary state; the answer is the copy taken in [first].
     */
    private class Search(private val grid: IntArray, private val limit: Int) {
        private val rowMask = IntArray(N)
        private val colMask = IntArray(N)
        private val boxMask = IntArray(N)

        var count = 0
            private set
        var first: IntArray? = null
            private set

        fun run() {
            for (i in 0 until CELLS) {
                val v = grid[i]
                if (v == 0) continue
                val bit = 1 shl v
                rowMask[i / N] = rowMask[i / N] or bit
                colMask[i % N] = colMask[i % N] or bit
                boxMask[boxOf(i)] = boxMask[boxOf(i)] or bit
            }
            step()
        }

        /** @return true once [limit] solutions are in hand and the search may stop. */
        private fun step(): Boolean {
            var bestIndex = -1
            var bestMask = 0
            var bestCount = 10

            for (i in 0 until CELLS) {
                if (grid[i] != 0) continue
                val avail = ALL and
                    (rowMask[i / N] or colMask[i % N] or boxMask[boxOf(i)]).inv()
                val n = Integer.bitCount(avail)
                // No legal digit anywhere below this node: abandon immediately.
                if (n == 0) return false
                if (n < bestCount) {
                    bestCount = n
                    bestMask = avail
                    bestIndex = i
                    if (n == 1) break   // cannot do better than forced
                }
            }

            if (bestIndex == -1) {          // no empty cell remains
                count++
                if (first == null) first = grid.copyOf()
                return count >= limit
            }

            val r = bestIndex / N
            val c = bestIndex % N
            val b = boxOf(bestIndex)
            var mask = bestMask
            while (mask != 0) {
                val bit = mask and (-mask)       // lowest set bit
                mask = mask xor bit
                val digit = Integer.numberOfTrailingZeros(bit)

                grid[bestIndex] = digit
                rowMask[r] = rowMask[r] or bit
                colMask[c] = colMask[c] or bit
                boxMask[b] = boxMask[b] or bit

                val stop = step()

                grid[bestIndex] = 0
                rowMask[r] = rowMask[r] and bit.inv()
                colMask[c] = colMask[c] and bit.inv()
                boxMask[b] = boxMask[b] and bit.inv()

                if (stop) return true
            }
            return false
        }
    }
}
