package com.sudokusolver.core

/** A puzzle that solves, together with the reading that produced it. */
class RecoveryResult(
    /** The clue grid actually used -- may differ from what OCR first reported. */
    val puzzle: IntArray,
    val solution: IntArray,
)

/**
 * Second-chance solving when the digits as read do not form a solvable puzzle.
 *
 * GridOCR is confident and wrong often enough that a single misread digit would
 * otherwise sink an entire photo.  Rather than give up, this re-reads the
 * puzzle using the *runner-up* predictions for the cells the network was least
 * sure about, and asks whether any of those readings solves.  Sudoku's
 * constraints are strong enough that a reading which solves is almost certainly
 * the correct one.
 *
 * Two strategies, in order:
 *
 *  1. Substitute alternatives into the least-confident cells -- one, then two,
 *     then three at a time.
 *  2. Failing that, blank every cell below a confidence threshold and let the
 *     solver reconstruct them, provided few enough are blanked that the puzzle
 *     stays determined.
 *
 * Ported from `recover_with_constraints` in the Python pipeline; the thresholds
 * are the measured ones and should not be nudged without re-running the eval.
 */
object ConstraintRecovery {

    /** Ignore alternatives the network gives less than this probability. */
    private const val MIN_ALT_PROB = 0.03f

    /** A cell is "uncertain", and worth substituting into, below this confidence. */
    private const val UNCERTAIN_BELOW = 0.60f

    /** Cap on cells considered; the search is combinatorial in this. */
    private const val MAX_UNCERTAIN = 12

    /** Substitute into at most this many cells at once. */
    private const val MAX_FLIPS = 3

    /** Blank-and-refill thresholds, tried in order. */
    private val CONFIDENCE_THRESHOLDS = floatArrayOf(0.80f, 0.70f, 0.60f)

    /** Refuse to blank more than this many cells: beyond it the puzzle is guesswork. */
    private const val MAX_BLANKED = 6

    /**
     * @param puzzle 81 digits as read, 0 meaning empty.
     * @param probs  81 x 10 softmax rows from GridOCR; index 0 is "empty".
     * @throws SudokuException if no reading solves.
     */
    fun recover(puzzle: IntArray, probs: Array<FloatArray>): RecoveryResult {
        require(puzzle.size == SudokuSolver.CELLS) { "Expected 81 cells, got ${puzzle.size}" }
        require(probs.size == SudokuSolver.CELLS) { "Expected 81 probability rows, got ${probs.size}" }

        val candidates = arrayOfNulls<IntArray>(SudokuSolver.CELLS)
        val top1 = FloatArray(SudokuSolver.CELLS)

        for (i in 0 until SudokuSolver.CELLS) {
            val row = probs[i]
            require(row.size == 10) { "Expected 10 classes at cell $i, got ${row.size}" }
            val ranked = (0..9).sortedByDescending { row[it] }
            top1[i] = row[ranked[0]]
            // Ranks 2-4 only: past that the network is guessing, and each extra
            // alternative multiplies the search.
            var alts = ranked.subList(1, 4).filter { row[it] >= MIN_ALT_PROB }
            // A cell read as a digit cannot be re-read as empty -- the solver
            // would simply fill it back in, so the substitution is not a test.
            if (puzzle[i] > 0) alts = alts.filter { it != 0 }
            candidates[i] = alts.toIntArray()
        }

        val uncertain = (0 until SudokuSolver.CELLS)
            .filter { candidates[it]!!.isNotEmpty() && top1[it] < UNCERTAIN_BELOW }
            .sortedBy { top1[it] }
            .take(MAX_UNCERTAIN)

        // ── Strategy 1: substitute alternatives ──────────────────────────────
        for (nFlip in 1..MAX_FLIPS) {
            for (combo in combinations(uncertain, nFlip)) {
                val altLists = combo.map { candidates[it]!! }
                for (choice in cartesian(altLists)) {
                    val trial = puzzle.copyOf()
                    for (k in combo.indices) trial[combo[k]] = choice[k]
                    SudokuSolver.solveOrNull(trial)?.let {
                        return RecoveryResult(trial, it)
                    }
                }
            }
        }

        // ── Strategy 2: blank the doubtful cells and let the solver refill ───
        for (threshold in CONFIDENCE_THRESHOLDS) {
            val blanked = (0 until SudokuSolver.CELLS)
                .count { top1[it] < threshold && puzzle[it] > 0 }
            // Blanking too much leaves the puzzle under-determined, and the
            // solver would answer with an arbitrary completion.
            if (blanked > MAX_BLANKED) continue

            val trial = puzzle.copyOf()
            for (i in 0 until SudokuSolver.CELLS) {
                if (top1[i] < threshold) trial[i] = 0
            }
            SudokuSolver.solveOrNull(trial)?.let {
                return RecoveryResult(trial, it)
            }
        }

        throw SudokuException(
            "No valid solution found after constraint recovery. " +
                "The puzzle image may be unclear or the grid detection may have failed."
        )
    }

    /** All size-[k] subsets of [items], in the input's order. */
    internal fun combinations(items: List<Int>, k: Int): Sequence<IntArray> = sequence {
        if (k <= 0 || k > items.size) return@sequence
        val idx = IntArray(k) { it }
        while (true) {
            yield(IntArray(k) { items[idx[it]] })
            var i = k - 1
            while (i >= 0 && idx[i] == i + items.size - k) i--
            if (i < 0) return@sequence
            idx[i]++
            for (j in i + 1 until k) idx[j] = idx[j - 1] + 1
        }
    }

    /** Cartesian product, one entry drawn from each list. Empty if any list is empty. */
    internal fun cartesian(lists: List<IntArray>): Sequence<IntArray> = sequence {
        if (lists.isEmpty() || lists.any { it.isEmpty() }) return@sequence
        val pos = IntArray(lists.size)
        while (true) {
            yield(IntArray(lists.size) { lists[it][pos[it]] })
            var i = lists.size - 1
            while (i >= 0 && pos[i] == lists[i].size - 1) { pos[i] = 0; i-- }
            if (i < 0) return@sequence
            pos[i]++
        }
    }
}
