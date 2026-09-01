package com.sudokusolver

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.min

/**
 * Draws a 9x9 sudoku grid with three digit roles, matching the Streamlit colours:
 *
 *   Blue  #0066cc — clues read from the photo (or user-confirmed values)
 *   Orange #E65100 — digits the user manually corrected in edit mode
 *   Red   #CC0000 — digits the solver filled in
 *
 * Edit mode is entered via [startEditing].  While active, tapping a cell
 * selects it (highlighted with a light-blue rectangle) and [setSelectedDigit]
 * writes a digit there, distinguishing it from the original OCR clue.
 */
class SudokuGridView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0,
) : View(context, attrs, defStyle) {

    // ── Display state (normal mode) ───────────────────────────────────────────
    private var clues: IntArray? = null
    private var solution: IntArray? = null

    // ── Edit state ────────────────────────────────────────────────────────────
    private var editable = false
    private var selectedCell = -1
    private var originalClues: IntArray? = null   // snapshot of OCR result
    private var editedDigits: IntArray? = null    // user's live corrections

    /** Called whenever the user taps a cell in edit mode. */
    var onCellTapped: ((cellIndex: Int) -> Unit)? = null

    // ── Paints ────────────────────────────────────────────────────────────────
    private val thin = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = 2f
        color = Color.argb(90, 128, 128, 128)
    }
    private val thick = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = 6f
        color = Color.argb(200, 48, 48, 48)
    }
    private val cluePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER; isFakeBoldText = true
        color = Color.rgb(0, 102, 204)      // #0066cc — OCR clues / confirmed
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        color = Color.rgb(204, 0, 0)        // #CC0000 — solver-filled
    }
    private val editPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER; isFakeBoldText = true
        color = Color.rgb(230, 81, 0)       // #E65100 — user-corrected
    }
    private val selectFill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.argb(35, 0, 102, 204)
    }
    private val selectBorder = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = 3f
        color = Color.rgb(0, 102, 204)
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /** Show [clues] (blue) and optionally [solution] filled digits (red). */
    fun show(clues: IntArray?, solution: IntArray?) {
        this.clues = clues
        this.solution = solution
        visibility = if (clues == null && solution == null) GONE else VISIBLE
        invalidate()
    }

    fun clear() = show(null, null)

    /**
     * Enter edit mode with [puzzle] as the starting point.
     * Taps select cells; [setSelectedDigit] writes digits.
     */
    fun startEditing(puzzle: IntArray) {
        originalClues = puzzle.copyOf()
        editedDigits = puzzle.copyOf()
        editable = true
        selectedCell = -1
        isClickable = true
        isFocusable = true
        visibility = VISIBLE
        invalidate()
    }

    /** Exit edit mode and deselect any cell. */
    fun stopEditing() {
        editable = false
        selectedCell = -1
        isClickable = false
        isFocusable = false
        invalidate()
    }

    /** Write [digit] (0 = clear) into the currently selected cell. */
    fun setSelectedDigit(digit: Int) {
        val cell = selectedCell
        if (cell < 0 || editedDigits == null) return
        editedDigits!![cell] = digit
        invalidate()
    }

    /** Returns the current 81-element puzzle array (edits included). */
    fun getEditedPuzzle(): IntArray = editedDigits?.copyOf() ?: clues?.copyOf() ?: IntArray(81)

    fun isEditing(): Boolean = editable

    // ── Sizing — always square ─────────────────────────────────────────────────
    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val w = MeasureSpec.getSize(widthMeasureSpec)
        val side = MeasureSpec.makeMeasureSpec(w, MeasureSpec.EXACTLY)
        super.onMeasure(side, side)
    }

    // ── Touch — cell selection ────────────────────────────────────────────────
    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (!editable) return super.onTouchEvent(event)
        if (event.action != MotionEvent.ACTION_UP) return true
        val cell = min(width, height).toFloat() / 9f
        val col = (event.x / cell).toInt().coerceIn(0, 8)
        val row = (event.y / cell).toInt().coerceIn(0, 8)
        selectedCell = row * 9 + col
        onCellTapped?.invoke(selectedCell)
        invalidate()
        return true
    }

    // ── Drawing ───────────────────────────────────────────────────────────────
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val size = min(width, height).toFloat()
        val cell = size / 9f
        val ts = cell * 0.58f
        cluePaint.textSize = ts; fillPaint.textSize = ts; editPaint.textSize = ts

        // Grid lines
        for (i in 0..9) {
            val p = if (i % 3 == 0) thick else thin
            val at = i * cell
            canvas.drawLine(at, 0f, at, size, p)
            canvas.drawLine(0f, at, size, at, p)
        }

        // Selection highlight
        if (editable && selectedCell >= 0) {
            val sc = selectedCell % 9; val sr = selectedCell / 9
            val rect = RectF(sc * cell + 2f, sr * cell + 2f,
                             (sc + 1) * cell - 2f, (sr + 1) * cell - 2f)
            canvas.drawRect(rect, selectFill)
            canvas.drawRect(rect, selectBorder)
        }

        val baseline = cell / 2f - (cluePaint.descent() + cluePaint.ascent()) / 2f

        if (editable && editedDigits != null) {
            // Edit mode: blue for unchanged OCR clues, orange for user changes.
            val edited = editedDigits!!
            val original = originalClues!!
            for (i in 0 until 81) {
                val v = edited[i]; if (v == 0) continue
                val col = i % 9; val row = i / 9
                val paint = if (original[i] != 0 && v == original[i]) cluePaint else editPaint
                canvas.drawText(v.toString(), col * cell + cell / 2f, row * cell + baseline, paint)
            }
        } else {
            // Normal mode: blue clues, red solver-filled.
            val c = clues; val s = solution
            if (c == null && s == null) return
            for (i in 0 until 81) {
                val clue = c?.getOrNull(i) ?: 0
                val v = if (clue != 0) clue else s?.getOrNull(i) ?: 0
                if (v == 0) continue
                val col = i % 9; val row = i / 9
                canvas.drawText(
                    v.toString(),
                    col * cell + cell / 2f, row * cell + baseline,
                    if (clue != 0) cluePaint else fillPaint,
                )
            }
        }
    }
}
