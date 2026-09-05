package com.sudokusolver

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import com.google.android.material.color.MaterialColors
import kotlin.math.min

/**
 * Draws a 9×9 sudoku grid.
 *
 * Digit roles:
 *   Blue  — clues from the photo (or user-confirmed values)
 *   Orange — digits the user manually corrected in edit mode
 *   Red   — digits the solver filled in
 *
 * Visual polish:
 *   - Alternating 3×3 box shading so regions read at a glance
 *   - Rounded-rect cell highlight in edit mode
 *   - Thick box borders, thin cell lines
 */
class SudokuGridView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0,
) : View(context, attrs, defStyle) {

    // ── Display state ─────────────────────────────────────────────────────────
    private var clues: IntArray? = null
    private var solution: IntArray? = null

    // ── Edit state ────────────────────────────────────────────────────────────
    private var editable = false
    private var selectedCell = -1
    private var originalClues: IntArray? = null
    private var editedDigits: IntArray? = null

    var onCellTapped: ((cellIndex: Int) -> Unit)? = null

    // ── Paints ────────────────────────────────────────────────────────────────

    // Box shading: every other 3×3 box gets a very faint tint
    private val boxShade = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val thin = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = 1.2f
    }
    private val thick = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = 5f
    }
    private val outerBorder = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = 5f
    }

    private val cluePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER; isFakeBoldText = true
        color = Color.rgb(27, 58, 107)      // navy — OCR clues
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        color = Color.rgb(192, 57, 43)      // red — solver-filled
    }
    private val editPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER; isFakeBoldText = true
        color = Color.rgb(230, 81, 0)       // orange — user-corrected
    }

    private val selectFill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val selectBorder = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = 2.5f
    }

    // ── Theme-aware colour init ───────────────────────────────────────────────
    init {
        val primary = MaterialColors.getColor(context, com.google.android.material.R.attr.colorPrimary, Color.rgb(27, 58, 107))
        val onSurface = MaterialColors.getColor(context, com.google.android.material.R.attr.colorOnSurface, Color.DKGRAY)
        val surfaceVariant = MaterialColors.getColor(context, com.google.android.material.R.attr.colorSurfaceVariant, Color.LTGRAY)

        boxShade.color = Color.argb(14, Color.red(onSurface), Color.green(onSurface), Color.blue(onSurface))
        thin.color = Color.argb(55, Color.red(onSurface), Color.green(onSurface), Color.blue(onSurface))
        thick.color = Color.argb(160, Color.red(onSurface), Color.green(onSurface), Color.blue(onSurface))
        outerBorder.color = Color.argb(220, Color.red(onSurface), Color.green(onSurface), Color.blue(onSurface))

        cluePaint.color = primary
        selectFill.color = Color.argb(28, Color.red(primary), Color.green(primary), Color.blue(primary))
        selectBorder.color = primary
    }

    // ── Public API ────────────────────────────────────────────────────────────

    fun show(clues: IntArray?, solution: IntArray?) {
        this.clues = clues
        this.solution = solution
        visibility = if (clues == null && solution == null) GONE else VISIBLE
        invalidate()
    }

    fun clear() = show(null, null)

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

    fun stopEditing() {
        editable = false
        selectedCell = -1
        isClickable = false
        isFocusable = false
        invalidate()
    }

    fun setSelectedDigit(digit: Int) {
        val cell = selectedCell
        if (cell < 0 || editedDigits == null) return
        editedDigits!![cell] = digit
        invalidate()
    }

    fun getEditedPuzzle(): IntArray = editedDigits?.copyOf() ?: clues?.copyOf() ?: IntArray(81)

    fun isEditing(): Boolean = editable

    // ── Sizing ────────────────────────────────────────────────────────────────
    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val w = MeasureSpec.getSize(widthMeasureSpec)
        val side = MeasureSpec.makeMeasureSpec(w, MeasureSpec.EXACTLY)
        super.onMeasure(side, side)
    }

    // ── Touch ─────────────────────────────────────────────────────────────────
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
        val ts = cell * 0.56f
        cluePaint.textSize = ts; fillPaint.textSize = ts; editPaint.textSize = ts

        // 3×3 box shading — shade the "dark" diagonal boxes: (0,0),(1,1),(2,2)
        // i.e., boxes where (boxRow + boxCol) is even
        for (boxRow in 0..2) {
            for (boxCol in 0..2) {
                if ((boxRow + boxCol) % 2 == 0) {
                    canvas.drawRect(
                        boxCol * 3 * cell, boxRow * 3 * cell,
                        (boxCol + 1) * 3 * cell, (boxRow + 1) * 3 * cell,
                        boxShade,
                    )
                }
            }
        }

        // Cell grid lines (thin)
        for (i in 1..8) {
            if (i % 3 == 0) continue   // box lines drawn separately
            val at = i * cell
            canvas.drawLine(at, 0f, at, size, thin)
            canvas.drawLine(0f, at, size, at, thin)
        }

        // Box border lines (thick)
        for (i in 0..3) {
            val at = i * 3 * cell
            canvas.drawLine(at, 0f, at, size, thick)
            canvas.drawLine(0f, at, size, at, thick)
        }

        // Outer border on top for crispness
        val half = outerBorder.strokeWidth / 2f
        canvas.drawRect(half, half, size - half, size - half, outerBorder)

        // Selection highlight (rounded rect)
        if (editable && selectedCell >= 0) {
            val sc = selectedCell % 9; val sr = selectedCell / 9
            val r = RectF(sc * cell + 2f, sr * cell + 2f,
                          (sc + 1) * cell - 2f, (sr + 1) * cell - 2f)
            canvas.drawRoundRect(r, 8f, 8f, selectFill)
            canvas.drawRoundRect(r, 8f, 8f, selectBorder)
        }

        val baseline = cell / 2f - (cluePaint.descent() + cluePaint.ascent()) / 2f

        if (editable && editedDigits != null) {
            val edited = editedDigits!!
            val original = originalClues!!
            for (i in 0 until 81) {
                val v = edited[i]; if (v == 0) continue
                val col = i % 9; val row = i / 9
                val paint = if (original[i] != 0 && v == original[i]) cluePaint else editPaint
                canvas.drawText(v.toString(), col * cell + cell / 2f, row * cell + baseline, paint)
            }
        } else {
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
