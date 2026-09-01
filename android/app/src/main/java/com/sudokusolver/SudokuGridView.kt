package com.sudokusolver

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/**
 * Draws a 9x9 grid, distinguishing the digits that were read from the photo
 * from the ones the solver filled in.
 *
 * Keeping the two visually separate is the point: it lets you see at a glance
 * whether a wrong answer came from a misread clue or from the solve.
 */
class SudokuGridView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0,
) : View(context, attrs, defStyle) {

    private var clues: IntArray? = null
    private var solution: IntArray? = null

    private val thin = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
        color = Color.argb(90, 128, 128, 128)
    }
    private val thick = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        color = Color.argb(180, 128, 128, 128)
    }
    private val cluePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        isFakeBoldText = true
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        color = Color.rgb(204, 0, 0)
    }

    /** @param clues digits read from the photo; @param solution the completed grid. */
    fun show(clues: IntArray?, solution: IntArray?) {
        this.clues = clues
        this.solution = solution
        invalidate()
    }

    fun clear() = show(null, null)

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val size = min(width, height).toFloat()
        val cell = size / 9f

        cluePaint.color = if (isDarkTheme()) Color.WHITE else Color.BLACK
        cluePaint.textSize = cell * 0.62f
        fillPaint.textSize = cell * 0.62f

        for (i in 0..9) {
            val p = if (i % 3 == 0) thick else thin
            val at = i * cell
            canvas.drawLine(at, 0f, at, size, p)
            canvas.drawLine(0f, at, size, at, p)
        }

        val c = clues
        val s = solution
        if (c == null && s == null) return

        // Baseline offset centres the glyph optically rather than by its box.
        val baseline = cell / 2f - (cluePaint.descent() + cluePaint.ascent()) / 2f
        for (i in 0 until 81) {
            val clue = c?.getOrNull(i) ?: 0
            val value = if (clue != 0) clue else s?.getOrNull(i) ?: 0
            if (value == 0) continue
            val col = i % 9
            val row = i / 9
            canvas.drawText(
                value.toString(),
                col * cell + cell / 2f,
                row * cell + baseline,
                if (clue != 0) cluePaint else fillPaint,
            )
        }
    }

    private fun isDarkTheme(): Boolean {
        val mode = resources.configuration.uiMode and
            android.content.res.Configuration.UI_MODE_NIGHT_MASK
        return mode == android.content.res.Configuration.UI_MODE_NIGHT_YES
    }
}
