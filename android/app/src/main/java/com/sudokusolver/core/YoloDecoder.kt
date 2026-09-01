package com.sudokusolver.core

import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * The postprocessing Ultralytics did for us in Python.
 *
 * On Android the raw tensor is all we get, so letterboxing, box decoding, NMS
 * and the mapping back to source pixels are ours to do.  Everything here was
 * derived from the exported models rather than assumed -- the outputs are
 * **normalised to 0..1** against the 640x640 letterboxed canvas, not pixels,
 * and the tensor is (1, channels, 8400) with channels down the *rows*.
 *
 * Channel layout, confirmed against the exported graphs:
 *
 *   cell_vision  (1,  6, 8400)   cx cy w h | empty filled
 *   grid_pose    (1, 17, 8400)   cx cy w h | grid | (x y conf) x 4 corners
 */
object YoloDecoder {

    const val INPUT_SIZE = 640

    /** Ultralytics' letterbox fill. */
    private const val PAD_VALUE = 114.0

    /** How a source image was fitted into the square input, so boxes can come back. */
    data class Letterbox(val scale: Double, val padX: Int, val padY: Int)

    /** One decoded detection in source-image pixels. */
    data class Raw(
        val x1: Double, val y1: Double, val x2: Double, val y2: Double,
        val score: Double, val cls: Int,
    )

    /**
     * Fit [image] into a square [INPUT_SIZE] canvas, preserving aspect ratio.
     *
     * Ultralytics resizes by the smaller ratio and pads the remainder to centre
     * the image.  Reproducing that exactly matters twice over: the network sees
     * the distribution it was trained on, and the padding offsets are what map
     * predictions back to source pixels.
     */
    fun letterbox(image: Mat): Pair<Mat, Letterbox> {
        val w = image.cols()
        val h = image.rows()
        val scale = min(INPUT_SIZE.toDouble() / w, INPUT_SIZE.toDouble() / h)
        val nw = (w * scale).roundToInt()
        val nh = (h * scale).roundToInt()

        val resized = Mat()
        Imgproc.resize(image, resized, Size(nw.toDouble(), nh.toDouble()))

        val canvas = Mat(INPUT_SIZE, INPUT_SIZE, image.type(), Scalar(PAD_VALUE, PAD_VALUE, PAD_VALUE))
        val padX = (INPUT_SIZE - nw) / 2
        val padY = (INPUT_SIZE - nh) / 2
        resized.copyTo(Mat(canvas, Rect(padX, padY, nw, nh)))

        return Pair(canvas, Letterbox(scale, padX, padY))
    }

    /**
     * Flatten a letterboxed RGB image into the model's input buffer.
     *
     * The exported models take **NCHW** -- (1, 3, 640, 640) -- so this writes
     * three separate colour planes, not interleaved pixels.  Feeding an
     * interleaved buffer does not throw; it silently produces nonsense.
     */
    fun toInputNCHW(letterboxed: Mat): FloatArray {
        val rgb = Mat()
        if (letterboxed.channels() == 3) letterboxed.copyTo(rgb)
        else Imgproc.cvtColor(letterboxed, rgb, Imgproc.COLOR_GRAY2RGB)

        val n = INPUT_SIZE * INPUT_SIZE
        val bytes = ByteArray(n * 3)
        rgb.get(0, 0, bytes)

        val out = FloatArray(n * 3)
        for (i in 0 until n) {
            out[i] = (bytes[i * 3].toInt() and 0xFF) / 255f            // R plane
            out[n + i] = (bytes[i * 3 + 1].toInt() and 0xFF) / 255f    // G plane
            out[2 * n + i] = (bytes[i * 3 + 2].toInt() and 0xFF) / 255f // B plane
        }
        return out
    }

    /** Undo the letterbox for one point, returning source-image pixels. */
    private fun unpad(x: Double, y: Double, lb: Letterbox): Pair<Double, Double> =
        Pair((x * INPUT_SIZE - lb.padX) / lb.scale, (y * INPUT_SIZE - lb.padY) / lb.scale)

    /**
     * Decode a detection head into source-pixel boxes, then suppress duplicates.
     *
     * @param output     flat (channels * 8400) tensor, channels-major.
     * @param channels   rows in the tensor: 4 box values plus one per class.
     * @param confThresh minimum class score to keep.
     */
    fun decodeDetections(
        output: FloatArray,
        channels: Int,
        lb: Letterbox,
        confThresh: Double = 0.25,
        iouThresh: Double = 0.5,
    ): List<Raw> {
        val anchors = output.size / channels
        val numClasses = channels - 4
        require(numClasses >= 1) { "Expected at least one class channel, got $channels" }

        val kept = ArrayList<Raw>()
        for (a in 0 until anchors) {
            var bestCls = 0
            var bestScore = -1.0
            for (c in 0 until numClasses) {
                val s = output[(4 + c) * anchors + a].toDouble()
                if (s > bestScore) { bestScore = s; bestCls = c }
            }
            if (bestScore < confThresh) continue

            val cx = output[a].toDouble()
            val cy = output[anchors + a].toDouble()
            val bw = output[2 * anchors + a].toDouble()
            val bh = output[3 * anchors + a].toDouble()

            val (x1, y1) = unpad(cx - bw / 2, cy - bh / 2, lb)
            val (x2, y2) = unpad(cx + bw / 2, cy + bh / 2, lb)
            kept.add(Raw(x1, y1, x2, y2, bestScore, bestCls))
        }
        return nms(kept, iouThresh)
    }

    /**
     * Decode the pose head to the four grid corners, in source-image pixels.
     *
     * Only the highest-scoring detection is used: a photo has one grid, and the
     * runner-up is invariably a duplicate of it.  The corners come back in the
     * model's trained order and are re-canonicalised by [GridGeometry.orderCorners]
     * so downstream code never depends on that ordering holding.
     *
     * @return TL, TR, BR, BL, or null if nothing passed [confThresh].
     */
    fun decodePoseCorners(
        output: FloatArray,
        lb: Letterbox,
        imageWidth: Int,
        imageHeight: Int,
        confThresh: Double = 0.25,
    ): Array<DoubleArray>? {
        val channels = 17
        val anchors = output.size / channels

        var best = -1
        var bestScore = confThresh
        for (a in 0 until anchors) {
            val s = output[4 * anchors + a].toDouble()
            if (s > bestScore) { bestScore = s; best = a }
        }
        if (best < 0) return null

        val corners = Array(4) { k ->
            val kx = output[(5 + k * 3) * anchors + best].toDouble()
            val ky = output[(6 + k * 3) * anchors + best].toDouble()
            val (x, y) = unpad(kx, ky, lb)
            // Clip to the image, as the Python pipeline does. The model happily
            // predicts corners a little outside the frame when the grid runs to
            // the edge, and letting those through would sample the warp from
            // outside the photo.
            doubleArrayOf(
                x.coerceIn(0.0, (imageWidth - 1).toDouble()),
                y.coerceIn(0.0, (imageHeight - 1).toDouble()),
            )
        }
        return GridGeometry.orderCorners(corners)
    }

    /**
     * Greedy non-maximum suppression, class-agnostic.
     *
     * Class-agnostic is deliberate for this pipeline: the cell detector's two
     * classes are empty and filled, and one physical cell predicted as both
     * should collapse to a single box rather than survive twice and fight for
     * the same lattice slot.
     */
    fun nms(boxes: List<Raw>, iouThresh: Double): List<Raw> {
        val order = boxes.sortedByDescending { it.score }
        val keep = ArrayList<Raw>()
        val dropped = BooleanArray(order.size)
        for (i in order.indices) {
            if (dropped[i]) continue
            val a = order[i]
            keep.add(a)
            for (j in i + 1 until order.size) {
                if (dropped[j]) continue
                if (iou(a, order[j]) > iouThresh) dropped[j] = true
            }
        }
        return keep
    }

    private fun iou(a: Raw, b: Raw): Double {
        val x1 = max(a.x1, b.x1)
        val y1 = max(a.y1, b.y1)
        val x2 = min(a.x2, b.x2)
        val y2 = min(a.y2, b.y2)
        val inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if (inter <= 0.0) return 0.0
        val areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
        val areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
        return inter / (areaA + areaB - inter)
    }
}
