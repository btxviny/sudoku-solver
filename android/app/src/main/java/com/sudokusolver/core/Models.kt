package com.sudokusolver.core

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Thin LiteRT wrappers around the three exported models.
 *
 * Accelerator choice: the GPU delegate where the device supports it, otherwise
 * multi-threaded CPU.  Notably *not* the NPU -- Tensor's TPU is not exposed to
 * third-party apps and NNAPI is deprecated as of Android 15.  At roughly 7 M
 * parameters across all three models this is not the bottleneck anyway.
 */
private fun buildOptions(context: Context): Pair<Interpreter.Options, GpuDelegate?> {
    val options = Interpreter.Options()
    var delegate: GpuDelegate? = null
    if (CompatibilityList().isDelegateSupportedOnThisDevice) {
        delegate = GpuDelegate(CompatibilityList().bestOptionsForThisDevice)
        options.addDelegate(delegate)
    } else {
        options.numThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(4)
    }
    return Pair(options, delegate)
}

/**
 * Memory-map a model straight out of the APK.
 *
 * Requires the asset to be stored uncompressed -- see `noCompress += "tflite"`
 * in build.gradle.kts. A compressed asset reports the wrong length here and the
 * interpreter fails to load it.
 */
private fun loadAsset(context: Context, name: String): MappedByteBuffer =
    context.assets.openFd(name).use { fd ->
        FileInputStream(fd.fileDescriptor).channel.use { channel ->
            channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
    }

private fun floatBuffer(size: Int): ByteBuffer =
    ByteBuffer.allocateDirect(size * 4).order(ByteOrder.nativeOrder())

/** Base for the single-input, fixed-shape models this pipeline uses. */
private abstract class TfliteModel(context: Context, asset: String) : Closeable {
    private val gpu: GpuDelegate?
    protected val interpreter: Interpreter

    init {
        val (options, delegate) = buildOptions(context)
        gpu = delegate
        interpreter = Interpreter(loadAsset(context, asset), options)
    }

    override fun close() {
        interpreter.close()
        gpu?.close()
    }
}

/**
 * Grid corner detector -- YOLOv8n-pose, (1, 3, 640, 640) in, (1, 17, 8400) out.
 *
 * Pose rather than seg is deliberate.  The seg model measured slightly better
 * on corner error, but it needs mask-prototype decoding on device: a 32-channel
 * matrix multiply, sigmoid, crop and upsample, then contour tracing.  Pose hands
 * back four points.  For the same job that is a large amount of surface area to
 * port and get wrong, and pose converted cleanly (23/23 detections, IoU 0.987).
 */
class GridCornerDetector(context: Context) : Closeable {
    private val model = object : TfliteModel(context, "grid_pose.tflite") {
        fun run(input: FloatArray): FloatArray {
            val inBuf = floatBuffer(input.size)
            inBuf.asFloatBuffer().put(input)
            val out = Array(1) { Array(17) { FloatArray(8400) } }
            interpreter.run(inBuf, out)
            val flat = FloatArray(17 * 8400)
            for (c in 0 until 17) out[0][c].copyInto(flat, c * 8400)
            return flat
        }
    }

    /** @return TL, TR, BR, BL in source-image pixels, or null if no grid found. */
    fun detect(image: org.opencv.core.Mat, conf: Double = 0.25): Array<DoubleArray>? {
        val (canvas, lb) = YoloDecoder.letterbox(image)
        val out = model.run(YoloDecoder.toInputNCHW(canvas))
        return YoloDecoder.decodePoseCorners(out, lb, image.cols(), image.rows(), conf)
    }

    override fun close() = model.close()
}

/**
 * Cell detector -- YOLOv8n, (1, 3, 640, 640) in, (1, 6, 8400) out.
 *
 * Runs on the *rectified* grid, so its input is already square and upright.
 */
class CellDetector(context: Context) : Closeable {
    private val model = object : TfliteModel(context, "cell_vision.tflite") {
        fun run(input: FloatArray): FloatArray {
            val inBuf = floatBuffer(input.size)
            inBuf.asFloatBuffer().put(input)
            val out = Array(1) { Array(6) { FloatArray(8400) } }
            interpreter.run(inBuf, out)
            val flat = FloatArray(6 * 8400)
            for (c in 0 until 6) out[0][c].copyInto(flat, c * 8400)
            return flat
        }
    }

    /** Detections in [rectified]'s own pixels, already NMS'd. */
    fun detect(
        rectified: org.opencv.core.Mat,
        conf: Double = 0.30,
        iou: Double = 0.50,
    ): List<Detection> {
        val (canvas, lb) = YoloDecoder.letterbox(rectified)
        val out = model.run(YoloDecoder.toInputNCHW(canvas))
        return YoloDecoder.decodeDetections(out, 6, lb, conf, iou)
            .map { Detection(it.x1, it.y1, it.x2, it.y2, it.cls) }
    }

    override fun close() = model.close()
}

/**
 * The digit readers that ship in the APK, mirroring `PIPELINE_PATHS` in the
 * Python pipeline.
 *
 * Both take (81, 1, 70, 70) and return (81, 10), and both are fed by the same
 * [CellPreprocessor], so they are interchangeable at this seam -- which is the
 * point: switching between them compares the networks and nothing else.
 *
 * @property asset the TFLite file in `assets/`, written by scripts/export_tflite.py
 * @property label what the picker shows
 */
enum class DigitModel(val asset: String, val label: String) {
    /** First generation: plain residual CNN, trained with MNIST handwriting. */
    GRID_OCR("gridocr.tflite", "GridOCR"),

    /** Second generation: squeeze-excitation CNN, EMNIST handwriting, MNIST held out. */
    CELL_OCR("cellocr.tflite", "CellOCR"),
}

/**
 * Digit reader -- (81, 1, 70, 70) in, (81, 10) out.
 *
 * The batch is fixed at a whole grid, matching the Python pipeline's single
 * forward pass over all 81 cells.
 */
class DigitReader(context: Context, digitModel: DigitModel = DigitModel.CELL_OCR) : Closeable {
    private val model = object : TfliteModel(context, digitModel.asset) {
        fun run(input: FloatArray): Array<FloatArray> {
            val inBuf = floatBuffer(input.size)
            inBuf.asFloatBuffer().put(input)
            val out = Array(81) { FloatArray(10) }
            interpreter.run(inBuf, out)
            return out
        }
    }

    /** @return 81 digits (0 = empty) and the 81x10 softmax probabilities. */
    fun read(patches: List<org.opencv.core.Mat>): Pair<IntArray, Array<FloatArray>> {
        val logits = model.run(CellPreprocessor.toModelInput(patches))
        val probs = Array(81) { i -> softmax(logits[i]) }
        val digits = IntArray(81) { i ->
            var best = 0
            for (d in 1 until 10) if (probs[i][d] > probs[i][best]) best = d
            best
        }
        return Pair(digits, probs)
    }

    /**
     * The exported graph ends at the logits -- the PyTorch model applied softmax
     * outside the module -- so it is applied here, shifted by the max for
     * numerical stability.
     */
    private fun softmax(logits: FloatArray): FloatArray {
        var max = Float.NEGATIVE_INFINITY
        for (v in logits) if (v > max) max = v
        var sum = 0.0
        val out = FloatArray(logits.size)
        for (i in logits.indices) {
            val e = Math.exp((logits[i] - max).toDouble())
            out[i] = e.toFloat()
            sum += e
        }
        for (i in out.indices) out[i] = (out[i] / sum).toFloat()
        return out
    }

    override fun close() = model.close()
}
