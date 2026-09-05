package com.sudokusolver

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.sudokusolver.core.DigitModel
import com.sudokusolver.core.SudokuException
import com.sudokusolver.core.SudokuPipeline
import com.sudokusolver.core.SudokuSolver
import com.sudokusolver.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var pipeline: SudokuPipeline? = null

    /** Which digit reader is loaded. Changing it rebuilds [pipeline]. */
    private var digitModel = DigitModel.CELL_OCR

    /** Last puzzle the OCR step produced — used as the seed for edit mode. */
    private var currentPuzzle: IntArray? = null
    private var editMode = false

    private val pickImage = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode != Activity.RESULT_OK) return@registerForActivityResult
        result.data?.data?.let { uri -> solve(loadBitmap(uri)) }
    }

    private val takePhoto = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode != Activity.RESULT_OK) return@registerForActivityResult
        @Suppress("DEPRECATION")
        (result.data?.extras?.get("data") as? Bitmap)?.let { solve(it) }
    }

    private val requestCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) launchCamera() else status("Camera permission denied.")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (!OpenCVLoader.initLocal()) {
            status("OpenCV failed to load.")
            binding.cameraButton.isEnabled = false
            binding.galleryButton.isEnabled = false
            return
        }

        binding.readerToggle.check(
            if (digitModel == DigitModel.CELL_OCR) R.id.readerCellOcr else R.id.readerGridOcr
        )
        binding.readerHint.setText(hintFor(digitModel))
        binding.readerToggle.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener
            val picked =
                if (checkedId == R.id.readerCellOcr) DigitModel.CELL_OCR else DigitModel.GRID_OCR
            if (picked != digitModel) loadPipeline(picked)
        }

        loadPipeline(digitModel)

        binding.cameraButton.setOnClickListener {
            requestCamera.launch(android.Manifest.permission.CAMERA)
        }
        binding.galleryButton.setOnClickListener {
            pickImage.launch(Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI))
        }

        // ── Edit button: toggle editing mode on the recognition grid ─────────
        binding.editButton.setOnClickListener {
            val puzzle = currentPuzzle ?: return@setOnClickListener
            if (!editMode) {
                binding.recognitionGrid.startEditing(puzzle)
                binding.numberPadCard.visibility = View.VISIBLE
                binding.editButton.text = getString(R.string.done_editing)
                editMode = true
            } else {
                binding.recognitionGrid.stopEditing()
                binding.numberPadCard.visibility = View.GONE
                binding.editButton.text = getString(R.string.edit_recognition)
                editMode = false
            }
        }

        // ── Number pad ────────────────────────────────────────────────────────
        val numButtons = listOf(
            binding.numBtn1, binding.numBtn2, binding.numBtn3,
            binding.numBtn4, binding.numBtn5, binding.numBtn6,
            binding.numBtn7, binding.numBtn8, binding.numBtn9,
        )
        numButtons.forEachIndexed { i, btn ->
            btn.setOnClickListener { binding.recognitionGrid.setSelectedDigit(i + 1) }
        }
        binding.numBtn0.setOnClickListener {
            binding.recognitionGrid.setSelectedDigit(0)
        }

        // ── Re-solve with edited digits ───────────────────────────────────────
        binding.resolveButton.setOnClickListener { resolveEdited() }
    }

    /**
     * Enable or disable the picker.
     *
     * The buttons are toggled individually: disabling the enclosing
     * [com.google.android.material.button.MaterialButtonToggleGroup] leaves its
     * children clickable, so a second switch could still arrive mid-load.
     */
    private fun setPickerEnabled(enabled: Boolean) {
        binding.readerGridOcr.isEnabled = enabled
        binding.readerCellOcr.isEnabled = enabled
    }

    private fun hintFor(model: DigitModel) = when (model) {
        DigitModel.GRID_OCR -> R.string.reader_hint_grid_ocr
        DigitModel.CELL_OCR -> R.string.reader_hint_cell_ocr
    }

    /**
     * Build the pipeline for [model], replacing whatever is loaded.
     *
     * Each reader is about 10 MB of mapped weights, so the old pipeline is
     * closed before the new one is built rather than holding both.  The picker
     * is disabled meanwhile: a second switch arriving mid-build would leave two
     * pipelines racing for the same interpreter resources.
     */
    private fun loadPipeline(model: DigitModel) {
        digitModel = model
        binding.readerHint.setText(hintFor(model))
        setPickerEnabled(false)
        lifecycleScope.launch {
            status(getString(R.string.reader_switching))
            val built = withContext(Dispatchers.Default) {
                pipeline?.close()
                pipeline = null
                runCatching { SudokuPipeline(applicationContext, model) }
            }
            setPickerEnabled(true)
            built.onSuccess {
                pipeline = it
                status(getString(R.string.subtitle))
            }.onFailure {
                // A missing asset is the likely cause: cellocr.tflite only
                // exists once training/cell_ocr has been exported.
                status("Could not load ${model.label}: ${it.message ?: "unknown error"}")
            }
        }
    }

    private fun launchCamera() = takePhoto.launch(Intent(MediaStore.ACTION_IMAGE_CAPTURE))

    private fun loadBitmap(uri: Uri): Bitmap? =
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }

    private fun solve(bitmap: Bitmap?) {
        val pipe = pipeline
        if (bitmap == null) return status("Could not read that image.")
        if (pipe == null) return status("Models are still loading…")

        resetSteps()
        binding.progress.visibility = View.VISIBLE
        binding.photoView.setImageBitmap(bitmap)
        binding.photoView.visibility = View.VISIBLE
        setBadge(binding.badge1, ok = true)
        status(getString(R.string.solving))

        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) {
                val mat = Mat()
                Utils.bitmapToMat(bitmap, mat)
                Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
                pipe.solve(mat)
            }
            binding.progress.visibility = View.GONE

            // Step 2: Grid detection
            result.rectified?.let { mat ->
                binding.rectifiedView.setImageBitmap(matToBitmap(mat))
                binding.rectifiedView.visibility = View.VISIBLE
                setBadge(binding.badge2, ok = true)
            } ?: setBadge(binding.badge2, ok = false)

            // Step 3: Recognition
            val puzzle = result.puzzle
            if (puzzle != null) {
                currentPuzzle = puzzle
                binding.recognitionGrid.show(puzzle, null)
                binding.editButton.visibility = View.VISIBLE
                setBadge(binding.badge3, ok = true)
            } else {
                setBadge(binding.badge3, ok = result.rectified != null, skipped = result.rectified == null)
            }

            // Step 4: Solution
            val solution = result.solution
            if (solution != null) {
                binding.gridView.show(puzzle, solution)
                binding.solutionCaption.visibility = View.VISIBLE
                setBadge(binding.badge4, ok = true)
            } else if (puzzle != null) {
                setBadge(binding.badge4, ok = false)
            } else {
                setBadge(binding.badge4, skipped = true)
            }

            // Timing
            if (result.timings.isNotEmpty()) {
                val stageNames = mapOf(
                    SudokuPipeline.Stage.GRID_DETECTION to "Grid detection",
                    SudokuPipeline.Stage.CELL_DETECTION to "Cell detection",
                    SudokuPipeline.Stage.DIGIT_READING to "Digit reading",
                    SudokuPipeline.Stage.SOLVING to "Solving",
                )
                val lines = result.timings.entries.joinToString("\n") { (stage, ms) ->
                    "${stageNames[stage] ?: stage.name.lowercase()}:  $ms ms"
                }
                binding.timingText.text = "$lines\nTotal:  ${result.timings.values.sum()} ms"
                binding.timingCard.visibility = View.VISIBLE
            }

            status(when {
                solution != null -> "Solved in ${result.timings.values.sum()} ms"
                result.message != null -> result.message!!
                else -> "Could not solve this photo."
            })
        }
    }

    /** Re-run the Kotlin solver on whatever the user has typed into the edit grid. */
    private fun resolveEdited() {
        val edited = binding.recognitionGrid.getEditedPuzzle()
        lifecycleScope.launch {
            val solution = withContext(Dispatchers.Default) {
                try { SudokuSolver.solve(edited) } catch (e: SudokuException) { null }
            }
            if (solution != null) {
                currentPuzzle = edited
                binding.gridView.show(edited, solution)
                binding.solutionCaption.visibility = View.VISIBLE
                setBadge(binding.badge4, ok = true)
                status("Re-solved")
            } else {
                setBadge(binding.badge4, ok = false)
                status("No solution — check your corrections.")
            }
        }
    }

    private fun resetSteps() {
        currentPuzzle = null
        editMode = false
        listOf(binding.badge1, binding.badge2, binding.badge3, binding.badge4)
            .forEach { it.text = "" }
        binding.photoView.visibility = View.GONE
        binding.rectifiedView.visibility = View.GONE
        binding.recognitionGrid.clear()
        binding.recognitionGrid.stopEditing()
        binding.editButton.visibility = View.GONE
        binding.editButton.text = getString(R.string.edit_recognition)
        binding.gridView.clear()
        binding.solutionCaption.visibility = View.GONE
        binding.numberPadCard.visibility = View.GONE
        binding.timingCard.visibility = View.GONE
    }

    private fun setBadge(view: TextView, ok: Boolean = false, skipped: Boolean = false) {
        when {
            skipped -> { view.text = "—"; view.setTextColor(Color.parseColor("#AAAAAA")) }
            ok      -> { view.text = "✓"; view.setTextColor(Color.parseColor("#1a9e5c")) }
            else    -> { view.text = "✗"; view.setTextColor(Color.parseColor("#d63031")) }
        }
    }

    private fun matToBitmap(mat: Mat): Bitmap {
        val bgra = Mat()
        Imgproc.cvtColor(mat, bgra, Imgproc.COLOR_RGB2BGRA)
        val bmp = Bitmap.createBitmap(bgra.cols(), bgra.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(bgra, bmp)
        bgra.release()
        return bmp
    }

    private fun status(text: String) { binding.status.text = text }

    override fun onDestroy() {
        super.onDestroy()
        pipeline?.close()
    }
}
