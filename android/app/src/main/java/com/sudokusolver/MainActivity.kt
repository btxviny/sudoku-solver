package com.sudokusolver

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.sudokusolver.core.SudokuPipeline
import com.sudokusolver.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import androidx.lifecycle.lifecycleScope

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var pipeline: SudokuPipeline? = null

    /**
     * Photos come back either as a full-resolution Uri (gallery) or a thumbnail
     * Bitmap (the lightweight camera intent).  Both funnel into [solve].
     */
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
            status("OpenCV failed to load - see android/README.md for setup.")
            binding.cameraButton.isEnabled = false
            binding.galleryButton.isEnabled = false
            return
        }

        // Loading three interpreters takes long enough to notice, so it happens
        // off the main thread rather than stalling the first frame.
        lifecycleScope.launch {
            status("Loading models…")
            pipeline = withContext(Dispatchers.Default) { SudokuPipeline(applicationContext) }
            status("Take or choose a photo of a sudoku puzzle.")
        }

        binding.cameraButton.setOnClickListener { requestCamera.launch(android.Manifest.permission.CAMERA) }
        binding.galleryButton.setOnClickListener {
            pickImage.launch(Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI))
        }
    }

    private fun launchCamera() = takePhoto.launch(Intent(MediaStore.ACTION_IMAGE_CAPTURE))

    private fun loadBitmap(uri: Uri): Bitmap? =
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }

    private fun solve(bitmap: Bitmap?) {
        val pipe = pipeline
        if (bitmap == null) return status("Could not read that image.")
        if (pipe == null) return status("Models are still loading…")

        binding.gridView.clear()
        binding.progress.visibility = android.view.View.VISIBLE
        status(getString(R.string.solving))

        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) {
                val mat = Mat()
                Utils.bitmapToMat(bitmap, mat)
                // bitmapToMat yields RGBA; every model and every preprocessing
                // constant assumes plain RGB.
                Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
                pipe.solve(mat)
            }
            binding.progress.visibility = android.view.View.GONE
            binding.gridView.show(result.puzzle, result.solution)

            val ms = result.timings.values.sum()
            status(
                when {
                    result.solved -> "Solved in $ms ms"
                    result.message != null -> result.message!!
                    else -> "Could not solve this photo."
                }
            )
        }
    }

    private fun status(text: String) { binding.status.text = text }

    override fun onDestroy() {
        super.onDestroy()
        pipeline?.close()
    }
}
