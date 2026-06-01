package com.aifencingcoach.runtime

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.net.Uri
import android.os.Environment
import androidx.media3.common.MediaItem
import androidx.media3.common.util.UnstableApi
import androidx.media3.effect.BitmapOverlay
import androidx.media3.effect.OverlayEffect
import androidx.media3.transformer.Composition
import androidx.media3.transformer.EditedMediaItem
import androidx.media3.transformer.Effects
import androidx.media3.transformer.ExportException
import androidx.media3.transformer.ExportResult
import androidx.media3.transformer.Transformer
import com.google.common.collect.ImmutableList
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@androidx.annotation.OptIn(UnstableApi::class)
class VideoAnnotator(private val context: Context) {
    fun exportVideo(
        sourceUri: Uri,
        frameStates: Map<Long, CoachFrameState>,
        videoWidth: Int,
        videoHeight: Int
    ): Flow<ExportProgress> = callbackFlow {
        val outputFile = createOutputFile()
        
        // Ensure even dimensions for video encoder
        val targetW = if (videoWidth % 2 != 0) videoWidth - 1 else videoWidth
        val targetH = if (videoHeight % 2 != 0) videoHeight - 1 else videoHeight

        val overlay = object : BitmapOverlay() {
            private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                strokeWidth = 5f
                style = Paint.Style.STROKE
            }
            private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.WHITE
                textSize = 40f
                setShadowLayer(5f, 0f, 0f, Color.BLACK)
            }
            // Reuse bitmap to avoid memory churn
            private val overlayBitmap = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
            private val canvas = Canvas(overlayBitmap)

            override fun getBitmap(presentationTimeUs: Long): Bitmap {
                canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
                
                // Find closest frame state (within 50ms)
                val closestTime = frameStates.keys.minByOrNull { Math.abs(it - presentationTimeUs) }
                val state = if (closestTime != null && Math.abs(closestTime - presentationTimeUs) < 50_000L) {
                    frameStates[closestTime]
                } else null

                if (state != null) {
                    // Draw Skeleton
                    state.targetSkeleton?.let { skeleton ->
                        paint.color = Color.parseColor("#4CAF50") // Green
                        drawSkeleton(canvas, skeleton, targetW, targetH, paint)
                    }

                    // Draw HUD
                    canvas.drawText("Action: ${state.action}", 50f, 80f, textPaint)
                    if (state.cue.isNotBlank()) {
                        textPaint.color = Color.parseColor("#E57373")
                        canvas.drawText("Cue: ${state.cue}", 50f, 140f, textPaint)
                        textPaint.color = Color.WHITE
                    }
                }
                
                return overlayBitmap
            }
        }

        val editedMediaItem = EditedMediaItem.Builder(MediaItem.fromUri(sourceUri))
            .setEffects(
                Effects(
                    mutableListOf<androidx.media3.common.audio.AudioProcessor>(),
                    mutableListOf<androidx.media3.common.Effect>(OverlayEffect(ImmutableList.of<androidx.media3.effect.TextureOverlay>(overlay)))
                )
            )
            .build()

        val transformer = Transformer.Builder(context)
            .addListener(object : Transformer.Listener {
                override fun onCompleted(composition: Composition, exportResult: ExportResult) {
                    trySend(ExportProgress.Completed(outputFile.absolutePath))
                    close()
                }

                override fun onError(composition: Composition, exportResult: ExportResult, exportException: ExportException) {
                    trySend(ExportProgress.Error(exportException))
                    close(exportException)
                }
            })
            .build()

        transformer.start(editedMediaItem, outputFile.absolutePath)

        awaitClose {
            transformer.cancel()
        }
    }.flowOn(Dispatchers.Main)

    private fun drawSkeleton(canvas: Canvas, skeleton: Skeleton, width: Int, height: Int, paint: Paint) {
        val connections = listOf(
            "left_shoulder" to "right_shoulder",
            "front_shoulder" to "front_elbow",
            "front_elbow" to "front_wrist",
            "left_shoulder" to "left_hip",
            "right_shoulder" to "right_hip",
            "left_hip" to "right_hip",
            "left_hip" to "left_knee",
            "left_knee" to "left_ankle",
            "right_hip" to "right_knee",
            "right_knee" to "right_ankle"
        )
        val sourceWidth = skeleton.values.maxOfOrNull { it.x }?.coerceAtLeast(1f) ?: 1f
        val sourceHeight = skeleton.values.maxOfOrNull { it.y }?.coerceAtLeast(1f) ?: 1f
        val scaleX = width / sourceWidth
        val scaleY = height / sourceHeight
        for ((p1, p2) in connections) {
            val k1 = skeleton[p1]
            val k2 = skeleton[p2]
            if (k1 != null && k2 != null) {
                canvas.drawLine(
                    k1.x * scaleX,
                    k1.y * scaleY,
                    k2.x * scaleX,
                    k2.y * scaleY,
                    paint
                )
            }
        }
    }

    private fun createOutputFile(): File {
        val moviesDir = context.getExternalFilesDir(Environment.DIRECTORY_MOVIES)
            ?: context.cacheDir
        val fencingDir = File(moviesDir, "AiFencingCoach").apply { mkdirs() }
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        return File(fencingDir, "Session_$timeStamp.mp4")
    }
}

sealed class ExportProgress {
    data class Completed(val filePath: String) : ExportProgress()
    data class Error(val exception: Exception) : ExportProgress()
}
