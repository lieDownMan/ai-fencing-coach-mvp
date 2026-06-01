package com.aifencingcoach.runtime

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

data class PostgameAnalysisProgress(
    val fraction: Float,
    val status: String
)

class PostgameVideoAnalyzer(private val context: Context) {
    suspend fun analyze(
        uri: Uri,
        targetSide: TargetSide,
        trainingMode: TrainingMode,
        poseBackend: PoseBackendKind,
        onProgress: suspend (PostgameAnalysisProgress) -> Unit
    ): Pair<PracticeReport, Map<Long, CoachFrameState>> = withContext(Dispatchers.Default) {
        val retriever = MediaMetadataRetriever()
        val pipeline = LiveCoachPipeline(context, poseBackend, targetSide, trainingMode)
        val frameStates = mutableMapOf<Long, CoachFrameState>()

        try {
            retriever.setDataSource(context, uri)
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull()
                ?.coerceAtLeast(1L)
                ?: 1L
            val durationUs = durationMs * 1_000L
            val totalFrames = maxOf(1, (durationUs / FrameStepUs).toInt())
            update(onProgress, 0f, "Reading clip")

            var processedFrames = 0
            var timeUs = 0L
            while (timeUs <= durationUs) {
                val bitmap = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)
                if (bitmap != null) {
                    val frame = bitmap.ensureArgb8888()
                    try {
                        val (state, _) = pipeline.processBitmap(frame, timestampNs = timeUs * 1_000L)
                        frameStates[timeUs] = state
                    } finally {
                        frame.recycle()
                        if (frame !== bitmap) bitmap.recycle()
                    }
                }

                processedFrames += 1
                if (processedFrames == 1 || processedFrames % ProgressEveryFrames == 0) {
                    update(
                        onProgress = onProgress,
                        fraction = (processedFrames.toFloat() / totalFrames).coerceIn(0f, 0.98f),
                        status = when {
                            processedFrames < totalFrames * 0.25f -> "Finding poses"
                            processedFrames < totalFrames * 0.6f -> "Scoring actions"
                            else -> "Building summary"
                        }
                    )
                }
                timeUs += FrameStepUs
            }

            update(onProgress, 1f, "Analysis ready")
            Pair(
                pipeline.practiceReport(elapsedSecondsOverride = maxOf(1L, durationMs / 1_000L)),
                frameStates
            )
        } finally {
            pipeline.close()
            retriever.release()
        }
    }

    private suspend fun update(
        onProgress: suspend (PostgameAnalysisProgress) -> Unit,
        fraction: Float,
        status: String
    ) {
        withContext(Dispatchers.Main) {
            onProgress(PostgameAnalysisProgress(fraction, status))
        }
    }

    private fun Bitmap.ensureArgb8888(): Bitmap =
        if (config == Bitmap.Config.ARGB_8888) this else copy(Bitmap.Config.ARGB_8888, false) ?: this

    private companion object {
        private const val FrameStepUs = 33_333L
        private const val ProgressEveryFrames = 5
    }
}
