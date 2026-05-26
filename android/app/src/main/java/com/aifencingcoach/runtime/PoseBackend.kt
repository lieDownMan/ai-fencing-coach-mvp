package com.aifencingcoach.runtime

import android.content.Context
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.MediaImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker

data class PoseBackendResult(
    val targetSkeleton: Skeleton?,
    val opponentSkeleton: Skeleton?
)

interface PoseBackend : AutoCloseable {
    val kind: PoseBackendKind
    val ready: Boolean
    val missingMessage: String

    fun detect(
        imageProxy: ImageProxy,
        targetSide: TargetSide
    ): PoseBackendResult
}

class MediaPipePoseBackend(context: Context) : PoseBackend {
    override val kind = PoseBackendKind.MEDIAPIPE
    override val missingMessage = "Add pose_landmarker_lite.task to assets."

    private val mapper = PoseMapper()
    private val poseLandmarker = createPoseLandmarker(context.applicationContext)

    override val ready: Boolean
        get() = poseLandmarker != null

    override fun detect(
        imageProxy: ImageProxy,
        targetSide: TargetSide
    ): PoseBackendResult {
        val landmarker = poseLandmarker ?: return PoseBackendResult(null, null)
        val mediaImage = imageProxy.image ?: return PoseBackendResult(null, null)
        val mpImage = MediaImageBuilder(mediaImage).build()
        val processingOptions = ImageProcessingOptions.builder()
            .setRotationDegrees(imageProxy.imageInfo.rotationDegrees)
            .build()
        val timestampMs = imageProxy.imageInfo.timestamp / 1_000_000
        val result = landmarker.detectForVideo(mpImage, processingOptions, timestampMs)
        val selected = mapper.selectFencers(
            poses = result.landmarks(),
            frameWidth = imageProxy.width,
            frameHeight = imageProxy.height,
            targetSide = targetSide
        )
        return PoseBackendResult(selected.first, selected.second)
    }

    override fun close() {
        poseLandmarker?.close()
    }

    companion object {
        private fun createPoseLandmarker(context: Context): PoseLandmarker? = runCatching {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("pose_landmarker_lite.task")
                .build()
            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.VIDEO)
                .setNumPoses(2)
                .setMinPoseDetectionConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .build()
            PoseLandmarker.createFromOptions(context, options)
        }.getOrNull()
    }
}

class YoloPoseBackend(private val context: Context) : PoseBackend {
    override val kind = PoseBackendKind.YOLO
    override val ready: Boolean
        get() = runCatching {
            context.assets.open("yolo_pose.onnx").close()
            false
        }.getOrDefault(false)
    override val missingMessage =
        "YOLO backend is selectable, but yolo_pose.onnx decoding is not wired yet."

    override fun detect(
        imageProxy: ImageProxy,
        targetSide: TargetSide
    ): PoseBackendResult = PoseBackendResult(null, null)

    override fun close() = Unit
}

fun createPoseBackend(context: Context, kind: PoseBackendKind): PoseBackend =
    when (kind) {
        PoseBackendKind.MEDIAPIPE -> MediaPipePoseBackend(context)
        PoseBackendKind.YOLO -> YoloPoseBackend(context.applicationContext)
    }
