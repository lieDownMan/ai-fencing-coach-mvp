package com.aifencingcoach.runtime

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.io.ByteArrayOutputStream
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

data class PoseBackendResult(
    val targetSkeleton: Skeleton?,
    val opponentSkeleton: Skeleton?,
    val detections: List<PoseDetection> = emptyList()
)

interface PoseBackend : AutoCloseable {
    val kind: PoseBackendKind
    val ready: Boolean
    val missingMessage: String

    fun detect(
        imageProxy: ImageProxy,
        targetSide: TargetSide
    ): PoseBackendResult

    fun detectBitmap(
        bitmap: Bitmap,
        targetSide: TargetSide,
        rotationDegrees: Int,
        timestampNs: Long
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
        val rawBitmap = imageProxy.toBitmapOrNull() ?: return PoseBackendResult(null, null)
        val rotation = imageProxy.imageInfo.rotationDegrees
        val bitmap = if (rotation != 0) rotateBitmap(rawBitmap, rotation) else rawBitmap
        return try {
            detectBitmap(
                bitmap = bitmap,
                targetSide = targetSide,
                rotationDegrees = 0,  // already rotated
                timestampNs = imageProxy.imageInfo.timestamp
            )
        } finally {
            bitmap.recycle()
            if (bitmap !== rawBitmap) rawBitmap.recycle()
        }
    }

    override fun detectBitmap(
        bitmap: Bitmap,
        targetSide: TargetSide,
        rotationDegrees: Int,
        timestampNs: Long
    ): PoseBackendResult {
        val landmarker = poseLandmarker ?: return PoseBackendResult(null, null)
        val mpImage = BitmapImageBuilder(bitmap).build()
        val processingOptions = ImageProcessingOptions.builder()
            .setRotationDegrees(rotationDegrees)
            .build()
        val timestampMs = timestampNs / 1_000_000
        val result = landmarker.detectForVideo(mpImage, processingOptions, timestampMs)
        val detections = mapper.mapDetections(
            poses = result.landmarks(),
            frameWidth = bitmap.width,
            frameHeight = bitmap.height,
            targetSide = targetSide
        )
        return PoseBackendResult(null, null, detections)
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

class YoloPoseBackend(context: Context) : PoseBackend {
    override val kind = PoseBackendKind.YOLO
    override val missingMessage = "Add yolo_pose.onnx to assets."

    private val appContext = context.applicationContext
    private val env = OrtEnvironment.getEnvironment()
    private val session = createSession(appContext)
    private val inputName = session?.inputNames?.firstOrNull().orEmpty()

    override val ready: Boolean
        get() = session != null

    override fun detect(
        imageProxy: ImageProxy,
        targetSide: TargetSide
    ): PoseBackendResult {
        val rawBitmap = imageProxy.toBitmapOrNull() ?: return PoseBackendResult(null, null)
        val rotation = imageProxy.imageInfo.rotationDegrees
        val bitmap = if (rotation != 0) rotateBitmap(rawBitmap, rotation) else rawBitmap
        return try {
            detectBitmap(bitmap, targetSide, 0, imageProxy.imageInfo.timestamp)  // already rotated
        } finally {
            bitmap.recycle()
            if (bitmap !== rawBitmap) rawBitmap.recycle()
        }
    }

    override fun detectBitmap(
        bitmap: Bitmap,
        targetSide: TargetSide,
        rotationDegrees: Int,
        timestampNs: Long
    ): PoseBackendResult {
        val activeSession = session ?: return PoseBackendResult(null, null)
        val prepared = prepareInput(bitmap)
        val detections = runInference(activeSession, prepared)
            .sortedByDescending { it.score }
            .take(8)
            .mapIndexedNotNull { index, detection -> detection.toPoseDetection(targetSide, index) }
        return PoseBackendResult(null, null, detections)
    }

    override fun close() {
        session?.close()
    }

    private fun runInference(session: OrtSession, input: LetterboxInput): List<YoloPoseDetection> {
        OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(input.tensor),
            longArrayOf(1, 3, ModelSize.toLong(), ModelSize.toLong())
        ).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val output = result.get(0).value as Array<Array<FloatArray>>
                return decodeOutput(output[0], input)
            }
        }
    }

    private fun decodeOutput(
        channels: Array<FloatArray>,
        input: LetterboxInput
    ): List<YoloPoseDetection> {
        if (channels.size < 56) return emptyList()
        val boxes = mutableListOf<YoloPoseDetection>()
        val candidateCount = channels[0].size
        for (index in 0 until candidateCount) {
            val score = channels[4][index]
            if (score < ConfidenceThreshold) continue
            val cx = unletterboxX(channels[0][index], input)
            val cy = unletterboxY(channels[1][index], input)
            val w = channels[2][index] / input.scale
            val h = channels[3][index] / input.scale
            val keypoints = Array(17) { kp ->
                val offset = 5 + kp * 3
                YoloKeypoint(
                    x = unletterboxX(channels[offset][index], input),
                    y = unletterboxY(channels[offset + 1][index], input),
                    confidence = channels[offset + 2][index]
                )
            }
            boxes.add(
                YoloPoseDetection(
                    x1 = (cx - w / 2f).coerceIn(0f, input.sourceWidth.toFloat()),
                    y1 = (cy - h / 2f).coerceIn(0f, input.sourceHeight.toFloat()),
                    x2 = (cx + w / 2f).coerceIn(0f, input.sourceWidth.toFloat()),
                    y2 = (cy + h / 2f).coerceIn(0f, input.sourceHeight.toFloat()),
                    score = score,
                    keypoints = keypoints
                )
            )
        }
        return nonMaxSuppression(boxes)
    }

    private fun prepareInput(bitmap: Bitmap): LetterboxInput {
        val sourceWidth = bitmap.width
        val sourceHeight = bitmap.height
        val scale = min(ModelSize / sourceWidth.toFloat(), ModelSize / sourceHeight.toFloat())
        val resizedWidth = max(1, (sourceWidth * scale).toInt())
        val resizedHeight = max(1, (sourceHeight * scale).toInt())
        val padX = (ModelSize - resizedWidth) / 2f
        val padY = (ModelSize - resizedHeight) / 2f

        val square = Bitmap.createBitmap(ModelSize, ModelSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(square)
        canvas.drawColor(Color.rgb(114, 114, 114))
        val destination = Rect(
            padX.toInt(),
            padY.toInt(),
            padX.toInt() + resizedWidth,
            padY.toInt() + resizedHeight
        )
        canvas.drawBitmap(bitmap, null, destination, Paint(Paint.FILTER_BITMAP_FLAG))

        val pixels = IntArray(ModelSize * ModelSize)
        square.getPixels(pixels, 0, ModelSize, 0, 0, ModelSize, ModelSize)
        val tensor = FloatArray(3 * ModelSize * ModelSize)
        val planeSize = ModelSize * ModelSize
        for (i in pixels.indices) {
            val color = pixels[i]
            tensor[i] = Color.red(color) / 255f
            tensor[planeSize + i] = Color.green(color) / 255f
            tensor[2 * planeSize + i] = Color.blue(color) / 255f
        }
        square.recycle()
        return LetterboxInput(tensor, scale, padX, padY, sourceWidth, sourceHeight)
    }

    private fun unletterboxX(x: Float, input: LetterboxInput): Float =
        ((x - input.padX) / input.scale).coerceIn(0f, input.sourceWidth.toFloat())

    private fun unletterboxY(y: Float, input: LetterboxInput): Float =
        ((y - input.padY) / input.scale).coerceIn(0f, input.sourceHeight.toFloat())

    private fun nonMaxSuppression(boxes: List<YoloPoseDetection>): List<YoloPoseDetection> {
        val selected = mutableListOf<YoloPoseDetection>()
        for (candidate in boxes.sortedByDescending { it.score }) {
            if (selected.any { iou(candidate, it) > IouThreshold }) continue
            selected.add(candidate)
            if (selected.size >= MaxDetections) break
        }
        return selected
    }

    private fun iou(a: YoloPoseDetection, b: YoloPoseDetection): Float {
        val x1 = max(a.x1, b.x1)
        val y1 = max(a.y1, b.y1)
        val x2 = min(a.x2, b.x2)
        val y2 = min(a.y2, b.y2)
        val intersection = max(0f, x2 - x1) * max(0f, y2 - y1)
        val areaA = max(0f, a.x2 - a.x1) * max(0f, a.y2 - a.y1)
        val areaB = max(0f, b.x2 - b.x1) * max(0f, b.y2 - b.y1)
        return intersection / (areaA + areaB - intersection).coerceAtLeast(1e-6f)
    }

    private data class LetterboxInput(
        val tensor: FloatArray,
        val scale: Float,
        val padX: Float,
        val padY: Float,
        val sourceWidth: Int,
        val sourceHeight: Int
    )

    private data class YoloKeypoint(
        val x: Float,
        val y: Float,
        val confidence: Float
    )

    private data class YoloPoseDetection(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val score: Float,
        val keypoints: Array<YoloKeypoint>
    ) {
        fun toPoseDetection(targetSide: TargetSide, sourceRank: Int): PoseDetection? {
            fun point(index: Int): Point2? {
                val keypoint = keypoints.getOrNull(index) ?: return null
                if (keypoint.confidence < KeypointThreshold) return null
                return Point2(keypoint.x, keypoint.y)
            }

            val nose = point(0) ?: return null
            val leftShoulder = point(5)
            val rightShoulder = point(6)
            val leftElbow = point(7)
            val rightElbow = point(8)
            val leftWrist = point(9)
            val rightWrist = point(10)
            val leftHip = point(11) ?: return null
            val rightHip = point(12) ?: return null
            val leftKnee = point(13) ?: return null
            val rightKnee = point(14) ?: return null
            val leftAnkle = point(15) ?: return null
            val rightAnkle = point(16) ?: return null

            val frontWrist = rightWrist
            val frontElbow = rightElbow
            val frontShoulder = rightShoulder
            val frontAnkle = rightAnkle
            val backWrist = leftWrist
            if (frontWrist == null || frontElbow == null || frontShoulder == null) return null

            val skeleton = linkedMapOf(
                "nose" to nose,
                "left_shoulder" to (leftShoulder ?: frontShoulder),
                "right_shoulder" to (rightShoulder ?: frontShoulder),
                "front_wrist" to frontWrist,
                "front_elbow" to frontElbow,
                "front_shoulder" to frontShoulder,
                "front_ankle" to frontAnkle,
                "left_hip" to leftHip,
                "right_hip" to rightHip,
                "left_knee" to leftKnee,
                "right_knee" to rightKnee,
                "left_ankle" to leftAnkle,
                "right_ankle" to rightAnkle
            )
            if (backWrist != null) skeleton["back_wrist"] = backWrist
            return PoseDetection(
                skeleton = skeleton,
                bbox = BoundingBox(x1, y1, x2, y2),
                confidence = score,
                sourceRank = sourceRank
            )
        }
    }

    companion object {
        private const val ModelSize = 640
        private const val ConfidenceThreshold = 0.35f
        private const val KeypointThreshold = 0.35f
        private const val IouThreshold = 0.45f
        private const val MaxDetections = 2

        private fun createSession(context: Context): OrtSession? = runCatching {
            val bytes = context.assets.open("yolo_pose.onnx").use { it.readBytes() }
            OrtEnvironment.getEnvironment().createSession(bytes, OrtSession.SessionOptions())
        }.getOrNull()
    }
}

fun createPoseBackend(context: Context, kind: PoseBackendKind): PoseBackend =
    when (kind) {
        PoseBackendKind.MEDIAPIPE -> MediaPipePoseBackend(context)
        PoseBackendKind.YOLO -> YoloPoseBackend(context.applicationContext)
    }

private fun ImageProxy.toBitmapOrNull(): Bitmap? {
    if (format != ImageFormat.YUV_420_888 || planes.size < 3) return null
    val nv21 = yuv420ToNv21(this)
    val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, width, height), 90, out)
    return BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
}

private fun yuv420ToNv21(image: ImageProxy): ByteArray {
    val yPlane = image.planes[0]
    val uPlane = image.planes[1]
    val vPlane = image.planes[2]
    val out = ByteArray(image.width * image.height * 3 / 2)
    copyPlane(yPlane, image.width, image.height, out, 0, 1)
    copyInterleavedChroma(vPlane, uPlane, image.width, image.height, out, image.width * image.height)
    return out
}

private fun copyPlane(
    plane: ImageProxy.PlaneProxy,
    width: Int,
    height: Int,
    out: ByteArray,
    offset: Int,
    pixelStrideOut: Int
) {
    val buffer = plane.buffer.duplicate()
    val rowStride = plane.rowStride
    val pixelStride = plane.pixelStride
    var outputIndex = offset
    for (row in 0 until height) {
        for (col in 0 until width) {
            out[outputIndex] = buffer.get(row * rowStride + col * pixelStride)
            outputIndex += pixelStrideOut
        }
    }
}

private fun copyInterleavedChroma(
    vPlane: ImageProxy.PlaneProxy,
    uPlane: ImageProxy.PlaneProxy,
    width: Int,
    height: Int,
    out: ByteArray,
    offset: Int
) {
    val chromaWidth = width / 2
    val chromaHeight = height / 2
    val vBuffer = vPlane.buffer.duplicate()
    val uBuffer = uPlane.buffer.duplicate()
    var outputIndex = offset
    for (row in 0 until chromaHeight) {
        for (col in 0 until chromaWidth) {
            out[outputIndex++] = vBuffer.get(row * vPlane.rowStride + col * vPlane.pixelStride)
            out[outputIndex++] = uBuffer.get(row * uPlane.rowStride + col * uPlane.pixelStride)
        }
    }
}

private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
    if (degrees == 0) return bitmap
    val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
}
