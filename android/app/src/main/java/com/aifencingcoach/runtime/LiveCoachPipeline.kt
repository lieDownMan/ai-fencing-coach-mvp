package com.aifencingcoach.runtime

import android.content.Context
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.MediaImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.util.ArrayDeque
import kotlin.system.measureNanoTime

class LiveCoachPipeline(
    context: Context,
    private var targetSide: TargetSide = TargetSide.LEFT,
    private var trainingMode: TrainingMode = TrainingMode.FREE_BOUTING
) : AutoCloseable {
    private val appContext = context.applicationContext
    private val mapper = PoseMapper()
    private val gatekeeper = ActivityGatekeeper(fps = 30)
    private val normalizer = SpatialNormalizer()
    private val heuristics = HeuristicsEngine(targetSide, trainingMode)
    private val scheduler = FeedbackScheduler(appContext, trainingMode)
    private val rawSkeletons = ArrayDeque<Skeleton>()
    private val normalizedFrames = ArrayDeque<FloatArray>()
    private val classifier = FenceNetClassifier.createOrNull(appContext)
    private val poseLandmarker = createPoseLandmarker(appContext)
    private var frameIndex = 0L
    private var lastInferenceFrame = 0L
    private var lastAction = "Idle"
    private var lastConfidence = 0f
    private var lastMetrics = PipelineMetrics()
    private var previousActive = false

    val ready: Boolean
        get() = poseLandmarker != null && classifier != null

    fun configure(targetSide: TargetSide, trainingMode: TrainingMode) {
        this.targetSide = targetSide
        this.trainingMode = trainingMode
        heuristics.configure(targetSide, trainingMode)
        scheduler.configure(trainingMode)
        reset()
    }

    fun reset() {
        gatekeeper.reset()
        normalizer.reset()
        rawSkeletons.clear()
        normalizedFrames.clear()
        scheduler.reset()
        frameIndex = 0L
        lastInferenceFrame = 0L
        lastAction = "Idle"
        lastConfidence = 0f
        previousActive = false
    }

    fun process(imageProxy: ImageProxy): Pair<CoachFrameState, FeedbackCue?> {
        val totalStart = System.nanoTime()
        val width = imageProxy.width
        val height = imageProxy.height

        if (!ready || poseLandmarker == null || classifier == null) {
            frameIndex += 1
            return CoachFrameState(
                state = "MODEL MISSING",
                action = "Idle",
                cue = "Add pose_landmarker_lite.task and fencenet_v2.onnx assets.",
                frameWidth = width,
                frameHeight = height,
                frameIndex = frameIndex,
                ready = false,
                metrics = lastMetrics
            ) to null
        }

        var targetSkeleton: Skeleton? = null
        var opponentSkeleton: Skeleton? = null
        var poseMs = 0L
        var classifierMs = 0L
        var decision = FeedbackDecision(null, emptyList())

        val shouldExtractPose = gatekeeper.shouldExtractPose()
        if (!shouldExtractPose) {
            frameIndex += 1
            return CoachFrameState(
                state = gatekeeper.state,
                action = lastAction,
                confidence = lastConfidence,
                cue = "",
                frameWidth = width,
                frameHeight = height,
                frameIndex = frameIndex,
                ready = true,
                metrics = lastMetrics
            ) to null
        }

        if (shouldExtractPose) {
            poseMs = measureNanoTime {
                val mediaImage = imageProxy.image
                if (mediaImage != null) {
                    val mpImage = MediaImageBuilder(mediaImage).build()
                    val processingOptions = ImageProcessingOptions.builder()
                        .setRotationDegrees(imageProxy.imageInfo.rotationDegrees)
                        .build()
                    val timestampMs = imageProxy.imageInfo.timestamp / 1_000_000
                    val result = poseLandmarker.detectForVideo(mpImage, processingOptions, timestampMs)
                    val selected = mapper.selectFencers(
                        poses = result.landmarks(),
                        frameWidth = width,
                        frameHeight = height,
                        targetSide = targetSide
                    )
                    targetSkeleton = selected.first
                    opponentSkeleton = selected.second
                }
            }.nanosToMillis()
        }

        val active = gatekeeper.update(targetSkeleton, opponentSkeleton, width, targetSide)
        if (active && !previousActive) {
            normalizer.reset()
        }
        previousActive = active

        appendSkeleton(targetSkeleton, active)

        if (
            normalizedFrames.size == FenceNetClassifier.WindowSize &&
            frameIndex - lastInferenceFrame >= FenceNetClassifier.Stride
        ) {
            classifierMs = measureNanoTime {
                classifier.classify(normalizedFrames.toList(), frameIndex.toInt())?.let { prediction ->
                    lastAction = prediction.action
                    lastConfidence = prediction.confidence
                    val activeErrors = if (prediction.action != "Idle") {
                        heuristics.evaluate(prediction.action, rawSkeletons.toList())
                    } else {
                        emptyList()
                    }
                    decision = scheduler.update(
                        activeErrorKeys = activeErrors,
                        nowSeconds = System.nanoTime() / 1_000_000_000.0
                    )
                }
            }.nanosToMillis()
            lastInferenceFrame = frameIndex
        }

        frameIndex += 1
        val totalMs = (System.nanoTime() - totalStart).nanosToMillis()
        lastMetrics = PipelineMetrics(
            poseMs = poseMs,
            classifierMs = classifierMs,
            totalMs = totalMs,
            droppedFrames = 0
        )

        val cue = decision.voiceCue
        return CoachFrameState(
            state = gatekeeper.state,
            action = lastAction,
            confidence = lastConfidence,
            cue = cue?.message ?: decision.visualCues.firstOrNull()?.message.orEmpty(),
            visualCues = decision.visualCues,
            targetSkeleton = targetSkeleton,
            opponentSkeleton = opponentSkeleton,
            frameWidth = width,
            frameHeight = height,
            frameIndex = frameIndex,
            ready = true,
            metrics = lastMetrics
        ) to cue
    }

    private fun appendSkeleton(targetSkeleton: Skeleton?, active: Boolean) {
        if (targetSkeleton != null) {
            rawSkeletons.addLast(targetSkeleton)
            if (rawSkeletons.size > FenceNetClassifier.WindowSize) rawSkeletons.removeFirst()

            val modelFrame = if (active) {
                runCatching {
                    if (normalizedFrames.isEmpty()) normalizer.fit(targetSkeleton)
                    normalizer.modelArray(normalizer.normalize(targetSkeleton))
                }.getOrDefault(FloatArray(FenceNetClassifier.Channels))
            } else {
                FloatArray(FenceNetClassifier.Channels)
            }
            normalizedFrames.addLast(modelFrame)
        } else {
            rawSkeletons.addLast(emptyMap())
            if (rawSkeletons.size > FenceNetClassifier.WindowSize) rawSkeletons.removeFirst()
            normalizedFrames.addLast(FloatArray(FenceNetClassifier.Channels))
        }
        if (normalizedFrames.size > FenceNetClassifier.WindowSize) normalizedFrames.removeFirst()
    }

    override fun close() {
        classifier?.close()
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

        private fun Long.nanosToMillis(): Long = this / 1_000_000
    }
}
