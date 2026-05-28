package com.aifencingcoach.runtime

import android.content.Context
import android.graphics.Bitmap
import androidx.camera.core.ImageProxy
import java.util.ArrayDeque
import kotlin.system.measureNanoTime

class LiveCoachPipeline(
    context: Context,
    private var poseBackendKind: PoseBackendKind = PoseBackendKind.MEDIAPIPE,
    private var targetSide: TargetSide = TargetSide.LEFT,
    private var trainingMode: TrainingMode = TrainingMode.FREE_BOUTING
) : AutoCloseable {
    private val appContext = context.applicationContext
    private val gatekeeper = ActivityGatekeeper(fps = 30)
    private val targetTracker = TargetTracker(targetSide)
    private val normalizer = SpatialNormalizer()
    private val heuristics = HeuristicsEngine(targetSide, trainingMode)
    private val scheduler = FeedbackScheduler(appContext, trainingMode)
    private val rawSkeletons = ArrayDeque<Skeleton>()
    private val normalizedFrames = ArrayDeque<FloatArray>()
    private val cueHistory = ArrayDeque<CueHistoryItem>()
    private val cueTimeline = ArrayDeque<CueHistoryItem>()
    private val classifier = FenceNetClassifier.createOrNull(appContext)
    private val poseBackend = createPoseBackend(appContext, poseBackendKind)
    private var frameIndex = 0L
    private var lastInferenceFrame = 0L
    private var lastAction = "Idle"
    private var lastConfidence = 0f
    private var lastMetrics = PipelineMetrics()
    private var lastVisualCues: List<FeedbackCue> = emptyList()
    private var lastTargetSkeleton: Skeleton? = null
    private var lastOpponentSkeleton: Skeleton? = null
    private var previousActive = false
    private var sessionStartedAtNs = System.nanoTime()
    private var activeFrames = 0L
    private var inferenceCount = 0L
    private var cueCount = 0L
    private var estimatedDroppedFrames = 0L
    private var lastImageTimestampNs: Long? = null
    private var lastFps = 0f
    private val actionCounts = linkedMapOf<String, Long>()
    private val cueCounts = linkedMapOf<String, CueCountItem>()

    val ready: Boolean
        get() = poseBackend.ready && classifier != null

    fun configure(targetSide: TargetSide, trainingMode: TrainingMode) {
        this.targetSide = targetSide
        this.trainingMode = trainingMode
        targetTracker.configure(targetSide)
        heuristics.configure(targetSide, trainingMode)
        scheduler.configure(trainingMode)
        reset()
    }

    fun reset() {
        gatekeeper.reset()
        targetTracker.reset()
        normalizer.reset()
        clearMotionBuffers()
        scheduler.reset()
        cueHistory.clear()
        cueTimeline.clear()
        frameIndex = 0L
        lastInferenceFrame = 0L
        lastAction = "Idle"
        lastConfidence = 0f
        lastVisualCues = emptyList()
        lastTargetSkeleton = null
        lastOpponentSkeleton = null
        previousActive = false
        sessionStartedAtNs = System.nanoTime()
        activeFrames = 0L
        inferenceCount = 0L
        cueCount = 0L
        estimatedDroppedFrames = 0L
        lastImageTimestampNs = null
        lastFps = 0f
        actionCounts.clear()
        cueCounts.clear()
        lastMetrics = PipelineMetrics()
    }

    fun practiceReport(elapsedSecondsOverride: Long? = null): PracticeReport =
        buildPracticeReport(
            trainingMode = trainingMode,
            poseBackend = poseBackendKind,
            targetSide = targetSide,
            elapsedSeconds = elapsedSecondsOverride ?: elapsedSeconds(),
            activeFrames = activeFrames,
            fps = 30,
            inferenceCount = inferenceCount,
            actionCounts = actionCounts,
            cueCounts = cueCounts,
            cueTimeline = cueTimeline.toList(),
            generatedAtFrame = frameIndex
        )

    fun process(imageProxy: ImageProxy): Pair<CoachFrameState, FeedbackCue?> =
        processFrame(
            width = imageProxy.width,
            height = imageProxy.height,
            timestampNs = imageProxy.imageInfo.timestamp,
            detectPose = { poseBackend.detect(imageProxy, targetSide) }
        )

    fun processBitmap(
        bitmap: Bitmap,
        timestampNs: Long,
        rotationDegrees: Int = 0
    ): Pair<CoachFrameState, FeedbackCue?> =
        processFrame(
            width = bitmap.width,
            height = bitmap.height,
            timestampNs = timestampNs,
            detectPose = { poseBackend.detectBitmap(bitmap, targetSide, rotationDegrees, timestampNs) }
        )

    private fun processFrame(
        width: Int,
        height: Int,
        timestampNs: Long,
        detectPose: () -> PoseBackendResult
    ): Pair<CoachFrameState, FeedbackCue?> {
        val totalStart = System.nanoTime()
        updateFrameCadence(timestampNs)

        if (!ready || classifier == null) {
            frameIndex += 1
            val missing = buildList {
                if (!poseBackend.ready) add(poseBackend.missingMessage)
                if (classifier == null) add("Add fencenet_v2.onnx to assets.")
            }.joinToString(" ")
            updateLastMetrics(totalStart)
            return CoachFrameState(
                state = "MODEL MISSING",
                action = "Idle",
                cue = missing,
                poseBackend = poseBackendKind,
                frameWidth = width,
                frameHeight = height,
                frameIndex = frameIndex,
                ready = false,
                metrics = lastMetrics,
                tracking = trackingStatus(),
                session = sessionStats(),
                cueHistory = cueHistoryList()
            ) to null
        }

        var targetSkeleton: Skeleton? = null
        var opponentSkeleton: Skeleton? = null
        var poseMs = 0L
        var classifierMs = 0L
        var decision = FeedbackDecision(null, lastVisualCues)

        val shouldExtractPose = gatekeeper.shouldExtractPose()
        if (!shouldExtractPose) {
            frameIndex += 1
            updateLastMetrics(totalStart)
            return CoachFrameState(
                state = gatekeeper.state,
                action = lastAction,
                confidence = lastConfidence,
                cue = "",
                poseBackend = poseBackendKind,
                visualCues = lastVisualCues,
                cueHistory = cueHistoryList(),
                targetSkeleton = lastTargetSkeleton,
                opponentSkeleton = lastOpponentSkeleton,
                frameWidth = width,
                frameHeight = height,
                frameIndex = frameIndex,
                ready = true,
                metrics = lastMetrics,
                tracking = trackingStatus(),
                session = sessionStats()
            ) to null
        }

        poseMs = measureNanoTime {
            val result = detectPose()
            val tracking = targetTracker.process(result.detections, frameIndex, targetSide)
            targetSkeleton = tracking.targetSkeleton ?: result.targetSkeleton
            opponentSkeleton = tracking.opponentSkeleton ?: result.opponentSkeleton
        }.nanosToMillis()

        if (targetSkeleton != null) lastTargetSkeleton = targetSkeleton
        if (opponentSkeleton != null) lastOpponentSkeleton = opponentSkeleton

        val active = gatekeeper.update(targetSkeleton, opponentSkeleton, width, targetSide)
        if (active && !previousActive) {
            clearMotionBuffers()
            normalizer.reset()
        } else if (!active && previousActive) {
            clearMotionBuffers()
            lastAction = "Idle"
            lastConfidence = 0f
            lastVisualCues = emptyList()
        }
        previousActive = active
        if (active) activeFrames += 1

        if (active && targetSkeleton != null) {
            appendActiveSkeleton(targetSkeleton)
        }

        if (
            active &&
            normalizedFrames.size == FenceNetClassifier.WindowSize &&
            frameIndex - lastInferenceFrame >= FenceNetClassifier.Stride
        ) {
            classifierMs = measureNanoTime {
                classifier.classify(normalizedFrames.toList(), frameIndex.toInt())?.let { prediction ->
                    lastAction = prediction.action
                    lastConfidence = prediction.confidence
                    inferenceCount += 1
                    actionCounts[prediction.action] = (actionCounts[prediction.action] ?: 0L) + 1L
                    val activeErrors = if (prediction.action != "Idle") {
                        heuristics.evaluate(prediction.action, rawSkeletons.toList())
                    } else {
                        emptyList()
                    }
                    decision = scheduler.update(
                        activeErrorKeys = activeErrors,
                        nowSeconds = timestampNs / 1_000_000_000.0
                    )
                    lastVisualCues = decision.visualCues
                    rememberCue(decision.voiceCue ?: decision.visualCues.firstOrNull())
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
            droppedFrames = estimatedDroppedFrames,
            fps = lastFps
        )

        val cue = decision.voiceCue
        val displayCue = displayCue(
            active = active,
            targetSkeleton = targetSkeleton,
            decision = decision
        )
        return CoachFrameState(
            state = gatekeeper.state,
            action = lastAction,
            confidence = lastConfidence,
            cue = displayCue,
            poseBackend = poseBackendKind,
            visualCues = decision.visualCues,
            cueHistory = cueHistoryList(),
            targetSkeleton = targetSkeleton,
            opponentSkeleton = opponentSkeleton,
            frameWidth = width,
            frameHeight = height,
            frameIndex = frameIndex,
            ready = true,
            metrics = lastMetrics,
            tracking = trackingStatus(),
            session = sessionStats()
        ) to cue
    }

    private fun appendActiveSkeleton(targetSkeleton: Skeleton) {
        val modelFrame = runCatching {
            if (normalizedFrames.isEmpty()) normalizer.fit(targetSkeleton)
            normalizer.modelArray(normalizer.normalize(targetSkeleton))
        }.getOrNull() ?: return

        rawSkeletons.addLast(targetSkeleton)
        if (rawSkeletons.size > FenceNetClassifier.WindowSize) rawSkeletons.removeFirst()
        normalizedFrames.addLast(modelFrame)
        if (normalizedFrames.size > FenceNetClassifier.WindowSize) normalizedFrames.removeFirst()
    }

    private fun clearMotionBuffers() {
        rawSkeletons.clear()
        normalizedFrames.clear()
    }

    private fun updateFrameCadence(timestampNs: Long) {
        val previous = lastImageTimestampNs
        if (previous != null && timestampNs > previous) {
            val delta = timestampNs - previous
            lastFps = 1_000_000_000f / delta.toFloat()
            val expected = 1_000_000_000L / 30L
            if (delta > expected * 3L / 2L) {
                estimatedDroppedFrames += maxOf(0L, delta / expected - 1L)
            }
        }
        lastImageTimestampNs = timestampNs
    }

    private fun updateLastMetrics(totalStart: Long) {
        lastMetrics = lastMetrics.copy(
            totalMs = (System.nanoTime() - totalStart).nanosToMillis(),
            droppedFrames = estimatedDroppedFrames,
            fps = lastFps
        )
    }

    private fun displayCue(
        active: Boolean,
        targetSkeleton: Skeleton?,
        decision: FeedbackDecision
    ): String {
        decision.voiceCue?.message?.let { return it }
        decision.visualCues.firstOrNull()?.message?.let { return it }
        if (targetSkeleton == null) return "Find target"
        if (!active) {
            return if (gatekeeper.state == ActivityGatekeeper.StateChecking) {
                "Hold en garde"
            } else {
                "Find stance"
            }
        }
        val fill = normalizedFrames.size
        if (fill < FenceNetClassifier.WindowSize) {
            return "Warming up $fill/${FenceNetClassifier.WindowSize}"
        }
        return ""
    }

    private fun rememberCue(cue: FeedbackCue?) {
        if (cue == null || cue.message.isBlank()) return
        val last = cueHistory.lastOrNull()
        if (last?.label == cue.label && frameIndex - last.frameIndex < FenceNetClassifier.Stride) return
        cueHistory.addLast(
            CueHistoryItem(
                frameIndex = frameIndex,
                errorKey = cue.errorKey,
                label = cue.label,
                message = cue.message,
                priority = cue.priority
            )
        )
        while (cueHistory.size > MaxCueHistory) cueHistory.removeFirst()
        cueTimeline.addLast(cueHistory.last())
        while (cueTimeline.size > MaxCueTimeline) cueTimeline.removeFirst()
        val previous = cueCounts[cue.errorKey]
        cueCounts[cue.errorKey] = CueCountItem(
            errorKey = cue.errorKey,
            label = cue.label,
            message = cue.message,
            count = (previous?.count ?: 0L) + 1L
        )
        cueCount += 1
    }

    private fun cueHistoryList(): List<CueHistoryItem> = cueHistory.toList().asReversed()

    private fun trackingStatus(): TrackingStatus {
        val fill = normalizedFrames.size
        return targetTracker.status.copy(
            bufferFill = fill,
            warmupFramesRemaining = maxOf(0, FenceNetClassifier.WindowSize - fill)
        )
    }

    private fun sessionStats(): SessionStats {
        val topAction = actionCounts
            .filterKeys { it != "Idle" }
            .maxByOrNull { it.value }
            ?.key ?: lastAction
        return SessionStats(
            elapsedSeconds = elapsedSeconds(),
            activeFrames = activeFrames,
            inferenceCount = inferenceCount,
            cueCount = cueCount,
            topAction = topAction
        )
    }

    private fun elapsedSeconds(): Long =
        (System.nanoTime() - sessionStartedAtNs) / 1_000_000_000L

    override fun close() {
        classifier?.close()
        poseBackend.close()
    }

    companion object {
        private const val MaxCueHistory = 5
        private const val MaxCueTimeline = 60
        private fun Long.nanosToMillis(): Long = this / 1_000_000
    }
}
