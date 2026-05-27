package com.aifencingcoach.runtime

import kotlin.math.abs
import kotlin.math.sqrt

data class Point2(val x: Float, val y: Float) {
    fun distanceTo(other: Point2): Float {
        val dx = x - other.x
        val dy = y - other.y
        return sqrt(dx * dx + dy * dy)
    }
}

typealias Skeleton = Map<String, Point2>

data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float
) {
    val width: Float get() = abs(x2 - x1)
    val height: Float get() = abs(y2 - y1)
    val area: Float get() = width * height
    val centerX: Float get() = (x1 + x2) / 2f
    val centerY: Float get() = (y1 + y2) / 2f

    companion object {
        fun fromSkeleton(skeleton: Skeleton): BoundingBox? {
            if (skeleton.isEmpty()) return null
            val xs = skeleton.values.map { it.x }
            val ys = skeleton.values.map { it.y }
            return BoundingBox(
                x1 = xs.minOrNull() ?: return null,
                y1 = ys.minOrNull() ?: return null,
                x2 = xs.maxOrNull() ?: return null,
                y2 = ys.maxOrNull() ?: return null
            )
        }
    }
}

data class PoseDetection(
    val skeleton: Skeleton,
    val bbox: BoundingBox,
    val confidence: Float = 1f,
    val sourceRank: Int = 0,
    val trackId: String? = null,
    val interpolated: Boolean = false
)

enum class TargetSide(val label: String) {
    LEFT("left"),
    RIGHT("right");

    companion object {
        fun fromLabel(label: String): TargetSide =
            entries.firstOrNull { it.label == label } ?: LEFT
    }
}

enum class TrainingMode(val label: String) {
    FOOTWORK("Footwork"),
    TARGET_PRACTICE("Target Practice"),
    FREE_BOUTING("Free Bouting");

    companion object {
        fun fromLabel(label: String): TrainingMode =
            entries.firstOrNull { it.label == label } ?: FREE_BOUTING
    }
}

enum class PoseBackendKind(val label: String) {
    MEDIAPIPE("MediaPipe"),
    YOLO("YOLO");

    companion object {
        fun fromLabel(label: String): PoseBackendKind =
            entries.firstOrNull { it.label == label } ?: MEDIAPIPE
    }
}

data class ActionPrediction(
    val action: String,
    val confidence: Float,
    val startFrame: Int,
    val endFrame: Int
)

data class FeedbackCue(
    val errorKey: String,
    val label: String,
    val message: String,
    val priority: String,
    val score: Float,
    val triggered: Boolean
)

data class FeedbackDecision(
    val voiceCue: FeedbackCue?,
    val visualCues: List<FeedbackCue>
)

data class PipelineMetrics(
    val poseMs: Long = 0,
    val classifierMs: Long = 0,
    val totalMs: Long = 0,
    val droppedFrames: Long = 0,
    val fps: Float = 0f
)

data class TrackingStatus(
    val lockState: String = "unlocked",
    val detectionCount: Int = 0,
    val targetInterpolated: Boolean = false,
    val missingFrames: Int = 0,
    val bufferFill: Int = 0,
    val warmupFramesRemaining: Int = 28
)

data class SessionStats(
    val elapsedSeconds: Long = 0,
    val activeFrames: Long = 0,
    val inferenceCount: Long = 0,
    val cueCount: Long = 0,
    val topAction: String = "Idle"
)

data class CueHistoryItem(
    val frameIndex: Long,
    val errorKey: String,
    val label: String,
    val message: String,
    val priority: String
)

data class ActionCountItem(
    val action: String,
    val count: Long,
    val percent: Int
)

data class CueCountItem(
    val errorKey: String,
    val label: String,
    val message: String,
    val count: Long
)

data class PracticeReport(
    val trainingMode: TrainingMode = TrainingMode.FREE_BOUTING,
    val poseBackend: PoseBackendKind = PoseBackendKind.MEDIAPIPE,
    val targetSide: TargetSide = TargetSide.LEFT,
    val elapsedSeconds: Long = 0,
    val activeSeconds: Long = 0,
    val activePercent: Int = 0,
    val inferenceCount: Long = 0,
    val cueCount: Long = 0,
    val topAction: String = "Idle",
    val actionCounts: List<ActionCountItem> = emptyList(),
    val topCues: List<CueCountItem> = emptyList(),
    val cueTimeline: List<CueHistoryItem> = emptyList(),
    val primaryTakeaway: String = "Build a longer active sample.",
    val generatedAtFrame: Long = 0
)

data class CoachFrameState(
    val state: String = "WARMING UP",
    val action: String = "Idle",
    val confidence: Float = 0f,
    val cue: String = "",
    val poseBackend: PoseBackendKind = PoseBackendKind.MEDIAPIPE,
    val visualCues: List<FeedbackCue> = emptyList(),
    val cueHistory: List<CueHistoryItem> = emptyList(),
    val targetSkeleton: Skeleton? = null,
    val opponentSkeleton: Skeleton? = null,
    val frameWidth: Int = 1,
    val frameHeight: Int = 1,
    val frameIndex: Long = 0,
    val ready: Boolean = false,
    val metrics: PipelineMetrics = PipelineMetrics(),
    val tracking: TrackingStatus = TrackingStatus(),
    val session: SessionStats = SessionStats()
)
