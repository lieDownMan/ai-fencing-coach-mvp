package com.aifencingcoach.runtime

import kotlin.math.sqrt

data class Point2(val x: Float, val y: Float) {
    fun distanceTo(other: Point2): Float {
        val dx = x - other.x
        val dy = y - other.y
        return sqrt(dx * dx + dy * dy)
    }
}

typealias Skeleton = Map<String, Point2>

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
    val droppedFrames: Long = 0
)

data class CoachFrameState(
    val state: String = "WARMING UP",
    val action: String = "Idle",
    val confidence: Float = 0f,
    val cue: String = "",
    val poseBackend: PoseBackendKind = PoseBackendKind.MEDIAPIPE,
    val visualCues: List<FeedbackCue> = emptyList(),
    val targetSkeleton: Skeleton? = null,
    val opponentSkeleton: Skeleton? = null,
    val frameWidth: Int = 1,
    val frameHeight: Int = 1,
    val frameIndex: Long = 0,
    val ready: Boolean = false,
    val metrics: PipelineMetrics = PipelineMetrics()
)
