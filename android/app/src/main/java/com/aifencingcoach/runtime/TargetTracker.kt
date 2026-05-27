package com.aifencingcoach.runtime

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.ln

data class TargetTrackingResult(
    val targetDetection: PoseDetection?,
    val opponentDetection: PoseDetection?,
    val status: TrackingStatus
) {
    val targetSkeleton: Skeleton? get() = targetDetection?.skeleton
    val opponentSkeleton: Skeleton? get() = opponentDetection?.skeleton
}

class TargetTracker(
    private var targetSide: TargetSide = TargetSide.LEFT,
    private val minBBoxArea: Float = 100f,
    private val minDetectionConfidence: Float = 0f,
    private val maxAspectRatio: Float = 8f
) {
    private var lockedTrackId: String? = null
    private var lockedFallbackBBox: BoundingBox? = null
    private var previousKnownSkeleton: Skeleton? = null
    private var previousKnownBBox: BoundingBox? = null
    private var lastKnownSkeleton: Skeleton? = null
    private var lastKnownBBox: BoundingBox? = null
    private var missingFramesCount = 0
    private val maxMissingFrames = 5
    private val maxPositionJump = 1.75f
    private val maxTrackJump = 2.5f

    var status = TrackingStatus()
        private set

    fun configure(targetSide: TargetSide) {
        if (this.targetSide != targetSide) {
            this.targetSide = targetSide
            reset()
        }
    }

    fun reset() {
        lockedTrackId = null
        lockedFallbackBBox = null
        previousKnownSkeleton = null
        previousKnownBBox = null
        lastKnownSkeleton = null
        lastKnownBBox = null
        missingFramesCount = 0
        status = TrackingStatus(lockState = "unlocked")
    }

    fun process(
        detections: List<PoseDetection>,
        frameIndex: Long,
        targetSide: TargetSide = this.targetSide
    ): TargetTrackingResult {
        this.targetSide = targetSide
        val validDetections = detections.filter(::isValidDetection)

        if (validDetections.isEmpty()) {
            val target = handleMissingTarget(validDetections.size)
            return TargetTrackingResult(target, null, status)
        }

        if (lockedTrackId == null && lockedFallbackBBox == null) {
            val initial = pickInitialTarget(validDetections)
            lockedTrackId = initial.trackId
            lockedFallbackBBox = initial.bbox
            status = status.copy(lockState = "locked", detectionCount = validDetections.size)
        }

        var target = matchByTrack(validDetections)
        if (target == null) {
            target = matchByPosition(validDetections)
            if (target?.trackId != null) lockedTrackId = target.trackId
        }

        val opponent = validDetections
            .filter { it !== target }
            .maxByOrNull { it.bbox.area }

        if (target != null) {
            rememberTarget(target, validDetections.size)
            return TargetTrackingResult(target, opponent, status)
        }

        val predicted = handleMissingTarget(validDetections.size)
        return TargetTrackingResult(predicted, opponent, status)
    }

    private fun isValidDetection(detection: PoseDetection): Boolean {
        val width = detection.bbox.width
        val height = detection.bbox.height
        if (width < 1f || height < 1f) return false
        if (detection.bbox.area < minBBoxArea) return false
        val aspect = maxOf(width / height, height / width)
        if (aspect > maxAspectRatio) return false
        return detection.confidence >= minDetectionConfidence
    }

    private fun pickInitialTarget(detections: List<PoseDetection>): PoseDetection =
        if (targetSide == TargetSide.LEFT) {
            detections.minByOrNull { it.bbox.centerX } ?: detections.first()
        } else {
            detections.maxByOrNull { it.bbox.centerX } ?: detections.first()
        }

    private fun matchByTrack(detections: List<PoseDetection>): PoseDetection? {
        val trackId = lockedTrackId ?: return null
        val candidate = detections.firstOrNull { it.trackId == trackId } ?: return null
        val reference = lastKnownBBox ?: return candidate
        return if (positionScore(candidate, reference) <= maxTrackJump) candidate else null
    }

    private fun matchByPosition(detections: List<PoseDetection>): PoseDetection? {
        val reference = lastKnownBBox ?: lockedFallbackBBox ?: return null
        val candidate = detections.minByOrNull { positionScore(it, reference) } ?: return null
        return if (positionScore(candidate, reference) <= maxPositionJump) candidate else null
    }

    private fun positionScore(detection: PoseDetection, reference: BoundingBox): Float {
        val dx = detection.bbox.centerX - reference.centerX
        val dy = detection.bbox.centerY - reference.centerY
        val diagonal = hypot(reference.width, reference.height).coerceAtLeast(1f)
        val centerDistance = hypot(dx, dy) / diagonal
        val areaRatio = abs(ln((detection.bbox.area / reference.area.coerceAtLeast(1f)).toDouble())).toFloat()
        return centerDistance + 0.25f * areaRatio
    }

    private fun rememberTarget(target: PoseDetection, detectionCount: Int) {
        previousKnownSkeleton = lastKnownSkeleton
        previousKnownBBox = lastKnownBBox
        lastKnownSkeleton = target.skeleton
        lastKnownBBox = target.bbox
        lockedFallbackBBox = target.bbox
        if (target.trackId != null) lockedTrackId = target.trackId
        missingFramesCount = 0
        status = TrackingStatus(
            lockState = "locked",
            detectionCount = detectionCount,
            targetInterpolated = false,
            missingFrames = 0
        )
    }

    private fun handleMissingTarget(detectionCount: Int): PoseDetection? {
        val lastSkeleton = lastKnownSkeleton
        val lastBBox = lastKnownBBox
        if (lastSkeleton != null && lastBBox != null && missingFramesCount < maxMissingFrames) {
            missingFramesCount += 1
            val predictedSkeleton = predictMissingSkeleton(lastSkeleton, missingFramesCount)
            val predictedBBox = predictMissingBBox(lastBBox, missingFramesCount)
            val detection = PoseDetection(
                skeleton = predictedSkeleton,
                bbox = predictedBBox,
                confidence = 0f,
                sourceRank = -1,
                trackId = lockedTrackId,
                interpolated = true
            )
            status = TrackingStatus(
                lockState = "interpolating",
                detectionCount = detectionCount,
                targetInterpolated = true,
                missingFrames = missingFramesCount
            )
            return detection
        }

        lockedTrackId = null
        lockedFallbackBBox = null
        status = TrackingStatus(
            lockState = if (lastKnownSkeleton == null) "unlocked" else "lost",
            detectionCount = detectionCount,
            targetInterpolated = false,
            missingFrames = missingFramesCount
        )
        return null
    }

    private fun predictMissingSkeleton(lastSkeleton: Skeleton, gapIndex: Int): Skeleton {
        val previousSkeleton = previousKnownSkeleton ?: return lastSkeleton
        return lastSkeleton.mapValues { (jointName, point) ->
            val previous = previousSkeleton[jointName] ?: return@mapValues point
            Point2(
                x = point.x + (point.x - previous.x) * gapIndex,
                y = point.y + (point.y - previous.y) * gapIndex
            )
        }
    }

    private fun predictMissingBBox(lastBBox: BoundingBox, gapIndex: Int): BoundingBox {
        val previous = previousKnownBBox ?: return lastBBox
        val dx = (lastBBox.centerX - previous.centerX) * gapIndex
        val dy = (lastBBox.centerY - previous.centerY) * gapIndex
        return BoundingBox(
            x1 = lastBBox.x1 + dx,
            y1 = lastBBox.y1 + dy,
            x2 = lastBBox.x2 + dx,
            y2 = lastBBox.y2 + dy
        )
    }
}
