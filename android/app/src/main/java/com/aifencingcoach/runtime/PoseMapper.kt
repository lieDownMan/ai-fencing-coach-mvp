package com.aifencingcoach.runtime

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

class PoseMapper(private val minVisibility: Float = 0.35f) {
    fun selectFencers(
        poses: List<List<NormalizedLandmark>>,
        frameWidth: Int,
        frameHeight: Int,
        targetSide: TargetSide
    ): Pair<Skeleton?, Skeleton?> {
        val candidates = mapDetections(poses, frameWidth, frameHeight, targetSide)
            .map { it.skeleton }
            .sortedBy { centerX(it) }

        if (candidates.isEmpty()) return null to null
        val target = if (targetSide == TargetSide.LEFT) candidates.first() else candidates.last()
        val opponent = candidates.firstOrNull { it !== target }
        return target to opponent
    }

    fun mapDetections(
        poses: List<List<NormalizedLandmark>>,
        frameWidth: Int,
        frameHeight: Int,
        targetSide: TargetSide
    ): List<PoseDetection> =
        poses.mapIndexedNotNull { index, landmarks ->
            val skeleton = mapPose(landmarks, frameWidth, frameHeight, targetSide) ?: return@mapIndexedNotNull null
            val bbox = BoundingBox.fromSkeleton(skeleton) ?: return@mapIndexedNotNull null
            PoseDetection(
                skeleton = skeleton,
                bbox = bbox,
                confidence = poseConfidence(landmarks),
                sourceRank = index
            )
        }

    fun mapPose(
        landmarks: List<NormalizedLandmark>,
        frameWidth: Int,
        frameHeight: Int,
        targetSide: TargetSide
    ): Skeleton? {
        fun point(index: Int): Point2? {
            if (index !in landmarks.indices) return null
            val landmark = landmarks[index]
            val visibility = landmark.visibility().orElse(1.0f)
            if (visibility < minVisibility) return null
            return Point2(landmark.x() * frameWidth, landmark.y() * frameHeight)
        }

        val nose = point(Nose) ?: return null
        val leftShoulder = point(LeftShoulder)
        val rightShoulder = point(RightShoulder)
        val leftElbow = point(LeftElbow)
        val rightElbow = point(RightElbow)
        val leftWrist = point(LeftWrist)
        val rightWrist = point(RightWrist)
        val leftHip = point(LeftHip) ?: return null
        val rightHip = point(RightHip) ?: return null
        val leftKnee = point(LeftKnee) ?: return null
        val rightKnee = point(RightKnee) ?: return null
        val leftAnkle = point(LeftAnkle) ?: return null
        val rightAnkle = point(RightAnkle) ?: return null

        val frontWrist = if (targetSide == TargetSide.LEFT) rightWrist else leftWrist
        val frontElbow = if (targetSide == TargetSide.LEFT) rightElbow else leftElbow
        val frontShoulder = if (targetSide == TargetSide.LEFT) rightShoulder else leftShoulder
        val backWrist = if (targetSide == TargetSide.LEFT) leftWrist else rightWrist
        val frontAnkle = if (targetSide == TargetSide.LEFT) rightAnkle else leftAnkle

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
        return skeleton
    }

    private fun centerX(skeleton: Skeleton): Float {
        val anchors = listOfNotNull(
            skeleton["left_hip"],
            skeleton["right_hip"],
            skeleton["left_shoulder"],
            skeleton["right_shoulder"],
            skeleton["left_ankle"],
            skeleton["right_ankle"]
        )
        return if (anchors.isEmpty()) 0f else anchors.map { it.x }.average().toFloat()
    }

    private fun poseConfidence(landmarks: List<NormalizedLandmark>): Float {
        val values = RequiredLandmarks.mapNotNull { index ->
            landmarks.getOrNull(index)?.visibility()?.orElse(1.0f)
        }
        return if (values.isEmpty()) 1f else values.average().toFloat()
    }

    companion object {
        private const val Nose = 0
        private const val LeftShoulder = 11
        private const val RightShoulder = 12
        private const val LeftElbow = 13
        private const val RightElbow = 14
        private const val LeftWrist = 15
        private const val RightWrist = 16
        private const val LeftHip = 23
        private const val RightHip = 24
        private const val LeftKnee = 25
        private const val RightKnee = 26
        private const val LeftAnkle = 27
        private const val RightAnkle = 28
        private val RequiredLandmarks = listOf(
            Nose,
            LeftShoulder,
            RightShoulder,
            LeftHip,
            RightHip,
            LeftKnee,
            RightKnee,
            LeftAnkle,
            RightAnkle
        )
    }
}
