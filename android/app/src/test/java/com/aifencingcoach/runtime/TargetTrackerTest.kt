package com.aifencingcoach.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TargetTrackerTest {
    @Test
    fun predictsShortMissingGapFromLastMotion() {
        val tracker = TargetTracker(TargetSide.LEFT)

        tracker.process(
            listOf(detection(centerX = 100f), detection(centerX = 800f)),
            frameIndex = 0
        )
        tracker.process(
            listOf(detection(centerX = 110f), detection(centerX = 800f)),
            frameIndex = 1
        )

        val result = tracker.process(emptyList(), frameIndex = 2)

        assertEquals("interpolating", result.status.lockState)
        assertTrue(result.status.targetInterpolated)
        assertEquals(120f, result.targetSkeleton?.get("nose")?.x ?: 0f, 0.001f)
    }

    private fun detection(centerX: Float): PoseDetection {
        val skeleton = mapOf(
            "nose" to Point2(centerX, 20f),
            "front_wrist" to Point2(centerX + 20f, 70f),
            "front_elbow" to Point2(centerX + 15f, 65f),
            "front_shoulder" to Point2(centerX + 10f, 60f),
            "front_ankle" to Point2(centerX + 12f, 190f),
            "left_shoulder" to Point2(centerX - 10f, 60f),
            "right_shoulder" to Point2(centerX + 10f, 60f),
            "left_hip" to Point2(centerX - 8f, 110f),
            "right_hip" to Point2(centerX + 8f, 110f),
            "left_knee" to Point2(centerX - 10f, 150f),
            "right_knee" to Point2(centerX + 10f, 150f),
            "left_ankle" to Point2(centerX - 12f, 190f),
            "right_ankle" to Point2(centerX + 12f, 190f)
        )
        return PoseDetection(
            skeleton = skeleton,
            bbox = BoundingBox(centerX - 40f, 0f, centerX + 40f, 210f),
            confidence = 1f
        )
    }
}
