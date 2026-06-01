package com.aifencingcoach.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ActivityGatekeeperTest {
    @Test
    fun usesKneeHysteresisAndExposesReasons() {
        val gatekeeper = ActivityGatekeeper(fps = 1)

        repeat(5) { frame ->
            gatekeeper.update(
                targetSkeleton = gateSkeleton(knee = "bent", shift = frame.toFloat() * 2f),
                opponentSkeleton = null,
                frameWidth = 640,
                targetSide = TargetSide.LEFT
            )
        }

        assertEquals(ActivityGatekeeper.StateActive, gatekeeper.state)
        assertTrue(gatekeeper.lastReasons["en_garde"] as Boolean)

        repeat(2) { frame ->
            gatekeeper.update(
                targetSkeleton = gateSkeleton(knee = "standing", shift = 20f + frame),
                opponentSkeleton = null,
                frameWidth = 640,
                targetSide = TargetSide.LEFT
            )
        }

        assertEquals(ActivityGatekeeper.StateIdle, gatekeeper.state)
        assertFalse(gatekeeper.update(null, null, 640, TargetSide.LEFT))
        assertTrue(gatekeeper.lastReasons["has_target"] == false)
    }

    private fun gateSkeleton(knee: String, shift: Float): Skeleton {
        val hip = Point2(110f + shift, 100f)
        val kneePoint = Point2(110f + shift, 140f)
        val ankle = if (knee == "bent") Point2(150f + shift, 140f) else Point2(110f + shift, 180f)
        return mapOf(
            "nose" to Point2(100f + shift, 40f),
            "left_shoulder" to Point2(80f + shift, 70f),
            "right_shoulder" to Point2(120f + shift, 70f),
            "front_shoulder" to Point2(120f + shift, 70f),
            "front_elbow" to Point2(130f + shift, 85f),
            "front_wrist" to Point2(140f + shift, 95f),
            "front_ankle" to ankle,
            "left_hip" to Point2(85f + shift, 100f),
            "right_hip" to hip,
            "left_knee" to Point2(85f + shift, 140f),
            "right_knee" to kneePoint,
            "left_ankle" to Point2(85f + shift, 180f),
            "right_ankle" to ankle
        )
    }
}
