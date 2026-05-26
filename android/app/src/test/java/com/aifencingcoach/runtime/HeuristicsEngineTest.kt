package com.aifencingcoach.runtime

import org.junit.Assert.assertTrue
import org.junit.Test

class HeuristicsEngineTest {
    @Test
    fun detectsStanceTooHighInFootworkWindow() {
        val skeletons = List(28) {
            mapOf(
                "nose" to Point2(100f, 20f),
                "front_wrist" to Point2(150f, 70f),
                "front_elbow" to Point2(140f, 65f),
                "front_shoulder" to Point2(130f, 60f),
                "front_ankle" to Point2(150f, 220f),
                "left_shoulder" to Point2(90f, 60f),
                "right_shoulder" to Point2(130f, 60f),
                "left_hip" to Point2(95f, 140f),
                "right_hip" to Point2(135f, 140f),
                "left_knee" to Point2(95f, 180f),
                "right_knee" to Point2(135f, 180f),
                "left_ankle" to Point2(95f, 220f),
                "right_ankle" to Point2(135f, 220f)
            )
        }

        val engine = HeuristicsEngine(TargetSide.LEFT, TrainingMode.FOOTWORK)
        assertTrue("stance_too_high" in engine.evaluate("SF", skeletons))
    }
}
