package com.aifencingcoach.runtime

import org.junit.Assert.assertTrue
import org.junit.Test

class HeuristicsEngineTest {
    @Test
    fun detectsStanceTooHighInFootworkWindow() {
        val skeletons = List(28) { baseSkeleton() }

        val engine = HeuristicsEngine(TargetSide.LEFT, TrainingMode.FOOTWORK)
        assertTrue("stance_too_high" in engine.evaluate("SF", skeletons))
    }

    @Test
    fun detectsLungeOverextensionAcrossModes() {
        val foldedKnee = baseSkeleton() + mapOf(
            "right_hip" to Point2(100f, 100f),
            "right_knee" to Point2(130f, 150f),
            "right_ankle" to Point2(160f, 100f),
            "front_ankle" to Point2(160f, 100f)
        )
        val engine = HeuristicsEngine(TargetSide.LEFT, TrainingMode.FREE_BOUTING)

        assertTrue("lunge_overextension" in engine.evaluate("SF", List(3) { foldedKnee }))
    }

    @Test
    fun detectsHandTooHigh() {
        val handHigh = baseSkeleton() + mapOf(
            "front_elbow" to Point2(130f, 100f),
            "front_wrist" to Point2(132f, 20f)
        )
        val engine = HeuristicsEngine(TargetSide.LEFT, TrainingMode.FOOTWORK)

        assertTrue("hand_too_high" in engine.evaluate("SF", List(3) { handHigh }))
    }

    @Test
    fun detectsVerticalOverParrySweep() {
        val skeletons = List(6) { index ->
            baseSkeleton() + mapOf(
                "front_wrist" to Point2(150f, 70f + index * 40f)
            )
        }
        val engine = HeuristicsEngine(TargetSide.LEFT, TrainingMode.FREE_BOUTING)

        assertTrue("over_parrying" in engine.evaluate("SB", skeletons))
    }

    private fun baseSkeleton(): Skeleton {
        val rightShoulder = Point2(130f, 60f)
        val rightElbow = Point2(140f, 65f)
        val rightWrist = Point2(150f, 70f)
        val rightAnkle = Point2(135f, 220f)
        return mapOf(
            "nose" to Point2(100f, 20f),
            "front_wrist" to rightWrist,
            "front_elbow" to rightElbow,
            "front_shoulder" to rightShoulder,
            "front_ankle" to rightAnkle,
            "left_shoulder" to Point2(90f, 60f),
            "right_shoulder" to rightShoulder,
            "left_hip" to Point2(95f, 140f),
            "right_hip" to Point2(135f, 140f),
            "left_knee" to Point2(95f, 180f),
            "right_knee" to Point2(135f, 180f),
            "left_ankle" to Point2(95f, 220f),
            "right_ankle" to rightAnkle
        )
    }
}
