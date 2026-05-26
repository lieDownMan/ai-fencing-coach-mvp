package com.aifencingcoach.runtime

import org.junit.Assert.assertEquals
import org.junit.Test

class SpatialNormalizerTest {
    @Test
    fun normalizesAroundFirstNoseAndFrontAnkleScale() {
        val skeleton = mapOf(
            "nose" to Point2(10f, 10f),
            "front_ankle" to Point2(10f, 30f),
            "front_wrist" to Point2(30f, 10f),
            "front_elbow" to Point2(20f, 10f),
            "front_shoulder" to Point2(10f, 10f),
            "left_hip" to Point2(8f, 20f),
            "right_hip" to Point2(12f, 20f),
            "left_knee" to Point2(8f, 25f),
            "right_knee" to Point2(12f, 25f),
            "left_ankle" to Point2(8f, 30f),
            "right_ankle" to Point2(12f, 30f)
        )

        val normalizer = SpatialNormalizer()
        normalizer.fit(skeleton)
        val normalized = normalizer.normalize(skeleton)
        val wrist = normalized.getValue("front_wrist")

        assertEquals(1.0f, wrist.x, 0.0001f)
        assertEquals(0.0f, wrist.y, 0.0001f)
        assertEquals(18, normalizer.modelArray(normalized).size)
    }
}
