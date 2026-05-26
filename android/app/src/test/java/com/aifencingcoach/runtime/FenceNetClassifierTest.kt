package com.aifencingcoach.runtime

import org.junit.Assert.assertEquals
import org.junit.Test

class FenceNetClassifierTest {
    @Test
    fun classOrderAndSoftmaxAreStable() {
        assertEquals(listOf("R", "IS", "WW", "JS", "SF", "SB"), FenceNetClassifier.ClassNames)
        val scores = FenceNetClassifier.softmax(floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f))
        assertEquals(6, scores.size)
        assertEquals(1.0f, scores.sum(), 0.0001f)
        assertEquals(5, scores.indices.maxBy { scores[it] })
    }
}
