package com.aifencingcoach.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class FeedbackSchedulerTest {
    @Test
    fun ranksAndRateLimitsVoiceCues() {
        val scheduler = FeedbackScheduler(
            trainingMode = TrainingMode.FREE_BOUTING,
            playbookEntries = mapOf(
                "stance_too_high" to PlaybookEntry("Stance too high", "Stay lower.", 10f),
                "guard_dropped" to PlaybookEntry("Guard dropped", "Guard up.", 9.7f)
            )
        )

        val first = scheduler.update(listOf("stance_too_high", "guard_dropped"), nowSeconds = 1.0)
        assertEquals("stance_too_high", first.voiceCue?.errorKey)
        assertEquals(2, first.visualCues.size)

        val second = scheduler.update(listOf("guard_dropped"), nowSeconds = 1.5)
        assertNull(second.voiceCue)

        val third = scheduler.update(listOf("guard_dropped"), nowSeconds = 2.3)
        assertEquals("guard_dropped", third.voiceCue?.errorKey)
    }
}
