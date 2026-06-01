package com.aifencingcoach.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PracticeReportBuilderTest {
    @Test
    fun ranksActionsAndCuesForPostPracticeReview() {
        val report = buildPracticeReport(
            trainingMode = TrainingMode.FOOTWORK,
            poseBackend = PoseBackendKind.YOLO,
            targetSide = TargetSide.RIGHT,
            elapsedSeconds = 60,
            activeFrames = 900,
            fps = 30,
            inferenceCount = 12,
            actionCounts = mapOf("SF" to 5, "SB" to 3, "Idle" to 4),
            cueCounts = mapOf(
                "guard_dropped" to CueCountItem(
                    errorKey = "guard_dropped",
                    label = "Guard dropped",
                    message = "Guard up.",
                    count = 3
                ),
                "stance_too_high" to CueCountItem(
                    errorKey = "stance_too_high",
                    label = "Stance too high",
                    message = "Stay lower.",
                    count = 5
                )
            ),
            cueTimeline = listOf(
                CueHistoryItem(10, "guard_dropped", "Guard dropped", "Guard up.", "primary"),
                CueHistoryItem(20, "stance_too_high", "Stance too high", "Stay lower.", "primary")
            ),
            generatedAtFrame = 1200
        )

        assertEquals(30, report.activeSeconds)
        assertEquals(50, report.activePercent)
        assertEquals("SF", report.topAction)
        assertEquals("stance_too_high", report.topCues.first().errorKey)
        assertEquals("Stay lower.", report.primaryTakeaway)
        assertEquals(2, report.actionCounts.size)
        assertTrue(report.cueTimeline.first().frameIndex > report.cueTimeline.last().frameIndex)
    }
}
