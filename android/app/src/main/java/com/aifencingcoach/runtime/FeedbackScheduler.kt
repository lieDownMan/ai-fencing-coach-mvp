package com.aifencingcoach.runtime

import android.content.Context

class FeedbackScheduler(
    context: Context? = null,
    private var trainingMode: TrainingMode = TrainingMode.FREE_BOUTING
) {
    private var lastVoiceAt: Double = 0.0

    fun configure(trainingMode: TrainingMode) {
        this.trainingMode = trainingMode
        lastVoiceAt = 0.0
    }

    fun reset() {
        lastVoiceAt = 0.0
    }

    fun update(activeErrorKeys: Iterable<String>, nowSeconds: Double): FeedbackDecision {
        val activeKeys = activeErrorKeys.distinct().filter { isAvailableForMode(it, trainingMode) }
        if (activeKeys.isEmpty()) {
            return FeedbackDecision(null, emptyList())
        }

        val firstError = activeKeys.first()
        val voiceMessage = kErrorVoice[firstError] ?: firstError
        val visualMessage = kErrorLabels[firstError] ?: firstError

        val cue = FeedbackCue(
            errorKey = firstError,
            label = visualMessage,
            message = voiceMessage,
            priority = "primary",
            score = 1f,
            triggered = true
        )

        var voiceCue: FeedbackCue? = null
        if (nowSeconds - lastVoiceAt >= 4.0) {
            voiceCue = cue
            lastVoiceAt = nowSeconds
        }

        return FeedbackDecision(voiceCue = voiceCue, visualCues = listOf(cue))
    }

    companion object {
        val kErrorLabels = mapOf(
            "lunge_overextension" to "Lunge Overextension",
            "guard_dropped" to "Guard Dropped",
            "bounce_excessive" to "Excessive Bounce",
            "foot_before_hand" to "Foot Before Hand",
            "over_parrying" to "Over-Parrying",
            "stance_too_high" to "Stance Too High",
            "incomplete_arm_extension" to "Incomplete Arm Extension",
            "wide_step" to "Wide Step",
            "narrow_step" to "Narrow Step",
            "center_of_mass_in_front" to "Center of Mass Forward",
            "center_of_mass_leaning_backward" to "Center of Mass Backward"
        )

        val kErrorVoice = mapOf(
            "lunge_overextension" to "Do not collapse the front knee.",
            "guard_dropped" to "Guard up.",
            "bounce_excessive" to "Stay level.",
            "foot_before_hand" to "Hand first.",
            "over_parrying" to "Keep the parry compact.",
            "stance_too_high" to "Stay lower.",
            "incomplete_arm_extension" to "Extend the arm.",
            "wide_step" to "Shorten the step.",
            "narrow_step" to "Widen the base.",
            "center_of_mass_in_front" to "Keep weight centered.",
            "center_of_mass_leaning_backward" to "Do not lean back."
        )

        val kErrorSupportedModes = mapOf(
            "foot_before_hand" to setOf(TrainingMode.TARGET_PRACTICE),
            "lunge_overextension" to setOf(TrainingMode.TARGET_PRACTICE),
            "incomplete_arm_extension" to setOf(TrainingMode.TARGET_PRACTICE),
            "guard_dropped" to setOf(TrainingMode.FOOTWORK, TrainingMode.TARGET_PRACTICE, TrainingMode.FREE_BOUTING),
            "stance_too_high" to setOf(TrainingMode.FOOTWORK, TrainingMode.TARGET_PRACTICE, TrainingMode.FREE_BOUTING),
            "bounce_excessive" to setOf(TrainingMode.FOOTWORK, TrainingMode.TARGET_PRACTICE, TrainingMode.FREE_BOUTING),
            "center_of_mass_in_front" to setOf(TrainingMode.FOOTWORK, TrainingMode.TARGET_PRACTICE, TrainingMode.FREE_BOUTING),
            "center_of_mass_leaning_backward" to setOf(TrainingMode.FOOTWORK, TrainingMode.TARGET_PRACTICE, TrainingMode.FREE_BOUTING),
            "over_parrying" to setOf(TrainingMode.FOOTWORK, TrainingMode.TARGET_PRACTICE, TrainingMode.FREE_BOUTING),
            "wide_step" to setOf(TrainingMode.FOOTWORK, TrainingMode.TARGET_PRACTICE, TrainingMode.FREE_BOUTING),
            "narrow_step" to setOf(TrainingMode.FOOTWORK, TrainingMode.TARGET_PRACTICE, TrainingMode.FREE_BOUTING)
        )

        private fun isAvailableForMode(errorKey: String, mode: TrainingMode): Boolean =
            kErrorSupportedModes[errorKey]?.contains(mode) == true
    }
}
