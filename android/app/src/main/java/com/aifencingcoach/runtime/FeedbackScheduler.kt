package com.aifencingcoach.runtime

class FeedbackScheduler(
    private var trainingMode: TrainingMode = TrainingMode.FREE_BOUTING,
    private val playbookEntries: Map<String, PlaybookEntry> = emptyMap()
) {
    constructor(
        playbookRepo: PlaybookRepository,
        trainingMode: TrainingMode = TrainingMode.FREE_BOUTING
    ) : this(trainingMode, playbookRepo.getAllEntries())

    private var lastVoiceAt: Double? = null
    private val lastSpokenByKey = mutableMapOf<String, Double>()

    fun configure(trainingMode: TrainingMode) {
        this.trainingMode = trainingMode
        reset()
    }

    fun reset() {
        lastVoiceAt = null
        lastSpokenByKey.clear()
    }

    fun update(activeErrorKeys: Iterable<String>, nowSeconds: Double): FeedbackDecision {
        val activeKeys = activeErrorKeys
            .distinct()
            .filter { isAvailableForMode(it, trainingMode) }
            .sortedWith(
                compareByDescending<String> { priorityFor(it) }
                    .thenBy { it }
            )
        if (activeKeys.isEmpty()) {
            return FeedbackDecision(null, emptyList())
        }

        val visualCues = activeKeys.take(3).mapIndexed { index, errorKey ->
            cueFor(errorKey, priority = if (index == 0) "primary" else "secondary")
        }

        val voiceCue = visualCues.firstOrNull { cue ->
            val lastGlobal = lastVoiceAt
            val lastForKey = lastSpokenByKey[cue.errorKey]
            (lastGlobal == null || nowSeconds - lastGlobal >= GlobalVoiceCooldownSeconds) &&
                (lastForKey == null || nowSeconds - lastForKey >= VoiceCooldownSeconds)
        }
        if (voiceCue != null) {
            lastVoiceAt = nowSeconds
            lastSpokenByKey[voiceCue.errorKey] = nowSeconds
        }

        return FeedbackDecision(voiceCue = voiceCue, visualCues = visualCues)
    }

    private fun cueFor(errorKey: String, priority: String): FeedbackCue {
        val entry = playbookEntries[errorKey]
        val label = entry?.label ?: kErrorLabels[errorKey] ?: errorKey
        val shortCue = entry?.shortCue ?: kErrorVoice[errorKey] ?: label
        return FeedbackCue(
            errorKey = errorKey,
            label = label,
            message = shortCue,
            priority = priority,
            score = priorityFor(errorKey),
            triggered = true,
            shortCue = shortCue,
            diagnosis = entry?.diagnosis.orEmpty(),
            practice = entry?.practice.orEmpty()
        )
    }

    private fun priorityFor(errorKey: String): Float {
        return playbookEntries[errorKey]?.weight
            ?: kErrorWeights[errorKey]
            ?: 5f
    }

    companion object {
        private const val GlobalVoiceCooldownSeconds = 1.2
        private const val VoiceCooldownSeconds = 4.0

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

        val kErrorWeights = mapOf(
            "lunge_overextension" to 9.5f,
            "guard_dropped" to 9.7f,
            "bounce_excessive" to 6.5f,
            "foot_before_hand" to 5.0f,
            "over_parrying" to 5.0f,
            "stance_too_high" to 10.0f,
            "incomplete_arm_extension" to 9.0f,
            "wide_step" to 4.0f,
            "narrow_step" to 9.0f,
            "center_of_mass_in_front" to 6.0f,
            "center_of_mass_leaning_backward" to 6.0f
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
