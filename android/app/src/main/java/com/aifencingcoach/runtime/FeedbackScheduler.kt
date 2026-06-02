package com.aifencingcoach.runtime

import kotlin.math.max
import kotlin.math.min

private data class FeedbackState(
    val errorKey: String,
    val baseWeight: Float,
    var skippedCount: Int = 0,
    var activeCount: Int = 0,
    var spokenCount: Int = 0,
    var firstPendingAt: Double? = null,
    var lastSeenAt: Double? = null,
    var lastSpokenAt: Double? = null
)

class FeedbackScheduler(
    private var trainingMode: TrainingMode = TrainingMode.FREE_BOUTING,
    private val playbookEntries: Map<String, PlaybookEntry> = emptyMap(),
    private var focusErrors: Set<String> = emptySet(),
    private var muteErrors: Set<String> = emptySet(),
    private var onlyErrors: Set<String> = emptySet()
) {
    constructor(
        playbookRepo: PlaybookRepository,
        trainingMode: TrainingMode = TrainingMode.FREE_BOUTING,
        focusErrors: Set<String> = emptySet(),
        muteErrors: Set<String> = emptySet(),
        onlyErrors: Set<String> = emptySet()
    ) : this(trainingMode, playbookRepo.getAllEntries(), focusErrors, muteErrors, onlyErrors)

    private var lastVoiceAt: Double? = null
    private val states = mutableMapOf<String, FeedbackState>()

    fun configure(
        trainingMode: TrainingMode,
        focusErrors: Set<String> = this.focusErrors,
        muteErrors: Set<String> = this.muteErrors,
        onlyErrors: Set<String> = this.onlyErrors
    ) {
        this.trainingMode = trainingMode
        this.focusErrors = focusErrors
        this.muteErrors = muteErrors
        this.onlyErrors = onlyErrors
        reset()
    }

    fun reset() {
        lastVoiceAt = null
        states.clear()
    }

    fun update(activeErrorKeys: Iterable<String>, nowSeconds: Double): FeedbackDecision {
        val activeSet = activeErrorKeys
            .asSequence()
            .filter { allows(it) }
            .toSet()

        for (errorKey in activeSet) {
            val state = stateFor(errorKey)
            if (!isPending(state, nowSeconds)) {
                state.firstPendingAt = nowSeconds
                state.skippedCount = 0
                state.activeCount = 0
            }
            state.lastSeenAt = nowSeconds
            state.activeCount += 1
        }

        for (state in states.values) {
            if (state.errorKey !in activeSet && !isPending(state, nowSeconds)) {
                state.activeCount = 0
                state.skippedCount = 0
                state.firstPendingAt = null
            }
        }

        val pendingStates = states.values
            .filter { isPending(it, nowSeconds) }
            .sortedWith(
                compareByDescending<FeedbackState> { score(it) }
                    .thenByDescending { it.baseWeight }
                    .thenByDescending { it.activeCount }
                    .thenBy { it.errorKey }
            )

        if (pendingStates.isEmpty()) return FeedbackDecision(null, emptyList())

        val voiceState = selectVoiceState(pendingStates, nowSeconds)
        val visualCues = pendingStates.take(VisualTopN).mapIndexed { index, state ->
            cueFor(
                errorKey = state.errorKey,
                priority = if (index == 0) "primary" else "secondary",
                dynamicScore = score(state),
                triggered = state.errorKey in activeSet
            )
        }

        val voiceCue = voiceState?.let { state ->
            state.lastSpokenAt = nowSeconds
            state.spokenCount += 1
            state.skippedCount = 0
            lastVoiceAt = nowSeconds
            for (other in pendingStates) {
                if (other.errorKey != state.errorKey) other.skippedCount += 1
            }
            cueFor(
                errorKey = state.errorKey,
                priority = "primary",
                dynamicScore = score(state),
                triggered = state.errorKey in activeSet
            )
        }

        return FeedbackDecision(voiceCue = voiceCue, visualCues = visualCues)
    }

    private fun allows(errorKey: String): Boolean {
        if (errorKey in muteErrors) return false
        if (onlyErrors.isNotEmpty() && errorKey !in onlyErrors) return false
        return isAvailableForMode(errorKey, trainingMode)
    }

    private fun stateFor(errorKey: String): FeedbackState =
        states.getOrPut(errorKey) {
            FeedbackState(errorKey = errorKey, baseWeight = priorityFor(errorKey))
        }

    private fun isPending(state: FeedbackState, nowSeconds: Double): Boolean {
        val lastSeen = state.lastSeenAt ?: return false
        return nowSeconds - lastSeen <= PendingTtlSeconds
    }

    private fun score(state: FeedbackState): Float {
        val novelty = if (state.spokenCount == 0) NoveltyBonus else 0f
        val repeat = RepeatPenalty * min(state.spokenCount, 3)
        val persistence = PersistenceFactor * min(state.activeCount, 8)
        val focus = if (state.errorKey in focusErrors) FocusBoost else 0f
        return state.baseWeight +
            AgingFactor * state.skippedCount +
            persistence +
            novelty +
            focus -
            repeat
    }

    private fun cooldownRemaining(state: FeedbackState, nowSeconds: Double): Double {
        val lastSpoken = state.lastSpokenAt ?: return 0.0
        return max(0.0, VoiceCooldownSeconds - (nowSeconds - lastSpoken))
    }

    private fun globalCooldownRemaining(nowSeconds: Double): Double {
        val lastVoice = lastVoiceAt ?: return 0.0
        return max(0.0, GlobalVoiceCooldownSeconds - (nowSeconds - lastVoice))
    }

    private fun selectVoiceState(
        rankedStates: List<FeedbackState>,
        nowSeconds: Double
    ): FeedbackState? {
        if (globalCooldownRemaining(nowSeconds) > 0.0) return null
        return rankedStates.firstOrNull { state ->
            state.activeCount >= MinActiveCount && cooldownRemaining(state, nowSeconds) <= 0.0
        }
    }

    private fun cueFor(
        errorKey: String,
        priority: String,
        dynamicScore: Float,
        triggered: Boolean
    ): FeedbackCue {
        val entry = playbookEntries[errorKey]
        val label = entry?.label ?: kErrorLabels[errorKey] ?: errorKey
        val shortCue = entry?.shortCue ?: kErrorVoice[errorKey] ?: label
        return FeedbackCue(
            errorKey = errorKey,
            label = label,
            message = shortCue,
            priority = priority,
            score = dynamicScore,
            triggered = triggered,
            shortCue = shortCue,
            diagnosis = entry?.diagnosis.orEmpty(),
            practice = entry?.practice.orEmpty()
        )
    }

    private fun priorityFor(errorKey: String): Float =
        playbookEntries[errorKey]?.weight
            ?: kErrorWeights[errorKey]
            ?: 5f

    companion object {
        private const val AgingFactor = 2.0f
        private const val PersistenceFactor = 0.25f
        private const val NoveltyBonus = 0.75f
        private const val RepeatPenalty = 1.0f
        private const val VoiceCooldownSeconds = 4.0
        private const val GlobalVoiceCooldownSeconds = 1.2
        private const val MinActiveCount = 1
        private const val VisualTopN = 3
        private const val PendingTtlSeconds = 5.0
        private const val FocusBoost = 4.0f

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
            "center_of_mass_leaning_backward" to "Center of Mass Backward",
            "hand_too_high" to "Hand Too High"
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
            "center_of_mass_leaning_backward" to "Do not lean back.",
            "hand_too_high" to "Lower the weapon hand."
        )

        val kErrorWeights = mapOf(
            "foot_before_hand" to 5.0f,
            "lunge_overextension" to 9.5f,
            "incomplete_arm_extension" to 9.0f,
            "guard_dropped" to 9.7f,
            "stance_too_high" to 10.0f,
            "bounce_excessive" to 6.5f,
            "center_of_mass_in_front" to 6.0f,
            "center_of_mass_leaning_backward" to 6.0f,
            "over_parrying" to 5.0f,
            "wide_step" to 4.0f,
            "narrow_step" to 9.0f,
            "hand_too_high" to 8.0f
        )

        val kErrorSupportedModes = mapOf(
            "foot_before_hand" to setOf(TrainingMode.TARGET_PRACTICE),
            "lunge_overextension" to TrainingMode.entries.toSet(),
            "incomplete_arm_extension" to setOf(TrainingMode.TARGET_PRACTICE),
            "guard_dropped" to TrainingMode.entries.toSet(),
            "stance_too_high" to TrainingMode.entries.toSet(),
            "bounce_excessive" to TrainingMode.entries.toSet(),
            "center_of_mass_in_front" to TrainingMode.entries.toSet(),
            "center_of_mass_leaning_backward" to TrainingMode.entries.toSet(),
            "over_parrying" to TrainingMode.entries.toSet(),
            "wide_step" to TrainingMode.entries.toSet(),
            "narrow_step" to TrainingMode.entries.toSet(),
            "hand_too_high" to TrainingMode.entries.toSet()
        )

        private fun isAvailableForMode(errorKey: String, mode: TrainingMode): Boolean =
            kErrorSupportedModes[errorKey]?.contains(mode) == true
    }
}
