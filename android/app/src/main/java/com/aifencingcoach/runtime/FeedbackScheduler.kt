package com.aifencingcoach.runtime

import android.content.Context
import org.json.JSONObject

data class PlaybookEntry(
    val label: String,
    val message: String,
    val weight: Float
)

private data class FeedbackState(
    val errorKey: String,
    val label: String,
    val message: String,
    val baseWeight: Float,
    var skippedCount: Int = 0,
    var activeCount: Int = 0,
    var spokenCount: Int = 0,
    var firstPendingAt: Double? = null,
    var lastSeenAt: Double? = null,
    var lastSpokenAt: Double? = null
)

class FeedbackScheduler(
    context: Context? = null,
    private var trainingMode: TrainingMode = TrainingMode.FREE_BOUTING,
    private val voiceCooldownSeconds: Double = 4.0,
    private val globalVoiceCooldownSeconds: Double = 1.2,
    private val pendingTtlSeconds: Double = 5.0,
    playbookEntries: Map<String, PlaybookEntry>? = null
) {
    private val playbook = playbookEntries ?: if (context != null) loadPlaybook(context) else emptyMap()
    private val states = linkedMapOf<String, FeedbackState>()
    private var lastVoiceAt: Double? = null

    fun configure(trainingMode: TrainingMode) {
        this.trainingMode = trainingMode
        states.entries.removeIf { !isAvailableForMode(it.key, trainingMode) }
        lastVoiceAt = null
    }

    fun reset() {
        states.clear()
        lastVoiceAt = null
    }

    fun update(activeErrorKeys: Iterable<String>, nowSeconds: Double): FeedbackDecision {
        val activeKeys = activeErrorKeys.distinct().filter { isAvailableForMode(it, trainingMode) }
        val activeSet = activeKeys.toSet()

        for (key in activeKeys) {
            val state = stateFor(key)
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

        val pending = states.values.filter { isPending(it, nowSeconds) }
        val scores = pending.associateWith { score(it) }
        val ranked = pending.sortedWith(
            compareByDescending<FeedbackState> { scores[it] ?: 0f }
                .thenByDescending { it.baseWeight }
                .thenByDescending { it.activeCount }
                .thenBy { it.errorKey }
        )

        val voiceState = selectVoice(ranked, scores, nowSeconds)
        val visual = ranked.take(3).mapIndexed { index, state ->
            makeCue(
                state = state,
                score = scores[state] ?: 0f,
                priority = if (index == 0) "primary" else "secondary",
                triggered = state.errorKey in activeSet
            )
        }

        val voiceCue = voiceState?.let { state ->
            state.lastSpokenAt = nowSeconds
            state.spokenCount += 1
            state.skippedCount = 0
            lastVoiceAt = nowSeconds
            for (other in pending) {
                if (other.errorKey != state.errorKey) other.skippedCount += 1
            }
            makeCue(state, scores[state] ?: 0f, "primary", true)
        }

        return FeedbackDecision(voiceCue = voiceCue, visualCues = visual)
    }

    private fun stateFor(errorKey: String): FeedbackState {
        states[errorKey]?.let { return it }
        val entry = playbook[errorKey] ?: PlaybookEntry(errorKey, errorKey, DefaultWeights[errorKey] ?: 5f)
        val state = FeedbackState(
            errorKey = errorKey,
            label = entry.label,
            message = entry.message,
            baseWeight = entry.weight
        )
        states[errorKey] = state
        return state
    }

    private fun isPending(state: FeedbackState, now: Double): Boolean {
        val lastSeen = state.lastSeenAt ?: return false
        return now - lastSeen <= pendingTtlSeconds
    }

    private fun score(state: FeedbackState): Float {
        val novelty = if (state.spokenCount == 0) 0.75f else 0f
        val repeatPenalty = minOf(state.spokenCount, 3) * 1f
        val persistence = minOf(state.activeCount, 8) * 0.25f
        return state.baseWeight + 2f * state.skippedCount + persistence + novelty - repeatPenalty
    }

    private fun selectVoice(
        ranked: List<FeedbackState>,
        scores: Map<FeedbackState, Float>,
        now: Double
    ): FeedbackState? {
        val globalLast = lastVoiceAt
        if (globalLast != null && now - globalLast < globalVoiceCooldownSeconds) return null
        return ranked.filter {
            it.activeCount >= 1 && cooldownRemaining(it, now) <= 0.0
        }.maxWithOrNull(
            compareBy<FeedbackState> { scores[it] ?: 0f }
                .thenBy { it.baseWeight }
                .thenBy { it.activeCount }
                .thenBy { it.errorKey }
        )
    }

    private fun cooldownRemaining(state: FeedbackState, now: Double): Double {
        val last = state.lastSpokenAt ?: return 0.0
        return maxOf(0.0, voiceCooldownSeconds - (now - last))
    }

    private fun makeCue(
        state: FeedbackState,
        score: Float,
        priority: String,
        triggered: Boolean
    ) = FeedbackCue(
        errorKey = state.errorKey,
        label = state.label,
        message = state.message,
        priority = priority,
        score = score,
        triggered = triggered
    )

    companion object {
        private val DefaultWeights = mapOf(
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
            "narrow_step" to 9.0f
        )

        private val Availability = mapOf(
            TrainingMode.FOOTWORK to setOf(
                "guard_dropped",
                "stance_too_high",
                "bounce_excessive",
                "center_of_mass_in_front",
                "center_of_mass_leaning_backward",
                "over_parrying",
                "wide_step",
                "narrow_step"
            ),
            TrainingMode.TARGET_PRACTICE to setOf(
                "foot_before_hand",
                "lunge_overextension",
                "incomplete_arm_extension",
                "guard_dropped",
                "stance_too_high",
                "bounce_excessive",
                "center_of_mass_in_front",
                "center_of_mass_leaning_backward",
                "over_parrying",
                "wide_step",
                "narrow_step"
            ),
            TrainingMode.FREE_BOUTING to setOf(
                "guard_dropped",
                "stance_too_high",
                "bounce_excessive",
                "center_of_mass_in_front",
                "center_of_mass_leaning_backward",
                "over_parrying",
                "wide_step",
                "narrow_step"
            )
        )

        private fun isAvailableForMode(errorKey: String, mode: TrainingMode): Boolean =
            errorKey in (Availability[mode] ?: DefaultWeights.keys)

        private fun loadPlaybook(context: Context): Map<String, PlaybookEntry> {
            return runCatching {
                val json = context.assets.open("coach_playbook.json").bufferedReader().use { it.readText() }
                val root = JSONObject(json)
                root.keys().asSequence().associateWith { key ->
                    val obj = root.getJSONObject(key)
                    PlaybookEntry(
                        label = obj.optString("error_name", key),
                        message = obj.optString("short_cue", key),
                        weight = DefaultWeights[key] ?: 5f
                    )
                }
            }.getOrDefault(emptyMap())
        }
    }
}
