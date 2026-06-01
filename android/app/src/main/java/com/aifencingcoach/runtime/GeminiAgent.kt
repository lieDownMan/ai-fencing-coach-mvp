package com.aifencingcoach.runtime

import com.aifencingcoach.BuildConfig
import com.google.ai.client.generativeai.GenerativeModel
import com.google.ai.client.generativeai.type.content
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

data class PlaybookEntry(
    val label: String,
    val shortCue: String,
    val weight: Float,
    val diagnosis: String = "",
    val practice: String = ""
)

enum class SummarySource {
    GEMINI,
    PLAYBOOK,
    DISABLED,
    FAILED
}

data class CoachingSummaryResult(
    val text: String,
    val source: SummarySource,
    val errorMessage: String? = null
)

class GeminiAgent(private val playbookRepository: PlaybookRepository) {
    companion object {
        private const val DEFAULT_MODEL_NAME = "gemini-2.5-flash"
    }

    private val apiKey = BuildConfig.GEMINI_API_KEY
    private val modelName = BuildConfig.GEMINI_MODEL.ifBlank { DEFAULT_MODEL_NAME }
    val isEnabled: Boolean = apiKey.isNotBlank() && modelName.isNotBlank()

    private val generativeModel by lazy {
        GenerativeModel(
            modelName = modelName,
            apiKey = apiKey
        )
    }

    fun playbookEntry(errorKey: String): PlaybookEntry? = playbookRepository.getEntry(errorKey)

    fun allPlaybookEntries(): Map<String, PlaybookEntry> = playbookRepository.getAllEntries()

    /**
     * Generate a structured coaching summary.
     */
    suspend fun generateSummary(
        trainingMode: String,
        targetSide: String,
        actionCounts: List<ActionCountItem>,
        cuesFired: List<CueHistoryItem>,
        userSettingsName: String,
        preferGemini: Boolean = true
    ): String = generateSummaryResult(
        trainingMode = trainingMode,
        targetSide = targetSide,
        actionCounts = actionCounts,
        cuesFired = cuesFired,
        userSettingsName = userSettingsName,
        preferGemini = preferGemini
    ).text

    suspend fun generateSummaryResult(
        trainingMode: String,
        targetSide: String,
        actionCounts: List<ActionCountItem>,
        cuesFired: List<CueHistoryItem>,
        userSettingsName: String,
        preferGemini: Boolean = true
    ): CoachingSummaryResult = withContext(Dispatchers.IO) {
        val totalActions = actionCounts.sumOf { it.count }
        val aggregatedErrors = aggregateErrors(cuesFired)
        val fallback = generateRuleBasedSummary(trainingMode, totalActions, aggregatedErrors)

        if (!preferGemini || !isEnabled) {
            return@withContext CoachingSummaryResult(
                text = fallback,
                source = if (preferGemini) SummarySource.DISABLED else SummarySource.PLAYBOOK
            )
        }

        val actionSummary = actionCounts
            .sortedWith(compareByDescending<ActionCountItem> { it.count }.thenBy { it.action })
            .map { "- ${it.action}: ${it.count} times (${it.percent}%)" }
            .joinToString("\n")

        val playbookBlock = formatPlaybookBlock(aggregatedErrors)
        val prompt = buildSessionSummaryPrompt(
            userName = userSettingsName,
            trainingMode = trainingMode,
            targetSide = targetSide,
            totalActions = totalActions,
            actionSummary = actionSummary,
            playbookBlock = playbookBlock,
            detectedProblemCount = aggregatedErrors.sumOf { it.count }
        )
        try {
            val response = generativeModel.generateContent(content { text(prompt) })
            val text = response.text?.trim()
            if (text.isNullOrBlank()) {
                CoachingSummaryResult(
                    text = fallback,
                    source = SummarySource.FAILED,
                    errorMessage = "Gemini returned an empty summary."
                )
            } else {
                CoachingSummaryResult(text = text, source = SummarySource.GEMINI)
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            CoachingSummaryResult(
                text = fallback,
                source = SummarySource.FAILED,
                errorMessage = e.message
            )
        }
    }

    /**
     * Generate an improvement analysis paragraph based on recent errors.
     */
    suspend fun generateImprovementAnalysis(
        userName: String,
        recapFocus: String,
        recentErrorsText: String,
        fallback: String,
        preferGemini: Boolean = true
    ): String = withContext(Dispatchers.IO) {
        if (!preferGemini || !isEnabled) return@withContext fallback

        val prompt = """You are an elite fencing coach reviewing recent training history. Use only the supplied session counts and playbook details; do not invent causes, injuries, tactics, or unseen technique.

[USER]
$userName

[RECAP FOCUS]
$recapFocus

[RECENT ERROR DATA]
$recentErrorsText

[OUTPUT RULES]
- Reply in Traditional Chinese.
- Write 2-3 concise sentences.
- Analyze the timeline of sessions (listed oldest to newest) to identify key trends.
- Focus on major mistakes, major improvements, or major declines across all errors. You do not need to focus on a single mistake.
- Do not define trends by percentage only; also define them by the absolute number of times (frequency).
- Focus your analysis context specifically on the timeframe or selection described in [RECAP FOCUS].
- End with one concrete practice recommendation from the playbook when present.
- Keep it under 250 Chinese characters if possible.
"""
        try {
            val response = generativeModel.generateContent(content { text(prompt) })
            response.text?.trim()?.ifBlank { null } ?: fallback
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            fallback
        }
    }

    private fun aggregateErrors(cues: List<CueHistoryItem>): List<AggregatedError> {
        val counts = cues.groupBy { it.errorKey }.mapValues { it.value.size }
        return counts.map { (key, count) ->
            val entry = playbookRepository.getEntry(key)
            AggregatedError(
                key = key,
                count = count,
                errorName = entry?.label ?: key,
                diagnosis = entry?.diagnosis ?: "",
                shortCue = entry?.shortCue ?: "",
                practice = entry?.practice ?: ""
            )
        }.sortedByDescending { it.count }
    }

    private fun generateRuleBasedSummary(
        trainingMode: String,
        totalActions: Long,
        errors: List<AggregatedError>
    ): String {
        val lines = mutableListOf("本次 $trainingMode 共辨識到 $totalActions 個動作。")
        if (errors.isEmpty()) {
            lines.add("未偵測到姿勢問題，表現良好！繼續保持！")
            return lines.joinToString("\n")
        }
        lines.add("偵測到的問題與頻率：")
        for (item in errors) {
            val detail = StringBuilder("- ${item.errorName}：${item.count} 次\n")
            if (item.diagnosis.isNotBlank()) detail.append("  診斷：${item.diagnosis}\n")
            if (item.practice.isNotBlank()) detail.append("  教練建議：${item.practice}\n")
            lines.add(detail.toString().trimEnd())
        }
        return lines.joinToString("\n")
    }

    private fun buildSessionSummaryPrompt(
        userName: String,
        trainingMode: String,
        targetSide: String,
        totalActions: Long,
        actionSummary: String,
        playbookBlock: String,
        detectedProblemCount: Int
    ): String {
        val modeFocus = when (trainingMode) {
            "Footwork" -> "balance, center-of-mass control, stance height, and step width"
            "Target Practice" -> "hand-before-foot timing, full extension, lunge depth, and knee safety"
            "Free Bouting" -> "guard discipline, setup quality, and stable movement under pressure"
            else -> "the detected playbook problems and the session action mix"
        }
        val actions = actionSummary.ifBlank { "- No classified actions." }
        return """You are an elite fencing coach. Create a post-session coaching summary from objective AI vision data.

[NON-NEGOTIABLE RULES]
- Use only the evidence in OBJECTIVE_DATA and PLAYBOOK_CONTEXT.
- Do not invent mistakes, timecodes, injuries, tactics, opponent behavior, or psychological advice.
- If detected_problem_count is 0, say no playbook-defined posture problems were detected and give only a light next-step cue for the training mode.
- If problems exist, mention every listed problem exactly once with its frequency, diagnosis, and practice recommendation.
- Reply in Traditional Chinese. Keep the whole answer under 160 Chinese words.
- Address the student directly as "$userName".
- Prefer compact paragraphs over long bullet lists.

[SESSION_CONTEXT]
training_mode: $trainingMode
target_side: $targetSide
mode_focus: $modeFocus

[OBJECTIVE_DATA]
total_actions: $totalActions
detected_problem_count: $detectedProblemCount
actions:
$actions

[PLAYBOOK_CONTEXT]
$playbookBlock

[RESPONSE SHAPE]
1. One sentence summarizing practice volume and action mix.
2. One compact sentence or bullet per detected problem: 問題｜次數｜診斷｜教練建議.
3. Final priority sentence naming the most frequent issue and the next drill.
"""
    }

    private fun formatPlaybookBlock(errors: List<AggregatedError>): String {
        if (errors.isEmpty()) return "No posture problems were detected."
        return errors.joinToString("\n") { item ->
            listOf(
                "- error_key: ${item.key}",
                "  frequency: ${item.count}",
                "  problem: ${item.errorName}",
                "  diagnosis: ${item.diagnosis.ifBlank { "No playbook diagnosis available." }}",
                "  short_cue: ${item.shortCue.ifBlank { "No playbook cue available." }}",
                "  practice: ${item.practice.ifBlank { "No playbook practice available." }}"
            ).joinToString("\n")
        }
    }

    private data class AggregatedError(
        val key: String,
        val count: Int,
        val errorName: String,
        val diagnosis: String,
        val shortCue: String,
        val practice: String
    )
}
