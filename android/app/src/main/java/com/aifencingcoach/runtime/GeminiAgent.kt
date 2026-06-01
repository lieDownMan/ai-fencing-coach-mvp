package com.aifencingcoach.runtime

import com.aifencingcoach.BuildConfig
import com.google.ai.client.generativeai.GenerativeModel
import com.google.ai.client.generativeai.type.content
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

data class PlaybookEntry(
    val label: String,
    val shortCue: String,
    val weight: Float,
    val diagnosis: String = "",
    val practice: String = ""
)

class GeminiAgent(private val playbookRepository: PlaybookRepository) {
    private val apiKey = BuildConfig.GEMINI_API_KEY
    private val modelName = BuildConfig.GEMINI_MODEL
    val isEnabled: Boolean = apiKey.isNotBlank()

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
    ): String = withContext(Dispatchers.IO) {
        val totalActions = actionCounts.sumOf { it.count }
        val aggregatedErrors = aggregateErrors(cuesFired)
        val fallback = generateRuleBasedSummary(trainingMode, totalActions, aggregatedErrors)

        if (!preferGemini || !isEnabled) {
            return@withContext fallback
        }

        val actionSummary = actionCounts
            .map { "${it.action}: ${it.count} times" }
            .joinToString("\n")

        val playbookBlock = formatPlaybookBlock(aggregatedErrors)

        val errorsSummary = if (aggregatedErrors.isEmpty()) {
            "0 Errors"
        } else {
            aggregatedErrors.joinToString(", ") { "${it.count}x ${it.errorName}" }
        }

        val prompt = """You are an elite, observant fencing coach. Your goal is to give a post-session summary based STRICTLY on objective biomechanical data extracted by our AI vision system. Do not invent errors or provide tactical advice that is not supported by the data.

[STUDENT PROFILE]
User: $userSettingsName

[SESSION CONTEXT]
Training Mode: $trainingMode
Target Side: $targetSide
* Context Guide for Coach:
  - If "Footwork", focus your advice on balance, center of mass stability, and stance width.
  - If "Target Practice", focus on kinetic chain (hand-before-foot), extension, and knee safety.
  - If "Free Bouting", focus on maintaining guard under pressure and action setup.

[OBJECTIVE ACTION STATS]
Total actions: $totalActions. Errors: $errorsSummary

## Actions Performed
$actionSummary

[COACH PLAYBOOK CONTEXT]
The following detected problems come from coach_playbook.json. Treat this as the source of truth for problem names, diagnoses, cue wording, practice recommendations, and frequency:
$playbookBlock

[INSTRUCTIONS]
Based on the stats and coach playbook context above, write a highly specific technical summary addressing the student directly.
1. Acknowledge the volume/type of actions they practiced.
2. List every detected problem and how many times it appeared.
3. For each problem, explicitly include the error_name, diagnosis, and **YOU MUST INCLUDE A BULLET POINT FOR 教練建議: [practice]**.
4. If multiple problems were detected, prioritize the most frequent one at the end.
5. Tone: Direct, professional, and encouraging.
6. Constraint: Strictly under 160 words. Do NOT list timecodes. Please reply in Traditional Chinese.
"""
        try {
            val response = generativeModel.generateContent(content { text(prompt) })
            response.text?.trim()?.ifBlank { null } ?: fallback
        } catch (e: Exception) {
            "Gemini 連線失敗，已改用離線摘要。\n\n$fallback"
        }
    }

    /**
     * Generate an improvement analysis paragraph based on recent errors.
     */
    suspend fun generateImprovementAnalysis(
        userName: String,
        recentErrorsText: String,
        fallback: String,
        preferGemini: Boolean = true
    ): String = withContext(Dispatchers.IO) {
        if (!preferGemini || !isEnabled) return@withContext fallback

        val prompt = """You are an elite fencing coach. Analyze the user's recent progress based on error frequencies over the last 5 sessions.
User: $userName
Recent Errors Data:
$recentErrorsText

INSTRUCTIONS:
1. Write a 2-3 sentence summary evaluating recent progress or consistency.
2. Use the per-session error counts and percentages to discuss whether the most important issue is improving, stable, or getting worse.
3. Mention the current focus area and the relevant practice recommendation when present.
4. Be encouraging but direct.
5. Reply in Traditional Chinese.
"""
        try {
            val response = generativeModel.generateContent(content { text(prompt) })
            response.text?.trim()?.ifBlank { null } ?: fallback
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
