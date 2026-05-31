package com.aifencingcoach.runtime

import android.content.Context
import com.aifencingcoach.BuildConfig
import com.google.ai.client.generativeai.GenerativeModel
import com.google.ai.client.generativeai.type.content
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

data class PlaybookEntry(
    val label: String,
    val shortCue: String,
    val weight: Float,
    val diagnosis: String = "",
    val practice: String = ""
)

class GeminiAgent(context: Context? = null) {
    private val apiKey = BuildConfig.GEMINI_API_KEY
    val isEnabled: Boolean = apiKey.isNotBlank()

    private val playbook: Map<String, PlaybookEntry> = if (context != null) loadPlaybook(context) else emptyMap()

    private val generativeModel by lazy {
        GenerativeModel(
            modelName = "gemini-1.5-flash",
            apiKey = apiKey
        )
    }

    /**
     * Generate a structured coaching summary.
     * If Gemini API is available, sends a rich prompt with playbook context.
     * Otherwise, returns a structured rule-based summary using coach_playbook.json data.
     */
    suspend fun generateSummary(
        trainingMode: String,
        targetSide: String,
        actionCounts: List<ActionCountItem>,
        cuesFired: List<CueHistoryItem>,
        userSettingsName: String
    ): String = withContext(Dispatchers.IO) {
        val totalActions = actionCounts.sumOf { it.count }
        val aggregatedErrors = aggregateErrors(cuesFired)

        if (!isEnabled) {
            return@withContext generateRuleBasedSummary(trainingMode, totalActions, aggregatedErrors)
        }

        // Build rich prompt with playbook context
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
3. Use the playbook diagnosis, short cue, and practice recommendation when explaining each problem.
4. If multiple problems were detected, prioritize the most frequent one at the end.
5. Tone: Direct, professional, and encouraging.
6. Constraint: Strictly under 160 words. Do NOT list timecodes. Please reply in Traditional Chinese.
"""
        try {
            val response = generativeModel.generateContent(
                content {
                    text(prompt)
                }
            )
            response.text ?: generateRuleBasedSummary(trainingMode, totalActions, aggregatedErrors)
        } catch (e: Exception) {
            val fallback = generateRuleBasedSummary(trainingMode, totalActions, aggregatedErrors)
            "**Gemini API Error:** ${e.localizedMessage}\n\n$fallback"
        }
    }

    /**
     * Aggregate cue history items into error counts with playbook data.
     */
    private fun aggregateErrors(cues: List<CueHistoryItem>): List<AggregatedError> {
        val counts = cues.groupBy { it.errorKey }.mapValues { it.value.size }
        return counts.map { (key, count) ->
            val entry = playbook[key]
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

    /**
     * Generate a structured Chinese coaching summary without any LLM.
     * Matches the logic from main branch's llm_agent.py: _generate_rule_based_summary()
     */
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
            val detail = StringBuilder("- ${item.errorName}：${item.count} 次")
            if (item.diagnosis.isNotBlank()) {
                detail.append("。${item.diagnosis}")
            }
            if (item.shortCue.isNotBlank()) {
                detail.append(" 教練提示：${item.shortCue}")
            }
            if (item.practice.isNotBlank()) {
                detail.append(" 練習建議：${item.practice}")
            }
            lines.add(detail.toString())
        }
        return lines.joinToString("\n")
    }

    /**
     * Format playbook errors into a structured block for the LLM prompt.
     */
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

    companion object {
        private fun loadPlaybook(context: Context): Map<String, PlaybookEntry> {
            return runCatching {
                val json = context.assets.open("coach_playbook.json").bufferedReader().use { it.readText() }
                val root = JSONObject(json)
                root.keys().asSequence().associateWith { key ->
                    val obj = root.getJSONObject(key)
                    PlaybookEntry(
                        label = obj.optString("error_name", key),
                        shortCue = obj.optString("short_cue", key),
                        diagnosis = obj.optString("diagnosis", ""),
                        practice = obj.optString("practice", ""),
                        weight = 5f
                    )
                }
            }.getOrDefault(emptyMap())
        }
    }
}
