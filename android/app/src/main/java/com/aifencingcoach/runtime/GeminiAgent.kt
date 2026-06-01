package com.aifencingcoach.runtime

import android.util.Log
import com.aifencingcoach.BuildConfig
import com.google.ai.client.generativeai.GenerativeModel
import com.google.ai.client.generativeai.type.content
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import org.json.JSONArray
import org.json.JSONObject

internal const val DEFAULT_GEMINI_MODEL_NAME = "gemini-flash-lite-latest"
internal const val DEFAULT_OPENAI_MODEL_NAME = "gpt-5.4-nano"

private const val LLM_TAG = "LlmCoachAgent"
private val FALLBACK_GEMINI_MODEL_NAMES = listOf(
    DEFAULT_GEMINI_MODEL_NAME,
    "gemini-2.0-flash-lite"
)
private val FALLBACK_OPENAI_MODEL_NAMES = listOf(
    DEFAULT_OPENAI_MODEL_NAME,
    "gpt-5.4-mini",
    "gpt-4.1-mini"
)

internal fun geminiModelCandidates(primaryModelName: String): List<String> =
    (listOf(primaryModelName) + FALLBACK_GEMINI_MODEL_NAMES)
        .map { it.trim() }
        .filter { it.isNotBlank() }
        .distinct()

internal fun openAiModelCandidates(primaryModelName: String): List<String> =
    (listOf(primaryModelName) + FALLBACK_OPENAI_MODEL_NAMES)
        .map { it.trim() }
        .filter { it.isNotBlank() }
        .distinct()

enum class LlmProviderKind(val label: String) {
    PLAYBOOK("Playbook"),
    GEMINI("Gemini"),
    OPENAI("OpenAI");

    companion object {
        fun fromLabel(label: String?): LlmProviderKind =
            entries.firstOrNull { it.label.equals(label, ignoreCase = true) } ?: GEMINI
    }
}

data class LlmProviderConfig(
    val provider: LlmProviderKind = LlmProviderKind.GEMINI,
    val apiKey: String = "",
    val modelName: String = "",
    val language: String = PlaybookRepository.DEFAULT_LANGUAGE,
    val useBundledGeminiKey: Boolean = true,
    val useBundledOpenAiKey: Boolean = true
)

data class PlaybookEntry(
    val label: String,
    val shortCue: String,
    val weight: Float,
    val diagnosis: String = "",
    val practice: String = ""
)

enum class SummarySource {
    GEMINI,
    OPENAI,
    PLAYBOOK,
    DISABLED,
    FAILED
}

data class CoachingSummaryResult(
    val text: String,
    val source: SummarySource,
    val errorMessage: String? = null
)

fun formatLlmErrorMessage(errorMessage: String?): String {
    val lower = errorMessage.orEmpty().lowercase()
    return when {
        "quota" in lower || "rate" in lower || "429" in lower || "too many requests" in lower ->
            "API quota or rate limit reached."
        "network error" in lower ||
            "timeout" in lower ||
            "unable to resolve host" in lower ||
            "unknown host" in lower ||
            "failed to connect" in lower ||
            "connectexception" in lower ||
            "socket" in lower ->
            "No internet connection."
        "api key" in lower || "permission" in lower || "unauthorized" in lower || "401" in lower || "403" in lower -> "API key rejected"
        "model" in lower || "not found" in lower || "404" in lower -> "Model unavailable."
        else -> "AI request failed; check internet connection, API key, or quota."
    }
}

class GeminiAgent(private val playbookRepository: PlaybookRepository) {
    private val _lastApiError = MutableStateFlow<String?>(null)
    val lastApiError: StateFlow<String?> = _lastApiError

    val isEnabled: Boolean
        get() = isEnabled(LlmProviderConfig())

    fun isEnabled(config: LlmProviderConfig): Boolean =
        config.provider != LlmProviderKind.PLAYBOOK &&
            effectiveApiKey(config).isNotBlank() &&
            effectiveModelName(config).isNotBlank()

    fun providerLabel(config: LlmProviderConfig): String = config.provider.label

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
        preferGemini: Boolean = true,
        llmConfig: LlmProviderConfig = LlmProviderConfig()
    ): String = generateSummaryResult(
        trainingMode = trainingMode,
        targetSide = targetSide,
        actionCounts = actionCounts,
        cuesFired = cuesFired,
        userSettingsName = userSettingsName,
        preferGemini = preferGemini,
        llmConfig = llmConfig
    ).text

    suspend fun generateSummaryResult(
        trainingMode: String,
        targetSide: String,
        actionCounts: List<ActionCountItem>,
        cuesFired: List<CueHistoryItem>,
        userSettingsName: String,
        preferGemini: Boolean = true,
        llmConfig: LlmProviderConfig = LlmProviderConfig()
    ): CoachingSummaryResult = withContext(Dispatchers.IO) {
        val totalActions = actionCounts.sumOf { it.count }
        val aggregatedErrors = aggregateErrors(cuesFired)
        val language = normalizePlaybookLanguage(llmConfig.language)
        val fallback = generateRuleBasedSummary(trainingMode, totalActions, aggregatedErrors, language)

        if (!preferGemini || !isEnabled(llmConfig)) {
            return@withContext CoachingSummaryResult(
                text = fallback,
                source = if (preferGemini) SummarySource.DISABLED else SummarySource.PLAYBOOK,
                errorMessage = if (preferGemini) "${providerLabel(llmConfig)} is not configured." else null
            )
        }

        val actionSummary = actionCounts
            .sortedWith(compareByDescending<ActionCountItem> { it.count }.thenBy { it.action })
            .map { "- ${it.action}: ${it.count} times (${it.percent}%)" }
            .joinToString("\n")

        val playbookBlock = formatPlaybookBlock(aggregatedErrors, language)
        val prompt = buildSessionSummaryPrompt(
            userName = userSettingsName,
            trainingMode = trainingMode,
            targetSide = targetSide,
            totalActions = totalActions,
            actionSummary = actionSummary,
            playbookBlock = playbookBlock,
            detectedProblemCount = aggregatedErrors.sumOf { it.count },
            language = language
        )
        val generation = generateTextWithFallback(prompt, llmConfig)
        if (generation.text.isNullOrBlank()) {
            CoachingSummaryResult(
                text = fallback,
                source = SummarySource.FAILED,
                errorMessage = generation.errorMessage ?: "Gemini returned an empty summary."
            )
        } else {
            CoachingSummaryResult(text = generation.text, source = generation.source)
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
        preferGemini: Boolean = true,
        llmConfig: LlmProviderConfig = LlmProviderConfig()
    ): String = withContext(Dispatchers.IO) {
        if (!preferGemini || !isEnabled(llmConfig)) return@withContext fallback
        val language = normalizePlaybookLanguage(llmConfig.language)
        val outputRules = if (language == PlaybookRepository.ENGLISH_LANGUAGE) {
            """
- Reply in English.
- Write 2-3 concise sentences.
- Analyze the timeline of sessions (listed oldest to newest) to identify key trends.
- Focus on major mistakes, major improvements, or major declines across all errors. You do not need to focus on a single mistake.
- Do not define trends by percentage only; also define them by the absolute number of times (frequency).
- Focus your analysis context specifically on the timeframe or selection described in [RECAP FOCUS].
- End with one concrete practice recommendation from the playbook when present.
- Keep it under 120 words if possible.
""".trimIndent()
        } else {
            """
- Reply in Traditional Chinese.
- Write 2-3 concise sentences.
- Analyze the timeline of sessions (listed oldest to newest) to identify key trends.
- Focus on major mistakes, major improvements, or major declines across all errors. You do not need to focus on a single mistake.
- Do not define trends by percentage only; also define them by the absolute number of times (frequency).
- Focus your analysis context specifically on the timeframe or selection described in [RECAP FOCUS].
- End with one concrete practice recommendation from the playbook when present.
- Keep it under 350 Chinese characters if possible.
""".trimIndent()
        }

        val prompt = """You are an elite fencing coach reviewing recent training history. Use only the supplied session counts and playbook details; do not invent causes, injuries, tactics, or unseen technique.

[USER]
$userName

[RECAP FOCUS]
$recapFocus

[RECENT ERROR DATA]
$recentErrorsText

[OUTPUT RULES]
$outputRules
"""
        generateTextWithFallback(prompt, llmConfig).text ?: fallback
    }

    private suspend fun generateTextWithFallback(
        prompt: String,
        config: LlmProviderConfig
    ): GeminiGeneration {
        return when (config.provider) {
            LlmProviderKind.GEMINI -> generateGeminiTextWithFallback(prompt, config)
            LlmProviderKind.OPENAI -> generateOpenAiTextWithFallback(prompt, config)
            LlmProviderKind.PLAYBOOK -> GeminiGeneration(
                text = null,
                source = SummarySource.PLAYBOOK,
                errorMessage = "Playbook provider selected."
            )
        }
    }

    private suspend fun generateGeminiTextWithFallback(
        prompt: String,
        config: LlmProviderConfig
    ): GeminiGeneration {
        val apiKey = effectiveApiKey(config)
        val failures = mutableListOf<String>()
        for (candidateModel in geminiModelCandidates(effectiveModelName(config))) {
            try {
                val response = GenerativeModel(
                    modelName = candidateModel,
                    apiKey = apiKey
                ).generateContent(content { text(prompt) })
                val text = response.text?.trim()
                if (!text.isNullOrBlank()) {
                    _lastApiError.value = null
                    return GeminiGeneration(text = text, source = SummarySource.GEMINI)
                }
                failures.add("$candidateModel: empty response")
                Log.w(LLM_TAG, "Gemini model $candidateModel returned an empty response.")
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                val summary = summarizeLlmException(e)
                failures.add("$candidateModel: $summary")
                Log.w(LLM_TAG, "Gemini model $candidateModel failed: $summary", e)
            }
        }
        val errMessage = failures.joinToString("; ")
        _lastApiError.value = formatLlmErrorMessage(errMessage)
        return GeminiGeneration(text = null, errorMessage = errMessage)
    }

    private fun generateOpenAiTextWithFallback(
        prompt: String,
        config: LlmProviderConfig
    ): GeminiGeneration {
        val apiKey = effectiveApiKey(config)
        val failures = mutableListOf<String>()
        for (candidateModel in openAiModelCandidates(effectiveModelName(config))) {
            try {
                val text = requestOpenAiText(apiKey, candidateModel, prompt)
                if (!text.isNullOrBlank()) {
                    _lastApiError.value = null
                    return GeminiGeneration(text = text, source = SummarySource.OPENAI)
                }
                failures.add("$candidateModel: empty response")
                Log.w(LLM_TAG, "OpenAI model $candidateModel returned an empty response.")
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                val summary = summarizeLlmException(e)
                failures.add("$candidateModel: $summary")
                Log.w(LLM_TAG, "OpenAI model $candidateModel failed: $summary", e)
            }
        }
        val errMessage = failures.joinToString("; ")
        _lastApiError.value = formatLlmErrorMessage(errMessage)
        return GeminiGeneration(text = null, errorMessage = errMessage)
    }

    private fun requestOpenAiText(apiKey: String, modelName: String, prompt: String): String? {
        val connection = (URL("https://api.openai.com/v1/responses").openConnection() as HttpURLConnection)
        connection.requestMethod = "POST"
        connection.connectTimeout = 30_000
        connection.readTimeout = 45_000
        connection.doOutput = true
        connection.setRequestProperty("Authorization", "Bearer $apiKey")
        connection.setRequestProperty("Content-Type", "application/json")

        val body = JSONObject()
            .put("model", modelName)
            .put("input", prompt)
            .put("max_output_tokens", 700)
            .toString()

        connection.outputStream.use { stream ->
            stream.write(body.toByteArray(Charsets.UTF_8))
        }

        val statusCode = connection.responseCode
        val responseBody = (if (statusCode in 200..299) {
            connection.inputStream
        } else {
            connection.errorStream
        })?.bufferedReader()?.use { it.readText() }.orEmpty()

        if (statusCode !in 200..299) {
            throw IOException("HTTP $statusCode: ${openAiErrorMessage(responseBody)}")
        }
        return openAiOutputTextFromResponse(responseBody)
    }

    private fun openAiErrorMessage(responseBody: String): String {
        return try {
            JSONObject(responseBody)
                .optJSONObject("error")
                ?.optString("message")
                ?.takeIf { it.isNotBlank() }
                ?: responseBody.take(180)
        } catch (_: Exception) {
            responseBody.take(180)
        }
    }

    private fun effectiveApiKey(config: LlmProviderConfig): String =
        when (config.provider) {
            LlmProviderKind.GEMINI -> config.apiKey.ifBlank {
                if (config.useBundledGeminiKey) BuildConfig.GEMINI_API_KEY else ""
            }
            LlmProviderKind.OPENAI -> config.apiKey.ifBlank {
                if (config.useBundledOpenAiKey) BuildConfig.OPENAI_API_KEY else ""
            }
            LlmProviderKind.PLAYBOOK -> ""
        }

    private fun effectiveModelName(config: LlmProviderConfig): String =
        when (config.provider) {
            LlmProviderKind.GEMINI -> config.modelName.ifBlank {
                BuildConfig.GEMINI_MODEL.ifBlank { DEFAULT_GEMINI_MODEL_NAME }
            }
            LlmProviderKind.OPENAI -> config.modelName.ifBlank {
                BuildConfig.OPENAI_MODEL.ifBlank { DEFAULT_OPENAI_MODEL_NAME }
            }
            LlmProviderKind.PLAYBOOK -> ""
        }

    private fun summarizeLlmException(e: Exception): String {
        val message = e.message.orEmpty()
        val lower = message.lowercase()
        return when {
            "429" in lower ||
                "too many requests" in lower ||
                "quota" in lower ||
                "rate" in lower ||
                "resource_exhausted" in lower -> "quota/rate limit"
            "401" in lower ||
                "403" in lower ||
                "invalid api key" in lower ||
                "api key" in lower ||
                "permission" in lower ||
                "unauthorized" in lower -> "API key rejected"
            "404" in lower ||
                "not found" in lower ||
                "model" in lower -> "model unavailable"
            "timeout" in lower ||
                "network" in lower ||
                "unable to resolve host" in lower ||
                "unknown host" in lower ||
                "failed to connect" in lower ||
                "connectexception" in lower ||
                "socket" in lower -> "network error"
            message.isNotBlank() -> message.take(120)
            else -> e::class.java.simpleName
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
        errors: List<AggregatedError>,
        language: String
    ): String {
        if (language == PlaybookRepository.ENGLISH_LANGUAGE) {
            val lines = mutableListOf("This $trainingMode session classified $totalActions actions.")
            if (errors.isEmpty()) {
                lines.add("No playbook-defined posture problems were detected. Keep the same stable form.")
                return lines.joinToString("\n")
            }
            lines.add("Detected playbook problems and frequency:")
            for (item in errors) {
                val detail = StringBuilder("- ${item.errorName}: ${item.count} times\n")
                if (item.diagnosis.isNotBlank()) detail.append("  Diagnosis: ${item.diagnosis}\n")
                if (item.practice.isNotBlank()) detail.append("  Coach suggestion: ${item.practice}\n")
                lines.add(detail.toString().trimEnd())
            }
            return lines.joinToString("\n")
        }

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
        detectedProblemCount: Int,
        language: String
    ): String {
        val modeFocus = when (trainingMode) {
            "Footwork" -> "balance, center-of-mass control, stance height, and step width"
            "Target Practice" -> "hand-before-foot timing, full extension, lunge depth, and knee safety"
            "Free Bouting" -> "guard discipline, setup quality, and stable movement under pressure"
            else -> "the detected playbook problems and the session action mix"
        }
        val actions = actionSummary.ifBlank { "- No classified actions." }
        val languageRule = if (language == PlaybookRepository.ENGLISH_LANGUAGE) {
            "- Reply in English. Keep the whole answer under 120 words."
        } else {
            "- Reply in Traditional Chinese. Keep the whole answer under 160 Chinese words."
        }
        val responseProblemShape = if (language == PlaybookRepository.ENGLISH_LANGUAGE) {
            "Problem | frequency | diagnosis | coaching suggestion."
        } else {
            "問題｜次數｜診斷｜教練建議."
        }
        return """You are an elite fencing coach. Create a post-session coaching summary from objective AI vision data.

[NON-NEGOTIABLE RULES]
- Use only the evidence in OBJECTIVE_DATA and PLAYBOOK_CONTEXT.
- Do not invent mistakes, timecodes, injuries, tactics, opponent behavior, or psychological advice.
- If detected_problem_count is 0, say no playbook-defined posture problems were detected and give only a light next-step cue for the training mode.
- If problems exist, mention every listed problem exactly once with its frequency, diagnosis, and practice recommendation.
$languageRule
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
2. One compact sentence or bullet per detected problem: $responseProblemShape
3. Final priority sentence naming the most frequent issue and the next drill.
"""
    }

    private fun formatPlaybookBlock(errors: List<AggregatedError>, language: String): String {
        if (errors.isEmpty()) {
            return if (language == PlaybookRepository.ENGLISH_LANGUAGE) {
                "No posture problems were detected."
            } else {
                "未偵測到姿勢問題。"
            }
        }
        val missingDiagnosis = if (language == PlaybookRepository.ENGLISH_LANGUAGE) {
            "No playbook diagnosis available."
        } else {
            "沒有可用的 playbook 診斷。"
        }
        val missingCue = if (language == PlaybookRepository.ENGLISH_LANGUAGE) {
            "No playbook cue available."
        } else {
            "沒有可用的 playbook 提示。"
        }
        val missingPractice = if (language == PlaybookRepository.ENGLISH_LANGUAGE) {
            "No playbook practice available."
        } else {
            "沒有可用的 playbook 練習。"
        }
        return errors.joinToString("\n") { item ->
            listOf(
                "- error_key: ${item.key}",
                "  frequency: ${item.count}",
                "  problem: ${item.errorName}",
                "  diagnosis: ${item.diagnosis.ifBlank { missingDiagnosis }}",
                "  short_cue: ${item.shortCue.ifBlank { missingCue }}",
                "  practice: ${item.practice.ifBlank { missingPractice }}"
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

    private data class GeminiGeneration(
        val text: String?,
        val source: SummarySource = SummarySource.FAILED,
        val errorMessage: String? = null
    )
}

internal fun openAiOutputTextFromResponse(responseBody: String): String? {
    val root = JSONObject(responseBody)
    root.optString("output_text").takeIf { it.isNotBlank() }?.let { return it.trim() }

    fun textFromContentArray(content: JSONArray?): String? {
        if (content == null) return null
        val pieces = buildList {
            for (index in 0 until content.length()) {
                val item = content.optJSONObject(index) ?: continue
                val text = item.optString("text").takeIf { it.isNotBlank() }
                if (text != null) add(text)
            }
        }
        return pieces.joinToString("\n").trim().ifBlank { null }
    }

    val output = root.optJSONArray("output") ?: return null
    val pieces = buildList {
        for (index in 0 until output.length()) {
            val item = output.optJSONObject(index) ?: continue
            textFromContentArray(item.optJSONArray("content"))?.let { add(it) }
        }
    }
    return pieces.joinToString("\n").trim().ifBlank { null }
}
