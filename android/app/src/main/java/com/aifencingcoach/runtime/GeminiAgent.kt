package com.aifencingcoach.runtime

import com.aifencingcoach.BuildConfig
import com.google.ai.client.generativeai.GenerativeModel
import com.google.ai.client.generativeai.type.content
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class GeminiAgent {
    // If the API key is not set, we will fail gracefully.
    private val apiKey = BuildConfig.GEMINI_API_KEY
    val isEnabled: Boolean = apiKey.isNotBlank()

    private val generativeModel by lazy {
        GenerativeModel(
            modelName = "gemini-1.5-flash",
            apiKey = apiKey
        )
    }

    suspend fun generateSummary(
        trainingMode: String,
        targetSide: String,
        actionCounts: List<ActionCountItem>,
        cuesFired: List<CueHistoryItem>,
        userSettingsName: String
    ): String = withContext(Dispatchers.IO) {
        if (!isEnabled) {
            return@withContext "*(No Gemini API Key found. Offline fallback summary)*\n" +
                    "Training Mode: $trainingMode\n" +
                    "Actions tracked: ${actionCounts.sumOf { it.count }}\n" +
                    "Feedback cues: ${cuesFired.size}"
        }

        val actionSummary = actionCounts
            .map { "${it.action}: ${it.count} times" }
            .joinToString("\n")

        val cueSummary = cuesFired.groupBy { it.label }
            .map { "- ${it.key}: ${it.value.size} times" }
            .joinToString("\n")

        val prompt = """
            You are an expert, professional fencing coach. The user just completed a training session.
            Provide a brief, encouraging, and highly technical summary of their session in Traditional Chinese.
            Do not exceed 3 paragraphs. Use markdown formatting.

            ## Session Details
            User: $userSettingsName
            Mode: $trainingMode
            Targeting: $targetSide

            ## Actions Performed
            $actionSummary

            ## Detected Mistakes (Frequency)
            $cueSummary
        """.trimIndent()

        try {
            val response = generativeModel.generateContent(
                content {
                    text(prompt)
                }
            )
            response.text ?: "無法生成總結 (API 回傳空值)。"
        } catch (e: Exception) {
            "**Gemini API Error:** ${e.localizedMessage}"
        }
    }
}
