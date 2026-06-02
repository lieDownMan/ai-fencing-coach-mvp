package com.aifencingcoach.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GeminiAgentTest {
    @Test
    fun modelCandidatesTryPrimaryBeforeFallbacks() {
        val candidates = geminiModelCandidates("gemini-2.5-flash")

        assertEquals("gemini-2.5-flash", candidates.first())
        assertTrue(candidates.contains(DEFAULT_GEMINI_MODEL_NAME))
        assertTrue(candidates.contains("gemini-2.0-flash-lite"))
    }

    @Test
    fun modelCandidatesDoNotDuplicateFallbackModel() {
        val candidates = geminiModelCandidates(DEFAULT_GEMINI_MODEL_NAME)

        assertEquals(DEFAULT_GEMINI_MODEL_NAME, candidates.first())
        assertEquals(candidates.distinct(), candidates)
    }

    @Test
    fun openAiModelCandidatesTryPrimaryBeforeFallbacks() {
        val candidates = openAiModelCandidates("gpt-5.4-mini")

        assertEquals("gpt-5.4-mini", candidates.first())
        assertTrue(candidates.contains(DEFAULT_OPENAI_MODEL_NAME))
        assertEquals(candidates.distinct(), candidates)
    }

    @Test
    fun parsesResponsesApiOutputTextShortcut() {
        val response = """{"output_text":"Ready."}"""

        assertEquals("Ready.", openAiOutputTextFromResponse(response))
    }

    @Test
    fun parsesResponsesApiMessageContent() {
        val response = """
            {
              "output": [
                {
                  "type": "message",
                  "content": [
                    {"type": "output_text", "text": "First line."},
                    {"type": "output_text", "text": "Second line."}
                  ]
                }
              ]
            }
        """.trimIndent()

        assertEquals("First line.\nSecond line.", openAiOutputTextFromResponse(response))
    }

    @Test
    fun mapsLanguageToPlaybookAsset() {
        assertEquals("coach_playbook.json", playbookAssetNameForLanguage("zh"))
        assertEquals("coach_playbook_en.json", playbookAssetNameForLanguage("en"))
        assertEquals("coach_playbook_en.json", playbookAssetNameForLanguage("English(en)"))
    }

    @Test
    fun formatsNetworkAndQuotaErrorsForUi() {
        assertEquals("No internet connection.", formatLlmErrorMessage("network error: timeout"))
        assertEquals("API quota or rate limit reached.", formatLlmErrorMessage("HTTP 429 quota exceeded"))
    }
}
