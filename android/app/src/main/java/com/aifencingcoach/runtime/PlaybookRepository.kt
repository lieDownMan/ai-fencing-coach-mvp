package com.aifencingcoach.runtime

import android.content.Context
import org.json.JSONObject

class PlaybookRepository(
    context: Context,
    language: String = DEFAULT_LANGUAGE
) {
    private val assetName = playbookAssetNameForLanguage(language)
    private val playbook: Map<String, PlaybookEntry> = loadPlaybook(context)

    fun getEntry(errorKey: String): PlaybookEntry? {
        return playbook[errorKey]
    }

    fun getAllEntries(): Map<String, PlaybookEntry> {
        return playbook
    }

    private fun loadPlaybook(context: Context): Map<String, PlaybookEntry> {
        return runCatching {
            val json = context.assets.open(assetName).bufferedReader(Charsets.UTF_8).use { it.readText() }
            val root = JSONObject(json)
            root.keys().asSequence().associateWith { key ->
                val obj = root.getJSONObject(key)
                PlaybookEntry(
                    label = obj.optString("error_name", key),
                    shortCue = obj.optString("short_cue", key),
                    diagnosis = obj.optString("diagnosis", ""),
                    practice = obj.optString("practice", ""),
                    weight = obj.optDouble(
                        "weight",
                        FeedbackScheduler.kErrorWeights[key]?.toDouble() ?: 5.0
                    ).toFloat()
                )
            }
        }.getOrDefault(emptyMap())
    }

    companion object {
        const val DEFAULT_LANGUAGE = "zh"
        const val ENGLISH_LANGUAGE = "en"

        fun displayName(language: String?): String =
            when (normalizePlaybookLanguage(language)) {
                ENGLISH_LANGUAGE -> "English(en)"
                else -> "中文(zh)"
            }
    }
}

internal fun normalizePlaybookLanguage(language: String?): String =
    when (language?.trim()?.lowercase()) {
        PlaybookRepository.ENGLISH_LANGUAGE,
        "english",
        "english(en)" -> PlaybookRepository.ENGLISH_LANGUAGE
        else -> PlaybookRepository.DEFAULT_LANGUAGE
    }

internal fun playbookAssetNameForLanguage(language: String?): String =
    when (normalizePlaybookLanguage(language)) {
        PlaybookRepository.ENGLISH_LANGUAGE -> "coach_playbook_en.json"
        else -> "coach_playbook.json"
    }
