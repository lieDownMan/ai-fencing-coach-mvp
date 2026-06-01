package com.aifencingcoach.runtime

import android.content.Context
import org.json.JSONObject

class PlaybookRepository(context: Context) {
    private val playbook: Map<String, PlaybookEntry> = loadPlaybook(context)

    fun getEntry(errorKey: String): PlaybookEntry? {
        return playbook[errorKey]
    }

    fun getAllEntries(): Map<String, PlaybookEntry> {
        return playbook
    }

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
                    weight = obj.optDouble(
                        "weight",
                        FeedbackScheduler.kErrorWeights[key]?.toDouble() ?: 5.0
                    ).toFloat()
                )
            }
        }.getOrDefault(emptyMap())
    }
}
