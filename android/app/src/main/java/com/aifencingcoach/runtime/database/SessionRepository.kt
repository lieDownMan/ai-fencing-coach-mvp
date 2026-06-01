package com.aifencingcoach.runtime.database

import android.content.Context
import com.aifencingcoach.runtime.PlaybookRepository
import com.aifencingcoach.runtime.PracticeReport
import com.aifencingcoach.runtime.CueHistoryItem
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class SessionRepository(private val context: Context) {
    private val dao = AppDatabase.getDatabase(context).sessionDao()

    suspend fun savePracticeReport(
        report: PracticeReport,
        cuesFired: List<CueHistoryItem>, // Using the PracticeReport model directly
        llmSummary: String? = null,
        userName: String = "Fencer",
        source: String = "Realtime",
        playbookLanguage: String = PlaybookRepository.DEFAULT_LANGUAGE
    ): Long = withContext(Dispatchers.IO) {
        val playbookRepository = PlaybookRepository(context, playbookLanguage)
        val sessionEntity = SessionEntity(
            timestamp = System.currentTimeMillis(),
            trainingMode = report.trainingMode.label,
            targetSide = report.targetSide.label,
            totalActiveSeconds = report.activeSeconds.toFloat(),
            elapsedSeconds = report.elapsedSeconds,
            inferenceCount = report.inferenceCount,
            cueCount = report.cueCount,
            topAction = report.topAction,
            geminiSummary = llmSummary,
            exportedVideoPath = null,
            userName = userName,
            source = source
        )
        val sessionId = dao.insertSession(sessionEntity)

        val actionCountEntities = report.actionCounts.map {
            ActionCountEntity(
                sessionId = sessionId,
                actionName = it.action,
                count = it.count
            )
        }
        if (actionCountEntities.isNotEmpty()) {
            dao.insertActionCounts(actionCountEntities)
        }

        val cueEntities = cuesFired.map { cue ->
            val playbookEntry = playbookRepository.getEntry(cue.errorKey)
            CueHistoryEntity(
                sessionId = sessionId,
                timestamp = cue.frameIndex / 30f,
                errorKey = cue.errorKey,
                errorName = playbookEntry?.label ?: cue.label,
                practiceSuggestion = cue.practice.ifBlank {
                    playbookEntry?.practice ?: cue.message
                }
            )
        }
        if (cueEntities.isNotEmpty()) {
            dao.insertCueHistory(cueEntities)
        }

        sessionId
    }

    suspend fun updateSummary(sessionId: Long, summary: String) = withContext(Dispatchers.IO) {
        dao.updateSessionSummary(sessionId, summary)
    }

    suspend fun updateVideoPath(sessionId: Long, videoPath: String) = withContext(Dispatchers.IO) {
        dao.updateSessionVideoPath(sessionId, videoPath)
    }

    suspend fun getRecentSessions(): List<SessionEntity> = withContext(Dispatchers.IO) {
        dao.getAllSessions()
    }

    suspend fun getDistinctUsers(): List<String> = withContext(Dispatchers.IO) {
        dao.getDistinctUsers()
    }

    suspend fun getSessionsByUser(userName: String): List<SessionEntity> = withContext(Dispatchers.IO) {
        dao.getSessionsByUser(userName)
    }

    suspend fun getSessionDetails(sessionId: Long): FullSessionData? = withContext(Dispatchers.IO) {
        dao.getFullSession(sessionId)
    }
    
    suspend fun deleteSession(sessionId: Long) = withContext(Dispatchers.IO) {
        dao.deleteSession(sessionId)
    }
}
