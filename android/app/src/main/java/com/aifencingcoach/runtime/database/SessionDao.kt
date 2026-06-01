package com.aifencingcoach.runtime.database

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import androidx.room.Transaction

@Dao
abstract class SessionDao {
    @Insert
    abstract suspend fun insertSession(session: SessionEntity): Long

    @Insert
    abstract suspend fun insertActionCounts(segments: List<ActionCountEntity>)

    @Insert
    abstract suspend fun insertCueHistory(cues: List<CueHistoryEntity>)

    @Query("SELECT * FROM sessions ORDER BY timestamp DESC")
    abstract suspend fun getAllSessions(): List<SessionEntity>

    @Query("SELECT DISTINCT userName FROM sessions ORDER BY userName ASC")
    abstract suspend fun getDistinctUsers(): List<String>

    @Query("SELECT * FROM sessions WHERE userName = :userName ORDER BY timestamp DESC")
    abstract suspend fun getSessionsByUser(userName: String): List<SessionEntity>

    @Query("SELECT * FROM sessions WHERE id = :sessionId LIMIT 1")
    abstract suspend fun getSessionById(sessionId: Long): SessionEntity?

    @Query("SELECT * FROM action_counts WHERE sessionId = :sessionId")
    abstract suspend fun getActionCountsForSession(sessionId: Long): List<ActionCountEntity>

    @Query("SELECT * FROM cue_history WHERE sessionId = :sessionId ORDER BY timestamp ASC")
    abstract suspend fun getCueHistoryForSession(sessionId: Long): List<CueHistoryEntity>

    @Query("UPDATE sessions SET geminiSummary = :summary WHERE id = :sessionId")
    abstract suspend fun updateSessionSummary(sessionId: Long, summary: String)

    @Query("UPDATE sessions SET exportedVideoPath = :videoPath WHERE id = :sessionId")
    abstract suspend fun updateSessionVideoPath(sessionId: Long, videoPath: String)
    
    @Query("DELETE FROM sessions WHERE id = :sessionId")
    abstract suspend fun deleteSession(sessionId: Long)

    @Transaction
    open suspend fun getFullSession(sessionId: Long): FullSessionData? {
        val session = getSessionById(sessionId) ?: return null
        val counts = getActionCountsForSession(sessionId)
        val cues = getCueHistoryForSession(sessionId)
        return FullSessionData(session, counts, cues)
    }
}

data class FullSessionData(
    val session: SessionEntity,
    val actionCounts: List<ActionCountEntity>,
    val cues: List<CueHistoryEntity>
)
