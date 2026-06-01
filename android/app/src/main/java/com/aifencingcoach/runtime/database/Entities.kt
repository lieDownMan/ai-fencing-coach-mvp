package com.aifencingcoach.runtime.database

import androidx.room.Entity
import androidx.room.PrimaryKey
import androidx.room.ForeignKey
import androidx.room.Index

@Entity(tableName = "sessions")
data class SessionEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val timestamp: Long,
    val trainingMode: String,
    val targetSide: String,
    val totalActiveSeconds: Float,
    val elapsedSeconds: Long = 0,
    val inferenceCount: Long = 0,
    val cueCount: Long = 0,
    val topAction: String = "Idle",
    val geminiSummary: String?, // Null until API returns
    val exportedVideoPath: String?, // Null if no video was exported
    val userName: String = "Fencer",
    val source: String = "Realtime" // "Realtime" or "Postgame"
)

@Entity(
    tableName = "action_counts",
    foreignKeys = [
        ForeignKey(
            entity = SessionEntity::class,
            parentColumns = ["id"],
            childColumns = ["sessionId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index("sessionId")]
)
data class ActionCountEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val sessionId: Long,
    val actionName: String,
    val count: Long
)

@Entity(
    tableName = "cue_history",
    foreignKeys = [
        ForeignKey(
            entity = SessionEntity::class,
            parentColumns = ["id"],
            childColumns = ["sessionId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index("sessionId")]
)
data class CueHistoryEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val sessionId: Long,
    val timestamp: Float, // Relative to session start
    val errorKey: String,
    val errorName: String, // Resolved from playbook
    val practiceSuggestion: String // So history is readable even if playbook changes
)
