package com.aifencingcoach.runtime

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import com.aifencingcoach.runtime.database.SessionRepository

class AnalysisManager(
    private val context: Context,
    private val sessionRepository: SessionRepository,
    private val geminiAgent: GeminiAgent
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val _isAnalyzing = MutableStateFlow(false)
    val isAnalyzing: StateFlow<Boolean> = _isAnalyzing

    private val _analysisProgress = MutableStateFlow(0f)
    val analysisProgress: StateFlow<Float> = _analysisProgress

    private val _analysisStatus = MutableStateFlow("Idle")
    val analysisStatus: StateFlow<String> = _analysisStatus

    private val _frameStates = MutableStateFlow<Map<Long, CoachFrameState>?>(null)
    val frameStates: StateFlow<Map<Long, CoachFrameState>?> = _frameStates

    private val _lastReport = MutableStateFlow<PracticeReport?>(null)
    val lastReport: StateFlow<PracticeReport?> = _lastReport
    
    private val _lastSessionId = MutableStateFlow<Long?>(null)
    val lastSessionId: StateFlow<Long?> = _lastSessionId

    fun startAnalysis(
        uri: Uri,
        targetSide: TargetSide,
        trainingMode: TrainingMode,
        poseBackend: PoseBackendKind,
        userSettingsName: String,
        useGeminiSummary: Boolean
    ) {
        if (_isAnalyzing.value) return

        _isAnalyzing.value = true
        _analysisProgress.value = 0f
        _analysisStatus.value = "Starting Analysis..."
        _frameStates.value = null
        _lastReport.value = null
        _lastSessionId.value = null

        scope.launch {
            try {
                val analyzer = PostgameVideoAnalyzer(context)
                val (report, states) = analyzer.analyze(
                    uri = uri,
                    targetSide = targetSide,
                    trainingMode = trainingMode,
                    poseBackend = poseBackend,
                    onProgress = { progress ->
                        _analysisProgress.value = progress.fraction
                        _analysisStatus.value = progress.status
                    }
                )

                _frameStates.value = states
                _lastReport.value = report

                _analysisStatus.value = "Saving to database..."

                var geminiSummaryText = ""
                _analysisStatus.value = "Generating Coach Summary..."
                geminiSummaryText = geminiAgent.generateSummary(
                    trainingMode = trainingMode.label,
                    targetSide = targetSide.label,
                    actionCounts = report.actionCounts,
                    cuesFired = report.cueTimeline,
                    userSettingsName = userSettingsName
                )

                val sessionId = sessionRepository.savePracticeReport(
                    report = report,
                    cuesFired = report.cueTimeline,
                    llmSummary = geminiSummaryText.ifEmpty { "No summary available." }
                )
                
                _lastSessionId.value = sessionId
                _analysisStatus.value = "Analysis Complete!"
            } catch (e: Exception) {
                e.printStackTrace()
                _analysisStatus.value = "Error: ${e.message}"
            } finally {
                _isAnalyzing.value = false
            }
        }
    }

    fun reset() {
        if (_isAnalyzing.value) return
        _analysisProgress.value = 0f
        _analysisStatus.value = "Idle"
        _frameStates.value = null
        _lastReport.value = null
        _lastSessionId.value = null
    }
}
