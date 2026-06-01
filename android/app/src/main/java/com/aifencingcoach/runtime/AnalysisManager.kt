package com.aifencingcoach.runtime

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import com.aifencingcoach.runtime.database.SessionRepository

data class AnalysisJob(
    val uri: Uri,
    val targetSide: TargetSide,
    val trainingMode: TrainingMode,
    val poseBackend: PoseBackendKind,
    val userSettingsName: String,
    val useGeminiSummary: Boolean,
    val llmConfig: LlmProviderConfig,
    val autoExport: Boolean
)

class AnalysisManager(
    private val context: Context,
    private val sessionRepository: SessionRepository,
    private val geminiAgent: GeminiAgent
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val jobChannel = Channel<AnalysisJob>(Channel.UNLIMITED)

    private val _queueSize = MutableStateFlow(0)
    val queueSize: StateFlow<Int> = _queueSize

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

    private val _lastSummary = MutableStateFlow<String?>(null)
    val lastSummary: StateFlow<String?> = _lastSummary

    private val _lastSummaryStatus = MutableStateFlow<String?>(null)
    val lastSummaryStatus: StateFlow<String?> = _lastSummaryStatus

    private val _lastSourceUri = MutableStateFlow<Uri?>(null)
    val lastSourceUri: StateFlow<Uri?> = _lastSourceUri

    init {
        scope.launch {
            for (job in jobChannel) {
                processJob(job)
                _queueSize.value = (_queueSize.value - 1).coerceAtLeast(0)
            }
        }
    }

    fun enqueueAnalysis(
        uris: List<Uri>,
        targetSide: TargetSide,
        trainingMode: TrainingMode,
        poseBackend: PoseBackendKind,
        userSettingsName: String,
        useGeminiSummary: Boolean,
        llmConfig: LlmProviderConfig,
        autoExport: Boolean
    ) {
        uris.forEach { uri ->
            jobChannel.trySend(
                AnalysisJob(
                    uri = uri,
                    targetSide = targetSide,
                    trainingMode = trainingMode,
                    poseBackend = poseBackend,
                    userSettingsName = userSettingsName,
                    useGeminiSummary = useGeminiSummary,
                    llmConfig = llmConfig,
                    autoExport = autoExport
                )
            )
            _queueSize.value = _queueSize.value + 1
        }
    }

    private suspend fun processJob(job: AnalysisJob) {
        _isAnalyzing.value = true
        _analysisProgress.value = 0f
        _analysisStatus.value = "Starting Analysis for ${job.uri.lastPathSegment}..."
        _frameStates.value = null
        _lastReport.value = null
        _lastSessionId.value = null
        _lastSummary.value = null
        _lastSummaryStatus.value = null
        _lastSourceUri.value = job.uri

        try {
            val analyzer = PostgameVideoAnalyzer(context)
            val (report, states) = analyzer.analyze(
                uri = job.uri,
                targetSide = job.targetSide,
                trainingMode = job.trainingMode,
                poseBackend = job.poseBackend,
                playbookLanguage = job.llmConfig.language,
                onProgress = { progress ->
                    _analysisProgress.value = progress.fraction
                    _analysisStatus.value = progress.status
                }
            )

            _frameStates.value = states
            _lastReport.value = report

            _analysisStatus.value = "Saving to database..."

            _analysisStatus.value = "Saving to database..."
            val fallbackResult = geminiAgent.generateSummaryResult(
                trainingMode = job.trainingMode.label,
                targetSide = job.targetSide.label,
                actionCounts = report.actionCounts,
                cuesFired = report.cueTimeline,
                userSettingsName = job.userSettingsName,
                preferGemini = false,
                llmConfig = job.llmConfig
            )
            val fallbackSummary = fallbackResult.text

            _lastSummary.value = fallbackSummary
            _lastSummaryStatus.value = if (!job.useGeminiSummary && job.llmConfig.provider != LlmProviderKind.PLAYBOOK) {
                "AI Summary is off. ${job.llmConfig.provider.label} is selected; showing playbook summary."
            } else {
                "Playbook summary ready."
            }

            val sessionId = sessionRepository.savePracticeReport(
                report = report,
                cuesFired = report.cueTimeline,
                llmSummary = fallbackSummary.ifEmpty { "No summary available." },
                userName = job.userSettingsName,
                source = "Postgame",
                playbookLanguage = job.llmConfig.language
            )

            _lastSessionId.value = sessionId

            if (job.useGeminiSummary) {
                val providerLabel = geminiAgent.providerLabel(job.llmConfig)
                _lastSummaryStatus.value = if (geminiAgent.isEnabled(job.llmConfig)) {
                    "Generating $providerLabel summary..."
                } else {
                    "AI unavailable for $providerLabel: ${geminiAgent.configurationError(job.llmConfig) ?: "$providerLabel is not configured."} Showing playbook summary."
                }
                scope.launch {
                    val geminiResult = geminiAgent.generateSummaryResult(
                        trainingMode = job.trainingMode.label,
                        targetSide = job.targetSide.label,
                        actionCounts = report.actionCounts,
                        cuesFired = report.cueTimeline,
                        userSettingsName = job.userSettingsName,
                        preferGemini = true,
                        llmConfig = job.llmConfig
                    )
                    sessionRepository.updateSummary(sessionId, geminiResult.text)
                    if (_lastSessionId.value == sessionId) {
                        _lastSummary.value = geminiResult.text
                        _lastSummaryStatus.value = summaryStatus(geminiResult)
                    }
                }
            }

            if (job.autoExport) {
                _analysisStatus.value = "Exporting Annotated Video..."
                _analysisProgress.value = 0f
                val videoAnnotator = VideoAnnotator(context)
                try {
                    videoAnnotator.exportVideo(
                        sourceUri = job.uri,
                        frameStates = states,
                        videoWidth = 720,
                        videoHeight = 1280
                    ).collect { progress ->
                        when (progress) {
                            is ExportProgress.Completed -> {
                                sessionRepository.updateVideoPath(sessionId, progress.filePath)
                            }
                            is ExportProgress.Error -> {
                                _analysisStatus.value = "Export Failed: ${progress.exception.message}"
                            }
                        }
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }

            _analysisStatus.value = "Completed ${job.uri.lastPathSegment}!"
        } catch (e: Exception) {
            e.printStackTrace()
            _analysisStatus.value = "Error: ${e.message}"
        } finally {
            _isAnalyzing.value = false
        }
    }

    fun reset() {
        if (_isAnalyzing.value) return
        _analysisProgress.value = 0f
        _analysisStatus.value = "Idle"
        _frameStates.value = null
        _lastReport.value = null
        _lastSessionId.value = null
        _lastSummary.value = null
        _lastSummaryStatus.value = null
        _lastSourceUri.value = null
    }

    private fun summaryStatus(result: CoachingSummaryResult): String =
        when (result.source) {
            SummarySource.GEMINI -> "Gemini summary ready."
            SummarySource.OPENAI -> "OpenAI summary ready."
            SummarySource.PLAYBOOK -> "Playbook summary ready."
            SummarySource.DISABLED -> result.errorMessage ?: "AI is not configured; showing playbook summary."
            SummarySource.FAILED -> "AI summary failed: ${formatLlmErrorMessage(result.errorMessage)}"
        }


}
