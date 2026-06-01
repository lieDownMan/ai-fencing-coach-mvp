package com.aifencingcoach

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.speech.tts.TextToSpeech
import androidx.activity.compose.BackHandler
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CheckboxDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.runtime.collectAsState
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.aifencingcoach.runtime.CoachFrameState
import com.aifencingcoach.runtime.FeedbackCue
import com.aifencingcoach.runtime.LiveCoachPipeline
import com.aifencingcoach.runtime.PoseBackendKind
import com.aifencingcoach.runtime.PracticeReport
import com.aifencingcoach.runtime.Skeleton
import com.aifencingcoach.runtime.TargetSide
import com.aifencingcoach.runtime.TrainingMode
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlinx.coroutines.launch

private enum class AppScreen {
    HOME,
    REALTIME,
    POSTGAME,
    HISTORY,
    SETTINGS,
    COACH,
    SESSION_DETAIL
}

private data class UserSettings(
    val name: String = "Fencer",
    val handedness: String = "right",
    val heightCm: String = "180",
    val processingProfile: String = "Balanced",
    val useGeminiSummary: Boolean = false,
    val onlyFocusedErrors: Boolean = false,
    val emphasizedErrors: Set<String> = emptySet(),
    val mutedErrors: Set<String> = emptySet(),
    val language: String = "zh",
    val autoExportVideo: Boolean = false,
    val showSkeletonOverlay: Boolean = true
)

class MainActivity : ComponentActivity() {
    private var tts: TextToSpeech? = null
    private var ttsReady = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        tts = TextToSpeech(this) { status ->
            ttsReady = status == TextToSpeech.SUCCESS
            if (ttsReady) {
                tts?.language = Locale.TAIWAN
            }
        }

        setContent {
            MaterialTheme {
                Surface(color = Color.Black) {
                    FencingCoachScreen(onSpeak = ::speak)
                }
            }
        }
    }

    override fun onDestroy() {
        tts?.shutdown()
        tts = null
        super.onDestroy()
    }

    private fun speak(text: String) {
        if (!ttsReady || text.isBlank()) return
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "cue-${System.nanoTime()}")
    }
}

@Composable
private fun FencingCoachScreen(onSpeak: (String) -> Unit) {
    val context = LocalContext.current
    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { granted -> hasPermission = granted }
    )

    LaunchedEffect(Unit) {
        if (!hasPermission) permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    if (!hasPermission) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color(0xFF101418)),
            contentAlignment = Alignment.Center
        ) {
            Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                Text("Allow camera")
            }
        }
        return
    }

    val prefs = remember {
        context.getSharedPreferences("ai_fencing_coach_settings", Context.MODE_PRIVATE)
    }
    
    val sessionRepository = remember { com.aifencingcoach.runtime.database.SessionRepository(context) }
    val playbookRepository = remember { com.aifencingcoach.runtime.PlaybookRepository(context) }
    val geminiAgent = remember { com.aifencingcoach.runtime.GeminiAgent(playbookRepository) }
    val analysisManager = remember { com.aifencingcoach.runtime.AnalysisManager(context, sessionRepository, geminiAgent) }

    var appScreen by remember { mutableStateOf(AppScreen.HOME) }
    var selectedSessionForDetail by remember { mutableStateOf<com.aifencingcoach.runtime.database.FullSessionData?>(null) }
    var targetSide by remember { mutableStateOf(TargetSide.LEFT) }
    var trainingMode by remember { mutableStateOf(TrainingMode.FREE_BOUTING) }
    var poseBackend by remember { mutableStateOf(PoseBackendKind.MEDIAPIPE) }
    var voiceEnabled by remember { mutableStateOf(prefs.getBoolean("voice_enabled", true)) }
    var lastPracticeReport by remember { mutableStateOf<PracticeReport?>(null) }
    var userSettings by remember {
        mutableStateOf(
            UserSettings(
                name = prefs.getString("user_name", "Fencer") ?: "Fencer",
                handedness = prefs.getString("handedness", "right") ?: "right",
                heightCm = prefs.getString("height_cm", "180") ?: "180",
                processingProfile = prefs.getString("processing_profile", "Balanced") ?: "Balanced",
                useGeminiSummary = prefs.getBoolean("use_gemini_summary", false),
                onlyFocusedErrors = prefs.getBoolean("only_focused_errors", false),
                emphasizedErrors = prefs.getStringSet("emphasized_errors", emptySet())?.toSet().orEmpty(),
                mutedErrors = prefs.getStringSet("muted_errors", emptySet())?.toSet().orEmpty(),
                language = prefs.getString("language", "zh") ?: "zh",
                autoExportVideo = prefs.getBoolean("auto_export_video", false),
                showSkeletonOverlay = prefs.getBoolean("show_skeleton_overlay", true)
            )
        )
    }

    fun saveSettings() {
        prefs.edit()
            .putString("user_name", userSettings.name.ifBlank { "Fencer" })
            .putString("handedness", userSettings.handedness)
            .putString("height_cm", userSettings.heightCm.ifBlank { "180" })
            .putString("processing_profile", userSettings.processingProfile)
            .putBoolean("use_gemini_summary", userSettings.useGeminiSummary)
            .putBoolean("only_focused_errors", userSettings.onlyFocusedErrors)
            .putStringSet("emphasized_errors", userSettings.emphasizedErrors)
            .putStringSet("muted_errors", userSettings.mutedErrors)
            .putBoolean("voice_enabled", voiceEnabled)
            .putString("language", userSettings.language)
            .putBoolean("auto_export_video", userSettings.autoExportVideo)
            .putBoolean("show_skeleton_overlay", userSettings.showSkeletonOverlay)
            .apply()
    }

    when (appScreen) {
        AppScreen.HOME -> HomeScreen(
            trainingMode = trainingMode,
            poseBackend = poseBackend,
            targetSide = targetSide,
            voiceEnabled = voiceEnabled,
            userSettings = userSettings,
            lastPracticeReport = lastPracticeReport,
            onRealtime = { appScreen = AppScreen.REALTIME },
            onPostgame = { appScreen = AppScreen.POSTGAME },
            onHistory = { appScreen = AppScreen.HISTORY },
            onSettings = { appScreen = AppScreen.SETTINGS }
        )
        AppScreen.REALTIME -> RealtimeSetupScreen(
            trainingMode = trainingMode,
            poseBackend = poseBackend,
            targetSide = targetSide,
            voiceEnabled = voiceEnabled,
            userSettings = userSettings,
            onSettings = { appScreen = AppScreen.SETTINGS },
            onBack = {
                saveSettings()
                appScreen = AppScreen.HOME
            },
            onStart = {
                saveSettings()
                appScreen = AppScreen.COACH
            }
        )
        AppScreen.POSTGAME -> PostgameScreen(
            userSettings = userSettings,
            trainingMode = trainingMode,
            targetSide = targetSide,
            poseBackend = poseBackend,
            lastPracticeReport = lastPracticeReport,
            sessionRepository = sessionRepository,
            analysisManager = analysisManager,
            geminiAgent = geminiAgent,
            onSettings = { appScreen = AppScreen.SETTINGS },
            onPracticeReport = { report -> lastPracticeReport = report },
            onBack = {
                saveSettings()
                appScreen = AppScreen.HOME
            }
        )
        AppScreen.HISTORY -> com.aifencingcoach.HistoryScreen(
            sessionRepo = sessionRepository,
            geminiAgent = geminiAgent,
            userName = userSettings.name,
            onBack = { appScreen = AppScreen.HOME },
            onSessionSelected = { fullSession ->
                selectedSessionForDetail = fullSession
                appScreen = AppScreen.SESSION_DETAIL
            }
        )
        AppScreen.SESSION_DETAIL -> {
            selectedSessionForDetail?.let { sessionData ->
                com.aifencingcoach.SessionDetailScreen(
                    sessionData = sessionData,
                    onBack = { appScreen = AppScreen.HISTORY }
                )
            } ?: run {
                appScreen = AppScreen.HISTORY
            }
        }
        AppScreen.SETTINGS -> UserSettingsScreen(
            userSettings = userSettings,
            trainingMode = trainingMode,
            targetSide = targetSide,
            voiceEnabled = voiceEnabled,
            poseBackend = poseBackend,
            onUserSettings = { userSettings = it },
            onTrainingMode = { trainingMode = it },
            onTargetSide = { targetSide = it },
            onVoiceEnabled = { voiceEnabled = it },
            onPoseBackend = { poseBackend = it },
            onBack = {
                saveSettings()
                appScreen = AppScreen.HOME
            }
        )
        AppScreen.COACH -> CoachScreen(
            context = context,
            targetSide = targetSide,
            trainingMode = trainingMode,
            poseBackend = poseBackend,
            voiceEnabled = voiceEnabled,
            userSettings = userSettings,
            geminiAgent = geminiAgent,
            onVoiceEnabled = {
                voiceEnabled = it
                prefs.edit().putBoolean("voice_enabled", it).apply()
            },
            onBackToMenu = {
                saveSettings()
                appScreen = AppScreen.HOME
            },
            onPracticeReport = { report ->
                lastPracticeReport = report
                
                // Save immediately with the playbook summary, then let Gemini update history in the background.
                val scope = kotlinx.coroutines.CoroutineScope(kotlinx.coroutines.Dispatchers.Default)
                scope.launch {
                    val fallbackSummary = geminiAgent.generateSummary(
                        trainingMode = trainingMode.label,
                        targetSide = targetSide.label,
                        actionCounts = report.actionCounts,
                        cuesFired = report.cueTimeline,
                        userSettingsName = userSettings.name,
                        preferGemini = false
                    )

                    val sessionId = sessionRepository.savePracticeReport(
                        report = report,
                        cuesFired = report.cueTimeline,
                        llmSummary = fallbackSummary.ifEmpty { "No summary available." },
                        userName = userSettings.name
                    )

                    if (userSettings.useGeminiSummary) {
                        val geminiSummaryText = geminiAgent.generateSummary(
                            trainingMode = trainingMode.label,
                            targetSide = targetSide.label,
                            actionCounts = report.actionCounts,
                            cuesFired = report.cueTimeline,
                            userSettingsName = userSettings.name,
                            preferGemini = true
                        )
                        sessionRepository.updateSummary(sessionId, geminiSummaryText)
                    }
                }
            },
            onSpeak = onSpeak
        )
    }
}

@Composable
@OptIn(ExperimentalLayoutApi::class)
private fun HomeScreen(
    trainingMode: TrainingMode,
    poseBackend: PoseBackendKind,
    targetSide: TargetSide,
    voiceEnabled: Boolean,
    userSettings: UserSettings,
    lastPracticeReport: PracticeReport?,
    onRealtime: () -> Unit,
    onPostgame: () -> Unit,
    onHistory: () -> Unit,
    onSettings: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(PageBackground)
            .padding(ScreenPadding),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        ScreenHeader(
            title = "AI Fencing Coach",
            subtitle = "${userSettings.name.ifBlank { "Fencer" }}  |  ${trainingMode.label}  |  ${poseBackend.label}  |  ${targetSide.label}  |  v1.1.1"
        )

        Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(14.dp)
            ) {
                HomeOption(
                    title = "Realtime",
                    accent = AccentGreen,
                    summary = liveSummary(trainingMode, poseBackend, voiceEnabled),
                    onClick = onRealtime,
                    modifier = Modifier.weight(1f)
                )
                HomeOption(
                    title = "Postgame",
                    accent = AccentGold,
                    summary = postgameSummary(lastPracticeReport),
                    onClick = onPostgame,
                    modifier = Modifier.weight(1f)
                )
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(14.dp)
            ) {
                HomeOption(
                    title = "History",
                    accent = Color(0xFF9E9E9E),
                    summary = "Review past sessions & summaries",
                    onClick = onHistory,
                    modifier = Modifier.weight(1f)
                )
                HomeOption(
                    title = "User Settings",
                    accent = AccentCoral,
                    summary = settingsSummary(userSettings),
                    onClick = onSettings,
                    modifier = Modifier.weight(1f)
                )
            }
        }

        FlowRow(
            modifier = Modifier
                .fillMaxWidth()
                .background(PanelColor, RoundedCornerShape(8.dp))
                .padding(10.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            StatusPill("target ${targetSide.label}", AccentGreen)
            StatusPill(if (voiceEnabled) "voice on" else "voice off", if (voiceEnabled) AccentGreen else MutedText)
            StatusPill("${userSettings.emphasizedErrors.size} emphasized", AccentGold)
            StatusPill("${userSettings.mutedErrors.size} muted", AccentCoral)
            lastPracticeReport?.let {
                StatusPill("last ${formatSeconds(it.elapsedSeconds)}", Color.White)
            }
        }
    }
}

@Composable
private fun HomeOption(
    title: String,
    accent: Color,
    summary: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Button(
        onClick = onClick,
        modifier = modifier.height(136.dp),
        shape = RoundedCornerShape(8.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = PanelColor,
            contentColor = Color.White
        )
    ) {
        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.SpaceBetween,
            horizontalAlignment = Alignment.Start
        ) {
            Column {
                Text(title, fontSize = 19.sp, fontWeight = FontWeight.Bold)
                Spacer(Modifier.height(8.dp))
                Text(summary, color = MutedText, fontSize = BodyTextSize)
            }
            Text("Open", color = accent, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
        }
    }
}

@Composable
@OptIn(ExperimentalLayoutApi::class)
private fun RealtimeSetupScreen(
    trainingMode: TrainingMode,
    poseBackend: PoseBackendKind,
    targetSide: TargetSide,
    voiceEnabled: Boolean,
    userSettings: UserSettings,
    onSettings: () -> Unit,
    onBack: () -> Unit,
    onStart: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(PageBackground)
            .padding(ScreenPadding),
        verticalArrangement = Arrangement.SpaceBetween
    ) {
        ScreenHeader(
            title = "Realtime",
            subtitle = "${userSettings.name.ifBlank { "Fencer" }} | ${trainingMode.label} | ${poseBackend.label} | target ${targetSide.label}",
            onBack = onBack
        )
        FlowRow(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            HudButton(text = "Start Camera", selected = true, onClick = onStart)
            HudButton(text = "User Settings", selected = false, onClick = onSettings)
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            MenuPanel("Current Setup", Modifier.weight(1f)) {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    StatusPill(trainingMode.label, AccentGreen)
                    StatusPill(poseBackend.label, AccentGreen)
                    StatusPill("target ${targetSide.label}", AccentGreen)
                    StatusPill(if (voiceEnabled) "voice on" else "voice off", if (voiceEnabled) AccentGreen else MutedText)
                }
                Spacer(Modifier.height(8.dp))
                Text(settingsSummary(userSettings), color = MutedText, fontSize = BodyTextSize)
            }
            MenuPanel("Feedback", Modifier.weight(1f)) {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    StatusPill("${userSettings.emphasizedErrors.size} emphasized", AccentGold)
                    StatusPill("${userSettings.mutedErrors.size} muted", AccentCoral)
                    StatusPill(if (userSettings.onlyFocusedErrors) "focused only" else "all cues", MutedText)
                }
                Spacer(Modifier.height(8.dp))
                Text(modeSummary(trainingMode), color = MutedText, fontSize = BodyTextSize)
            }
        }

        MenuPanel("Feedback Focus") {
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                availableErrorsForMode(trainingMode).take(5).forEach { option ->
                    val color = when (option.key) {
                        in userSettings.mutedErrors -> AccentCoral
                        in userSettings.emphasizedErrors -> AccentGold
                        else -> MutedText
                    }
                    StatusPill(option.label, color)
                }
                if (availableErrorsForMode(trainingMode).size > 5) {
                    StatusPill("+${availableErrorsForMode(trainingMode).size - 5}", MutedText)
                }
            }
            Spacer(Modifier.height(10.dp))
            Text(
                text = "${userSettings.emphasizedErrors.size} emphasized  |  ${userSettings.mutedErrors.size} muted  |  ${if (userSettings.onlyFocusedErrors) "focused only" else "all cues"}",
                color = MutedText,
                fontSize = BodyTextSize
            )
        }
    }
}

@Composable
@OptIn(ExperimentalLayoutApi::class)
private fun PostgameScreen(
    userSettings: UserSettings,
    trainingMode: TrainingMode,
    targetSide: TargetSide,
    poseBackend: PoseBackendKind,
    lastPracticeReport: PracticeReport?,
    sessionRepository: com.aifencingcoach.runtime.database.SessionRepository,
    geminiAgent: com.aifencingcoach.runtime.GeminiAgent,
    analysisManager: com.aifencingcoach.runtime.AnalysisManager,
    onSettings: () -> Unit,
    onPracticeReport: (PracticeReport) -> Unit,
    onBack: () -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val analysisRunning by analysisManager.isAnalyzing.collectAsStateWithLifecycle()
    val analysisProgress by analysisManager.analysisProgress.collectAsStateWithLifecycle()
    val analysisStatus by analysisManager.analysisStatus.collectAsStateWithLifecycle()
    val frameStates by analysisManager.frameStates.collectAsStateWithLifecycle()
    val lastReport by analysisManager.lastReport.collectAsStateWithLifecycle()
    val lastSessionId by analysisManager.lastSessionId.collectAsStateWithLifecycle()
    val lastSummary by analysisManager.lastSummary.collectAsStateWithLifecycle()
    val lastSourceUri by analysisManager.lastSourceUri.collectAsStateWithLifecycle()

    val queueSize by analysisManager.queueSize.collectAsStateWithLifecycle()
    var selectedVideoUris by remember { mutableStateOf<List<Uri>>(emptyList()) }
    var isExporting by remember { mutableStateOf(false) }
    val videoAnnotator = remember { com.aifencingcoach.runtime.VideoAnnotator(context) }
    val videoPicker = rememberLauncherForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris ->
        if (uris.isNotEmpty()) {
            selectedVideoUris = (selectedVideoUris + uris).distinct()
        }
    }

    LaunchedEffect(lastReport) {
        lastReport?.let(onPracticeReport)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(PageBackground)
            .verticalScroll(rememberScrollState())
            .padding(ScreenPadding),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        ScreenHeader(
            title = "Postgame",
            subtitle = "${userSettings.name.ifBlank { "Fencer" }}  |  ${trainingMode.label}  |  ${userSettings.processingProfile}",
            onBack = onBack
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            MenuPanel("Analysis Defaults", Modifier.weight(1f)) {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    StatusPill(trainingMode.label, AccentGreen)
                    StatusPill("target ${targetSide.label}", AccentGreen)
                    StatusPill(userSettings.processingProfile, AccentGold)
                    StatusPill(if (userSettings.useGeminiSummary) "Gemini" else "Playbook", AccentGold)
                }
                Spacer(Modifier.height(10.dp))
                Text(
                    text = "Focus ${userSettings.emphasizedErrors.size}  |  Mute ${userSettings.mutedErrors.size}  |  ${if (userSettings.onlyFocusedErrors) "focused only" else "all errors"}",
                    color = MutedText,
                    fontSize = BodyTextSize
                )
                Spacer(Modifier.height(10.dp))
                HudButton(text = "Edit Settings", selected = false, onClick = onSettings)
            }
            MenuPanel("Video Queue", Modifier.weight(2f)) {
                val statusText = if (selectedVideoUris.isEmpty()) {
                    "No videos selected"
                } else {
                    "${selectedVideoUris.size} video(s) selected"
                }
                Text(
                    text = statusText,
                    color = if (selectedVideoUris.isEmpty()) MutedText else Color.White,
                    fontSize = BodyTextSize
                )
                if (selectedVideoUris.isNotEmpty()) {
                    Spacer(Modifier.height(8.dp))
                    selectedVideoUris.take(5).forEach { uri ->
                        val name = remember(uri) { videoDisplayName(context, uri) }
                        Text(
                            text = "Selected: $name",
                            color = AccentGreen,
                            fontSize = 12.sp,
                            maxLines = 1
                        )
                    }
                    if (selectedVideoUris.size > 5) {
                        Text(
                            text = "+${selectedVideoUris.size - 5} more selected",
                            color = MutedText,
                            fontSize = 12.sp
                        )
                    }
                }
                if (queueSize > 0) {
                    Text(
                        text = "$queueSize video(s) remaining in queue",
                        color = AccentGold,
                        fontSize = BodyTextSize,
                        fontWeight = FontWeight.Bold
                    )
                }
                Spacer(Modifier.height(10.dp))
                StatusPill(
                    if (userSettings.autoExportVideo) "auto export on" else "manual export",
                    if (userSettings.autoExportVideo) AccentGreen else MutedText
                )
                Spacer(Modifier.height(10.dp))
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    HudButton(text = "Add Videos", selected = false, onClick = { videoPicker.launch("video/*") })
                    if (selectedVideoUris.isNotEmpty()) {
                        HudButton(text = "Clear Selection", selected = false, onClick = { selectedVideoUris = emptyList() })
                    }
                    HudButton(
                        text = if (analysisRunning) "Add to Queue" else "Process ${selectedVideoUris.size} Video(s)",
                        selected = selectedVideoUris.isNotEmpty(),
                        onClick = {
                            if (selectedVideoUris.isNotEmpty()) {
                                analysisManager.enqueueAnalysis(
                                    uris = selectedVideoUris,
                                    targetSide = targetSide,
                                    trainingMode = trainingMode,
                                    poseBackend = poseBackend,
                                    userSettingsName = userSettings.name,
                                    useGeminiSummary = userSettings.useGeminiSummary,
                                    autoExport = userSettings.autoExportVideo
                                )
                                selectedVideoUris = emptyList() // clear after queuing
                            }
                        }
                    )
                }
                Spacer(Modifier.height(12.dp))
                Text(
                    text = analysisStatus,
                    color = if (analysisRunning) AccentGreen else MutedText,
                    fontSize = BodyTextSize,
                    fontWeight = FontWeight.SemiBold
                )
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(
                    progress = { analysisProgress.coerceIn(0f, 1f) },
                    modifier = Modifier.fillMaxWidth(),
                    color = AccentGreen,
                    trackColor = Color(0xFF263039)
                )
            }
        }

        MenuPanel("Latest Session", Modifier.fillMaxWidth()) {
            val reportToShow = lastReport ?: lastPracticeReport
            if (reportToShow == null) {
                Text("No practice report yet", color = MutedText, fontSize = BodyTextSize)
            } else {
                PracticeReportSummary(
                    report = reportToShow,
                    summary = if (lastReport != null) lastSummary else null
                )
                
                val currentFrameStates = frameStates
                val currentSourceUri = lastSourceUri
                if (currentFrameStates != null && currentSourceUri != null) {
                    Spacer(Modifier.height(16.dp))
                    HudButton(
                        text = if (isExporting) "Exporting Video..." else "Save Annotated Video",
                        selected = false,
                        onClick = {
                            if (!isExporting) {
                                isExporting = true
                                scope.launch {
                                    try {
                                        videoAnnotator.exportVideo(
                                            sourceUri = currentSourceUri,
                                            frameStates = currentFrameStates,
                                            videoWidth = 720,
                                            videoHeight = 1280
                                        ).collect { progress ->
                                            when (progress) {
                                                is com.aifencingcoach.runtime.ExportProgress.Completed -> {
                                                    isExporting = false
                                                    val outputPath = progress.filePath

                                                    val currentSessionId = lastSessionId
                                                    scope.launch(kotlinx.coroutines.Dispatchers.IO) {
                                                        if (currentSessionId != null) {
                                                            com.aifencingcoach.runtime.database.AppDatabase.getDatabase(context)
                                                                .sessionDao().updateSessionVideoPath(currentSessionId, outputPath)
                                                        }
                                                    }

                                                    android.widget.Toast.makeText(context, "Annotated video saved.", android.widget.Toast.LENGTH_LONG).show()
                                                }
                                                is com.aifencingcoach.runtime.ExportProgress.Error -> {
                                                    isExporting = false
                                                    android.widget.Toast.makeText(context, "Export failed: ${progress.exception.message}", android.widget.Toast.LENGTH_LONG).show()
                                                }
                                            }
                                        }
                                    } catch (e: Exception) {
                                        isExporting = false
                                        android.widget.Toast.makeText(context, "Export failed: ${e.message}", android.widget.Toast.LENGTH_LONG).show()
                                    }
                                }
                            }
                        }
                    )
                }
            }
        }
    }
}

@Composable
@OptIn(ExperimentalLayoutApi::class)
private fun UserSettingsScreen(
    userSettings: UserSettings,
    trainingMode: TrainingMode,
    targetSide: TargetSide,
    voiceEnabled: Boolean,
    poseBackend: PoseBackendKind,
    onUserSettings: (UserSettings) -> Unit,
    onTrainingMode: (TrainingMode) -> Unit,
    onTargetSide: (TargetSide) -> Unit,
    onVoiceEnabled: (Boolean) -> Unit,
    onPoseBackend: (PoseBackendKind) -> Unit,
    onBack: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(PageBackground)
            .verticalScroll(rememberScrollState())
            .padding(ScreenPadding),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        ScreenHeader(
            title = "User Settings",
            subtitle = "${userSettings.name.ifBlank { "Fencer" }}  |  ${trainingMode.label}",
            onBack = onBack
        )

        MenuPanel("User Information", Modifier.fillMaxWidth()) {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(
                    value = userSettings.name,
                    onValueChange = { onUserSettings(userSettings.copy(name = it)) },
                    label = { Text("Name", color = MutedText) },
                    textStyle = androidx.compose.ui.text.TextStyle(color = Color.White),
                    singleLine = true,
                    modifier = Modifier.weight(1f)
                )
                OutlinedTextField(
                    value = userSettings.heightCm,
                    onValueChange = { value ->
                        onUserSettings(userSettings.copy(heightCm = value.filter(Char::isDigit).take(3)))
                    },
                    label = { Text("Height cm", color = MutedText) },
                    textStyle = androidx.compose.ui.text.TextStyle(color = Color.White),
                    singleLine = true,
                    modifier = Modifier.width(140.dp)
                )
            }
            Spacer(Modifier.height(10.dp))
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                SegmentedGroup(
                    labels = listOf("left", "right"),
                    selected = userSettings.handedness,
                    onSelected = { onUserSettings(userSettings.copy(handedness = it)) }
                )
                SegmentedGroup(
                    labels = TrainingMode.entries.map { it.label },
                    selected = trainingMode.label,
                    onSelected = { onTrainingMode(TrainingMode.fromLabel(it)) }
                )
                SegmentedGroup(
                    labels = listOf("left", "right"),
                    selected = targetSide.label,
                    onSelected = { onTargetSide(TargetSide.fromLabel(it)) }
                )
                SegmentedGroup(
                    labels = listOf("zh", "en"),
                    selected = userSettings.language,
                    onSelected = { onUserSettings(userSettings.copy(language = it)) }
                )
            }
        }

        MenuPanel("App Defaults", Modifier.fillMaxWidth()) {
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                SegmentedGroup(
                    labels = PoseBackendKind.entries.map { it.label },
                    selected = poseBackend.label,
                    onSelected = { onPoseBackend(PoseBackendKind.fromLabel(it)) }
                )
                SegmentedGroup(
                    labels = ProcessingProfiles,
                    selected = userSettings.processingProfile,
                    onSelected = { onUserSettings(userSettings.copy(processingProfile = it)) }
                )
                HudButton(
                    text = if (voiceEnabled) "Voice on" else "Voice off",
                    selected = voiceEnabled,
                    onClick = { onVoiceEnabled(!voiceEnabled) }
                )
                HudButton(
                    text = if (userSettings.useGeminiSummary) "Gemini on" else "Playbook",
                    selected = userSettings.useGeminiSummary,
                    onClick = { onUserSettings(userSettings.copy(useGeminiSummary = !userSettings.useGeminiSummary)) }
                )
            }
            Spacer(Modifier.height(8.dp))
            CheckboxLine(
                label = "Auto-export annotated postgame videos",
                checked = userSettings.autoExportVideo,
                onCheckedChange = { onUserSettings(userSettings.copy(autoExportVideo = it)) }
            )
            CheckboxLine(
                label = "Show skeleton overlay in realtime",
                checked = userSettings.showSkeletonOverlay,
                onCheckedChange = { onUserSettings(userSettings.copy(showSkeletonOverlay = it)) }
            )
        }

        MenuPanel("Feedback Controls", Modifier.fillMaxWidth()) {
            CheckboxLine(
                label = "Only focused errors",
                checked = userSettings.onlyFocusedErrors,
                onCheckedChange = { onUserSettings(userSettings.copy(onlyFocusedErrors = it)) }
            )
            Spacer(Modifier.height(8.dp))
            availableErrorsForMode(trainingMode).forEach { option ->
                ErrorPreferenceRow(
                    option = option,
                    emphasized = option.key in userSettings.emphasizedErrors,
                    muted = option.key in userSettings.mutedErrors,
                    onEmphasized = { checked ->
                        val emphasized = userSettings.emphasizedErrors.toggle(option.key, checked)
                        val muted = if (checked) userSettings.mutedErrors - option.key else userSettings.mutedErrors
                        onUserSettings(userSettings.copy(emphasizedErrors = emphasized, mutedErrors = muted))
                    },
                    onMuted = { checked ->
                        val muted = userSettings.mutedErrors.toggle(option.key, checked)
                        val emphasized = if (checked) userSettings.emphasizedErrors - option.key else userSettings.emphasizedErrors
                        onUserSettings(userSettings.copy(emphasizedErrors = emphasized, mutedErrors = muted))
                    }
                )
            }
        }

        FlowRow(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            HudButton(text = "Done", selected = true, onClick = onBack)
        }
        Spacer(Modifier.height(8.dp))
    }
}

@Composable
private fun MenuPanel(
    title: String,
    modifier: Modifier = Modifier,
    content: @Composable () -> Unit
) {
    Column(
        modifier = modifier
            .background(PanelColor, RoundedCornerShape(8.dp))
            .padding(10.dp)
    ) {
        Text(
            text = title,
            color = Color.White,
            fontSize = 15.sp,
            fontWeight = FontWeight.Bold
        )
        Spacer(Modifier.height(10.dp))
        content()
    }
}

@Composable
private fun ScreenHeader(
    title: String,
    subtitle: String,
    onBack: (() -> Unit)? = null
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.Top
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(title, color = Color.White, fontSize = 22.sp, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(6.dp))
            Text(subtitle, color = AccentGreen, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
        }
        if (onBack != null) {
            HudButton(text = "Home", selected = false, onClick = onBack)
        }
    }
}

@Composable
private fun StatusPill(text: String, color: Color) {
    Text(
        text = text,
        color = color,
        fontSize = 11.sp,
        fontWeight = FontWeight.SemiBold,
        modifier = Modifier
            .background(Color(0xFF222B31), RoundedCornerShape(6.dp))
            .padding(horizontal = 10.dp, vertical = 6.dp)
    )
}

@Composable
@OptIn(ExperimentalLayoutApi::class)
private fun PracticeReportSummary(report: PracticeReport, summary: String? = null) {
    SessionFeedbackPanel(report = report, summary = summary)
}

@Composable
private fun CheckboxLine(
    label: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Checkbox(checked = checked, onCheckedChange = onCheckedChange)
        Text(label, color = Color.White, fontSize = BodyTextSize, fontWeight = FontWeight.SemiBold)
    }
}

@Composable
private fun ErrorPreferenceRow(
    option: FeedbackErrorOption,
    emphasized: Boolean,
    muted: Boolean,
    onEmphasized: (Boolean) -> Unit,
    onMuted: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(option.label, color = Color.White, fontSize = BodyTextSize, fontWeight = FontWeight.SemiBold)
            Text(option.key, color = MutedText, fontSize = 11.sp)
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Checkbox(checked = emphasized, onCheckedChange = onEmphasized)
            Text("Emphasize", color = AccentGold, fontSize = 12.sp)
            Spacer(Modifier.width(12.dp))
            Checkbox(checked = muted, onCheckedChange = onMuted)
            Text("Mute", color = AccentCoral, fontSize = 12.sp)
        }
    }
}

@Composable
private fun CoachScreen(
    context: Context,
    targetSide: TargetSide,
    trainingMode: TrainingMode,
    poseBackend: PoseBackendKind,
    voiceEnabled: Boolean,
    userSettings: UserSettings,
    geminiAgent: com.aifencingcoach.runtime.GeminiAgent,
    onVoiceEnabled: (Boolean) -> Unit,
    onBackToMenu: () -> Unit,
    onPracticeReport: (PracticeReport) -> Unit,
    onSpeak: (String) -> Unit
) {
    var analysisPaused by remember { mutableStateOf(false) }
    var frameState by remember { mutableStateOf(CoachFrameState()) }
    var reviewReport by remember { mutableStateOf<PracticeReport?>(null) }
    var reviewSummary by remember { mutableStateOf<String?>(null) }
    var resetToken by remember { mutableStateOf(0) }
    var sessionRecorded by remember { mutableStateOf(false) }

    LaunchedEffect(reviewReport) {
        if (reviewReport != null) {
            reviewSummary = "Generating Coach Summary..."
            reviewSummary = geminiAgent.generateSummary(
                trainingMode = trainingMode.label,
                targetSide = targetSide.label,
                actionCounts = reviewReport!!.actionCounts,
                cuesFired = reviewReport!!.cueTimeline,
                userSettingsName = userSettings.name,
                preferGemini = userSettings.useGeminiSummary
            )
        } else {
            reviewSummary = null
        }
    }

    val pipeline = remember(poseBackend, targetSide, trainingMode, resetToken) {
        LiveCoachPipeline(context, poseBackend, targetSide, trainingMode)
    }

    DisposableEffect(pipeline) {
        onDispose { pipeline.close() }
    }

    val isReviewing = reviewReport != null
    val displayedState = when {
        isReviewing -> frameState.copy(state = "REVIEW", cue = "Practice complete")
        analysisPaused -> frameState.copy(state = "PAUSED", cue = "Analysis paused")
        else -> frameState
    }

    fun recordCurrentSessionIfNeeded() {
        if (sessionRecorded) return
        val report = reviewReport ?: pipeline.practiceReport()
        reviewReport = report
        onPracticeReport(report)
        sessionRecorded = true
    }

    fun leaveRealtime() {
        recordCurrentSessionIfNeeded()
        analysisPaused = true
        onBackToMenu()
    }

    BackHandler {
        leaveRealtime()
    }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        Column(modifier = Modifier.fillMaxSize()) {
            // ── Camera + overlays (top 60%) ─────────────────────────────
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(0.6f)
            ) {
                CameraPreview(
                    pipeline = pipeline,
                    analysisPaused = analysisPaused || isReviewing,
                    onFrameState = { state, cue ->
                        frameState = state
                        if (voiceEnabled && cue != null) onSpeak(cue.shortCue.ifBlank { cue.message })
                    }
                )
                if (userSettings.showSkeletonOverlay) {
                    SkeletonOverlay(state = displayedState)
                }

                // Action pill overlay (top center, like Flutter)
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 12.dp),
                    contentAlignment = Alignment.TopCenter
                ) {
                    ActionPill(
                        action = displayedState.action,
                        confidence = displayedState.confidence,
                        state = displayedState.state
                    )
                }

                // Status badge (top left)
                Box(
                    modifier = Modifier
                        .padding(12.dp)
                        .align(Alignment.TopStart)
                ) {
                    StatusBadge(
                        poseBackend = poseBackend,
                        ready = displayedState.ready,
                        fps = displayedState.metrics.fps
                    )
                }
            }

            // ── Feedback panel (bottom 40%) ─────────────────────────────
            FeedbackPanel(
                state = displayedState,
                targetSide = targetSide,
                trainingMode = trainingMode,
                voiceEnabled = voiceEnabled,
                analysisPaused = analysisPaused,
                userSettings = userSettings,
                onVoiceEnabled = onVoiceEnabled,
                onAnalysisPaused = { analysisPaused = it },
                onFinishPractice = {
                    val report = pipeline.practiceReport()
                    reviewReport = report
                    if (!sessionRecorded) {
                        onPracticeReport(report)
                        sessionRecorded = true
                    }
                    analysisPaused = true
                },
                onReset = {
                    pipeline.reset()
                    resetToken += 1
                    sessionRecorded = false
                },
                onBackToMenu = { leaveRealtime() },
                modifier = Modifier.weight(0.4f)
            )
        }
        reviewReport?.let { report ->
            PostPracticeReview(
                report = report,
                onResume = {
                    reviewReport = null
                    analysisPaused = false
                },
                onNewSession = {
                    reviewReport = null
                    analysisPaused = false
                    sessionRecorded = false
                    frameState = CoachFrameState()
                    pipeline.reset()
                    resetToken += 1
                },
                onHome = { leaveRealtime() },
                llmSummary = reviewSummary
            )
        }
    }
}

@Composable
private fun CameraPreview(
    pipeline: LiveCoachPipeline,
    analysisPaused: Boolean,
    onFrameState: (CoachFrameState, FeedbackCue?) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val mainExecutor = remember { ContextCompat.getMainExecutor(context) }
    val pausedState = rememberUpdatedState(analysisPaused)
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.PERFORMANCE
        }
    }

    DisposableEffect(pipeline) {
        val analyzerExecutor = Executors.newSingleThreadExecutor()
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener(
            {
                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                analysis.setAnalyzer(analyzerExecutor) { imageProxy ->
                    try {
                        if (pausedState.value) return@setAnalyzer
                        try {
                            val (state, cue) = pipeline.process(imageProxy)
                            mainExecutor.execute { onFrameState(state, cue) }
                        } catch (e: Exception) {
                            // Ignore pipeline exceptions (e.g. MediaPipe paused state) to prevent crashing
                        }
                    } finally {
                        imageProxy.close()
                    }
                }
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis
                )
            },
            mainExecutor
        )

        onDispose {
            runCatching { ProcessCameraProvider.getInstance(context).get().unbindAll() }
            analyzerExecutor.shutdownSafely()
        }
    }

    AndroidView(
        factory = { previewView },
        modifier = Modifier.fillMaxSize()
    )
}

@Composable
private fun SkeletonOverlay(state: CoachFrameState) {
    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val viewWidth = constraints.maxWidth.toFloat()
        val viewHeight = constraints.maxHeight.toFloat()
        val videoWidth = state.frameWidth.toFloat().coerceAtLeast(1f)
        val videoHeight = state.frameHeight.toFloat().coerceAtLeast(1f)

        // FIT_CENTER logic
        val scale = minOf(viewWidth / videoWidth, viewHeight / videoHeight)
        val scaledWidth = videoWidth * scale
        val scaledHeight = videoHeight * scale
        val padX = (viewWidth - scaledWidth) / 2f
        val padY = (viewHeight - scaledHeight) / 2f

        Canvas(modifier = Modifier.fillMaxSize()) {
            fun drawSkeleton(skeleton: Skeleton?, color: Color) {
                if (skeleton == null) return
                
                fun offset(name: String): Offset? {
                    val point = skeleton[name] ?: return null
                    return Offset(
                        x = point.x * scale + padX,
                        y = point.y * scale + padY
                    )
                }

                for ((a, b) in SkeletonEdges) {
                    val start = offset(a)
                    val end = offset(b)
                    if (start != null && end != null) {
                        drawLine(color, start, end, strokeWidth = 5f, cap = StrokeCap.Round)
                    }
                }
                for ((name, point) in skeleton) {
                    val center = Offset(
                        x = point.x * scale + padX,
                        y = point.y * scale + padY
                    )
                    if (name == "front_wrist" || name == "front_ankle") {
                        drawCircle(Color(0xFFFF4D00), radius = 18f, center = center, alpha = 0.6f)
                        drawCircle(Color.White, radius = 8f, center = center)
                    } else {
                        drawCircle(color, radius = 6f, center = center)
                    }
                }
            }

            drawSkeleton(state.opponentSkeleton, Color(0x88FFFFFF))
            drawSkeleton(state.targetSkeleton, Color(0xFFD7FF5F))
        }

        if (state.visualCues.isNotEmpty()) {
            val primaryCue = state.visualCues.first()
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color(0x40FF0000)), // Flash overlay effect
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = primaryCue.label,
                    color = Color.White,
                    fontSize = 28.sp,
                    fontWeight = FontWeight.ExtraBold,
                    modifier = Modifier
                        .background(Color(0xCCB71C1C), RoundedCornerShape(12.dp))
                        .padding(horizontal = 24.dp, vertical = 12.dp)
                )
            }
        }
    }
}

@Composable
private fun ActionPill(action: String, confidence: Float, state: String) {
    val offensiveActions = setOf("R", "JS", "WW", "IS")
    val footworkActions = setOf("SF", "SB")
    val isIdle = action == "Idle" || state == "IDLE"
    val isOffensive = action in offensiveActions
    val pillColor = when {
        state == "PAUSED" -> Color(0xFF888888)
        state == "REVIEW" -> AccentGold
        isIdle -> Color(0x80FFFFFF)
        isOffensive -> Color(0xFFFF4D00)
        action in footworkActions -> Color(0xFF00D4FF)
        else -> Color.White
    }
    val emoji = when {
        state == "PAUSED" -> "⏸"
        state == "REVIEW" -> "📋"
        isIdle -> "⏸"
        isOffensive -> "⚔️"
        else -> "🏃"
    }
    val label = if (isIdle && state != "PAUSED" && state != "REVIEW") {
        "$emoji  Idle"
    } else if (state == "PAUSED") {
        "$emoji  Paused"
    } else if (state == "REVIEW") {
        "$emoji  Review"
    } else {
        "$emoji  $action  (${(confidence * 100f).toInt()}%)"
    }

    Text(
        text = label,
        color = pillColor,
        fontSize = 13.sp,
        fontWeight = FontWeight.Bold,
        modifier = Modifier
            .background(Color(0xAA000000), RoundedCornerShape(20.dp))
            .padding(horizontal = 14.dp, vertical = 6.dp)
    )
}

@Composable
private fun StatusBadge(poseBackend: PoseBackendKind, ready: Boolean, fps: Float) {
    val badgeColor = if (ready) Color(0xFF00E676) else Color(0xFFFF9100)
    val label = if (ready) "✓ ${poseBackend.label} ${fps.toInt()}fps" else "⚠ Loading..."
    Text(
        text = label,
        color = badgeColor,
        fontSize = 10.sp,
        fontWeight = FontWeight.SemiBold,
        modifier = Modifier
            .background(Color(0x8C000000), RoundedCornerShape(8.dp))
            .padding(horizontal = 8.dp, vertical = 4.dp)
    )
}

@Composable
@OptIn(ExperimentalLayoutApi::class)
private fun FeedbackPanel(
    state: CoachFrameState,
    targetSide: TargetSide,
    trainingMode: TrainingMode,
    voiceEnabled: Boolean,
    analysisPaused: Boolean,
    userSettings: UserSettings,
    onVoiceEnabled: (Boolean) -> Unit,
    onAnalysisPaused: (Boolean) -> Unit,
    onFinishPractice: () -> Unit,
    onReset: () -> Unit,
    onBackToMenu: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .background(Color(0xFF0A0A10))
    ) {
        // ── Control bar ──────────────────────────────────────────────────
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color(0xFF141420))
                .padding(horizontal = 12.dp, vertical = 6.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Info text
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "${userSettings.name.ifBlank { "Fencer" }} | ${trainingMode.label} | ${targetSide.label}",
                    color = Color.White,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    text = "pose ${state.metrics.poseMs}ms | total ${state.metrics.totalMs}ms | cues ${state.session.cueCount}",
                    color = MutedText,
                    fontSize = 10.sp
                )
            }
            // Buttons
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                HudButton(
                    text = if (analysisPaused) "Resume" else "Pause",
                    selected = analysisPaused,
                    onClick = { onAnalysisPaused(!analysisPaused) }
                )
                HudButton(
                    text = if (voiceEnabled) "Voice" else "Silent",
                    selected = voiceEnabled,
                    onClick = { onVoiceEnabled(!voiceEnabled) }
                )
                HudButton(text = "Finish", selected = false, onClick = onFinishPractice)
                HudButton(text = "Reset", selected = false, onClick = onReset)
                HudButton(text = "Home", selected = false, onClick = onBackToMenu)
            }
        }

        // ── Error cards / Good technique indicator ───────────────────────
        val activeCues = state.visualCues
        if (activeCues.isEmpty()) {
            // Good technique - centered indicator
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Box(
                        modifier = Modifier
                            .background(Color(0x1A00E676), RoundedCornerShape(50))
                            .padding(16.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text("✓", fontSize = 28.sp, color = Color(0xFF00E676))
                    }
                    Spacer(Modifier.height(10.dp))
                    Text(
                        text = if (state.state == "IDLE") "Waiting for fencer..." else "Good Technique",
                        color = Color(0xFF00E676),
                        fontSize = 16.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = if (state.state == "IDLE") "Position yourself in the camera frame" else "Keep it up!",
                        color = Color(0xFFB6C2CC),
                        fontSize = 12.sp
                    )
                }
            }
        } else {
            // Error cards list
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                activeCues.forEachIndexed { index, cue ->
                    ErrorCard(
                        label = cue.label,
                        message = cue.message,
                        isPrimary = index == 0
                    )
                }

                // Cue history
                if (state.cueHistory.isNotEmpty()) {
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = "Recent: ${state.cueHistory.take(3).joinToString("  |  ") { it.label }}",
                        color = Color(0xFF8DA2B2),
                        fontSize = 11.sp
                    )
                }
            }
        }
    }
}

@Composable
private fun ErrorCard(label: String, message: String, isPrimary: Boolean) {
    val bgColors = if (isPrimary) {
        listOf(Color(0xFF5A0000), Color(0xFF2A0000))
    } else {
        listOf(Color(0xFF3A2000), Color(0xFF1A1000))
    }
    val borderColor = if (isPrimary) Color(0xCCFF5252) else Color(0x66FF9800)

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                brush = androidx.compose.ui.graphics.Brush.horizontalGradient(bgColors),
                shape = RoundedCornerShape(10.dp)
            )
            .padding(1.dp) // border effect
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 10.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = if (isPrimary) "⚠️" else "📌",
                    fontSize = 14.sp
                )
                Text(
                    text = label,
                    color = Color.White,
                    fontSize = if (isPrimary) 14.sp else 12.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

@Composable
@OptIn(ExperimentalLayoutApi::class)
private fun PostPracticeReview(
    report: PracticeReport,
    onResume: () -> Unit,
    onNewSession: () -> Unit,
    onHome: () -> Unit,
    llmSummary: String? = null
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xDD05080B))
            .padding(ScreenPadding),
        contentAlignment = Alignment.Center
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth(0.9f)
                .fillMaxHeight(0.9f)
                .background(Color(0xF0141A20), RoundedCornerShape(8.dp))
                .verticalScroll(rememberScrollState())
                .padding(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Top
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Practice Review",
                        color = Color.White,
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = "${report.trainingMode.label}  |  ${report.poseBackend.label}  |  ${report.targetSide.label}",
                        color = Color(0xFFD7FF5F),
                        fontSize = BodyTextSize
                    )
                }
                Row {
                    HudButton(text = "Home", selected = false, onClick = onHome)
                    Spacer(Modifier.width(8.dp))
                    HudButton(text = "Resume", selected = false, onClick = onResume)
                    Spacer(Modifier.width(8.dp))
                    HudButton(text = "New", selected = true, onClick = onNewSession)
                }
            }

            Spacer(Modifier.height(14.dp))
            SessionFeedbackPanel(report = report, summary = llmSummary)
            Spacer(Modifier.height(16.dp))
        }
    }
}

@Composable
private fun MetricBlock(label: String, value: String) {
    Column(
        modifier = Modifier
            .background(Color(0xFF263039), RoundedCornerShape(6.dp))
            .padding(horizontal = 12.dp, vertical = 8.dp)
    ) {
        Text(text = label, color = Color(0xFF8DA2B2), fontSize = 11.sp)
        Text(text = value, color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
    }
}

@Composable
private fun ActionRow(action: String, count: Long, percent: Int) {
    Column(modifier = Modifier.padding(bottom = 8.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(text = action, color = Color.White, fontSize = BodyTextSize, fontWeight = FontWeight.SemiBold)
            Text(text = "$count  $percent%", color = Color(0xFFB6C2CC), fontSize = 11.sp)
        }
        Spacer(Modifier.height(4.dp))
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(7.dp)
                .background(Color(0xFF263039), RoundedCornerShape(4.dp))
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth((percent / 100f).coerceIn(0.02f, 1f))
                    .height(7.dp)
                    .background(Color(0xFFD7FF5F), RoundedCornerShape(4.dp))
            )
        }
    }
}

@Composable
private fun SegmentedGroup(
    labels: List<String>,
    selected: String,
    onSelected: (String) -> Unit
) {
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        for (label in labels) {
            HudButton(
                text = label,
                selected = label == selected,
                onClick = { onSelected(label) }
            )
        }
    }
}

@Composable
private fun HudButton(text: String, selected: Boolean, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        shape = RoundedCornerShape(6.dp),
        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 5.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = if (selected) Color(0xFFD7FF5F) else Color(0xFF263039),
            contentColor = if (selected) Color(0xFF101418) else Color.White
        )
    ) {
        Text(text = text, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
    }
}

private fun ExecutorService.shutdownSafely() {
    shutdown()
}

private fun formatSeconds(totalSeconds: Long): String {
    val minutes = totalSeconds / 60
    val seconds = totalSeconds % 60
    return "%d:%02d".format(Locale.US, minutes, seconds)
}

private fun modeSummary(mode: TrainingMode): String =
    when (mode) {
        TrainingMode.FOOTWORK -> "Footwork cues prioritize stance, bounce, step width, and center of mass."
        TrainingMode.TARGET_PRACTICE -> "Target practice includes lunge timing, arm extension, and guard position."
        TrainingMode.FREE_BOUTING -> "Free bouting keeps cues broad for mixed movement and opponent context."
    }

private fun modelSummary(model: PoseBackendKind): String =
    when (model) {
        PoseBackendKind.MEDIAPIPE -> "MediaPipe lite is the default low-latency pose model."
        PoseBackendKind.YOLO -> "YOLO pose runs from yolo_pose.onnx through ONNX Runtime."
    }

private fun liveSummary(
    mode: TrainingMode,
    model: PoseBackendKind,
    voiceEnabled: Boolean
): String =
    "${mode.label} with ${model.label}, target lock, skeleton overlay, cues, ${if (voiceEnabled) "voice" else "silent"}."

private fun postgameSummary(report: PracticeReport?): String =
    if (report == null) {
        "Clip review setup, report view, history controls, and summary options."
    } else {
        "Latest: ${formatSeconds(report.elapsedSeconds)}, ${report.cueCount} cues, top ${report.topAction}."
    }

private fun settingsSummary(settings: UserSettings): String =
    "${settings.handedness}, ${settings.heightCm.ifBlank { "180" }}cm, ${settings.emphasizedErrors.size} focus, ${settings.mutedErrors.size} muted."

private fun videoDisplayName(context: Context, uri: Uri): String {
    context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
        val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
        if (index >= 0 && cursor.moveToFirst()) {
            cursor.getString(index)?.takeIf { it.isNotBlank() }?.let { return it }
        }
    }
    return uri.lastPathSegment ?: uri.toString()
}

private data class FeedbackErrorOption(
    val key: String,
    val label: String,
    val modes: Set<TrainingMode>
)

private fun availableErrorsForMode(mode: TrainingMode): List<FeedbackErrorOption> =
    FeedbackErrorOptions.filter { mode in it.modes }

private fun Set<String>.toggle(value: String, enabled: Boolean): Set<String> =
    if (enabled) this + value else this - value

private val ProcessingProfiles = listOf("Balanced", "Fast", "Full Quality")

private val FeedbackErrorOptions = listOf(
    FeedbackErrorOption("foot_before_hand", "Foot before hand", setOf(TrainingMode.TARGET_PRACTICE)),
    FeedbackErrorOption("lunge_overextension", "Lunge overextension", setOf(TrainingMode.TARGET_PRACTICE)),
    FeedbackErrorOption("incomplete_arm_extension", "Incomplete arm extension", setOf(TrainingMode.TARGET_PRACTICE)),
    FeedbackErrorOption("guard_dropped", "Guard dropped", TrainingMode.entries.toSet()),
    FeedbackErrorOption("stance_too_high", "Stance too high", TrainingMode.entries.toSet()),
    FeedbackErrorOption("bounce_excessive", "Bounce excessive", TrainingMode.entries.toSet()),
    FeedbackErrorOption("center_of_mass_in_front", "Center of mass in front", TrainingMode.entries.toSet()),
    FeedbackErrorOption("center_of_mass_leaning_backward", "Center of mass leaning backward", TrainingMode.entries.toSet()),
    FeedbackErrorOption("over_parrying", "Over parrying", TrainingMode.entries.toSet()),
    FeedbackErrorOption("wide_step", "Wide step", TrainingMode.entries.toSet()),
    FeedbackErrorOption("narrow_step", "Narrow step", TrainingMode.entries.toSet())
)

private val PageBackground = Color(0xFF0D1115)
private val PanelColor = Color(0xEE182026)
private val MutedText = Color(0xFFB6C2CC)
private val AccentGreen = Color(0xFFD7FF5F)
private val AccentGold = Color(0xFFFFD166)
private val AccentCoral = Color(0xFFFF7D6E)
private val ScreenPadding = 14.dp
private val BodyTextSize = 12.sp

private const val FenceNetWarmup = 28

private val SkeletonEdges = listOf(
    "nose" to "front_shoulder",
    "front_shoulder" to "front_elbow",
    "front_elbow" to "front_wrist",
    "left_shoulder" to "right_shoulder",
    "left_shoulder" to "left_hip",
    "right_shoulder" to "right_hip",
    "left_hip" to "right_hip",
    "left_hip" to "left_knee",
    "left_knee" to "left_ankle",
    "right_hip" to "right_knee",
    "right_knee" to "right_ankle",
    "left_ankle" to "right_ankle"
)
