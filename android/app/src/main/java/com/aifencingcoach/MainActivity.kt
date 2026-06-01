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
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.Icon
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CheckboxDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Divider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
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
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.aifencingcoach.runtime.CoachFrameState
import com.aifencingcoach.runtime.CoachingSummaryResult
import com.aifencingcoach.runtime.FeedbackCue
import com.aifencingcoach.runtime.LiveCoachPipeline
import com.aifencingcoach.runtime.LlmProviderConfig
import com.aifencingcoach.runtime.LlmProviderKind
import com.aifencingcoach.runtime.PlaybookRepository
import com.aifencingcoach.runtime.PoseBackendKind
import com.aifencingcoach.runtime.PracticeReport
import com.aifencingcoach.runtime.Skeleton
import com.aifencingcoach.runtime.SummarySource
import com.aifencingcoach.runtime.TargetSide
import com.aifencingcoach.runtime.TrainingMode
import com.aifencingcoach.runtime.normalizePlaybookLanguage
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
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
    val age: String = "25",
    val handedness: String = "right",
    val heightCm: String = "180",
    val weightKg: String = "70",
    val processingProfile: String = "Balanced",
    val useGeminiSummary: Boolean = false,
    val llmProvider: LlmProviderKind = LlmProviderKind.GEMINI,
    val geminiApiKey: String = "",
    val openAiApiKey: String = "",
    val onlyFocusedErrors: Boolean = false,
    val emphasizedErrors: Set<String> = emptySet(),
    val mutedErrors: Set<String> = emptySet(),
    val language: String = "zh",
    val autoExportVideo: Boolean = false,
    val showSkeletonOverlay: Boolean = true
)

private fun UserSettings.llmConfig(): LlmProviderConfig {
    val providerApiKey = when (llmProvider) {
        LlmProviderKind.GEMINI -> geminiApiKey
        LlmProviderKind.OPENAI -> openAiApiKey
        LlmProviderKind.PLAYBOOK -> ""
    }
    return LlmProviderConfig(
        provider = llmProvider,
        apiKey = providerApiKey.trim(),
        language = normalizePlaybookLanguage(language)
    )
}

class MainActivity : ComponentActivity() {
    private var tts: TextToSpeech? = null
    private var ttsReady = false

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        tts = TextToSpeech(this) { status ->
            ttsReady = status == TextToSpeech.SUCCESS
            if (ttsReady) {
                tts?.language = Locale.TAIWAN
            }
        }

        setContent {
            MaterialTheme {
                Surface(color = Color.Black, modifier = Modifier.fillMaxSize().systemBarsPadding()) {
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

    private fun speak(text: String, language: String) {
        if (!ttsReady || text.isBlank()) return
        tts?.language = ttsLocaleForLanguage(language)
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "cue-${System.nanoTime()}")
    }
}

@Composable
private fun FencingCoachScreen(onSpeak: (String, String) -> Unit) {
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
    val persistenceScope = rememberCoroutineScope()

    var appScreen by remember { mutableStateOf(AppScreen.HOME) }
    var selectedSessionForDetail by remember { mutableStateOf<com.aifencingcoach.runtime.database.FullSessionData?>(null) }
    var selectedHistoryUser by remember { mutableStateOf<String?>(null) }
    var targetSide by remember { mutableStateOf(TargetSide.LEFT) }
    var trainingMode by remember { mutableStateOf(TrainingMode.FREE_BOUTING) }
    var poseBackend by remember { mutableStateOf(PoseBackendKind.YOLO) }
    var voiceEnabled by remember { mutableStateOf(prefs.getBoolean("voice_enabled", true)) }
    var lastPracticeReport by remember { mutableStateOf<PracticeReport?>(null) }
    var userSettings by remember {
        mutableStateOf(
            UserSettings(
                name = prefs.getString("user_name", "Fencer") ?: "Fencer",
                age = prefs.getString("age", "25") ?: "25",
                handedness = prefs.getString("handedness", "right") ?: "right",
                heightCm = prefs.getString("height_cm", "180") ?: "180",
                weightKg = prefs.getString("weight_kg", "70") ?: "70",
                processingProfile = prefs.getString("processing_profile", "Balanced") ?: "Balanced",
                useGeminiSummary = prefs.getBoolean(
                    "use_ai_summary",
                    prefs.getBoolean("use_gemini_summary", false)
                ),
                llmProvider = LlmProviderKind.fromLabel(prefs.getString("llm_provider", LlmProviderKind.GEMINI.label)),
                geminiApiKey = prefs.getString("gemini_api_key", "") ?: "",
                openAiApiKey = prefs.getString("openai_api_key", "") ?: "",
                onlyFocusedErrors = prefs.getBoolean("only_focused_errors", false),
                emphasizedErrors = prefs.getStringSet("emphasized_errors", emptySet())?.toSet().orEmpty(),
                mutedErrors = prefs.getStringSet("muted_errors", emptySet())?.toSet().orEmpty(),
                language = normalizePlaybookLanguage(prefs.getString("language", PlaybookRepository.DEFAULT_LANGUAGE)),
                autoExportVideo = prefs.getBoolean("auto_export_video", false),
                showSkeletonOverlay = prefs.getBoolean("show_skeleton_overlay", true)
            )
        )
    }
    val playbookRepository = remember(userSettings.language) {
        com.aifencingcoach.runtime.PlaybookRepository(context, userSettings.language)
    }
    val geminiAgent = remember(playbookRepository) {
        com.aifencingcoach.runtime.GeminiAgent(playbookRepository)
    }
    val analysisManager = remember(geminiAgent) {
        com.aifencingcoach.runtime.AnalysisManager(context, sessionRepository, geminiAgent)
    }
    
    val lastApiError by geminiAgent.lastApiError.collectAsState()

    fun saveSettings() {
        prefs.edit()
            .putString("user_name", userSettings.name.ifBlank { "Fencer" })
            .putString("age", userSettings.age.ifBlank { "25" })
            .putString("handedness", userSettings.handedness)
            .putString("height_cm", userSettings.heightCm.ifBlank { "180" })
            .putString("weight_kg", userSettings.weightKg.ifBlank { "70" })
            .putString("processing_profile", userSettings.processingProfile)
            .putBoolean("use_ai_summary", userSettings.useGeminiSummary)
            .putBoolean("use_gemini_summary", userSettings.useGeminiSummary)
            .putString("llm_provider", userSettings.llmProvider.label)
            .putString("gemini_api_key", userSettings.geminiApiKey.trim())
            .putString("openai_api_key", userSettings.openAiApiKey.trim())
            .putBoolean("only_focused_errors", userSettings.onlyFocusedErrors)
            .putStringSet("emphasized_errors", userSettings.emphasizedErrors)
            .putStringSet("muted_errors", userSettings.mutedErrors)
            .putBoolean("voice_enabled", voiceEnabled)
            .putString("language", normalizePlaybookLanguage(userSettings.language))
            .putBoolean("auto_export_video", userSettings.autoExportVideo)
            .putBoolean("show_skeleton_overlay", userSettings.showSkeletonOverlay)
            .apply()
    }

    var backPressedOnce by remember { mutableStateOf(false) }
    LaunchedEffect(backPressedOnce) {
        if (backPressedOnce) {
            kotlinx.coroutines.delay(2000)
            backPressedOnce = false
        }
    }

    val activity = context as? android.app.Activity
    BackHandler {
        if (appScreen == AppScreen.HOME) {
            if (backPressedOnce) {
                activity?.finish()
            } else {
                backPressedOnce = true
                android.widget.Toast.makeText(context, "再次點按即可退出", android.widget.Toast.LENGTH_SHORT).show()
            }
        } else if (appScreen == AppScreen.SESSION_DETAIL) {
            appScreen = AppScreen.HISTORY
        } else {
            saveSettings()
            appScreen = AppScreen.HOME
        }
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
            onHistory = {
                selectedHistoryUser = null
                appScreen = AppScreen.HISTORY
            },
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
            voiceEnabled = voiceEnabled,
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
            useGeminiSummary = userSettings.useGeminiSummary,
            llmConfig = userSettings.llmConfig(),
            selectedUser = selectedHistoryUser,
            onSelectedUserChange = { selectedHistoryUser = it },
            onBack = {
                selectedHistoryUser = null
                appScreen = AppScreen.HOME
            },
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
            lastApiError = lastApiError,
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
                
                // Save immediately with the playbook summary, then let AI update history in the background.
                persistenceScope.launch {
                    val fallbackResult = geminiAgent.generateSummaryResult(
                        trainingMode = trainingMode.label,
                        targetSide = targetSide.label,
                        actionCounts = report.actionCounts,
                        cuesFired = report.cueTimeline,
                        userSettingsName = userSettings.name,
                        preferGemini = false,
                        llmConfig = userSettings.llmConfig()
                    )
                    val fallbackSummary = fallbackResult.text

                    val sessionId = sessionRepository.savePracticeReport(
                        report = report,
                        cuesFired = report.cueTimeline,
                        llmSummary = fallbackSummary.ifEmpty { "No summary available." },
                        userName = userSettings.name,
                        playbookLanguage = userSettings.language
                    )

                    if (userSettings.useGeminiSummary) {
                        val geminiResult = geminiAgent.generateSummaryResult(
                            trainingMode = trainingMode.label,
                            targetSide = targetSide.label,
                            actionCounts = report.actionCounts,
                            cuesFired = report.cueTimeline,
                            userSettingsName = userSettings.name,
                            preferGemini = true,
                            llmConfig = userSettings.llmConfig()
                        )
                        sessionRepository.updateSummary(sessionId, geminiResult.text)
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
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(PageBackground)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(ScreenPadding),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            ScreenHeader(
                title = "AI Fencing Coach",
                subtitle = ""
            )
            UserBox(userName = userSettings.name)
    
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
        }

        Text(
            text = "v1.1.1",
            color = MutedText,
            fontSize = 11.sp,
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .padding(16.dp)
        )
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
        Column(
            modifier = Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            ScreenHeader(
                title = "Realtime",
                subtitle = "",
                onBack = onBack
            )
            UserBox(userName = userSettings.name)
            
            ExpandableMenuPanel(title = "Current Setup", initiallyExpanded = true) {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    StatusPill(trainingMode.label, AccentGreen)
                    StatusPill(poseBackend.label, AccentGreen)
                    StatusPill("target ${targetSide.label}", AccentGreen)
                    StatusPill(if (voiceEnabled) "voice on" else "voice off", if (voiceEnabled) AccentGreen else MutedText)
                }

            }
            
            ExpandableMenuPanel(title = "Feedback", initiallyExpanded = true) {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    StatusPill("${userSettings.emphasizedErrors.size} emphasized", AccentGold)
                    StatusPill("${userSettings.mutedErrors.size} muted", AccentCoral)
                    StatusPill(if (userSettings.onlyFocusedErrors) "focused only" else "all cues", MutedText)
                }

            }
            
            ExpandableMenuPanel(title = "Feedback Focus", initiallyExpanded = true) {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    availableErrorsForMode(trainingMode).forEach { option ->
                        val color = when (option.key) {
                            in userSettings.mutedErrors -> AccentCoral
                            in userSettings.emphasizedErrors -> AccentGold
                            else -> MutedText
                        }
                        StatusPill(localizedErrorLabel(option, userSettings.language), color)
                    }
                }
            }
        }
        
        // Start Button at the bottom
        HudButton(
            text = "Start Camera",
            selected = true,
            modifier = Modifier.fillMaxWidth().padding(top = 16.dp),
            onClick = onStart
        )
    }
}

@Composable
@OptIn(ExperimentalLayoutApi::class)
private fun PostgameScreen(
    userSettings: UserSettings,
    trainingMode: TrainingMode,
    targetSide: TargetSide,
    poseBackend: PoseBackendKind,
    voiceEnabled: Boolean,
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
    val lastSummaryStatus by analysisManager.lastSummaryStatus.collectAsStateWithLifecycle()
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
            subtitle = "",
            onBack = onBack
        )
        UserBox(userName = userSettings.name)

        ExpandableMenuPanel("Analysis Setup", initiallyExpanded = true) {
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                StatusPill(trainingMode.label, AccentGreen)
                StatusPill(poseBackend.label, AccentGreen)
                StatusPill("target ${targetSide.label}", AccentGreen)
                StatusPill(if (voiceEnabled) "voice on" else "voice off", if (voiceEnabled) AccentGreen else MutedText)
            }
        }
        
        ExpandableMenuPanel("Video Queue", initiallyExpanded = true) {
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
                                    llmConfig = userSettings.llmConfig(),
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

        ExpandableMenuPanel("Latest Session", initiallyExpanded = true) {
            val reportToShow = lastReport ?: lastPracticeReport
            if (reportToShow == null) {
                Text("No practice report yet", color = MutedText, fontSize = BodyTextSize)
            } else {
                PracticeReportSummary(
                    report = reportToShow,
                    summary = if (lastReport != null) lastSummary else null,
                    summaryStatus = if (lastReport != null) lastSummaryStatus else null
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
private fun IgSectionHeader(title: String) {
    Text(
        text = title,
        color = Color(0xFFAAAAAA),
        fontSize = 14.sp,
        fontWeight = FontWeight.Normal,
        modifier = Modifier.padding(start = 16.dp, end = 16.dp, top = 24.dp, bottom = 8.dp)
    )
}

@Composable
private fun IgSettingRow(label: String, content: @Composable () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text = label, color = Color.White, fontSize = 16.sp)
        content()
    }
}

@Composable
private fun IgTextFieldRow(label: String, value: String, isPassword: Boolean = false, onValueChange: (String) -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text = label, color = Color.White, fontSize = 16.sp)
        BasicTextField(
            value = value,
            onValueChange = onValueChange,
            textStyle = androidx.compose.ui.text.TextStyle(color = Color(0xFFAAAAAA), fontSize = 16.sp, textAlign = TextAlign.End),
            visualTransformation = if (isPassword) PasswordVisualTransformation() else androidx.compose.ui.text.input.VisualTransformation.None,
            keyboardOptions = KeyboardOptions(keyboardType = if (isPassword) KeyboardType.Password else KeyboardType.Text),
            modifier = Modifier.weight(1f).padding(start = 16.dp),
            singleLine = true,
            cursorBrush = SolidColor(Color.White)
        )
    }
}

private enum class SettingsSubpage {
    USER_INFORMATION,
    PRACTICE_INFORMATION,
    APP_DEFAULT,
    FEEDBACK_CONTROL,
    HAND_SELECTION,
    LANGUAGE_SELECTION,
    TRAINING_MODE_SELECTION,
    TARGET_SIDE_SELECTION,
    CV_MODEL_SELECTION,
    EFFICIENCY_SELECTION,
    LLM_MODEL_SELECTION
}

@Composable
private fun IgCategoryRow(label: String, value: String? = null, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 16.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text = label, color = Color.White, fontSize = 16.sp)
        Row(verticalAlignment = Alignment.CenterVertically) {
            if (value != null) {
                Text(text = value, color = Color(0xFFAAAAAA), fontSize = 16.sp, modifier = Modifier.padding(end = 8.dp))
            }
            Icon(
                imageVector = Icons.Default.KeyboardArrowRight,
                contentDescription = "Go",
                tint = Color(0xFFAAAAAA)
            )
        }
    }
}

@Composable
private fun SelectionList(
    items: List<String>,
    selected: String,
    labelFor: (String) -> String = { it },
    onSelect: (String) -> Unit
) {
    Column {
        items.forEach { item ->
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { onSelect(item) }
                    .padding(horizontal = 16.dp, vertical = 16.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(text = labelFor(item), color = Color.White, fontSize = 16.sp)
                if (item == selected) {
                    Icon(
                        imageVector = Icons.Default.Check,
                        contentDescription = "Selected",
                        tint = Color(0xFF0095F6)
                    )
                }
            }
            androidx.compose.material3.Divider(color = Color(0xFF262626), thickness = 1.dp)
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
    lastApiError: String?,
    onUserSettings: (UserSettings) -> Unit,
    onTrainingMode: (TrainingMode) -> Unit,
    onTargetSide: (TargetSide) -> Unit,
    onVoiceEnabled: (Boolean) -> Unit,
    onPoseBackend: (PoseBackendKind) -> Unit,
    onBack: () -> Unit
) {
    val subpageStack = androidx.compose.runtime.remember { androidx.compose.runtime.mutableStateListOf<SettingsSubpage>() }
    val currentSubpage = subpageStack.lastOrNull()
    val onBackSubpage: () -> Unit = {
        if (subpageStack.isNotEmpty()) subpageStack.removeLast() else onBack()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
            .verticalScroll(rememberScrollState())
    ) {
        if (currentSubpage == null) {
            ScreenHeader(
                title = "User Settings",
                subtitle = "",
                onBack = onBack
            )

            IgCategoryRow("User Information") { subpageStack.add(SettingsSubpage.USER_INFORMATION) }
            IgCategoryRow("Practice Information") { subpageStack.add(SettingsSubpage.PRACTICE_INFORMATION) }
            IgCategoryRow("App Default") { subpageStack.add(SettingsSubpage.APP_DEFAULT) }
            IgCategoryRow("Feedback Control") { subpageStack.add(SettingsSubpage.FEEDBACK_CONTROL) }
        } else {
            ScreenHeader(
                title = when (currentSubpage) {
                    SettingsSubpage.USER_INFORMATION -> "User Information"
                    SettingsSubpage.PRACTICE_INFORMATION -> "Practice Information"
                    SettingsSubpage.APP_DEFAULT -> "App Default"
                    SettingsSubpage.FEEDBACK_CONTROL -> "Feedback Control"
                    SettingsSubpage.HAND_SELECTION -> "Hand"
                    SettingsSubpage.LANGUAGE_SELECTION -> "Language"
                    SettingsSubpage.TRAINING_MODE_SELECTION -> "Training Mode"
                    SettingsSubpage.TARGET_SIDE_SELECTION -> "Target Side"
                    SettingsSubpage.CV_MODEL_SELECTION -> "CV Model"
                    SettingsSubpage.EFFICIENCY_SELECTION -> "Efficiency"
                    SettingsSubpage.LLM_MODEL_SELECTION -> "LLM Model"
                },
                subtitle = "",
                onBack = onBackSubpage
            )

            when (currentSubpage) {
                SettingsSubpage.USER_INFORMATION -> {
                    IgTextFieldRow("Name", userSettings.name) { onUserSettings(userSettings.copy(name = it)) }
                    IgTextFieldRow("Age", userSettings.age) { onUserSettings(userSettings.copy(age = it.filter(Char::isDigit).take(3))) }
                    IgTextFieldRow("Height cm", userSettings.heightCm) { onUserSettings(userSettings.copy(heightCm = it.filter(Char::isDigit).take(3))) }
                    IgTextFieldRow("Weight kg", userSettings.weightKg) { onUserSettings(userSettings.copy(weightKg = it.filter(Char::isDigit).take(3))) }
                    IgCategoryRow("Hand", userSettings.handedness) { subpageStack.add(SettingsSubpage.HAND_SELECTION) }
                    IgCategoryRow("Language", PlaybookRepository.displayName(userSettings.language)) {
                        subpageStack.add(SettingsSubpage.LANGUAGE_SELECTION)
                    }
                }
                SettingsSubpage.PRACTICE_INFORMATION -> {
                    IgCategoryRow("Training Mode", trainingMode.label) { subpageStack.add(SettingsSubpage.TRAINING_MODE_SELECTION) }
                    IgCategoryRow("Target Side", targetSide.label) { subpageStack.add(SettingsSubpage.TARGET_SIDE_SELECTION) }
                }
                SettingsSubpage.APP_DEFAULT -> {
                    IgCategoryRow("CV Model", poseBackend.label) { subpageStack.add(SettingsSubpage.CV_MODEL_SELECTION) }
                    IgCategoryRow("Efficiency", userSettings.processingProfile) { subpageStack.add(SettingsSubpage.EFFICIENCY_SELECTION) }
                    IgSettingRow("Voice Cues") {
                        Switch(
                            checked = voiceEnabled,
                            onCheckedChange = { onVoiceEnabled(it) },
                            colors = SwitchDefaults.colors(checkedThumbColor = Color.White, checkedTrackColor = Color(0xFF0095F6))
                        )
                    }
                    IgCategoryRow("LLM Model", userSettings.llmProvider.label) { subpageStack.add(SettingsSubpage.LLM_MODEL_SELECTION) }
                    val keyStatus = when (userSettings.llmProvider) {
                        LlmProviderKind.PLAYBOOK -> "offline"
                        LlmProviderKind.GEMINI -> if (userSettings.geminiApiKey.isBlank()) {
                            if (com.aifencingcoach.BuildConfig.GEMINI_API_KEY.isBlank()) "missing bundled key" else "bundled key"
                        } else "user key"
                        LlmProviderKind.OPENAI -> if (userSettings.openAiApiKey.isBlank()) {
                            if (com.aifencingcoach.BuildConfig.OPENAI_API_KEY.isBlank()) "missing bundled key" else "bundled key"
                        } else "user key"
                    }
                    IgSettingRow("API Key Status") {
                        StatusPill(keyStatus, if (keyStatus.contains("missing")) AccentCoral else AccentGreen)
                    }
                    
                    if (userSettings.llmProvider != LlmProviderKind.PLAYBOOK) {
                        val isOpenAi = userSettings.llmProvider == LlmProviderKind.OPENAI
                        IgTextFieldRow(
                            label = "${userSettings.llmProvider.label} API key",
                            value = if (isOpenAi) userSettings.openAiApiKey else userSettings.geminiApiKey,
                            isPassword = true,
                            onValueChange = { value ->
                                onUserSettings(
                                    if (isOpenAi) userSettings.copy(openAiApiKey = value.trim())
                                    else userSettings.copy(geminiApiKey = value.trim())
                                )
                            }
                        )
                    }
                    
                    if (lastApiError != null && userSettings.llmProvider != LlmProviderKind.PLAYBOOK) {
                        Text(
                            "AI unavailable: $lastApiError",
                            color = AccentCoral,
                            fontSize = 12.sp,
                            modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)
                        )
                    }

                    IgSettingRow("Auto-export annotated video") {
                        Switch(
                            checked = userSettings.autoExportVideo,
                            onCheckedChange = { onUserSettings(userSettings.copy(autoExportVideo = it)) },
                            colors = SwitchDefaults.colors(checkedThumbColor = Color.White, checkedTrackColor = Color(0xFF0095F6))
                        )
                    }
                    IgSettingRow("Show skeleton overlay") {
                        Switch(
                            checked = userSettings.showSkeletonOverlay,
                            onCheckedChange = { onUserSettings(userSettings.copy(showSkeletonOverlay = it)) },
                            colors = SwitchDefaults.colors(checkedThumbColor = Color.White, checkedTrackColor = Color(0xFF0095F6))
                        )
                    }
                }
                SettingsSubpage.FEEDBACK_CONTROL -> {
                    Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)) {
                        CheckboxLine(
                            label = "Only focused errors",
                            checked = userSettings.onlyFocusedErrors,
                            onCheckedChange = { onUserSettings(userSettings.copy(onlyFocusedErrors = it)) }
                        )
                        Spacer(Modifier.height(16.dp))
                        availableErrorsForMode(trainingMode).forEach { option ->
                            ErrorPreferenceRow(
                                option = option,
                                language = userSettings.language,
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
                }
                SettingsSubpage.HAND_SELECTION -> {
                    SelectionList(listOf("left", "right"), userSettings.handedness) {
                        onUserSettings(userSettings.copy(handedness = it))
                        onBackSubpage()
                    }
                }
                SettingsSubpage.LANGUAGE_SELECTION -> {
                    SelectionList(
                        items = listOf(PlaybookRepository.DEFAULT_LANGUAGE, PlaybookRepository.ENGLISH_LANGUAGE),
                        selected = userSettings.language,
                        labelFor = { PlaybookRepository.displayName(it) }
                    ) {
                        onUserSettings(userSettings.copy(language = normalizePlaybookLanguage(it)))
                        onBackSubpage()
                    }
                }
                SettingsSubpage.TRAINING_MODE_SELECTION -> {
                    SelectionList(TrainingMode.entries.map { it.label }, trainingMode.label) {
                        onTrainingMode(TrainingMode.fromLabel(it))
                        onBackSubpage()
                    }
                }
                SettingsSubpage.TARGET_SIDE_SELECTION -> {
                    SelectionList(listOf("left", "right"), targetSide.label) {
                        onTargetSide(TargetSide.fromLabel(it))
                        onBackSubpage()
                    }
                }
                SettingsSubpage.CV_MODEL_SELECTION -> {
                    SelectionList(PoseBackendKind.entries.map { it.label }, poseBackend.label) {
                        onPoseBackend(PoseBackendKind.fromLabel(it))
                        onBackSubpage()
                    }
                }
                SettingsSubpage.EFFICIENCY_SELECTION -> {
                    SelectionList(ProcessingProfiles, userSettings.processingProfile) {
                        onUserSettings(userSettings.copy(processingProfile = it))
                        onBackSubpage()
                    }
                }
                SettingsSubpage.LLM_MODEL_SELECTION -> {
                    SelectionList(LlmProviderKind.entries.map { it.label }, userSettings.llmProvider.label) {
                        onUserSettings(userSettings.copy(llmProvider = LlmProviderKind.fromLabel(it)))
                        onBackSubpage()
                    }
                }
            }
        }

        Spacer(Modifier.height(32.dp))
    }
}

@Composable
private fun ExpandableMenuPanel(
    title: String,
    modifier: Modifier = Modifier,
    initiallyExpanded: Boolean = false,
    content: @Composable () -> Unit
) {
    var expanded by remember { mutableStateOf(initiallyExpanded) }
    Column(
        modifier = modifier
            .background(PanelColor, RoundedCornerShape(8.dp))
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clickable { expanded = !expanded }
                .padding(10.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = title,
                color = Color.White,
                fontSize = 15.sp,
                fontWeight = FontWeight.Bold
            )
            Icon(
                imageVector = if (expanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                contentDescription = if (expanded) "Collapse" else "Expand",
                tint = Color.White
            )
        }
        if (expanded) {
            Column(modifier = Modifier.padding(start = 10.dp, end = 10.dp, bottom = 10.dp)) {
                content()
            }
        }
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
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 16.dp, end = 16.dp, top = 18.dp, bottom = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        if (onBack != null) {
            androidx.compose.material3.IconButton(onClick = onBack) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = Color.White)
            }
            Spacer(Modifier.width(8.dp))
        }
        Column(modifier = Modifier.weight(1f)) {
            Text(title, color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)
            if (subtitle.isNotBlank()) {
                Spacer(Modifier.height(2.dp))
                Text(subtitle, color = AccentGreen, fontSize = 10.sp, fontWeight = FontWeight.SemiBold)
            }
        }
    }
}

@Composable
private fun UserBox(userName: String) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier.size(36.dp).background(AccentGreen, androidx.compose.foundation.shape.CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = userName.take(1).uppercase(),
                color = Color.White,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )
        }
        Spacer(Modifier.width(12.dp))
        Text(
            text = userName.ifBlank { "Fencer" },
            color = Color.White,
            fontSize = 15.sp,
            fontWeight = FontWeight.SemiBold
        )
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

private fun summaryStatusText(result: CoachingSummaryResult): String =
    when (result.source) {
        SummarySource.GEMINI -> "Gemini summary ready."
        SummarySource.OPENAI -> "OpenAI summary ready."
        SummarySource.PLAYBOOK -> "Playbook summary ready."
        SummarySource.DISABLED -> result.errorMessage ?: "AI is not configured; showing playbook summary."
        SummarySource.FAILED -> "AI summary failed: ${summaryErrorLabel(result.errorMessage)}."
    }

private fun summaryErrorLabel(errorMessage: String?): String {
    val lower = errorMessage.orEmpty().lowercase()
    return when {
        lower.contains("quota") || lower.contains("rate") -> "API quota or rate limit reached"
        lower.contains("api key") || lower.contains("permission") || lower.contains("unauthorized") -> "API key rejected"
        lower.contains("model") || lower.contains("not found") -> "model unavailable"
        lower.contains("network") ||
            lower.contains("timeout") ||
            lower.contains("unable to resolve host") ||
            lower.contains("unknown host") ||
            lower.contains("failed to connect") ||
            lower.contains("connectexception") ||
            lower.contains("socket") -> "No internet connection"
        else -> "see logcat"
    }
}

private fun summaryStatusColor(status: String): Color =
    when {
        status.contains("ready", ignoreCase = true) && (
            status.contains("Gemini", ignoreCase = true) ||
                status.contains("OpenAI", ignoreCase = true)
            ) -> AccentGreen
        status.contains("Generating", ignoreCase = true) -> AccentGold
        status.contains("failed", ignoreCase = true) || status.contains("not configured", ignoreCase = true) -> AccentCoral
        else -> MutedText
    }

@Composable
@OptIn(ExperimentalLayoutApi::class)
private fun PracticeReportSummary(
    report: PracticeReport,
    summary: String? = null,
    summaryStatus: String? = null
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        if (!summaryStatus.isNullOrBlank()) {
            StatusPill(summaryStatus, summaryStatusColor(summaryStatus))
        }
        SessionFeedbackPanel(report = report, summary = summary)
    }
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
    language: String,
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
            Text(localizedErrorLabel(option, language), color = Color.White, fontSize = BodyTextSize, fontWeight = FontWeight.SemiBold)
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
    onSpeak: (String, String) -> Unit
) {
    var analysisPaused by remember { mutableStateOf(false) }
    var frameState by remember { mutableStateOf(CoachFrameState()) }
    var reviewReport by remember { mutableStateOf<PracticeReport?>(null) }
    var reviewSummary by remember { mutableStateOf<String?>(null) }
    var reviewSummaryStatus by remember { mutableStateOf<String?>(null) }
    var resetToken by remember { mutableStateOf(0) }
    var sessionRecorded by remember { mutableStateOf(false) }
    val llmConfig = userSettings.llmConfig()

    LaunchedEffect(reviewReport, userSettings.useGeminiSummary, llmConfig) {
        val report = reviewReport
        if (report != null) {
            val fallbackResult = geminiAgent.generateSummaryResult(
                trainingMode = trainingMode.label,
                targetSide = targetSide.label,
                actionCounts = report.actionCounts,
                cuesFired = report.cueTimeline,
                userSettingsName = userSettings.name,
                preferGemini = false,
                llmConfig = llmConfig
            )
            reviewSummary = fallbackResult.text
            reviewSummaryStatus = summaryStatusText(fallbackResult)

            if (userSettings.useGeminiSummary) {
                val providerLabel = geminiAgent.providerLabel(llmConfig)
                reviewSummaryStatus = if (geminiAgent.isEnabled(llmConfig)) {
                    "Generating $providerLabel summary..."
                } else {
                    "$providerLabel is not configured; showing playbook summary."
                }
                if (geminiAgent.isEnabled(llmConfig)) {
                    val geminiResult = geminiAgent.generateSummaryResult(
                        trainingMode = trainingMode.label,
                        targetSide = targetSide.label,
                        actionCounts = report.actionCounts,
                        cuesFired = report.cueTimeline,
                        userSettingsName = userSettings.name,
                        preferGemini = true,
                        llmConfig = llmConfig
                    )
                    reviewSummary = geminiResult.text
                    reviewSummaryStatus = summaryStatusText(geminiResult)
                }
            }
        } else {
            reviewSummary = null
            reviewSummaryStatus = null
        }
    }

    val pipeline = remember(
        poseBackend,
        targetSide,
        trainingMode,
        userSettings.emphasizedErrors,
        userSettings.mutedErrors,
        userSettings.onlyFocusedErrors,
        userSettings.language,
        resetToken
    ) {
        LiveCoachPipeline(
            context = context,
            poseBackendKind = poseBackend,
            targetSide = targetSide,
            trainingMode = trainingMode,
            focusErrors = userSettings.emphasizedErrors,
            muteErrors = userSettings.mutedErrors,
            onlyErrors = if (userSettings.onlyFocusedErrors) userSettings.emphasizedErrors else emptySet(),
            playbookLanguage = userSettings.language
        )
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
                        if (voiceEnabled && cue != null) {
                            onSpeak(cue.shortCue.ifBlank { cue.message }, userSettings.language)
                        }
                    }
                )
                if (userSettings.showSkeletonOverlay) {
                    SkeletonOverlay(state = displayedState)
                }

                // Action pill overlay (top center, like Flutter)
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 28.dp),
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
                        .padding(start = 12.dp, top = 28.dp, end = 12.dp, bottom = 12.dp)
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
                llmSummary = reviewSummary,
                summaryStatus = reviewSummaryStatus
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
    llmSummary: String? = null,
    summaryStatus: String? = null
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
            PracticeReportSummary(
                report = report,
                summary = llmSummary,
                summaryStatus = summaryStatus
            )
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
    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
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
private fun HudButton(text: String, selected: Boolean, modifier: Modifier = Modifier, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        modifier = modifier,
        shape = RoundedCornerShape(6.dp),
        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = if (selected) Color(0xFFD7FF5F) else Color(0xFF263039),
            contentColor = if (selected) Color(0xFF101418) else Color.White
        )
    ) {
        Text(text = text, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
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

private fun ttsLocaleForLanguage(language: String): Locale =
    if (normalizePlaybookLanguage(language) == PlaybookRepository.ENGLISH_LANGUAGE) {
        Locale.US
    } else {
        Locale.TAIWAN
    }

private fun modeSummary(mode: TrainingMode): String =
    when (mode) {
        TrainingMode.FOOTWORK -> "Footwork cues prioritize stance, bounce, step width, and center of mass."
        TrainingMode.TARGET_PRACTICE -> "Target practice includes lunge timing, arm extension, and guard position."
        TrainingMode.FREE_BOUTING -> "Free bouting keeps cues broad for mixed movement and opponent context."
    }

private fun modelSummary(model: PoseBackendKind): String =
    when (model) {
        PoseBackendKind.MEDIAPIPE -> "MediaPipe lite provides low-latency single-person pose tracking."
        PoseBackendKind.YOLO -> "YOLO pose is the default bout-oriented model through ONNX Runtime."
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
    "${PlaybookRepository.displayName(settings.language)}, ${settings.handedness}, ${settings.heightCm.ifBlank { "180" }}cm, ${settings.emphasizedErrors.size} focus, ${settings.mutedErrors.size} muted."

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

private fun localizedErrorLabel(option: FeedbackErrorOption, language: String): String =
    if (normalizePlaybookLanguage(language) == PlaybookRepository.ENGLISH_LANGUAGE) {
        option.label
    } else {
        ChineseFeedbackLabels[option.key] ?: option.label
    }

private fun Set<String>.toggle(value: String, enabled: Boolean): Set<String> =
    if (enabled) this + value else this - value

private val ProcessingProfiles = listOf("Balanced", "Fast", "Full Quality")

private val FeedbackErrorOptions = listOf(
    FeedbackErrorOption("foot_before_hand", "Foot before hand", setOf(TrainingMode.TARGET_PRACTICE)),
    FeedbackErrorOption("lunge_overextension", "Lunge overextension", TrainingMode.entries.toSet()),
    FeedbackErrorOption("incomplete_arm_extension", "Incomplete arm extension", setOf(TrainingMode.TARGET_PRACTICE)),
    FeedbackErrorOption("guard_dropped", "Guard dropped", TrainingMode.entries.toSet()),
    FeedbackErrorOption("stance_too_high", "Stance too high", TrainingMode.entries.toSet()),
    FeedbackErrorOption("bounce_excessive", "Bounce excessive", TrainingMode.entries.toSet()),
    FeedbackErrorOption("center_of_mass_in_front", "Center of mass in front", TrainingMode.entries.toSet()),
    FeedbackErrorOption("center_of_mass_leaning_backward", "Center of mass leaning backward", TrainingMode.entries.toSet()),
    FeedbackErrorOption("over_parrying", "Over parrying", TrainingMode.entries.toSet()),
    FeedbackErrorOption("wide_step", "Wide step", TrainingMode.entries.toSet()),
    FeedbackErrorOption("narrow_step", "Narrow step", TrainingMode.entries.toSet()),
    FeedbackErrorOption("hand_too_high", "Hand too high", TrainingMode.entries.toSet())
)

private val ChineseFeedbackLabels = mapOf(
    "foot_before_hand" to "腳先於手",
    "lunge_overextension" to "弓步過度伸展",
    "incomplete_arm_extension" to "手臂伸展不足",
    "guard_dropped" to "護手下掉",
    "stance_too_high" to "姿勢太高",
    "bounce_excessive" to "彈跳過多",
    "center_of_mass_in_front" to "重心過前",
    "center_of_mass_leaning_backward" to "重心後仰",
    "over_parrying" to "格擋過大",
    "wide_step" to "步幅過大",
    "narrow_step" to "步距過窄",
    "hand_too_high" to "手位過高"
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
