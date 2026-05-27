package com.aifencingcoach

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
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
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
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
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
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

private enum class AppScreen {
    MENU,
    COACH
}

private data class UserSettings(
    val name: String = "Fencer",
    val handedness: String = "right",
    val heightCm: String = "180"
)

class MainActivity : ComponentActivity() {
    private var tts: TextToSpeech? = null
    private var ttsReady = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        tts = TextToSpeech(this) { status ->
            ttsReady = status == TextToSpeech.SUCCESS
            if (ttsReady) {
                tts?.language = Locale.US
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
    var appScreen by remember { mutableStateOf(AppScreen.MENU) }
    var targetSide by remember { mutableStateOf(TargetSide.LEFT) }
    var trainingMode by remember { mutableStateOf(TrainingMode.FREE_BOUTING) }
    var poseBackend by remember { mutableStateOf(PoseBackendKind.MEDIAPIPE) }
    var voiceEnabled by remember { mutableStateOf(prefs.getBoolean("voice_enabled", true)) }
    var userSettings by remember {
        mutableStateOf(
            UserSettings(
                name = prefs.getString("user_name", "Fencer") ?: "Fencer",
                handedness = prefs.getString("handedness", "right") ?: "right",
                heightCm = prefs.getString("height_cm", "180") ?: "180"
            )
        )
    }

    fun saveSettings() {
        prefs.edit()
            .putString("user_name", userSettings.name.ifBlank { "Fencer" })
            .putString("handedness", userSettings.handedness)
            .putString("height_cm", userSettings.heightCm.ifBlank { "180" })
            .putBoolean("voice_enabled", voiceEnabled)
            .apply()
    }

    when (appScreen) {
        AppScreen.MENU -> MainMenuScreen(
            trainingMode = trainingMode,
            poseBackend = poseBackend,
            targetSide = targetSide,
            voiceEnabled = voiceEnabled,
            userSettings = userSettings,
            onTrainingMode = { trainingMode = it },
            onPoseBackend = { poseBackend = it },
            onTargetSide = { targetSide = it },
            onVoiceEnabled = { voiceEnabled = it },
            onUserSettings = { userSettings = it },
            onStart = {
                saveSettings()
                appScreen = AppScreen.COACH
            }
        )
        AppScreen.COACH -> CoachScreen(
            context = context,
            targetSide = targetSide,
            trainingMode = trainingMode,
            poseBackend = poseBackend,
            voiceEnabled = voiceEnabled,
            userSettings = userSettings,
            onTargetSide = { targetSide = it },
            onTrainingMode = { trainingMode = it },
            onPoseBackend = { poseBackend = it },
            onVoiceEnabled = {
                voiceEnabled = it
                prefs.edit().putBoolean("voice_enabled", it).apply()
            },
            onBackToMenu = {
                saveSettings()
                appScreen = AppScreen.MENU
            },
            onSpeak = onSpeak
        )
    }
}

@Composable
@OptIn(ExperimentalLayoutApi::class)
private fun MainMenuScreen(
    trainingMode: TrainingMode,
    poseBackend: PoseBackendKind,
    targetSide: TargetSide,
    voiceEnabled: Boolean,
    userSettings: UserSettings,
    onTrainingMode: (TrainingMode) -> Unit,
    onPoseBackend: (PoseBackendKind) -> Unit,
    onTargetSide: (TargetSide) -> Unit,
    onVoiceEnabled: (Boolean) -> Unit,
    onUserSettings: (UserSettings) -> Unit,
    onStart: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF101418))
            .padding(22.dp),
        verticalArrangement = Arrangement.SpaceBetween
    ) {
        Column {
            Text(
                text = "AI Fencing Coach",
                color = Color.White,
                fontSize = 30.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(Modifier.height(6.dp))
            Text(
                text = "${trainingMode.label}  |  ${poseBackend.label}  |  ${targetSide.label}",
                color = Color(0xFFD7FF5F),
                fontSize = 15.sp,
                fontWeight = FontWeight.SemiBold
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            MenuPanel("Mode", Modifier.weight(1f)) {
                SegmentedGroup(
                    labels = TrainingMode.entries.map { it.label },
                    selected = trainingMode.label,
                    onSelected = { onTrainingMode(TrainingMode.fromLabel(it)) }
                )
                Spacer(Modifier.height(10.dp))
                Text(
                    text = modeSummary(trainingMode),
                    color = Color(0xFFB6C2CC),
                    fontSize = 13.sp
                )
            }

            MenuPanel("Model", Modifier.weight(1f)) {
                SegmentedGroup(
                    labels = PoseBackendKind.entries.map { it.label },
                    selected = poseBackend.label,
                    onSelected = { onPoseBackend(PoseBackendKind.fromLabel(it)) }
                )
                Spacer(Modifier.height(10.dp))
                Text(
                    text = modelSummary(poseBackend),
                    color = Color(0xFFB6C2CC),
                    fontSize = 13.sp
                )
            }

            MenuPanel("User", Modifier.weight(1f)) {
                OutlinedTextField(
                    value = userSettings.name,
                    onValueChange = { onUserSettings(userSettings.copy(name = it)) },
                    label = { Text("Name", color = Color(0xFFB6C2CC)) },
                    textStyle = androidx.compose.ui.text.TextStyle(color = Color.White),
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(Modifier.height(8.dp))
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    SegmentedGroup(
                        labels = listOf("left", "right"),
                        selected = userSettings.handedness,
                        onSelected = { onUserSettings(userSettings.copy(handedness = it)) }
                    )
                    OutlinedTextField(
                        value = userSettings.heightCm,
                        onValueChange = { value ->
                            onUserSettings(userSettings.copy(heightCm = value.filter(Char::isDigit).take(3)))
                        },
                        label = { Text("Height cm", color = Color(0xFFB6C2CC)) },
                        textStyle = androidx.compose.ui.text.TextStyle(color = Color.White),
                        singleLine = true,
                        modifier = Modifier.width(132.dp)
                    )
                }
            }
        }

        FlowRow(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color(0xAA182028), RoundedCornerShape(8.dp))
                .padding(12.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            SegmentedGroup(
                labels = listOf("left", "right"),
                selected = targetSide.label,
                onSelected = { onTargetSide(TargetSide.fromLabel(it)) }
            )
            HudButton(
                text = if (voiceEnabled) "Voice on" else "Voice off",
                selected = voiceEnabled,
                onClick = { onVoiceEnabled(!voiceEnabled) }
            )
            HudButton(text = "Start", selected = true, onClick = onStart)
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
            .background(Color(0xAA182028), RoundedCornerShape(8.dp))
            .padding(14.dp)
    ) {
        Text(
            text = title,
            color = Color.White,
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold
        )
        Spacer(Modifier.height(10.dp))
        content()
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
    onTargetSide: (TargetSide) -> Unit,
    onTrainingMode: (TrainingMode) -> Unit,
    onPoseBackend: (PoseBackendKind) -> Unit,
    onVoiceEnabled: (Boolean) -> Unit,
    onBackToMenu: () -> Unit,
    onSpeak: (String) -> Unit
) {
    var analysisPaused by remember { mutableStateOf(false) }
    var frameState by remember { mutableStateOf(CoachFrameState()) }
    var reviewReport by remember { mutableStateOf<PracticeReport?>(null) }
    var resetToken by remember { mutableStateOf(0) }

    val pipeline = remember(poseBackend, targetSide, trainingMode, resetToken) {
        LiveCoachPipeline(context, poseBackend, targetSide, trainingMode)
    }

    DisposableEffect(pipeline) {
        onDispose { pipeline.close() }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        val isReviewing = reviewReport != null
        val displayedState = when {
            isReviewing -> frameState.copy(state = "REVIEW", cue = "Practice complete")
            analysisPaused -> frameState.copy(state = "PAUSED", cue = "Analysis paused")
            else -> frameState
        }
        CameraPreview(
            pipeline = pipeline,
            analysisPaused = analysisPaused || isReviewing,
            onFrameState = { state, cue ->
                frameState = state
                if (voiceEnabled && cue != null) onSpeak(cue.message)
            }
        )
        SkeletonOverlay(state = displayedState)
        HudPanel(
            state = displayedState,
            poseBackend = poseBackend,
            targetSide = targetSide,
            trainingMode = trainingMode,
            voiceEnabled = voiceEnabled,
            analysisPaused = analysisPaused,
            userSettings = userSettings,
            onPoseBackend = onPoseBackend,
            onTargetSide = onTargetSide,
            onTrainingMode = onTrainingMode,
            onVoiceEnabled = onVoiceEnabled,
            onAnalysisPaused = { analysisPaused = it },
            onFinishPractice = {
                reviewReport = pipeline.practiceReport()
                analysisPaused = true
            },
            onReset = {
                pipeline.reset()
                resetToken += 1
            },
            onBackToMenu = onBackToMenu
        )
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
                    frameState = CoachFrameState()
                    pipeline.reset()
                    resetToken += 1
                }
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
            scaleType = PreviewView.ScaleType.FILL_CENTER
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
                        val (state, cue) = pipeline.process(imageProxy)
                        mainExecutor.execute { onFrameState(state, cue) }
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
    Canvas(modifier = Modifier.fillMaxSize()) {
        fun drawSkeleton(skeleton: Skeleton?, color: Color) {
            if (skeleton == null) return
            val scaleX = size.width / state.frameWidth.coerceAtLeast(1)
            val scaleY = size.height / state.frameHeight.coerceAtLeast(1)

            fun offset(name: String): Offset? {
                val point = skeleton[name] ?: return null
                return Offset(point.x * scaleX, point.y * scaleY)
            }

            for ((a, b) in SkeletonEdges) {
                val start = offset(a)
                val end = offset(b)
                if (start != null && end != null) {
                    drawLine(color, start, end, strokeWidth = 5f, cap = StrokeCap.Round)
                }
            }
            for (point in skeleton.values) {
                drawCircle(color, radius = 6f, center = Offset(point.x * scaleX, point.y * scaleY))
            }
        }

        drawSkeleton(state.opponentSkeleton, Color(0x88FFFFFF))
        drawSkeleton(state.targetSkeleton, Color(0xFFD7FF5F))
    }
}

@Composable
@OptIn(ExperimentalLayoutApi::class)
private fun HudPanel(
    state: CoachFrameState,
    poseBackend: PoseBackendKind,
    targetSide: TargetSide,
    trainingMode: TrainingMode,
    voiceEnabled: Boolean,
    analysisPaused: Boolean,
    userSettings: UserSettings,
    onPoseBackend: (PoseBackendKind) -> Unit,
    onTargetSide: (TargetSide) -> Unit,
    onTrainingMode: (TrainingMode) -> Unit,
    onVoiceEnabled: (Boolean) -> Unit,
    onAnalysisPaused: (Boolean) -> Unit,
    onFinishPractice: () -> Unit,
    onReset: () -> Unit,
    onBackToMenu: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(18.dp),
        verticalArrangement = Arrangement.SpaceBetween
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.Top
        ) {
            Column(
                modifier = Modifier
                    .weight(1f)
                    .background(Color(0xAA101418), RoundedCornerShape(8.dp))
                    .padding(16.dp)
            ) {
                Text(
                    text = state.cue.ifBlank { if (state.ready) "Find stance" else "Model assets missing" },
                    color = Color.White,
                    fontSize = 28.sp,
                    fontWeight = FontWeight.Bold
                )
                Spacer(Modifier.height(6.dp))
                Text(
                    text = "${state.poseBackend.label}  |  ${state.state}  |  ${state.action} ${(state.confidence * 100f).toInt()}%",
                    color = Color(0xFFD7FF5F),
                    fontSize = 16.sp
                )
                Text(
                    text = "${userSettings.name.ifBlank { "Fencer" }}  |  ${userSettings.handedness}  |  ${userSettings.heightCm.ifBlank { "180" }}cm",
                    color = Color(0xFFB6C2CC),
                    fontSize = 13.sp
                )
                if (state.tracking.warmupFramesRemaining > 0 && state.state == "ACTIVE") {
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = "warmup ${FenceNetWarmup - state.tracking.warmupFramesRemaining}/$FenceNetWarmup",
                        color = Color(0xFFB6C2CC),
                        fontSize = 13.sp
                    )
                }
                CueStack(state = state)
            }

            Column(
                modifier = Modifier
                    .widthIn(min = 160.dp, max = 260.dp)
                    .background(Color(0xAA101418), RoundedCornerShape(8.dp))
                    .padding(12.dp),
                horizontalAlignment = Alignment.End
            ) {
                Text(
                    text = "pose ${state.metrics.poseMs}ms  model ${state.metrics.classifierMs}ms",
                    color = Color.White,
                    fontSize = 13.sp
                )
                Text(
                    text = "total ${state.metrics.totalMs}ms  fps ${state.metrics.fps.toInt()}",
                    color = Color(0xFFB6C2CC),
                    fontSize = 13.sp
                )
                Text(
                    text = "target ${state.tracking.lockState.uppercase(Locale.US)}  poses ${state.tracking.detectionCount}",
                    color = if (state.tracking.targetInterpolated) Color(0xFFFFC857) else Color(0xFFD7FF5F),
                    fontSize = 13.sp
                )
                Text(
                    text = "buffer ${state.tracking.bufferFill}/$FenceNetWarmup  drops ${state.metrics.droppedFrames}",
                    color = Color(0xFFB6C2CC),
                    fontSize = 13.sp
                )
                Text(
                    text = "session ${formatSeconds(state.session.elapsedSeconds)}  inf ${state.session.inferenceCount}  cues ${state.session.cueCount}",
                    color = Color(0xFFB6C2CC),
                    fontSize = 13.sp
                )
            }
        }

        FlowRow(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color(0xAA101418), RoundedCornerShape(8.dp))
                .padding(10.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            SegmentedGroup(
                labels = TrainingMode.entries.map { it.label },
                selected = trainingMode.label,
                onSelected = { onTrainingMode(TrainingMode.fromLabel(it)) }
            )
            SegmentedGroup(
                labels = PoseBackendKind.entries.map { it.label },
                selected = poseBackend.label,
                onSelected = { onPoseBackend(PoseBackendKind.fromLabel(it)) }
            )
            SegmentedGroup(
                labels = listOf("left", "right"),
                selected = targetSide.label,
                onSelected = { onTargetSide(TargetSide.fromLabel(it)) }
            )
            Row(verticalAlignment = Alignment.CenterVertically) {
                HudButton(
                    text = if (analysisPaused) "Resume" else "Pause",
                    selected = analysisPaused,
                    onClick = { onAnalysisPaused(!analysisPaused) }
                )
                Spacer(Modifier.width(8.dp))
                HudButton(
                    text = if (voiceEnabled) "Voice on" else "Voice off",
                    selected = voiceEnabled,
                    onClick = { onVoiceEnabled(!voiceEnabled) }
                )
                Spacer(Modifier.width(8.dp))
                HudButton(text = "Finish", selected = false, onClick = onFinishPractice)
                Spacer(Modifier.width(8.dp))
                HudButton(text = "Reset", selected = false, onClick = onReset)
                Spacer(Modifier.width(8.dp))
                HudButton(text = "Menu", selected = false, onClick = onBackToMenu)
            }
        }
    }
}

@Composable
@OptIn(ExperimentalLayoutApi::class)
private fun PostPracticeReview(
    report: PracticeReport,
    onResume: () -> Unit,
    onNewSession: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xDD05080B))
            .padding(22.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth(0.9f)
                .background(Color(0xF0141A20), RoundedCornerShape(8.dp))
                .padding(18.dp)
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
                        fontSize = 26.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = "${report.trainingMode.label}  |  ${report.poseBackend.label}  |  ${report.targetSide.label}",
                        color = Color(0xFFD7FF5F),
                        fontSize = 14.sp
                    )
                }
                Row {
                    HudButton(text = "Resume", selected = false, onClick = onResume)
                    Spacer(Modifier.width(8.dp))
                    HudButton(text = "New", selected = true, onClick = onNewSession)
                }
            }

            Spacer(Modifier.height(14.dp))
            Text(
                text = report.primaryTakeaway,
                color = Color(0xFFFFE066),
                fontSize = 20.sp,
                fontWeight = FontWeight.SemiBold
            )
            Spacer(Modifier.height(14.dp))

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                MetricBlock("time", formatSeconds(report.elapsedSeconds))
                MetricBlock("active", "${report.activeSeconds}s ${report.activePercent}%")
                MetricBlock("checks", report.inferenceCount.toString())
                MetricBlock("cues", report.cueCount.toString())
                MetricBlock("top", report.topAction)
            }

            Spacer(Modifier.height(16.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Actions",
                        color = Color.White,
                        fontSize = 17.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                    Spacer(Modifier.height(8.dp))
                    if (report.actionCounts.isEmpty()) {
                        Text("No model actions yet", color = Color(0xFFB6C2CC), fontSize = 13.sp)
                    } else {
                        report.actionCounts.take(5).forEach { item ->
                            ActionRow(item.action, item.count, item.percent)
                        }
                    }
                }
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Cues",
                        color = Color.White,
                        fontSize = 17.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                    Spacer(Modifier.height(8.dp))
                    if (report.topCues.isEmpty()) {
                        Text("No repeated cues yet", color = Color(0xFFB6C2CC), fontSize = 13.sp)
                    } else {
                        report.topCues.take(5).forEach { cue ->
                            Text(
                                text = "${cue.count}x  ${cue.label}",
                                color = Color(0xFFFFE066),
                                fontSize = 14.sp,
                                fontWeight = FontWeight.SemiBold
                            )
                            Text(
                                text = cue.message,
                                color = Color(0xFFB6C2CC),
                                fontSize = 12.sp
                            )
                            Spacer(Modifier.height(7.dp))
                        }
                    }
                }
            }

            if (report.cueTimeline.isNotEmpty()) {
                Spacer(Modifier.height(10.dp))
                Text(
                    text = "Timeline",
                    color = Color.White,
                    fontSize = 17.sp,
                    fontWeight = FontWeight.SemiBold
                )
                Spacer(Modifier.height(6.dp))
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    report.cueTimeline.take(8).forEach { item ->
                        Text(
                            text = "${item.frameIndex}: ${item.label}",
                            color = Color(0xFFB6C2CC),
                            fontSize = 12.sp,
                            modifier = Modifier
                                .background(Color(0xFF263039), RoundedCornerShape(6.dp))
                                .padding(horizontal = 8.dp, vertical = 5.dp)
                        )
                    }
                }
            }
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
        Text(text = value, color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
    }
}

@Composable
private fun ActionRow(action: String, count: Long, percent: Int) {
    Column(modifier = Modifier.padding(bottom = 8.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(text = action, color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
            Text(text = "$count  $percent%", color = Color(0xFFB6C2CC), fontSize = 13.sp)
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
private fun CueStack(state: CoachFrameState) {
    val cues = state.visualCues.take(3)
    if (cues.isNotEmpty()) {
        Spacer(Modifier.height(10.dp))
        cues.forEachIndexed { index, cue ->
            Text(
                text = if (index == 0) cue.message else "Next: ${cue.message}",
                color = if (index == 0) Color(0xFFFFE066) else Color(0xFFB6C2CC),
                fontSize = if (index == 0) 15.sp else 13.sp,
                fontWeight = if (index == 0) FontWeight.SemiBold else FontWeight.Normal
            )
        }
    }

    if (state.cueHistory.isNotEmpty()) {
        Spacer(Modifier.height(10.dp))
        Text(
            text = state.cueHistory.take(2).joinToString("  |  ") { it.label },
            color = Color(0xFF8DA2B2),
            fontSize = 12.sp
        )
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
        colors = ButtonDefaults.buttonColors(
            containerColor = if (selected) Color(0xFFD7FF5F) else Color(0xFF263039),
            contentColor = if (selected) Color(0xFF101418) else Color.White
        )
    ) {
        Text(text = text, fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
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
