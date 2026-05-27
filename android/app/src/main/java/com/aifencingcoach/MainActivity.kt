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
import com.aifencingcoach.runtime.Skeleton
import com.aifencingcoach.runtime.TargetSide
import com.aifencingcoach.runtime.TrainingMode
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

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

    var targetSide by remember { mutableStateOf(TargetSide.LEFT) }
    var trainingMode by remember { mutableStateOf(TrainingMode.FREE_BOUTING) }
    var poseBackend by remember { mutableStateOf(PoseBackendKind.MEDIAPIPE) }
    var voiceEnabled by remember { mutableStateOf(true) }
    var analysisPaused by remember { mutableStateOf(false) }
    var frameState by remember { mutableStateOf(CoachFrameState()) }
    var resetToken by remember { mutableStateOf(0) }

    val pipeline = remember(poseBackend, targetSide, trainingMode, resetToken) {
        LiveCoachPipeline(context, poseBackend, targetSide, trainingMode)
    }

    DisposableEffect(pipeline) {
        onDispose { pipeline.close() }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        val displayedState = if (analysisPaused) {
            frameState.copy(state = "PAUSED", cue = "Analysis paused")
        } else {
            frameState
        }
        CameraPreview(
            pipeline = pipeline,
            analysisPaused = analysisPaused,
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
            onPoseBackend = { poseBackend = it },
            onTargetSide = { targetSide = it },
            onTrainingMode = { trainingMode = it },
            onVoiceEnabled = { voiceEnabled = it },
            onAnalysisPaused = { analysisPaused = it },
            onReset = {
                pipeline.reset()
                resetToken += 1
            }
        )
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
    onPoseBackend: (PoseBackendKind) -> Unit,
    onTargetSide: (TargetSide) -> Unit,
    onTrainingMode: (TrainingMode) -> Unit,
    onVoiceEnabled: (Boolean) -> Unit,
    onAnalysisPaused: (Boolean) -> Unit,
    onReset: () -> Unit
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
                HudButton(text = "Reset", selected = false, onClick = onReset)
            }
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
