package com.aifencingcoach

import android.net.Uri
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.common.MediaItem
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import com.aifencingcoach.runtime.database.FullSessionData
import com.aifencingcoach.runtime.ActionCountItem
import com.aifencingcoach.runtime.PlaybookRepository
import com.aifencingcoach.runtime.PracticeReport
import java.io.File

@Composable
fun SessionFeedbackPanel(
    sessionData: FullSessionData? = null,
    report: PracticeReport? = null,
    summary: String? = null,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val playbookRepository = remember { PlaybookRepository(context) }
    val bgColor = Color(0xFF141420)
    val primaryColor = Color(0xFFFF6600)
    val secondaryColor = Color(0xFF00D4FF)

    val elapsedSeconds = sessionData?.session?.elapsedSeconds?.takeIf { it > 0 }
        ?: report?.elapsedSeconds
        ?: sessionData?.session?.totalActiveSeconds?.toLong()
        ?: 0L
    val activeSeconds = sessionData?.session?.totalActiveSeconds?.toLong() ?: report?.activeSeconds ?: 0L
    val mode = sessionData?.session?.trainingMode ?: report?.trainingMode?.label ?: "Free Bouting"
    val source = sessionData?.session?.source ?: if (report != null) "Realtime" else "Session"
    val inferenceCount = sessionData?.session?.inferenceCount?.takeIf { it > 0 }
        ?: report?.inferenceCount
        ?: sessionData?.actionCounts?.sumOf { it.count }
        ?: 0L
    val cues = sessionData?.cues ?: report?.cueTimeline ?: emptyList()
    val uiCues = mapToUICues(cues, playbookRepository)
    val cueCount = sessionData?.session?.cueCount?.takeIf { it > 0 }
        ?: report?.cueCount
        ?: uiCues.size.toLong()
    val topAction = report?.topAction
        ?: sessionData?.session?.topAction?.takeIf { it.isNotBlank() && it != "Idle" }
        ?: sessionData?.actionCounts?.maxByOrNull { it.count }?.actionName
        ?: "Idle"
    val actionItems = report?.actionCounts ?: sessionData?.actionCounts?.toActionItems().orEmpty()
    val coachSummary = (sessionData?.session?.geminiSummary ?: summary)
        ?.takeUnless { it.isBlank() || it == "No summary available." }

    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            AssistChip(
                onClick = {},
                label = { Text(mode, color = secondaryColor, fontWeight = FontWeight.SemiBold) },
                colors = AssistChipDefaults.assistChipColors(containerColor = secondaryColor.copy(alpha = 0.15f)),
                border = AssistChipDefaults.assistChipBorder(borderColor = secondaryColor.copy(alpha = 0.5f), enabled = true)
            )
            AssistChip(
                onClick = {},
                label = { Text(source, color = primaryColor, fontWeight = FontWeight.SemiBold) },
                colors = AssistChipDefaults.assistChipColors(containerColor = primaryColor.copy(alpha = 0.15f)),
                border = AssistChipDefaults.assistChipBorder(borderColor = primaryColor.copy(alpha = 0.5f), enabled = true)
            )
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = bgColor)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(14.dp),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                SummaryMetric("time", formatPanelSeconds(elapsedSeconds))
                SummaryMetric("active", "${activeSeconds}s")
                SummaryMetric("checks", inferenceCount.toString())
                SummaryMetric("cues", cueCount.toString())
                SummaryMetric("top", topAction)
            }
        }

        if (actionItems.isNotEmpty()) {
            ActionBreakdownPanel(actionItems)
        }

        if (uiCues.isNotEmpty()) {
            ErrorBreakdownPanel(uiCues)
        }

        if (!coachSummary.isNullOrEmpty()) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = bgColor),
                elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
            ) {
                Column(Modifier.padding(20.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .background(primaryColor, RoundedCornerShape(50))
                        )
                        Spacer(Modifier.width(8.dp))
                        Text("Coach Summary", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 18.sp)
                    }
                    Spacer(Modifier.height(12.dp))
                    Text(coachSummary, color = Color.White.copy(alpha = 0.85f), fontSize = 15.sp, lineHeight = 22.sp)
                }
            }
        }

        if (uiCues.isNotEmpty()) {
            CollapsibleTimelineCueList(uiCues)
        }
    }
}

data class UICue(
    val timestampStr: String,
    val errorKey: String,
    val errorName: String,
    val shortCue: String,
    val diagnosis: String,
    val practice: String
)

fun mapToUICues(cues: List<Any>, playbookRepository: PlaybookRepository? = null): List<UICue> {
    return cues.mapNotNull {
        when (it) {
            is com.aifencingcoach.runtime.CueHistoryItem -> {
                val entry = playbookRepository?.getEntry(it.errorKey)
                UICue(
                    timestampStr = String.format("%.1fs", it.frameIndex / 30f),
                    errorKey = it.errorKey,
                    errorName = entry?.label ?: it.label,
                    shortCue = it.message.ifBlank { entry?.shortCue.orEmpty() },
                    diagnosis = it.diagnosis.ifBlank { entry?.diagnosis.orEmpty() },
                    practice = it.practice.ifBlank { entry?.practice ?: it.message }
                )
            }
            is com.aifencingcoach.runtime.database.CueHistoryEntity -> {
                val entry = playbookRepository?.getEntry(it.errorKey)
                UICue(
                    timestampStr = String.format("%.1fs", it.timestamp),
                    errorKey = it.errorKey,
                    errorName = entry?.label ?: it.errorName,
                    shortCue = entry?.shortCue ?: it.practiceSuggestion,
                    diagnosis = entry?.diagnosis.orEmpty(),
                    practice = it.practiceSuggestion.ifBlank { entry?.practice.orEmpty() }
                )
            }
            else -> null
        }
    }
}

@Composable
fun ActionPieChart(cues: List<Any>) {
    val context = LocalContext.current
    val playbookRepository = remember { PlaybookRepository(context) }
    ErrorBreakdownPanel(mapToUICues(cues, playbookRepository))
}

@Composable
private fun ErrorBreakdownPanel(cues: List<UICue>) {
    val errorCounts = cues.groupBy { it.errorName }.mapValues { it.value.size }.toList().sortedByDescending { it.second }
    if (errorCounts.isEmpty()) return

    val totalCues = errorCounts.sumOf { it.second }
    val colors = listOf(Color(0xFFFF3D3D), Color(0xFFFF9100), Color(0xFFFFE066), Color(0xFF00D4FF), Color(0xFFE040FB))

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF141420))
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text("Mistake Breakdown", color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(12.dp))

            Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Box(modifier = Modifier.size(84.dp).padding(8.dp)) {
                    Canvas(modifier = Modifier.fillMaxSize()) {
                        var startAngle = -90f
                        errorCounts.forEachIndexed { index, pair ->
                            val sweepAngle = (pair.second.toFloat() / totalCues) * 360f
                            val color = colors[index % colors.size]
                            drawArc(
                                color = color,
                                startAngle = startAngle,
                                sweepAngle = sweepAngle,
                                useCenter = false,
                                style = Stroke(width = 18f)
                            )
                            startAngle += sweepAngle
                        }
                    }
                }

                Spacer(modifier = Modifier.width(12.dp))

                Column(modifier = Modifier.weight(1f)) {
                    errorCounts.take(4).forEachIndexed { index, pair ->
                        val color = colors[index % colors.size]
                        val percentage = (pair.second.toFloat() / totalCues * 100).toInt()
                        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 2.dp)) {
                            Box(modifier = Modifier.size(8.dp).background(color, RoundedCornerShape(50)))
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(pair.first, color = Color.White, fontSize = 11.sp, modifier = Modifier.weight(1f), maxLines = 1)
                            Text("${pair.second} | $percentage%", color = Color.Gray, fontSize = 11.sp)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ActionBreakdownPanel(actions: List<ActionCountItem>) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF141420))
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Action Percentage", color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)
            actions.take(6).forEach { action ->
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text(action.action, color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                        Text("${action.count}  |  ${action.percent}%", color = Color.Gray, fontSize = 13.sp)
                    }
                    LinearProgressIndicator(
                        progress = { (action.percent / 100f).coerceIn(0f, 1f) },
                        modifier = Modifier.fillMaxWidth(),
                        color = Color(0xFF00D4FF),
                        trackColor = Color.White.copy(alpha = 0.10f)
                    )
                }
            }
        }
    }
}

@Composable
private fun SummaryMetric(label: String, value: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.widthIn(min = 54.dp)) {
        Text(value, color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold, maxLines = 1)
        Text(label, color = Color.Gray, fontSize = 11.sp)
    }
}

@Composable
fun CollapsibleTimelineCues(cues: List<Any>) {
    val context = LocalContext.current
    val playbookRepository = remember { PlaybookRepository(context) }
    CollapsibleTimelineCueList(mapToUICues(cues, playbookRepository))
}

@Composable
private fun CollapsibleTimelineCueList(uiCues: List<UICue>) {
    if (uiCues.isEmpty()) return

    var expanded by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier.fillMaxWidth().clickable { expanded = !expanded },
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF141420))
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("Timeline Cues (${uiCues.size})", color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                Icon(
                    imageVector = if (expanded) Icons.Filled.KeyboardArrowUp else Icons.Filled.KeyboardArrowDown,
                    contentDescription = "Expand",
                    tint = Color.White
                )
            }

            AnimatedVisibility(visible = expanded) {
                Column(modifier = Modifier.padding(top = 16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    uiCues.forEach { cue ->
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Box(
                                modifier = Modifier
                                    .background(Color(0xFFFF6600).copy(alpha = 0.2f), RoundedCornerShape(6.dp))
                                    .padding(horizontal = 8.dp, vertical = 4.dp)
                            ) {
                                Text(cue.timestampStr, color = Color(0xFFFF6600), fontWeight = FontWeight.Bold, fontSize = 12.sp)
                            }
                            Column {
                                Text(cue.errorName, color = Color(0xFFFF3D3D), fontWeight = FontWeight.Bold, fontSize = 14.sp)
                                if (cue.shortCue.isNotBlank()) {
                                    Text(cue.shortCue, color = Color.Gray, fontSize = 12.sp)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

private fun List<com.aifencingcoach.runtime.database.ActionCountEntity>.toActionItems(): List<ActionCountItem> {
    val total = sumOf { it.count }.coerceAtLeast(1L)
    return sortedWith(compareByDescending<com.aifencingcoach.runtime.database.ActionCountEntity> { it.count }.thenBy { it.actionName })
        .map {
            ActionCountItem(
                action = it.actionName,
                count = it.count,
                percent = ((it.count * 100L) / total).toInt()
            )
        }
}

private fun formatPanelSeconds(totalSeconds: Long): String {
    val minutes = totalSeconds / 60
    val seconds = totalSeconds % 60
    return if (minutes > 0) "${minutes}m ${seconds}s" else "${seconds}s"
}

@Composable
fun SharedVideoPlayer(videoPath: String) {
    val context = LocalContext.current
    val exoPlayer = remember {
        ExoPlayer.Builder(context).build().apply {
            val uri = Uri.parse(videoPath).takeIf { it.scheme != null } ?: Uri.fromFile(File(videoPath))
            setMediaItem(MediaItem.fromUri(uri))
            prepare()
        }
    }

    DisposableEffect(Unit) {
        onDispose { exoPlayer.release() }
    }

    Card(
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp),
        modifier = Modifier.fillMaxWidth().aspectRatio(16f / 9f) // Prevents huge black bars
    ) {
        AndroidView(
            factory = { ctx ->
                PlayerView(ctx).apply {
                    player = exoPlayer
                }
            },
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black)
        )
    }
}
