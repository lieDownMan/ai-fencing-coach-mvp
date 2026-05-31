package com.aifencingcoach

import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.common.MediaItem
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import com.aifencingcoach.runtime.database.FullSessionData
import java.text.SimpleDateFormat
import java.util.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SessionDetailScreen(
    sessionData: FullSessionData,
    onBack: () -> Unit
) {
    val formatter = SimpleDateFormat("yyyy/MM/dd HH:mm", Locale.getDefault())
    val dateString = formatter.format(Date(sessionData.session.timestamp))

    val bgColor = Color(0xFF0A0A0F)
    val cardColor = Color(0xFF141420)
    val primaryColor = Color(0xFFFF6600)
    val secondaryColor = Color(0xFF00D4FF)
    val errorColor = Color(0xFFFF3D3D)

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(dateString, color = Color.White, fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = primaryColor)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = bgColor)
            )
        },
        containerColor = bgColor
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 24.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.spacedBy(20.dp)
        ) {
            // Video Player
            val videoPath = sessionData.session.exportedVideoPath
            if (videoPath != null) {
                item {
                    Text(
                        "Annotated Video",
                        color = Color.White,
                        fontWeight = FontWeight.Bold,
                        fontSize = 18.sp,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    VideoPlayerCard(videoPath)
                }
            }

            // Overview Stats
            item {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    AssistChip(
                        onClick = {},
                        label = { Text(sessionData.session.trainingMode, color = secondaryColor, fontWeight = FontWeight.SemiBold) },
                        colors = AssistChipDefaults.assistChipColors(containerColor = secondaryColor.copy(alpha = 0.15f)),
                        border = AssistChipDefaults.assistChipBorder(borderColor = secondaryColor.copy(alpha = 0.5f), enabled = true)
                    )
                    AssistChip(
                        onClick = {},
                        label = { Text("Target ${sessionData.session.targetSide}", color = primaryColor, fontWeight = FontWeight.SemiBold) },
                        colors = AssistChipDefaults.assistChipColors(containerColor = primaryColor.copy(alpha = 0.15f)),
                        border = AssistChipDefaults.assistChipBorder(borderColor = primaryColor.copy(alpha = 0.5f), enabled = true)
                    )
                    AssistChip(
                        onClick = {},
                        label = { Text("${sessionData.session.totalActiveSeconds.toInt()}s Active", color = Color.White, fontWeight = FontWeight.SemiBold) },
                        colors = AssistChipDefaults.assistChipColors(containerColor = Color.White.copy(alpha = 0.1f)),
                        border = AssistChipDefaults.assistChipBorder(borderColor = Color.White.copy(alpha = 0.3f), enabled = true)
                    )
                }
            }

            // AI Summary
            val summary = sessionData.session.geminiSummary
            if (!summary.isNullOrEmpty()) {
                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(16.dp),
                        colors = CardDefaults.cardColors(containerColor = cardColor),
                        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
                    ) {
                        Column(Modifier.padding(20.dp)) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Box(
                                    modifier = Modifier
                                        .size(8.dp)
                                        .background(primaryColor, RoundedCornerShape(50))
                                )
                                Spacer(Modifier.width(8.dp))
                                Text("Coach Summary", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 20.sp)
                            }
                            Spacer(Modifier.height(12.dp))
                            Text(summary, color = Color.White.copy(alpha = 0.8f), fontSize = 15.sp, lineHeight = 22.sp)
                        }
                    }
                }
            }

            // Timeline of Cues
            if (sessionData.cues.isNotEmpty()) {
                item {
                    Text(
                        "Timeline Cues",
                        color = Color.White,
                        fontWeight = FontWeight.Bold,
                        fontSize = 18.sp,
                        modifier = Modifier.padding(top = 8.dp, bottom = 4.dp)
                    )
                }
                items(sessionData.cues) { cue ->
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(12.dp),
                        colors = CardDefaults.cardColors(containerColor = cardColor)
                    ) {
                        Row(
                            Modifier.padding(16.dp),
                            horizontalArrangement = Arrangement.spacedBy(16.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Box(
                                modifier = Modifier
                                    .background(
                                        brush = Brush.verticalGradient(
                                            colors = listOf(primaryColor.copy(alpha = 0.8f), primaryColor.copy(alpha = 0.2f))
                                        ),
                                        shape = RoundedCornerShape(8.dp)
                                    )
                                    .padding(horizontal = 12.dp, vertical = 8.dp)
                            ) {
                                Text(
                                    text = "${cue.timestamp}s",
                                    color = Color.White,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                            Column {
                                Text(cue.errorName, color = errorColor, fontWeight = FontWeight.Bold, fontSize = 16.sp)
                                Spacer(Modifier.height(4.dp))
                                Text(cue.practiceSuggestion, color = Color.White.copy(alpha = 0.6f), fontSize = 14.sp)
                            }
                        }
                    }
                }
            } else {
                item {
                    Text("No cues recorded during this session.", color = Color.White.copy(alpha = 0.5f), fontSize = 15.sp)
                }
            }
            
            item {
                Spacer(Modifier.height(32.dp)) // Bottom padding
            }
        }
    }
}

@Composable
fun VideoPlayerCard(videoPath: String) {
    val context = LocalContext.current
    val exoPlayer = remember {
        ExoPlayer.Builder(context).build().apply {
            setMediaItem(MediaItem.fromUri(Uri.parse(videoPath)))
            prepare()
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            exoPlayer.release()
        }
    }

    Card(
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        AndroidView(
            factory = { ctx ->
                PlayerView(ctx).apply {
                    player = exoPlayer
                }
            },
            modifier = Modifier
                .fillMaxWidth()
                .height(280.dp)
                .background(Color.Black)
        )
    }
}
