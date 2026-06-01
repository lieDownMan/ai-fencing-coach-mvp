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
import androidx.compose.ui.graphics.Color
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
                    SharedVideoPlayer(videoPath)
                }
            }

            item {
                SessionFeedbackPanel(sessionData = sessionData)
            }
            
            item {
                Spacer(Modifier.height(32.dp)) // Bottom padding
            }
        }
    }
}
