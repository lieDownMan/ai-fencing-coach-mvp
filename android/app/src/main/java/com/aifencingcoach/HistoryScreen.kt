package com.aifencingcoach

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.aifencingcoach.runtime.database.FullSessionData
import com.aifencingcoach.runtime.database.SessionEntity
import com.aifencingcoach.runtime.database.SessionRepository
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HistoryScreen(
    sessionRepo: SessionRepository,
    onBack: () -> Unit,
    onSessionSelected: (FullSessionData) -> Unit
) {
    var sessions by remember { mutableStateOf<List<SessionEntity>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }
    val scope = rememberCoroutineScope()

    LaunchedEffect(Unit) {
        sessions = sessionRepo.getRecentSessions()
        isLoading = false
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("訓練歷史紀錄", color = Color.White) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Filled.ArrowBack, contentDescription = "返回", tint = Color.White)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color(0xFF1E262F)
                )
            )
        },
        containerColor = Color(0xFF101418)
    ) { padding ->
        if (isLoading) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator(color = Color(0xFF2E6DD1))
            }
        } else if (sessions.isEmpty()) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("尚無訓練紀錄", color = Color.Gray, fontSize = 18.sp)
            }
        } else {
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                items(sessions) { session ->
                    SessionCard(
                        session = session,
                        onClick = {
                            scope.launch {
                                val fullData = sessionRepo.getSessionDetails(session.id)
                                if (fullData != null) {
                                    onSessionSelected(fullData)
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
private fun SessionCard(session: SessionEntity, onClick: () -> Unit) {
    val formatter = SimpleDateFormat("yyyy/MM/dd HH:mm", Locale.getDefault())
    val dateString = formatter.format(Date(session.timestamp))
    
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF1E262F)),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = dateString,
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    fontSize = 16.sp
                )
                Text(
                    text = "${session.totalActiveSeconds.toInt()} 秒",
                    color = Color.Gray,
                    fontSize = 14.sp
                )
            }
            Spacer(modifier = Modifier.height(8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                AssistChip(
                    onClick = { },
                    label = { Text(session.trainingMode, color = Color.White) },
                    colors = AssistChipDefaults.assistChipColors(containerColor = Color(0xFF2E6DD1).copy(alpha = 0.2f))
                )
                AssistChip(
                    onClick = { },
                    label = { Text(session.targetSide, color = Color.White) },
                    colors = AssistChipDefaults.assistChipColors(containerColor = Color(0xFF2E6DD1).copy(alpha = 0.2f))
                )
            }
            if (session.geminiSummary != null) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "✓ 已生成 AI 總結",
                    color = Color(0xFFE57373), // Reusing existing color as an accent
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}
