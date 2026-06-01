package com.aifencingcoach

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
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
import com.aifencingcoach.runtime.database.FullSessionData
import com.aifencingcoach.runtime.database.SessionEntity
import com.aifencingcoach.runtime.database.SessionRepository
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*

import com.aifencingcoach.runtime.GeminiAgent

@Composable
fun HistoryScreen(
    sessionRepo: SessionRepository,
    geminiAgent: GeminiAgent,
    userName: String,
    onBack: () -> Unit,
    onSessionSelected: (FullSessionData) -> Unit
) {
    var selectedUser by remember { mutableStateOf<String?>(userName) }

    if (selectedUser == null || selectedUser!!.isEmpty()) {
        UserSelectionScreen(
            sessionRepo = sessionRepo,
            onBack = onBack,
            onUserSelected = { selectedUser = it }
        )
    } else {
        UserHistoryScreen(
            userName = selectedUser!!,
            sessionRepo = sessionRepo,
            geminiAgent = geminiAgent,
            onBack = {
                if (userName.isEmpty()) selectedUser = null else onBack()
            },
            onSessionSelected = onSessionSelected
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun UserSelectionScreen(
    sessionRepo: SessionRepository,
    onBack: () -> Unit,
    onUserSelected: (String) -> Unit
) {
    var users by remember { mutableStateOf<List<String>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }

    LaunchedEffect(Unit) {
        users = sessionRepo.getDistinctUsers()
        isLoading = false
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Select User", color = Color.White) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = Color.White)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = Color(0xFF1E262F))
            )
        },
        containerColor = Color(0xFF101418)
    ) { padding ->
        if (isLoading) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator(color = Color(0xFF2E6DD1))
            }
        } else if (users.isEmpty()) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("No users found.", color = Color.Gray, fontSize = 18.sp)
            }
        } else {
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                items(users) { userName ->
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { onUserSelected(userName) },
                        colors = CardDefaults.cardColors(containerColor = Color(0xFF1E262F)),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(20.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(40.dp)
                                    .background(Color(0xFF2E6DD1), RoundedCornerShape(20.dp)),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(userName.take(1).uppercase(), color = Color.White, fontWeight = FontWeight.Bold)
                            }
                            Spacer(Modifier.width(16.dp))
                            Text(userName, color = Color.White, fontSize = 18.sp, fontWeight = FontWeight.SemiBold)
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun UserHistoryScreen(
    userName: String,
    sessionRepo: SessionRepository,
    geminiAgent: GeminiAgent,
    onBack: () -> Unit,
    onSessionSelected: (FullSessionData) -> Unit
) {
    var sessions by remember { mutableStateOf<List<SessionEntity>>(emptyList()) }
    var fullRecentSessions by remember { mutableStateOf<List<FullSessionData>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }
    val scope = rememberCoroutineScope()

    LaunchedEffect(userName) {
        sessions = sessionRepo.getSessionsByUser(userName)
        val recentFull = sessions.take(5).mapNotNull { sessionRepo.getSessionDetails(it.id) }
        fullRecentSessions = recentFull
        isLoading = false
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("$userName's History", color = Color.White) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = Color.White)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = Color(0xFF1E262F))
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
                Text("No sessions found.", color = Color.Gray, fontSize = 18.sp)
            }
        } else {
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                item {
                    UserRecapCard(fullRecentSessions, geminiAgent, userName)
                }

                item {
                    Text(
                        text = "Session History",
                        color = Color.White,
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(top = 8.dp)
                    )
                }

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
private fun UserRecapCard(
    recentSessions: List<FullSessionData>,
    geminiAgent: GeminiAgent,
    userName: String
) {
    if (recentSessions.isEmpty()) return

    val allCues = recentSessions.flatMap { it.cues }
    var aiAnalysis by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(allCues) {
        val fallback = buildRecapFallback(recentSessions, geminiAgent)
        aiAnalysis = if (geminiAgent.isEnabled && allCues.isNotEmpty()) {
            "Generating AI Analysis..."
            geminiAgent.generateImprovementAnalysis(
                userName = userName,
                recentErrorsText = buildRecentErrorPromptData(recentSessions, geminiAgent),
                fallback = fallback
            )
        } else {
            fallback
        }
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF141420)),
        shape = RoundedCornerShape(12.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Text("Last ${recentSessions.size} Sessions Recap", color = Color.White, fontSize = 18.sp, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(16.dp))

            if (aiAnalysis != null) {
                Text(
                    text = aiAnalysis!!,
                    color = Color(0xFFB6C2CC),
                    fontSize = 15.sp,
                    lineHeight = 22.sp
                )
                Spacer(Modifier.height(20.dp))
            }

            if (allCues.isNotEmpty()) {
                ActionPieChart(cues = allCues)
            }
        }
    }
}

private fun buildRecapFallback(
    recentSessions: List<FullSessionData>,
    geminiAgent: GeminiAgent
): String {
    val allCues = recentSessions.flatMap { it.cues }
    if (allCues.isEmpty()) {
        return "最近五場沒有明顯重複錯誤，保持目前節奏。"
    }

    val topError = allCues
        .groupingBy { it.errorKey }
        .eachCount()
        .maxByOrNull { it.value }
        ?: return "最近五場沒有明顯重複錯誤，保持目前節奏。"

    val entry = geminiAgent.playbookEntry(topError.key)
    val label = entry?.label ?: allCues.firstOrNull { it.errorKey == topError.key }?.errorName ?: topError.key
    val lastTwoCount = recentSessions.take(2).sumOf { session ->
        session.cues.count { it.errorKey == topError.key }
    }
    val previousThreeCount = recentSessions.drop(2).take(3).sumOf { session ->
        session.cues.count { it.errorKey == topError.key }
    }
    val improvementPercent = improvementPercent(
        recentCount = lastTwoCount,
        recentSessions = minOf(2, recentSessions.size),
        previousCount = previousThreeCount,
        previousSessions = minOf(3, (recentSessions.size - 2).coerceAtLeast(0))
    )
    val trend = when {
        recentSessions.size < 3 -> "資料還少，先把這個問題當作近期主軸。"
        improvementPercent > 0 -> "近兩場平均比前三場改善 $improvementPercent%。"
        improvementPercent < 0 -> "近兩場平均比前三場增加 ${-improvementPercent}%，需要優先處理。"
        else -> "近兩場和前三場大致持平。"
    }
    val details = buildList {
        add("近期重點：$label，共 ${topError.value} 次。$trend")
        if (recentSessions.size >= 3) {
            add("近兩場 $lastTwoCount 次，前三場 $previousThreeCount 次。")
        }
        entry?.diagnosis?.takeIf { it.isNotBlank() }?.let { add("診斷：$it") }
        entry?.practice?.takeIf { it.isNotBlank() }?.let { add("練習建議：$it") }
    }
    return details.joinToString("\n")
}

private fun buildRecentErrorPromptData(
    recentSessions: List<FullSessionData>,
    geminiAgent: GeminiAgent
): String {
    return recentSessions.mapIndexed { index, session ->
        val total = session.cues.size.coerceAtLeast(1)
        val counts = session.cues
            .groupingBy { it.errorKey }
            .eachCount()
            .entries
            .sortedByDescending { it.value }
            .joinToString(", ") { (key, count) ->
                val label = geminiAgent.playbookEntry(key)?.label ?: key
                val percent = (count * 100) / total
                "$label($key): $count/$total, $percent%"
            }
        "Session ${index + 1}: ${counts.ifBlank { "no errors" }}"
    }.joinToString("\n") + "\n\nImprovement:\n" + buildImprovementPromptLine(recentSessions, geminiAgent) +
        "\n\nFocus details:\n" + geminiAgent.allPlaybookEntries()
        .entries
        .joinToString("\n") { (key, entry) ->
            "$key: ${entry.label}; diagnosis=${entry.diagnosis}; practice=${entry.practice}"
        }
}

private fun buildImprovementPromptLine(
    recentSessions: List<FullSessionData>,
    geminiAgent: GeminiAgent
): String {
    val allCues = recentSessions.flatMap { it.cues }
    val topError = allCues.groupingBy { it.errorKey }.eachCount().maxByOrNull { it.value } ?: return "No repeated error."
    val lastTwoCount = recentSessions.take(2).sumOf { session -> session.cues.count { it.errorKey == topError.key } }
    val previousThreeCount = recentSessions.drop(2).take(3).sumOf { session -> session.cues.count { it.errorKey == topError.key } }
    val percent = improvementPercent(
        recentCount = lastTwoCount,
        recentSessions = minOf(2, recentSessions.size),
        previousCount = previousThreeCount,
        previousSessions = minOf(3, (recentSessions.size - 2).coerceAtLeast(0))
    )
    val label = geminiAgent.playbookEntry(topError.key)?.label ?: topError.key
    return "$label: last_two_count=$lastTwoCount, previous_three_count=$previousThreeCount, improvement_percent=$percent"
}

private fun improvementPercent(
    recentCount: Int,
    recentSessions: Int,
    previousCount: Int,
    previousSessions: Int
): Int {
    if (recentSessions <= 0 || previousSessions <= 0 || previousCount <= 0) return 0
    val recentAverage = recentCount.toFloat() / recentSessions.toFloat()
    val previousAverage = previousCount.toFloat() / previousSessions.toFloat()
    if (previousAverage <= 0f) return 0
    return (((previousAverage - recentAverage) / previousAverage) * 100f).toInt()
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
                    text = "${session.totalActiveSeconds.toInt()}s",
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
                    label = { Text(session.source, color = Color.White) },
                    colors = AssistChipDefaults.assistChipColors(containerColor = Color(0xFF2E6DD1).copy(alpha = 0.2f))
                )
            }
            if (session.geminiSummary != null) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "✓ AI Summary",
                    color = Color(0xFFE57373),
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}
