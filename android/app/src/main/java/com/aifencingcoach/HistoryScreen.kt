package com.aifencingcoach

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.aifencingcoach.runtime.database.FullSessionData
import com.aifencingcoach.runtime.database.SessionEntity
import com.aifencingcoach.runtime.database.SessionRepository
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.roundToInt

import com.aifencingcoach.runtime.GeminiAgent
import com.aifencingcoach.runtime.LlmProviderConfig

sealed class RecapConfig {
    data class ByCount(val count: Int) : RecapConfig()
    data class ByTime(val hours: Double) : RecapConfig()
    data class CustomSelection(val sessionIds: Set<Long>) : RecapConfig()
}

@Composable
fun HistoryScreen(
    sessionRepo: SessionRepository,
    geminiAgent: GeminiAgent,
    useGeminiSummary: Boolean,
    llmConfig: LlmProviderConfig,
    selectedUser: String?,
    onSelectedUserChange: (String?) -> Unit,
    onBack: () -> Unit,
    onSessionSelected: (FullSessionData) -> Unit
) {
    if (selectedUser.isNullOrEmpty()) {
        UserSelectionScreen(
            sessionRepo = sessionRepo,
            onBack = onBack,
            onUserSelected = onSelectedUserChange
        )
    } else {
        UserHistoryScreen(
            userName = selectedUser,
            sessionRepo = sessionRepo,
            geminiAgent = geminiAgent,
            useGeminiSummary = useGeminiSummary,
            llmConfig = llmConfig,
            onBack = { onSelectedUserChange(null) },
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
    useGeminiSummary: Boolean,
    llmConfig: LlmProviderConfig,
    onBack: () -> Unit,
    onSessionSelected: (FullSessionData) -> Unit
) {
    var sessions by remember { mutableStateOf<List<SessionEntity>>(emptyList()) }
    var fullRecentSessions by remember { mutableStateOf<List<FullSessionData>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }
    var refreshToken by remember { mutableStateOf(0) }
    var selectionMode by remember { mutableStateOf(false) }
    var selectedSessionIds by remember { mutableStateOf<Set<Long>>(emptySet()) }
    var showDeleteConfirm by remember { mutableStateOf(false) }
    var showCustomCountDialog by remember { mutableStateOf(false) }
    var showCustomTimeDialog by remember { mutableStateOf(false) }
    var recapConfig by remember { mutableStateOf<RecapConfig>(RecapConfig.ByCount(5)) }
    var isHistoryExpanded by remember { mutableStateOf(true) }
    val scope = rememberCoroutineScope()

    LaunchedEffect(userName, refreshToken, recapConfig) {
        isLoading = true
        sessions = sessionRepo.getSessionsByUser(userName)
        val sessionsToRecap = when (val config = recapConfig) {
            is RecapConfig.ByCount -> sessions.take(config.count)
            is RecapConfig.ByTime -> {
                val cutoff = System.currentTimeMillis() - (config.hours * 3600 * 1000).toLong()
                sessions.filter { it.timestamp >= cutoff }
            }
            is RecapConfig.CustomSelection -> sessions.filter { it.id in config.sessionIds }
        }
        val recentFull = sessionsToRecap.mapNotNull { sessionRepo.getSessionDetails(it.id) }
        fullRecentSessions = recentFull
        selectedSessionIds = selectedSessionIds.intersect(sessions.map { it.id }.toSet())
        if (sessions.isEmpty()) {
            selectionMode = false
            selectedSessionIds = emptySet()
        }
        isLoading = false
    }

    if (showDeleteConfirm) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirm = false },
            title = { Text("Delete selected history?") },
            text = { Text("Delete ${selectedSessionIds.size} selected session(s). This cannot be undone.") },
            confirmButton = {
                TextButton(
                    onClick = {
                        val idsToDelete = selectedSessionIds
                        showDeleteConfirm = false
                        scope.launch {
                            idsToDelete.forEach { sessionRepo.deleteSession(it) }
                            selectedSessionIds = emptySet()
                            selectionMode = false
                            refreshToken += 1
                        }
                    }
                ) {
                    Text("Delete", color = Color(0xFFFF6B6B))
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirm = false }) {
                    Text("Cancel")
                }
            }
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                modifier = Modifier.padding(top = 16.dp),
                title = {
                    Text(
                        text = if (selectionMode) "${selectedSessionIds.size} selected" else "$userName's History",
                        color = Color.White
                    )
                },
                navigationIcon = {
                    IconButton(
                        onClick = {
                            if (selectionMode) {
                                selectionMode = false
                                selectedSessionIds = emptySet()
                            } else {
                                onBack()
                            }
                        }
                    ) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = Color.White)
                    }
                },
                actions = {
                    if (selectionMode) {
                        TextButton(
                            onClick = { selectedSessionIds = sessions.map { it.id }.toSet() },
                            enabled = sessions.isNotEmpty()
                        ) {
                            Text("All", color = Color.White)
                        }
                        TextButton(
                            onClick = {
                                recapConfig = RecapConfig.CustomSelection(selectedSessionIds)
                                selectionMode = false
                                selectedSessionIds = emptySet()
                            },
                            enabled = selectedSessionIds.isNotEmpty()
                        ) {
                            Text(
                                text = "Recap",
                                color = if (selectedSessionIds.isNotEmpty()) Color(0xFF64B5F6) else Color.Gray
                            )
                        }
                        TextButton(
                            onClick = { showDeleteConfirm = true },
                            enabled = selectedSessionIds.isNotEmpty()
                        ) {
                            Text(
                                text = "Delete",
                                color = if (selectedSessionIds.isNotEmpty()) Color(0xFFFF6B6B) else Color.Gray
                            )
                        }
                    } else {
                        TextButton(
                            onClick = { selectionMode = true },
                            enabled = sessions.isNotEmpty()
                        ) {
                            Text("Select", color = if (sessions.isNotEmpty()) Color.White else Color.Gray)
                        }
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
                    UserRecapCard(
                        recentSessions = fullRecentSessions,
                        geminiAgent = geminiAgent,
                        useGeminiSummary = useGeminiSummary,
                        llmConfig = llmConfig,
                        userName = userName,
                        recapConfig = recapConfig,
                        onConfigChange = { recapConfig = it },
                        onRequestCustomCount = { showCustomCountDialog = true },
                        onRequestCustomTime = { showCustomTimeDialog = true }
                    )
                }

                item {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 8.dp, bottom = 8.dp)
                            .background(Color(0xFF1E262F), RoundedCornerShape(8.dp))
                            .clickable { isHistoryExpanded = !isHistoryExpanded }
                            .padding(horizontal = 16.dp, vertical = 12.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = "Session History",
                            color = Color(0xFF64B5F6),
                            fontSize = 18.sp,
                            fontWeight = FontWeight.ExtraBold,
                            letterSpacing = 0.5.sp
                        )
                        Icon(
                            imageVector = if (isHistoryExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                            contentDescription = "Expand/Collapse",
                            tint = Color(0xFF64B5F6)
                        )
                    }
                }

                if (isHistoryExpanded) {
                    items(sessions) { session ->
                        SessionCard(
                            session = session,
                            selectionMode = selectionMode,
                            selected = session.id in selectedSessionIds,
                            onSelectedChange = { selected ->
                                selectedSessionIds = if (selected) {
                                    selectedSessionIds + session.id
                                } else {
                                    selectedSessionIds - session.id
                                }
                            },
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

    if (showCustomCountDialog) {
        var input by remember { mutableStateOf("") }
        AlertDialog(
            onDismissRequest = { showCustomCountDialog = false },
            title = { Text("Custom Session Count") },
            text = {
                OutlinedTextField(
                    value = input,
                    onValueChange = { input = it },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    label = { Text("Number of sessions") }
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    input.toIntOrNull()?.let { count ->
                        if (count > 0) recapConfig = RecapConfig.ByCount(count)
                    }
                    showCustomCountDialog = false
                }) { Text("Apply") }
            },
            dismissButton = {
                TextButton(onClick = { showCustomCountDialog = false }) { Text("Cancel") }
            }
        )
    }

    if (showCustomTimeDialog) {
        var input by remember { mutableStateOf("") }
        AlertDialog(
            onDismissRequest = { showCustomTimeDialog = false },
            title = { Text("Custom Days") },
            text = {
                OutlinedTextField(
                    value = input,
                    onValueChange = { input = it },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    label = { Text("Days") }
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    input.toDoubleOrNull()?.let { days ->
                        if (days > 0.0) recapConfig = RecapConfig.ByTime(days * 24.0)
                    }
                    showCustomTimeDialog = false
                }) { Text("Apply") }
            },
            dismissButton = {
                TextButton(onClick = { showCustomTimeDialog = false }) { Text("Cancel") }
            }
        )
    }
}

@Composable
private fun UserRecapCard(
    recentSessions: List<FullSessionData>,
    geminiAgent: GeminiAgent,
    useGeminiSummary: Boolean,
    llmConfig: LlmProviderConfig,
    userName: String,
    recapConfig: RecapConfig,
    onConfigChange: (RecapConfig) -> Unit,
    onRequestCustomCount: () -> Unit,
    onRequestCustomTime: () -> Unit
) {
    val allCues = recentSessions.flatMap { it.cues }
    var aiAnalysis by remember { mutableStateOf<String?>(null) }
    var isGenerating by remember { mutableStateOf(false) }
    val recapRefreshKey = remember(recentSessions) {
        recentSessions.joinToString("|") { session ->
            "${session.session.id}:${session.session.timestamp}:${session.cues.size}"
        }
    }

    LaunchedEffect(recapRefreshKey, llmConfig, useGeminiSummary) {
        val fallback = buildRecapFallback(recentSessions, geminiAgent, recapConfig)
        if (useGeminiSummary && geminiAgent.isEnabled(llmConfig) && allCues.isNotEmpty()) {
            isGenerating = true
            val recapFocus = when (recapConfig) {
                is RecapConfig.ByCount -> "Focusing on the trend across the last ${recapConfig.count} sessions."
                is RecapConfig.ByTime -> {
                    val hours = recapConfig.hours
                    if (hours >= 24.0 && hours % 24.0 == 0.0) "Focusing on the trend over the last ${(hours / 24.0).toInt()} days."
                    else "Focusing on the trend over the last $hours hours."
                }
                is RecapConfig.CustomSelection -> "Focusing on the trend across these specifically selected sessions."
            }
            aiAnalysis = geminiAgent.generateImprovementAnalysis(
                userName = userName,
                recapFocus = recapFocus,
                recentErrorsText = buildRecentErrorPromptData(recentSessions, geminiAgent),
                fallback = fallback,
                preferGemini = useGeminiSummary,
                llmConfig = llmConfig
            )
            isGenerating = false
        } else {
            aiAnalysis = fallback
            isGenerating = false
        }
    }

    val titleText = when (recapConfig) {
        is RecapConfig.ByCount -> "Last ${recapConfig.count} Sessions Recap"
        is RecapConfig.ByTime -> {
            val hours = recapConfig.hours
            if (hours >= 24.0 && hours % 24.0 == 0.0) {
                val days = (hours / 24.0).toInt()
                if (days == 1) "Last 1 Day Recap" else "Last $days Days Recap"
            } else {
                if (hours == 1.0) "Last 1 Hour Recap" else "Last ${if (hours % 1.0 == 0.0) hours.toInt().toString() else hours.toString()} Hours Recap"
            }
        }
        is RecapConfig.CustomSelection -> "Selected Sessions Recap"
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF141420)),
        shape = RoundedCornerShape(12.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                Text(titleText, color = Color(0xFF64B5F6), fontSize = 18.sp, fontWeight = FontWeight.ExtraBold, letterSpacing = 0.5.sp, modifier = Modifier.weight(1f))
                
                var expanded by remember { mutableStateOf(false) }
                Box {
                    IconButton(onClick = { expanded = true }) {
                        Icon(Icons.Filled.Settings, contentDescription = "Settings", tint = Color.White)
                    }
                    DropdownMenu(
                        expanded = expanded,
                        onDismissRequest = { expanded = false },
                        modifier = Modifier.background(Color(0xFF263647))
                    ) {
                        DropdownMenuItem(
                            text = { Text("Last 5 Sessions", color = Color.White) },
                            onClick = { onConfigChange(RecapConfig.ByCount(5)); expanded = false }
                        )
                        DropdownMenuItem(
                            text = { Text("Last 10 Sessions", color = Color.White) },
                            onClick = { onConfigChange(RecapConfig.ByCount(10)); expanded = false }
                        )
                        DropdownMenuItem(
                            text = { Text("Last 15 Sessions", color = Color.White) },
                            onClick = { onConfigChange(RecapConfig.ByCount(15)); expanded = false }
                        )
                        DropdownMenuItem(
                            text = { Text("Custom Session Count...", color = Color.White) },
                            onClick = { onRequestCustomCount(); expanded = false }
                        )
                        HorizontalDivider(color = Color.Gray)
                        DropdownMenuItem(
                            text = { Text("Last 1 Hour", color = Color.White) },
                            onClick = { onConfigChange(RecapConfig.ByTime(1.0)); expanded = false }
                        )
                        DropdownMenuItem(
                            text = { Text("Last 3 Hours", color = Color.White) },
                            onClick = { onConfigChange(RecapConfig.ByTime(3.0)); expanded = false }
                        )
                        DropdownMenuItem(
                            text = { Text("Last 1 Day", color = Color.White) },
                            onClick = { onConfigChange(RecapConfig.ByTime(24.0)); expanded = false }
                        )
                        DropdownMenuItem(
                            text = { Text("Custom Days...", color = Color.White) },
                            onClick = { onRequestCustomTime(); expanded = false }
                        )
                    }
                }
            }
            Spacer(Modifier.height(16.dp))

            if (recentSessions.isEmpty()) {
                Text("No sessions found for this filter.", color = Color.Gray)
            } else {
                if (isGenerating) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(modifier = Modifier.size(16.dp), color = Color(0xFF64B5F6), strokeWidth = 2.dp)
                        Spacer(Modifier.width(12.dp))
                        Text("Generating Recap...", color = Color.Gray, fontSize = 15.sp)
                    }
                    Spacer(Modifier.height(20.dp))
                } else if (aiAnalysis != null) {
                    val annotatedText = buildAnnotatedString {
                        val text = aiAnalysis!!
                        val highlightRegex = Regex("【 近期重點:.*?】")
                        var lastIndex = 0
                        for (match in highlightRegex.findAll(text)) {
                            append(text.substring(lastIndex, match.range.first))
                            withStyle(style = SpanStyle(color = Color(0xFFFFB74D), fontWeight = FontWeight.Bold)) {
                                append(match.value)
                            }
                            lastIndex = match.range.last + 1
                        }
                        append(text.substring(lastIndex))
                    }

                    Text(
                        text = annotatedText,
                        color = Color(0xFFB6C2CC),
                        fontSize = 15.sp,
                        lineHeight = 22.sp
                    )
                    Spacer(Modifier.height(20.dp))
                }

                if (allCues.isNotEmpty()) {
                    Text("Mistake Breakdown", color = Color(0xFF64B5F6), fontSize = 16.sp, fontWeight = FontWeight.ExtraBold, letterSpacing = 0.5.sp)
                    Spacer(Modifier.height(8.dp))
                    ActionPieChart(cues = allCues)
                }
            }
        }
    }
}

private fun buildRecapFallback(
    recentSessions: List<FullSessionData>,
    geminiAgent: GeminiAgent,
    recapConfig: RecapConfig
): String {
    val allCues = recentSessions.flatMap { it.cues }
    if (allCues.isEmpty()) {
        return "最近沒有明顯重複錯誤，保持目前節奏。"
    }

    val topError = allCues
        .groupingBy { it.errorKey }
        .eachCount()
        .maxByOrNull { it.value }
        ?: return "最近沒有明顯重複錯誤，保持目前節奏。"

    val entry = geminiAgent.playbookEntry(topError.key)
    val label = entry?.label ?: allCues.firstOrNull { it.errorKey == topError.key }?.errorName ?: topError.key

    val (recentPeriodCount, previousPeriodCount, trendText) = when (recapConfig) {
        is RecapConfig.ByTime -> {
            val cutoffTime = System.currentTimeMillis() - (recapConfig.hours * 3600 * 1000).toLong()
            val totalDuration = System.currentTimeMillis() - cutoffTime
            val recentDuration = (totalDuration * 0.4).toLong()
            val midTime = System.currentTimeMillis() - recentDuration
            
            val latestSessions = recentSessions.filter { it.session.timestamp >= midTime }
            val pastSessions = recentSessions.filter { it.session.timestamp < midTime }

            val rCount = latestSessions.sumOf { s -> s.cues.count { it.errorKey == topError.key } }
            val pCount = pastSessions.sumOf { s -> s.cues.count { it.errorKey == topError.key } }
            
            val rAvg = if (latestSessions.isNotEmpty()) rCount.toFloat() / latestSessions.size else 0f
            val pAvg = if (pastSessions.isNotEmpty()) pCount.toFloat() / pastSessions.size else 0f
            
            val pct = if (pAvg > 0) (((pAvg - rAvg) / pAvg) * 100f).roundToInt() else 0
            
            val tText = when {
                pastSessions.isEmpty() -> "資料還少，先專注在這個問題就好!"
                pct > 0 -> "近期平均比之前改善 $pct%。"
                pct < 0 -> "近期平均比之前增加 ${-pct}%，需要優先處理。"
                else -> "近期和之前大致持平。"
            }
            Triple(rCount, pCount, tText)
        }
        else -> {
            val totalSessions = recentSessions.size
            val recentCount = maxOf(1, (totalSessions * 0.4).roundToInt())
            val latestSessions = recentSessions.take(recentCount)
            val pastSessions = recentSessions.drop(recentCount)

            val rCount = latestSessions.sumOf { s -> s.cues.count { it.errorKey == topError.key } }
            val pCount = pastSessions.sumOf { s -> s.cues.count { it.errorKey == topError.key } }
            
            val rAvg = if (latestSessions.isNotEmpty()) rCount.toFloat() / latestSessions.size else 0f
            val pAvg = if (pastSessions.isNotEmpty()) pCount.toFloat() / pastSessions.size else 0f
            
            val pct = if (pAvg > 0) (((pAvg - rAvg) / pAvg) * 100f).roundToInt() else 0
            
            val tText = when {
                pastSessions.isEmpty() -> "不過資料還少，先專注在這個問題就好!"
                pct > 0 -> "近期(${recentCount}場)平均比之前(${totalSessions - recentCount}場)改善 $pct%。"
                pct < 0 -> "近期(${recentCount}場)平均比之前(${totalSessions - recentCount}場)增加 ${-pct}%，需要優先處理。"
                else -> "近期(${recentCount}場)和之前(${totalSessions - recentCount}場)大致持平。"
            }
            Triple(rCount, pCount, tText)
        }
    }

    val details = buildList {
        add("【 近期重點: $label】\n$trendText")
        if (!trendText.contains("資料還少") && (previousPeriodCount > 0 || recentPeriodCount > 0)) {
            add("近期 $recentPeriodCount 次，之前 $previousPeriodCount 次。")
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
    val formatter = SimpleDateFormat("yyyy/MM/dd HH:mm", Locale.getDefault())
    return recentSessions.reversed().map { session ->
        val dateString = formatter.format(Date(session.session.timestamp))
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
        "Session on $dateString: ${counts.ifBlank { "no errors" }}"
    }.joinToString("\n") + "\n\nFocus details:\n" + geminiAgent.allPlaybookEntries()
        .entries
        .joinToString("\n") { (key, entry) ->
            "$key: ${entry.label}; diagnosis=${entry.diagnosis}; practice=${entry.practice}"
        }
}

@Composable
private fun SessionCard(
    session: SessionEntity,
    selectionMode: Boolean,
    selected: Boolean,
    onSelectedChange: (Boolean) -> Unit,
    onClick: () -> Unit
) {
    val formatter = SimpleDateFormat("yyyy/MM/dd HH:mm", Locale.getDefault())
    val dateString = formatter.format(Date(session.timestamp))
    val clickAction = {
        if (selectionMode) {
            onSelectedChange(!selected)
        } else {
            onClick()
        }
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = clickAction),
        colors = CardDefaults.cardColors(
            containerColor = if (selected) Color(0xFF263647) else Color(0xFF1E262F)
        ),
        shape = RoundedCornerShape(12.dp)
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (selectionMode) {
                Checkbox(
                    checked = selected,
                    onCheckedChange = onSelectedChange,
                    colors = CheckboxDefaults.colors(checkedColor = Color(0xFF00D4FF))
                )
                Spacer(modifier = Modifier.width(10.dp))
            }

            Column(modifier = Modifier.weight(1f)) {
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
}
