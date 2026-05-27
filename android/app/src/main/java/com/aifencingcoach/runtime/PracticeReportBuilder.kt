package com.aifencingcoach.runtime

fun buildPracticeReport(
    trainingMode: TrainingMode,
    poseBackend: PoseBackendKind,
    targetSide: TargetSide,
    elapsedSeconds: Long,
    activeFrames: Long,
    fps: Int,
    inferenceCount: Long,
    actionCounts: Map<String, Long>,
    cueCounts: Map<String, CueCountItem>,
    cueTimeline: List<CueHistoryItem>,
    generatedAtFrame: Long
): PracticeReport {
    val activeSeconds = activeFrames / fps.coerceAtLeast(1)
    val activePercent = if (elapsedSeconds > 0) {
        ((activeSeconds * 100L) / elapsedSeconds).toInt().coerceIn(0, 100)
    } else {
        0
    }
    val nonIdleActionCounts = actionCounts.filterKeys { it != "Idle" }
    val totalActions = nonIdleActionCounts.values.sum().coerceAtLeast(1L)
    val actionItems = nonIdleActionCounts
        .entries
        .sortedWith(compareByDescending<Map.Entry<String, Long>> { it.value }.thenBy { it.key })
        .map { entry ->
            ActionCountItem(
                action = entry.key,
                count = entry.value,
                percent = ((entry.value * 100L) / totalActions).toInt()
            )
        }
    val topAction = actionItems.firstOrNull()?.action
        ?: actionCounts.maxByOrNull { it.value }?.key
        ?: "Idle"
    val topCues = cueCounts.values
        .sortedWith(compareByDescending<CueCountItem> { it.count }.thenBy { it.label })
        .take(5)
    val primaryTakeaway = when {
        topCues.isNotEmpty() -> topCues.first().message
        actionItems.isNotEmpty() -> "Keep building clean ${actionItems.first().action} reps."
        else -> "Build a longer active sample."
    }

    return PracticeReport(
        trainingMode = trainingMode,
        poseBackend = poseBackend,
        targetSide = targetSide,
        elapsedSeconds = elapsedSeconds,
        activeSeconds = activeSeconds,
        activePercent = activePercent,
        inferenceCount = inferenceCount,
        cueCount = topCues.sumOf { it.count },
        topAction = topAction,
        actionCounts = actionItems,
        topCues = topCues,
        cueTimeline = cueTimeline.takeLast(MaxReportTimeline).asReversed(),
        primaryTakeaway = primaryTakeaway,
        generatedAtFrame = generatedAtFrame
    )
}

private const val MaxReportTimeline = 20
