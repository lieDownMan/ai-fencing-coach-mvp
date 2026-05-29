class ActionCountItem {
  final String action;
  final int count;

  ActionCountItem({required this.action, required this.count});

  Map<String, dynamic> toJson() => {
        'action': action,
        'count': count,
      };

  factory ActionCountItem.fromJson(Map<String, dynamic> json) =>
      ActionCountItem(
        action: json['action'] as String,
        count: json['count'] as int,
      );
}

class CueHistoryItem {
  final int timestampMs;
  final String label;

  CueHistoryItem({required this.timestampMs, required this.label});

  Map<String, dynamic> toJson() => {
        'timestampMs': timestampMs,
        'label': label,
      };

  factory CueHistoryItem.fromJson(Map<String, dynamic> json) => CueHistoryItem(
        timestampMs: json['timestampMs'] as int,
        label: json['label'] as String,
      );
}

class PracticeReport {
  final String id;
  final int startTimeMs;
  final int endTimeMs;
  final int elapsedSeconds;
  final List<ActionCountItem> actionCounts;
  final List<CueHistoryItem> cueTimeline;
  final String llmSummary;

  PracticeReport({
    required this.id,
    required this.startTimeMs,
    required this.endTimeMs,
    required this.elapsedSeconds,
    required this.actionCounts,
    required this.cueTimeline,
    this.llmSummary = '',
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'startTimeMs': startTimeMs,
        'endTimeMs': endTimeMs,
        'elapsedSeconds': elapsedSeconds,
        'actionCounts': actionCounts.map((e) => e.toJson()).toList(),
        'cueTimeline': cueTimeline.map((e) => e.toJson()).toList(),
        'llmSummary': llmSummary,
      };

  factory PracticeReport.fromJson(Map<String, dynamic> json) => PracticeReport(
        id: json['id'] as String,
        startTimeMs: json['startTimeMs'] as int,
        endTimeMs: json['endTimeMs'] as int,
        elapsedSeconds: json['elapsedSeconds'] as int,
        actionCounts: (json['actionCounts'] as List<dynamic>?)
                ?.map((e) => ActionCountItem.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
        cueTimeline: (json['cueTimeline'] as List<dynamic>?)
                ?.map((e) => CueHistoryItem.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
        llmSummary: json['llmSummary'] as String? ?? '',
      );
}
