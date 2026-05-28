import 'dart:convert';
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart' as p;
import '../postgame/postgame_analyzer.dart';

/// A lightweight record representing a saved session.
class SessionRecord {
  final int id;
  final DateTime date;
  final String trainingMode;
  final String targetSide;
  final int durationMs;
  final int framesAnalyzed;
  final int modelChecks;
  final String topAction;
  final String? llmSummary;
  final Map<String, int> actionCounts;
  final Map<String, int> errorCounts;
  final List<PostgameTimelineItem> timeline;

  const SessionRecord({
    required this.id,
    required this.date,
    required this.trainingMode,
    required this.targetSide,
    required this.durationMs,
    required this.framesAnalyzed,
    required this.modelChecks,
    required this.topAction,
    this.llmSummary,
    this.actionCounts = const {},
    this.errorCounts = const {},
    this.timeline = const [],
  });

  /// Reconstruct a [PostgameReport] so we can reuse the existing report UI.
  PostgameReport toReport() {
    return PostgameReport(
      videoName: 'Session #$id',
      duration: Duration(milliseconds: durationMs),
      framesAnalyzed: framesAnalyzed,
      modelChecks: modelChecks,
      actionCounts: actionCounts,
      errorCounts: errorCounts,
      timeline: timeline,
    );
  }
}

class DatabaseHelper {
  static final DatabaseHelper _instance = DatabaseHelper._internal();
  factory DatabaseHelper() => _instance;
  DatabaseHelper._internal();

  Database? _db;

  Future<Database> get database async {
    if (_db != null) return _db!;
    _db = await _initDatabase();
    return _db!;
  }

  Future<Database> _initDatabase() async {
    final dbPath = await getDatabasesPath();
    final path = p.join(dbPath, 'fencing_coach.db');
    return openDatabase(
      path,
      version: 1,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            training_mode TEXT NOT NULL,
            target_side TEXT NOT NULL,
            duration_ms INTEGER NOT NULL,
            frames_analyzed INTEGER NOT NULL,
            model_checks INTEGER NOT NULL,
            top_action TEXT NOT NULL,
            llm_summary TEXT,
            action_counts_json TEXT,
            error_counts_json TEXT,
            timeline_json TEXT
          )
        ''');
      },
    );
  }

  /// Insert a new session from a [PostgameReport] and config.
  Future<int> insertSession({
    required PostgameReport report,
    required PostgameAnalysisConfig config,
  }) async {
    final db = await database;
    final timelineJson = jsonEncode(
      report.timeline.map((item) => {
        'time': item.timeSeconds,
        'action': item.action,
        'confidence': item.confidence,
        'errors': item.errors,
      }).toList(),
    );
    return db.insert('sessions', {
      'date': DateTime.now().toIso8601String(),
      'training_mode': config.trainingMode,
      'target_side': config.targetSide,
      'duration_ms': report.duration.inMilliseconds,
      'frames_analyzed': report.framesAnalyzed,
      'model_checks': report.modelChecks,
      'top_action': report.topAction,
      'action_counts_json': jsonEncode(report.actionCounts),
      'error_counts_json': jsonEncode(report.errorCounts),
      'timeline_json': timelineJson,
    });
  }

  /// Update the LLM summary for an existing session.
  Future<void> updateSummary(int sessionId, String summary) async {
    final db = await database;
    await db.update(
      'sessions',
      {'llm_summary': summary},
      where: 'id = ?',
      whereArgs: [sessionId],
    );
  }

  /// Get all sessions ordered by most recent first.
  Future<List<SessionRecord>> getSessions() async {
    final db = await database;
    final rows = await db.query('sessions', orderBy: 'date DESC');
    return rows.map(_rowToRecord).toList();
  }

  /// Get a single session by ID.
  Future<SessionRecord?> getSession(int id) async {
    final db = await database;
    final rows = await db.query('sessions', where: 'id = ?', whereArgs: [id]);
    if (rows.isEmpty) return null;
    return _rowToRecord(rows.first);
  }

  SessionRecord _rowToRecord(Map<String, Object?> row) {
    final actionCountsRaw = row['action_counts_json'] as String?;
    final errorCountsRaw = row['error_counts_json'] as String?;
    final timelineRaw = row['timeline_json'] as String?;

    Map<String, int> actionCounts = {};
    if (actionCountsRaw != null && actionCountsRaw.isNotEmpty) {
      final decoded = jsonDecode(actionCountsRaw) as Map<String, dynamic>;
      actionCounts = decoded.map((k, v) => MapEntry(k, (v as num).toInt()));
    }

    Map<String, int> errorCounts = {};
    if (errorCountsRaw != null && errorCountsRaw.isNotEmpty) {
      final decoded = jsonDecode(errorCountsRaw) as Map<String, dynamic>;
      errorCounts = decoded.map((k, v) => MapEntry(k, (v as num).toInt()));
    }

    List<PostgameTimelineItem> timeline = [];
    if (timelineRaw != null && timelineRaw.isNotEmpty) {
      final decoded = jsonDecode(timelineRaw) as List<dynamic>;
      timeline = decoded.map((item) {
        final map = item as Map<String, dynamic>;
        return PostgameTimelineItem(
          timeSeconds: (map['time'] as num).toDouble(),
          action: map['action'] as String,
          confidence: (map['confidence'] as num).toDouble(),
          errors: (map['errors'] as List<dynamic>).cast<String>(),
        );
      }).toList();
    }

    return SessionRecord(
      id: row['id'] as int,
      date: DateTime.parse(row['date'] as String),
      trainingMode: row['training_mode'] as String,
      targetSide: row['target_side'] as String,
      durationMs: row['duration_ms'] as int,
      framesAnalyzed: row['frames_analyzed'] as int,
      modelChecks: row['model_checks'] as int,
      topAction: row['top_action'] as String,
      llmSummary: row['llm_summary'] as String?,
      actionCounts: actionCounts,
      errorCounts: errorCounts,
      timeline: timeline,
    );
  }
}
