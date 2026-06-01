import 'dart:async';
import 'dart:convert';
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart' as p;

// ---------------------------------------------------------------------------
// Data models (mirrors Android Entities.kt)
// ---------------------------------------------------------------------------

class SessionRecord {
  final int id;
  final DateTime date;
  final String trainingMode;
  final String targetSide;
  final int durationMs;
  final int framesAnalyzed;
  final int modelChecks;
  final String topAction;
  final int cueCount;
  final String? llmSummary;
  final String? exportedVideoPath;
  final String userName;
  final String source; // 'Realtime' or 'Postgame'
  final Map<String, int> actionCounts;
  final Map<String, int> errorCounts;

  const SessionRecord({
    required this.id,
    required this.date,
    required this.trainingMode,
    required this.targetSide,
    required this.durationMs,
    required this.framesAnalyzed,
    required this.modelChecks,
    required this.topAction,
    this.cueCount = 0,
    this.llmSummary,
    this.exportedVideoPath,
    this.userName = 'Fencer',
    this.source = 'Realtime',
    this.actionCounts = const {},
    this.errorCounts = const {},
  });
}

class CueHistoryItem {
  final int id;
  final int sessionId;
  final double timeSeconds;
  final String errorKey;
  final String errorName;
  final String practiceSuggestion;

  const CueHistoryItem({
    required this.id,
    required this.sessionId,
    required this.timeSeconds,
    required this.errorKey,
    required this.errorName,
    required this.practiceSuggestion,
  });
}

class FullSessionData {
  final SessionRecord session;
  final List<CueHistoryItem> cues;

  const FullSessionData({required this.session, required this.cues});
}

// ---------------------------------------------------------------------------
// DatabaseHelper (singleton)
// ---------------------------------------------------------------------------

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
    final path = p.join(dbPath, 'fencing_coach_v2.db');
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
            duration_ms INTEGER NOT NULL DEFAULT 0,
            frames_analyzed INTEGER NOT NULL DEFAULT 0,
            model_checks INTEGER NOT NULL DEFAULT 0,
            top_action TEXT NOT NULL DEFAULT 'Idle',
            cue_count INTEGER NOT NULL DEFAULT 0,
            llm_summary TEXT,
            exported_video_path TEXT,
            user_name TEXT NOT NULL DEFAULT 'Fencer',
            source TEXT NOT NULL DEFAULT 'Realtime',
            action_counts_json TEXT,
            error_counts_json TEXT
          )
        ''');
        await db.execute('''
          CREATE TABLE cue_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            time_seconds REAL NOT NULL,
            error_key TEXT NOT NULL,
            error_name TEXT NOT NULL,
            practice_suggestion TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
          )
        ''');
        await db.execute(
          'CREATE INDEX idx_cue_session ON cue_history (session_id)',
        );
        await db.execute(
          'CREATE INDEX idx_session_user ON sessions (user_name)',
        );
      },
    );
  }

  // ── Write ─────────────────────────────────────────────────────────────────

  /// Save a live or postgame session and return the newly created session ID.
  Future<int> insertSession({
    required String trainingMode,
    required String targetSide,
    required int durationMs,
    required int framesAnalyzed,
    required int modelChecks,
    required String topAction,
    required Map<String, int> actionCounts,
    required Map<String, int> errorCounts,
    required List<CueHistoryItem> cues,
    String? llmSummary,
    String? exportedVideoPath,
    String userName = 'Fencer',
    String source = 'Realtime',
  }) async {
    final db = await database;
    final sessionId = await db.insert('sessions', {
      'date': DateTime.now().toIso8601String(),
      'training_mode': trainingMode,
      'target_side': targetSide,
      'duration_ms': durationMs,
      'frames_analyzed': framesAnalyzed,
      'model_checks': modelChecks,
      'top_action': topAction,
      'cue_count': cues.length,
      'llm_summary': llmSummary,
      'exported_video_path': exportedVideoPath,
      'user_name': userName,
      'source': source,
      'action_counts_json': jsonEncode(actionCounts),
      'error_counts_json': jsonEncode(errorCounts),
    });

    for (final cue in cues) {
      await db.insert('cue_history', {
        'session_id': sessionId,
        'time_seconds': cue.timeSeconds,
        'error_key': cue.errorKey,
        'error_name': cue.errorName,
        'practice_suggestion': cue.practiceSuggestion,
      });
    }

    return sessionId;
  }

  Future<void> updateSummary(int sessionId, String summary) async {
    final db = await database;
    await db.update(
      'sessions',
      {'llm_summary': summary},
      where: 'id = ?',
      whereArgs: [sessionId],
    );
  }

  Future<void> updateVideoPath(int sessionId, String path) async {
    final db = await database;
    await db.update(
      'sessions',
      {'exported_video_path': path},
      where: 'id = ?',
      whereArgs: [sessionId],
    );
  }

  Future<void> deleteSession(int sessionId) async {
    final db = await database;
    await db.delete('sessions', where: 'id = ?', whereArgs: [sessionId]);
  }

  // ── Read ──────────────────────────────────────────────────────────────────

  Future<List<String>> getDistinctUsers() async {
    final db = await database;
    final rows = await db.rawQuery(
      'SELECT DISTINCT user_name FROM sessions ORDER BY user_name ASC',
    );
    return rows.map((r) => r['user_name'] as String).toList();
  }

  Future<List<SessionRecord>> getAllSessions() async {
    final db = await database;
    final rows = await db.query('sessions', orderBy: 'date DESC');
    return rows.map(_rowToRecord).toList();
  }

  Future<List<SessionRecord>> getSessionsByUser(String userName) async {
    final db = await database;
    final rows = await db.query(
      'sessions',
      where: 'user_name = ?',
      whereArgs: [userName],
      orderBy: 'date DESC',
    );
    return rows.map(_rowToRecord).toList();
  }

  Future<SessionRecord?> getSession(int id) async {
    final db = await database;
    final rows =
        await db.query('sessions', where: 'id = ?', whereArgs: [id]);
    if (rows.isEmpty) return null;
    return _rowToRecord(rows.first);
  }

  Future<FullSessionData?> getFullSession(int sessionId) async {
    final session = await getSession(sessionId);
    if (session == null) return null;
    final db = await database;
    final cueRows = await db.query(
      'cue_history',
      where: 'session_id = ?',
      whereArgs: [sessionId],
      orderBy: 'time_seconds ASC',
    );
    final cues = cueRows.map(_rowToCue).toList();
    return FullSessionData(session: session, cues: cues);
  }

  // ── Row parsers ───────────────────────────────────────────────────────────

  SessionRecord _rowToRecord(Map<String, Object?> row) {
    Map<String, int> _decodeIntMap(String? json) {
      if (json == null || json.isEmpty) return {};
      final decoded = jsonDecode(json) as Map<String, dynamic>;
      return decoded.map((k, v) => MapEntry(k, (v as num).toInt()));
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
      cueCount: row['cue_count'] as int,
      llmSummary: row['llm_summary'] as String?,
      exportedVideoPath: row['exported_video_path'] as String?,
      userName: row['user_name'] as String? ?? 'Fencer',
      source: row['source'] as String? ?? 'Realtime',
      actionCounts: _decodeIntMap(row['action_counts_json'] as String?),
      errorCounts: _decodeIntMap(row['error_counts_json'] as String?),
    );
  }

  CueHistoryItem _rowToCue(Map<String, Object?> row) {
    return CueHistoryItem(
      id: row['id'] as int,
      sessionId: row['session_id'] as int,
      timeSeconds: (row['time_seconds'] as num).toDouble(),
      errorKey: row['error_key'] as String,
      errorName: row['error_name'] as String,
      practiceSuggestion: row['practice_suggestion'] as String,
    );
  }
}
