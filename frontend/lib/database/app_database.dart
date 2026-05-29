import 'dart:convert';
import 'package:path/path.dart';
import 'package:sqflite/sqflite.dart';
import 'package:path_provider/path_provider.dart';
import 'entities.dart';

class AppDatabase {
  static final AppDatabase instance = AppDatabase._init();
  static Database? _database;

  AppDatabase._init();

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDB('fencing_coach_history.db');
    return _database!;
  }

  Future<Database> _initDB(String filePath) async {
    final dbPath = await getApplicationDocumentsDirectory();
    final path = join(dbPath.path, filePath);

    return await openDatabase(
      path,
      version: 1,
      onCreate: _createDB,
    );
  }

  Future _createDB(Database db, int version) async {
    const idType = 'TEXT PRIMARY KEY';
    const textType = 'TEXT NOT NULL';
    const integerType = 'INTEGER NOT NULL';

    await db.execute('''
CREATE TABLE sessions (
  id $idType,
  startTimeMs $integerType,
  endTimeMs $integerType,
  elapsedSeconds $integerType,
  actionCounts $textType,
  cueTimeline $textType,
  llmSummary TEXT
)
''');
  }

  Future<void> savePracticeReport(PracticeReport report) async {
    final db = await instance.database;

    await db.insert(
      'sessions',
      {
        'id': report.id,
        'startTimeMs': report.startTimeMs,
        'endTimeMs': report.endTimeMs,
        'elapsedSeconds': report.elapsedSeconds,
        'actionCounts': jsonEncode(report.actionCounts.map((e) => e.toJson()).toList()),
        'cueTimeline': jsonEncode(report.cueTimeline.map((e) => e.toJson()).toList()),
        'llmSummary': report.llmSummary,
      },
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  Future<List<PracticeReport>> getAllPracticeReports() async {
    final db = await instance.database;
    final result = await db.query('sessions', orderBy: 'startTimeMs DESC');

    return result.map((json) {
      return PracticeReport(
        id: json['id'] as String,
        startTimeMs: json['startTimeMs'] as int,
        endTimeMs: json['endTimeMs'] as int,
        elapsedSeconds: json['elapsedSeconds'] as int,
        actionCounts: (jsonDecode(json['actionCounts'] as String) as List<dynamic>)
            .map((e) => ActionCountItem.fromJson(e as Map<String, dynamic>))
            .toList(),
        cueTimeline: (jsonDecode(json['cueTimeline'] as String) as List<dynamic>)
            .map((e) => CueHistoryItem.fromJson(e as Map<String, dynamic>))
            .toList(),
        llmSummary: json['llmSummary'] as String? ?? '',
      );
    }).toList();
  }

  Future<void> close() async {
    final db = await instance.database;
    db.close();
  }
}
