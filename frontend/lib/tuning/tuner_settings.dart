/// Persisted settings for the macOS video threshold-tuner: where the cue
/// videos live and where the YOLO pose mlpackage is. Prefilled from
/// --dart-define (see tool/run_tuner.sh), overridable in the UI, and stored
/// in Application Support so they survive restarts.

library;

import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';

class TunerSettings {
  String videoDir;
  String modelPath;

  TunerSettings({required this.videoDir, required this.modelPath});

  static const String _defineVideoDir = String.fromEnvironment('TUNE_VIDEO_DIR');
  static const String _defineModelPath = String.fromEnvironment('TUNE_MODEL_PATH');

  static Future<File> _file() async {
    final dir = await getApplicationSupportDirectory();
    return File('${dir.path}/tuner_settings.json');
  }

  static Future<TunerSettings> load() async {
    String videoDir = _defineVideoDir;
    String modelPath = _defineModelPath;
    try {
      final f = await _file();
      if (await f.exists()) {
        final raw = jsonDecode(await f.readAsString()) as Map<String, dynamic>;
        // Saved values win over defines only when they still exist on disk —
        // a moved repo shouldn't pin the tuner to dead paths forever.
        final savedVideoDir = raw['videoDir'] as String? ?? '';
        final savedModelPath = raw['modelPath'] as String? ?? '';
        if (savedVideoDir.isNotEmpty && Directory(savedVideoDir).existsSync()) {
          videoDir = savedVideoDir;
        }
        if (savedModelPath.isNotEmpty &&
            FileSystemEntity.typeSync(savedModelPath) !=
                FileSystemEntityType.notFound) {
          modelPath = savedModelPath;
        }
      }
    } catch (_) {}
    return TunerSettings(videoDir: videoDir, modelPath: modelPath);
  }

  Future<void> save() async {
    final f = await _file();
    await f.writeAsString(jsonEncode({
      'videoDir': videoDir,
      'modelPath': modelPath,
    }));
  }
}
