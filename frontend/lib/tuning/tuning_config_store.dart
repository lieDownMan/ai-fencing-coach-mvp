/// Persists in-app tuned HeuristicsConfig values as JSON under
/// Documents/tuning/config_overrides.json (visible in the Files app), so
/// adjustments made in the Tuning tab survive restarts and can be copied
/// back into the code as the new shipped defaults.

library;

import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';

import '../heuristics/heuristics_engine.dart';

class TuningConfigStore {
  static Future<File> _file() async {
    final docs = await getApplicationDocumentsDirectory();
    final dir = Directory('${docs.path}/tuning');
    await dir.create(recursive: true);
    return File('${dir.path}/config_overrides.json');
  }

  static Future<HeuristicsConfig> load() async {
    try {
      final f = await _file();
      if (!await f.exists()) return const HeuristicsConfig();
      final raw = jsonDecode(await f.readAsString()) as Map<String, dynamic>;
      final map = raw.map((k, v) => MapEntry(k, (v as num).toDouble()));
      return HeuristicsConfig.fromMap(map);
    } catch (_) {
      return const HeuristicsConfig();
    }
  }

  static Future<void> save(HeuristicsConfig config) async {
    final f = await _file();
    await f.writeAsString(jsonEncode(config.toMap()));
  }

  static Future<void> reset() async {
    final f = await _file();
    if (await f.exists()) await f.delete();
  }

  /// Dart snippet of the current values, ready to paste over the defaults in
  /// HeuristicsConfig's constructor.
  static String asDartSnippet(HeuristicsConfig config) {
    final buf = StringBuffer();
    config.toMap().forEach((k, v) {
      buf.writeln('    this.$k = $v,');
    });
    return buf.toString();
  }
}
