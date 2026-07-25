/// Records per-frame pose skeletons + classifier output to a JSONL file so
/// heuristic thresholds can be tuned offline against real data.
///
/// One line per pose frame:
///   `{"t":<epoch ms>,"action":"SF","conf":0.83,`
///   `"joints":{"front_wrist":[0.7123,0.4411], ...}}`
///
/// Files land in `<app documents>/tuning/session__<timestamp>.jsonl` and are
/// visible in the iOS Files app (UIFileSharingEnabled) for AirDrop/export.
/// Labeling happens later by renaming: good__*.jsonl, stance_too_high__*.jsonl…

library;

import 'dart:convert';
import 'dart:io';
import 'dart:ui' show Offset;

import 'package:path_provider/path_provider.dart';

class SessionRecorder {
  final List<String> _lines = [];
  bool _recording = false;

  bool get isRecording => _recording;
  int get frameCount => _lines.length;

  void start() {
    _lines.clear();
    _recording = true;
  }

  double _round4(double v) => (v * 10000).roundToDouble() / 10000;

  void addFrame({
    required String action,
    required double confidence,
    required Map<String, Offset> joints,
  }) {
    if (!_recording) return;
    final jointMap = <String, List<double>>{
      for (final e in joints.entries)
        e.key: [_round4(e.value.dx), _round4(e.value.dy)],
    };
    _lines.add(jsonEncode({
      't': DateTime.now().millisecondsSinceEpoch,
      'action': action,
      'conf': _round4(confidence),
      'joints': jointMap,
    }));
  }

  /// Stop recording and write the buffered frames to disk.
  /// Returns null if nothing was recorded.
  Future<File?> stop() async {
    _recording = false;
    if (_lines.isEmpty) return null;

    final docs = await getApplicationDocumentsDirectory();
    final dir = Directory('${docs.path}/tuning');
    await dir.create(recursive: true);

    final ts = DateTime.now()
        .toIso8601String()
        .replaceAll(':', '-')
        .split('.')
        .first;
    final file = File('${dir.path}/session__$ts.jsonl');
    await file.writeAsString('${_lines.join('\n')}\n');
    _lines.clear();
    return file;
  }
}
