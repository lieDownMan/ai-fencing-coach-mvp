/// Offline replay of recorded sessions for threshold tuning.
///
/// Records come from the app's Debug-tab recorder (JSONL, one pose frame per
/// line). Label a file by renaming it: `<label>__whatever.jsonl` where label
/// is `good` or an error key (`stance_too_high`, `narrow_step`, …).
///
/// Run (prints a per-file + per-label report, never fails):
///   flutter test test/tuning/replay_recordings_test.dart \
///     --dart-define=TUNE_DIR=/path/to/recordings --plain-name replay
///
/// Optional overrides to A/B a candidate threshold, e.g.:
///     --dart-define=TUNE_TARGET_SIDE=right
///     --dart-define=TUNE_MODE="Target Practice"
///
/// The report shows, for every rule metric, the distribution (p05/p50/p95,
/// min/max) per label — put the threshold where the `good` distribution and
/// the error-labeled distribution separate.
library;

import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/heuristics/heuristics_engine.dart';

const String kTuneDir = String.fromEnvironment('TUNE_DIR');
const String kTargetSide =
    String.fromEnvironment('TUNE_TARGET_SIDE', defaultValue: 'left');
const String kMode =
    String.fromEnvironment('TUNE_MODE', defaultValue: 'Footwork');

// Mirrors the live loop in main.dart.
const int kBufferSize = 60;
const int kEvalEveryNFrames = 10;

class Frame {
  final int t;
  final String action;
  final double conf;
  final Skeleton joints;
  Frame(this.t, this.action, this.conf, this.joints);
}

List<Frame> loadJsonl(File f) {
  final frames = <Frame>[];
  for (final line in f.readAsLinesSync()) {
    if (line.trim().isEmpty) continue;
    final obj = jsonDecode(line) as Map<String, dynamic>;
    final joints = <String, Offset>{};
    (obj['joints'] as Map<String, dynamic>).forEach((k, v) {
      final xy = (v as List).cast<num>();
      joints[k] = Offset(xy[0].toDouble(), xy[1].toDouble());
    });
    frames.add(Frame(
      obj['t'] as int,
      obj['action'] as String? ?? 'Idle',
      (obj['conf'] as num?)?.toDouble() ?? 0.0,
      joints,
    ));
  }
  return frames;
}

String pct(List<double> sorted, double p) {
  if (sorted.isEmpty) return '-';
  final i = ((sorted.length - 1) * p).round();
  return sorted[i].toStringAsFixed(3);
}

void main() {
  test('replay recordings and report metric distributions', () {
    if (kTuneDir.isEmpty) {
      // ignore: avoid_print
      print('TUNE_DIR not set — skipping. See file header for usage.');
      return;
    }

    final files = Directory(kTuneDir)
        .listSync()
        .whereType<File>()
        .where((f) => f.path.endsWith('.jsonl'))
        .toList()
      ..sort((a, b) => a.path.compareTo(b.path));
    expect(files, isNotEmpty, reason: 'no .jsonl files in $kTuneDir');

    // label -> metric name -> all observed values
    final metricsByLabel = <String, Map<String, List<double>>>{};
    // label -> error key -> windows triggered
    final triggersByLabel = <String, Map<String, int>>{};
    // label -> total eval windows
    final windowsByLabel = <String, int>{};

    for (final f in files) {
      final name = f.uri.pathSegments.last;
      final label = name.contains('__') ? name.split('__').first : 'unlabeled';
      final frames = loadJsonl(f);
      if (frames.length < 5) continue;

      final spanMs = frames.last.t - frames.first.t;
      final fps =
          spanMs > 0 ? (frames.length - 1) * 1000.0 / spanMs : kDefaultFps;

      final engine =
          HeuristicsEngine(targetSide: kTargetSide, trainingMode: kMode);
      final buffer = <Skeleton>[];
      final fileTriggers = <String, int>{};
      var fileWindows = 0;

      for (var i = 0; i < frames.length; i++) {
        buffer.add(frames[i].joints);
        if (buffer.length > kBufferSize) buffer.removeAt(0);
        if (i % kEvalEveryNFrames != 0 || buffer.length < kBufferSize) {
          continue;
        }
        fileWindows++;
        final errors = engine.evaluateWindow(
          action: frames[i].action,
          skeletons: List.of(buffer),
          fps: fps,
        );
        for (final e in errors) {
          fileTriggers[e] = (fileTriggers[e] ?? 0) + 1;
        }
        engine.computeWindowMetrics(List.of(buffer), fps: fps).forEach((k, v) {
          metricsByLabel
              .putIfAbsent(label, () => {})
              .putIfAbsent(k, () => [])
              .add(v);
        });
      }

      windowsByLabel[label] = (windowsByLabel[label] ?? 0) + fileWindows;
      fileTriggers.forEach((k, v) {
        triggersByLabel.putIfAbsent(label, () => {})[k] =
            (triggersByLabel[label]?[k] ?? 0) + v;
      });

      // ignore: avoid_print
      print('▶ $name  [$label]  ${frames.length} frames, '
          '${(spanMs / 1000).toStringAsFixed(1)}s, fps=${fps.toStringAsFixed(1)}, '
          '$fileWindows windows, triggers: '
          '${fileTriggers.isEmpty ? '(none)' : fileTriggers}');
    }

    // ignore: avoid_print
    print('\n===== Trigger rate per label (windows triggered / total) =====');
    for (final label in windowsByLabel.keys) {
      final total = windowsByLabel[label]!;
      final trig = triggersByLabel[label] ?? {};
      // ignore: avoid_print
      print('[$label] $total windows');
      for (final e in trig.entries) {
        final rate = (100 * e.value / total).toStringAsFixed(1);
        // ignore: avoid_print
        print('    ${e.key}: ${e.value} ($rate%)');
      }
    }

    // ignore: avoid_print
    print('\n===== Metric distributions per label =====');
    final allMetricNames = metricsByLabel.values
        .expand((m) => m.keys)
        .toSet()
        .toList()
      ..sort();
    for (final metric in allMetricNames) {
      // ignore: avoid_print
      print(metric);
      for (final label in metricsByLabel.keys) {
        final values = metricsByLabel[label]![metric];
        if (values == null || values.isEmpty) continue;
        final sorted = List<double>.from(values)..sort();
        // ignore: avoid_print
        print('    [$label] n=${sorted.length} '
            'min=${sorted.first.toStringAsFixed(3)} '
            'p05=${pct(sorted, 0.05)} p50=${pct(sorted, 0.50)} '
            'p95=${pct(sorted, 0.95)} '
            'max=${sorted.last.toStringAsFixed(3)}');
      }
    }

    // Reference: thresholds currently shipped.
    const c = HeuristicsConfig();
    // ignore: avoid_print
    print('\n===== Current thresholds =====\n'
        'bounce_ratio > ${c.bounceRatioThreshold}\n'
        'lunge_knee_angle_deg < ${c.lungeKneeMinAngleDeg}\n'
        'guard_below_pelvis_max_run_s > ${c.guardDroppedSeconds} '
        '(FreeBouting ${c.guardDroppedFreeBoutingSeconds})\n'
        'avg_front_knee_angle_deg > ${c.stanceTooHighAngleDeg}\n'
        'arm_extension_angle_deg < ${c.incompleteArmExtensionAngleDeg}\n'
        'parry_sweep_torso_ratio > ${c.overParryTorsoRatioThreshold}\n'
        'step_ratio > ${c.wideStepRatioThreshold} (wide) / '
        '< ${c.narrowStepRatioThreshold} (narrow), sustained ${c.stepSustainedSeconds}s\n'
        'com_ratio > ${c.comInFrontRatioThreshold} (front) / '
        '< ${c.comLeaningBackRatioThreshold} (back), sustained ${c.comSustainedSeconds}s');
  }, timeout: const Timeout(Duration(minutes: 5)));
}
