import 'dart:async';
import 'dart:io' show File;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:video_player/video_player.dart';
import 'package:video_thumbnail/video_thumbnail.dart';

import '../heuristics/fencenet_channel.dart';
import '../heuristics/heuristics_engine.dart';
import '../pose/pose_service.dart';
import '../pose/yolo_pose_service.dart';

typedef PostgameProgressCallback = void Function(double fraction, String status);

const List<String> _modelJoints = [
  'front_wrist',
  'front_elbow',
  'front_shoulder',
  'left_hip',
  'right_hip',
  'left_knee',
  'right_knee',
  'left_ankle',
  'right_ankle',
];

class PostgameAnalysisConfig {
  final String targetSide;
  final String trainingMode;
  final List<String> focusErrors;
  final List<String> muteErrors;
  final bool onlySelected;

  const PostgameAnalysisConfig({
    required this.targetSide,
    required this.trainingMode,
    this.focusErrors = const [],
    this.muteErrors = const [],
    this.onlySelected = false,
  });
}

class PostgameTimelineItem {
  final double timeSeconds;
  final String action;
  final double confidence;
  final List<String> errors;

  const PostgameTimelineItem({
    required this.timeSeconds,
    required this.action,
    required this.confidence,
    required this.errors,
  });
}

class PostgameReport {
  final String videoName;
  final Duration duration;
  final int framesAnalyzed;
  final int modelChecks;
  final Map<String, int> actionCounts;
  final Map<String, int> errorCounts;
  final List<PostgameTimelineItem> timeline;

  const PostgameReport({
    required this.videoName,
    required this.duration,
    required this.framesAnalyzed,
    required this.modelChecks,
    required this.actionCounts,
    required this.errorCounts,
    required this.timeline,
  });

  String get topAction {
    final entries = actionCounts.entries.where((entry) => entry.key != 'Idle').toList();
    if (entries.isEmpty) return 'Idle';
    entries.sort((a, b) => b.value.compareTo(a.value));
    return entries.first.key;
  }

  List<MapEntry<String, int>> get topErrors {
    final entries = errorCounts.entries.toList();
    entries.sort((a, b) {
      final byCount = b.value.compareTo(a.value);
      return byCount != 0 ? byCount : a.key.compareTo(b.key);
    });
    return entries;
  }

  String primaryTakeaway(Map<String, String> labels) {
    final sortedErrors = topErrors;
    if (sortedErrors.isNotEmpty) {
      final topError = sortedErrors.first;
      return '${labels[topError.key] ?? topError.key} appeared ${topError.value} times.';
    }
    if (modelChecks > 0) {
      return 'No posture cues were detected in the sampled windows.';
    }
    return 'The clip did not contain enough usable fencing poses for a model check.';
  }
}

class PostgameAnalyzer {
  static const int windowSize = 28;
  static const int stride = 10;
  static const int frameStepMs = 67;

  final YoloPoseService poseService;
  final FenceNetChannel fenceNet;
  final Map<String, List<String>> supportedModes;

  PostgameAnalyzer({
    required this.poseService,
    required this.fenceNet,
    required this.supportedModes,
  });

  Future<PostgameReport> analyzeVideo({
    required String path,
    required String videoName,
    required PostgameAnalysisConfig config,
    PostgameProgressCallback? onProgress,
  }) async {
    if (!poseService.isLoaded) {
      throw StateError('Pose model is not loaded yet.');
    }
    if (!fenceNet.isLoaded) {
      throw StateError('FenceNet action model is not loaded yet.');
    }

    final controller = VideoPlayerController.file(File(path));
    try {
      onProgress?.call(0.02, 'Reading clip');
      await controller.initialize();
      final duration = controller.value.duration;
      final durationMs = duration.inMilliseconds > 0 ? duration.inMilliseconds : frameStepMs;
      final sampleTimes = _sampleTimes(durationMs);

      final heuristics = HeuristicsEngine(
        targetSide: config.targetSide,
        trainingMode: config.trainingMode,
      );
      final skeletonWindow = <Skeleton>[];
      final classifierWindow = <Skeleton>[];
      final actionCounts = <String, int>{};
      final errorCounts = <String, int>{};
      final timeline = <PostgameTimelineItem>[];

      Offset? referenceNose;
      double? referenceScale;
      int framesAnalyzed = 0;
      int modelChecks = 0;

      poseService.resetTracking();

      for (var index = 0; index < sampleTimes.length; index++) {
        final timeMs = sampleTimes[index];
        onProgress?.call(
          0.05 + 0.75 * (index / sampleTimes.length),
          'Analyzing frame ${index + 1}/${sampleTimes.length}',
        );

        final thumbnail = await VideoThumbnail.thumbnailData(
          video: path,
          imageFormat: ImageFormat.PNG,
          maxWidth: 640,
          quality: 85,
          timeMs: timeMs,
        );
        if (thumbnail == null || thumbnail.isEmpty) continue;

        final frame = await _decodeThumbnail(thumbnail);
        if (frame == null) continue;

        final skeleton = await poseService.processImageBytes(
          bytes: frame.bgraBytes,
          width: frame.width,
          height: frame.height,
          bytesPerRow: frame.width * 4,
          targetSide: config.targetSide,
          isFrontCamera: false,
        );
        if (skeleton == null) continue;

        framesAnalyzed += 1;
        referenceNose ??= skeleton.nose;
        referenceScale ??= skeleton.scale;

        skeletonWindow.add(skeleton.joints);
        if (skeletonWindow.length > 60) skeletonWindow.removeAt(0);
        classifierWindow.add(skeleton.joints);
        if (classifierWindow.length > windowSize) classifierWindow.removeAt(0);

        if (classifierWindow.length != windowSize || framesAnalyzed % stride != 0) {
          continue;
        }

        final input = _buildFenceNetInput(
          classifierWindow,
          referenceNose: referenceNose,
          referenceScale: referenceScale,
        );
        final result = await fenceNet.classify(input);
        final action = result.action;
        final confidence = result.confidence;

        modelChecks += 1;
        actionCounts[action] = (actionCounts[action] ?? 0) + 1;
        if (action == 'Idle') continue;

        final errors = _filterErrors(
          heuristics.evaluateWindow(
            action: action,
            skeletons: List<Skeleton>.from(skeletonWindow),
          ),
          config,
        );
        for (final key in errors) {
          errorCounts[key] = (errorCounts[key] ?? 0) + 1;
        }
        timeline.add(
          PostgameTimelineItem(
            timeSeconds: timeMs / 1000.0,
            action: action,
            confidence: confidence,
            errors: errors,
          ),
        );
      }

      onProgress?.call(1.0, 'Analysis ready');
      return PostgameReport(
        videoName: videoName,
        duration: duration,
        framesAnalyzed: framesAnalyzed,
        modelChecks: modelChecks,
        actionCounts: actionCounts,
        errorCounts: errorCounts,
        timeline: timeline,
      );
    } finally {
      await controller.dispose();
    }
  }

  List<int> _sampleTimes(int durationMs) {
    final times = <int>[];
    for (var timeMs = 0; timeMs <= durationMs; timeMs += frameStepMs) {
      times.add(timeMs);
    }
    if (times.isEmpty) times.add(0);
    return times;
  }

  List<String> _filterErrors(
    Iterable<String> errors,
    PostgameAnalysisConfig config,
  ) {
    final focus = config.focusErrors.toSet();
    final mute = config.muteErrors.toSet();
    final filtered = errors.where((key) {
      final modes = supportedModes[key] ?? const <String>[];
      if (!modes.contains(config.trainingMode)) return false;
      if (mute.contains(key)) return false;
      if (config.onlySelected && focus.isNotEmpty && !focus.contains(key)) {
        return false;
      }
      return true;
    }).toList();

    filtered.sort((a, b) {
      final aFocus = focus.contains(a) ? 0 : 1;
      final bFocus = focus.contains(b) ? 0 : 1;
      final focusOrder = aFocus.compareTo(bFocus);
      return focusOrder != 0 ? focusOrder : a.compareTo(b);
    });
    return filtered;
  }

  List<double> _buildFenceNetInput(
    List<Skeleton> window, {
    required ui.Offset? referenceNose,
    required double? referenceScale,
  }) {
    final input = List<double>.filled(18 * windowSize, 0.0);
    for (var t = 0; t < windowSize; t++) {
      final skeleton = window[t];
      for (var j = 0; j < _modelJoints.length; j++) {
        final point = skeleton[_modelJoints[j]];
        var x = 0.0;
        var y = 0.0;
        if (point != null &&
            referenceNose != null &&
            referenceScale != null &&
            referenceScale > 1e-6) {
          x = (point.dx - referenceNose.dx) / referenceScale;
          y = (point.dy - referenceNose.dy) / referenceScale;
        }
        input[(j * 2) * windowSize + t] = x;
        input[(j * 2 + 1) * windowSize + t] = y;
      }
    }
    return input;
  }

  Future<_DecodedFrame?> _decodeThumbnail(Uint8List bytes) async {
    final image = await _decodeImage(bytes);
    final byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    final width = image.width;
    final height = image.height;
    image.dispose();
    if (byteData == null) return null;
    return _DecodedFrame(
      width: width,
      height: height,
      bgraBytes: _rgbaToBgra(byteData.buffer.asUint8List()),
    );
  }

  Future<ui.Image> _decodeImage(Uint8List bytes) {
    final completer = Completer<ui.Image>();
    ui.decodeImageFromList(bytes, completer.complete);
    return completer.future;
  }

  Uint8List _rgbaToBgra(Uint8List rgba) {
    final bgra = Uint8List(rgba.length);
    for (var i = 0; i + 3 < rgba.length; i += 4) {
      bgra[i] = rgba[i + 2];
      bgra[i + 1] = rgba[i + 1];
      bgra[i + 2] = rgba[i];
      bgra[i + 3] = rgba[i + 3];
    }
    return bgra;
  }
}

class _DecodedFrame {
  final int width;
  final int height;
  final Uint8List bgraBytes;

  const _DecodedFrame({
    required this.width,
    required this.height,
    required this.bgraBytes,
  });
}
