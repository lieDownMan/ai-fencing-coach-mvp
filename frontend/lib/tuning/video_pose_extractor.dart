/// Offline video → per-frame YOLO pose detections, via the macOS
/// YoloVideoPoseBridge (see macos/Runner/MainFlutterWindow.swift).
///
/// Results are cached as `<video>.poses.json` next to the video file, storing
/// the RAW detection lists (pre-target-tracking), so switching targetSide in
/// the tuner never requires re-running CoreML.

library;

import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';

/// One decoded video frame: presentation time + raw YOLO detections in the
/// exact shape the native bridges emit (list of {bbox, confidence,
/// sourceRank, keypoints:[{index,x,y,confidence}]}).
class VideoPoseFrame {
  final int tMs;
  final List<dynamic> detections;

  const VideoPoseFrame({required this.tMs, required this.detections});
}

class VideoPoseData {
  final String videoPath;
  final List<VideoPoseFrame> frames;

  const VideoPoseData({required this.videoPath, required this.frames});
}

class VideoPoseExtractor {
  static const _cacheVersion = 1;
  static const MethodChannel _channel =
      MethodChannel('fencing_coach/yolo_video_pose');

  static void Function(int done, int total)? _onProgress;

  static bool _handlerInstalled = false;

  static void _ensureHandler() {
    if (_handlerInstalled) return;
    _handlerInstalled = true;
    _channel.setMethodCallHandler((call) async {
      if (call.method == 'progress') {
        final args = call.arguments as Map;
        _onProgress?.call(args['done'] as int, args['total'] as int);
      }
    });
  }

  static File _cacheFile(String videoPath) => File('$videoPath.poses.json');

  /// Extract poses for [videoPath], using the on-disk cache when present.
  static Future<VideoPoseData> extract({
    required String videoPath,
    required String modelPath,
    void Function(int done, int total)? onProgress,
    bool forceReextract = false,
  }) async {
    final cache = _cacheFile(videoPath);
    if (!forceReextract && await cache.exists()) {
      try {
        final raw = jsonDecode(await cache.readAsString()) as Map<String, dynamic>;
        if (raw['version'] == _cacheVersion) {
          final frames = (raw['frames'] as List)
              .map((f) => VideoPoseFrame(
                    tMs: f['tMs'] as int,
                    detections: f['detections'] as List<dynamic>,
                  ))
              .toList();
          return VideoPoseData(videoPath: videoPath, frames: frames);
        }
      } catch (_) {
        // Corrupt/stale cache → fall through to re-extraction.
      }
    }

    _ensureHandler();
    _onProgress = onProgress;
    try {
      final raw = await _channel.invokeMethod<List<dynamic>>('analyzeVideo', {
        'videoPath': videoPath,
        'modelPath': modelPath,
      });
      final frames = (raw ?? [])
          .map((f) => VideoPoseFrame(
                tMs: (f['tMs'] as num).toInt(),
                detections: f['detections'] as List<dynamic>,
              ))
          .toList();

      await cache.writeAsString(jsonEncode({
        'version': _cacheVersion,
        'videoPath': videoPath,
        'frames': [
          for (final f in frames) {'tMs': f.tMs, 'detections': f.detections},
        ],
      }));

      return VideoPoseData(videoPath: videoPath, frames: frames);
    } finally {
      _onProgress = null;
    }
  }
}
