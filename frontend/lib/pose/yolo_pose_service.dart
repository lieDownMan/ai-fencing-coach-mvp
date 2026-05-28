/// YOLOv8-Pose service for the Flutter app.
///
/// This mirrors the main-branch Python contract:
/// YOLO COCO keypoints -> fencing skeleton keys -> side-based target tracking.

library;

import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Offset;

import 'package:flutter/services.dart';

import 'pose_service.dart';

class YoloPoseService {
  static const MethodChannel _channel = MethodChannel('fencing_coach/yolo_pose');

  final _TargetTracker _tracker = _TargetTracker();
  bool _isLoaded = false;
  String? _configuredTargetSide;

  bool get isLoaded => _isLoaded;

  Future<bool> load() async {
    try {
      final result = await _channel.invokeMethod<bool>('load');
      _isLoaded = result == true;
      return _isLoaded;
    } catch (_) {
      _isLoaded = false;
      return false;
    }
  }

  void resetTracking() {
    _tracker.reset();
    _configuredTargetSide = null;
  }

  /// Process a camera frame and return the selected fencer skeleton.
  Future<FencingSkeleton?> processImageBytes({
    required Uint8List bytes,
    required int width,
    required int height,
    required int bytesPerRow,
    required String targetSide,
    required bool isFrontCamera,
  }) async {
    if (!_isLoaded) return null;

    if (_configuredTargetSide != targetSide) {
      _tracker.reset();
      _configuredTargetSide = targetSide;
    }

    try {
      final raw = await _channel.invokeMethod<List<dynamic>>(
        'detectPose',
        {
          'bytes': bytes,
          'width': width,
          'height': height,
          'bytesPerRow': bytesPerRow,
        },
      );

      final detections = _parseDetections(raw, isFrontCamera);
      return _tracker.process(detections, targetSide)?.skeleton;
    } catch (_) {
      return null;
    }
  }

  List<_PoseDetection> _parseDetections(
    List<dynamic>? raw,
    bool isFrontCamera,
  ) {
    if (raw == null || raw.isEmpty) return const [];

    // Backward compatibility for the older bridge shape that returned only a
    // keypoint list: [{"index": 0, "x": ..., "y": ..., "confidence": ...}].
    final first = raw.first;
    if (first is Map && first.containsKey('index')) {
      final detection = _detectionFromKeypointItems(
        keypointItems: raw,
        confidence: 1.0,
        sourceRank: 0,
        isFrontCamera: isFrontCamera,
      );
      return detection == null ? const [] : [detection];
    }

    final detections = <_PoseDetection>[];
    for (final item in raw) {
      if (item is! Map) continue;
      final keypointItems = item['keypoints'];
      if (keypointItems is! List) continue;

      final detection = _detectionFromKeypointItems(
        keypointItems: keypointItems,
        confidence: (item['confidence'] as num?)?.toDouble() ?? 0.0,
        sourceRank: (item['sourceRank'] as num?)?.toInt() ?? detections.length,
        isFrontCamera: isFrontCamera,
      );
      if (detection != null) detections.add(detection);
    }

    detections.sort((a, b) => b.area.compareTo(a.area));
    return detections;
  }

  _PoseDetection? _detectionFromKeypointItems({
    required List<dynamic> keypointItems,
    required double confidence,
    required int sourceRank,
    required bool isFrontCamera,
  }) {
    final keypoints = <int, _Keypoint>{};
    for (final item in keypointItems) {
      if (item is! Map) continue;
      final idx = (item['index'] as num?)?.toInt();
      if (idx == null) continue;
      keypoints[idx] = _Keypoint(
        x: (item['x'] as num?)?.toDouble() ?? 0.0,
        y: (item['y'] as num?)?.toDouble() ?? 0.0,
        confidence: (item['confidence'] as num?)?.toDouble() ?? 0.0,
      );
    }

    final skeleton = _buildFencingSkeleton(keypoints, isFrontCamera);
    if (skeleton == null) return null;
    final bbox = _BoundingBox.fromSkeleton(skeleton);
    if (bbox == null) return null;

    return _PoseDetection(
      skeleton: skeleton,
      bbox: bbox,
      confidence: confidence,
      sourceRank: sourceRank,
    );
  }

  FencingSkeleton? _buildFencingSkeleton(
    Map<int, _Keypoint> keypoints,
    bool isFrontCamera,
  ) {
    const imgW = 1.0;
    const imgH = 1.0;

    Offset? lm(int idx) {
      final kp = keypoints[idx];
      if (kp == null || kp.confidence < 0.35) return null;

      final rawX = kp.x.clamp(0.0, 1.0).toDouble();
      final rawY = kp.y.clamp(0.0, 1.0).toDouble();

      if (isFrontCamera) {
        // Current: (imgH - rawY, imgW - rawX)
        // Rotated 90 CCW (y, 1 - x): (imgW - rawX, rawY)
        return Offset(imgW - rawX, rawY);
      }
      // Current: (imgH - rawY, rawX)
      // Rotated 90 CCW (y, 1 - x): (rawX, rawY)
      return Offset(rawX, rawY);
    }

    final nose = lm(0);
    final leftShoulder = lm(5);
    final rightShoulder = lm(6);
    final leftElbow = lm(7);
    final rightElbow = lm(8);
    final leftWrist = lm(9);
    final rightWrist = lm(10);
    final leftHip = lm(11);
    final rightHip = lm(12);
    final leftKnee = lm(13);
    final rightKnee = lm(14);
    final leftAnkle = lm(15);
    final rightAnkle = lm(16);

    // Main-branch YOLO assumes the front/sword side is the COCO right side.
    final frontWrist = rightWrist;
    final frontElbow = rightElbow;
    final frontShoulder = rightShoulder;
    final frontAnkle = rightAnkle;
    final backWrist = leftWrist;

    if (nose == null ||
        frontWrist == null ||
        frontElbow == null ||
        frontShoulder == null ||
        frontAnkle == null ||
        leftHip == null ||
        rightHip == null ||
        leftKnee == null ||
        rightKnee == null ||
        leftAnkle == null ||
        rightAnkle == null) {
      return null;
    }

    final joints = <String, Offset>{
      'nose': nose,
      'front_wrist': frontWrist,
      'front_elbow': frontElbow,
      'front_shoulder': frontShoulder,
      'front_ankle': frontAnkle,
      'left_hip': leftHip,
      'right_hip': rightHip,
      'left_knee': leftKnee,
      'right_knee': rightKnee,
      'left_ankle': leftAnkle,
      'right_ankle': rightAnkle,
      'left_shoulder': leftShoulder ?? frontShoulder,
      'right_shoulder': rightShoulder ?? frontShoulder,
      'left_elbow': leftElbow ?? frontElbow,
      'right_elbow': rightElbow ?? frontElbow,
      'left_wrist': leftWrist ?? frontWrist,
      'right_wrist': rightWrist ?? frontWrist,
    };
    if (backWrist != null) joints['back_wrist'] = backWrist;

    final scale = (frontAnkle.dy - nose.dy).abs();

    return FencingSkeleton(
      joints: joints,
      nose: nose,
      scale: scale < 1e-6 ? null : scale,
      imageWidth: imgW,
      imageHeight: imgH,
    );
  }
}

class _TargetTracker {
  String targetSide = 'left';
  _BoundingBox? _lockedFallbackBBox;
  FencingSkeleton? _lastKnownSkeleton;
  _BoundingBox? _lastKnownBBox;
  int _missingFramesCount = 0;

  static const int _maxMissingFrames = 5;
  static const double _maxPositionJump = 1.75;

  void reset() {
    _lockedFallbackBBox = null;
    _lastKnownSkeleton = null;
    _lastKnownBBox = null;
    _missingFramesCount = 0;
  }

  _PoseDetection? process(List<_PoseDetection> detections, String side) {
    if (targetSide != side) {
      targetSide = side;
      reset();
    }

    final valid = detections.where((d) => d.area > 0).toList();
    if (valid.isEmpty) return _handleMissingTarget();

    if (_lockedFallbackBBox == null) {
      final initial = _pickInitialTarget(valid);
      _lockedFallbackBBox = initial.bbox;
    }

    final target = _matchByPosition(valid) ?? _pickInitialTarget(valid);
    _rememberTarget(target);
    return target;
  }

  _PoseDetection _pickInitialTarget(List<_PoseDetection> detections) {
    if (targetSide == 'left') {
      return detections.reduce(
        (best, d) => d.bbox.centerX < best.bbox.centerX ? d : best,
      );
    }
    return detections.reduce(
      (best, d) => d.bbox.centerX > best.bbox.centerX ? d : best,
    );
  }

  _PoseDetection? _matchByPosition(List<_PoseDetection> detections) {
    final reference = _lastKnownBBox ?? _lockedFallbackBBox;
    if (reference == null) return null;

    _PoseDetection? best;
    double bestScore = double.infinity;
    for (final detection in detections) {
      final score = _positionScore(detection, reference);
      if (score < bestScore) {
        best = detection;
        bestScore = score;
      }
    }

    return bestScore <= _maxPositionJump ? best : null;
  }

  double _positionScore(_PoseDetection detection, _BoundingBox reference) {
    final dx = detection.bbox.centerX - reference.centerX;
    final dy = detection.bbox.centerY - reference.centerY;
    final diagonal = reference.diagonal.clamp(1e-6, double.infinity).toDouble();
    final centerDistance = math.sqrt(dx * dx + dy * dy) / diagonal;
    final areaRatio = detection.area /
        reference.area.clamp(1e-6, double.infinity).toDouble();
    return centerDistance + 0.25 * math.log(areaRatio).abs();
  }

  void _rememberTarget(_PoseDetection target) {
    _lastKnownSkeleton = target.skeleton;
    _lastKnownBBox = target.bbox;
    _lockedFallbackBBox = target.bbox;
    _missingFramesCount = 0;
  }

  _PoseDetection? _handleMissingTarget() {
    final skeleton = _lastKnownSkeleton;
    final bbox = _lastKnownBBox;
    if (skeleton != null && bbox != null && _missingFramesCount < _maxMissingFrames) {
      _missingFramesCount += 1;
      return _PoseDetection(
        skeleton: skeleton,
        bbox: bbox,
        confidence: 0.0,
        sourceRank: -1,
      );
    }
    return null;
  }
}

class _PoseDetection {
  final FencingSkeleton skeleton;
  final _BoundingBox bbox;
  final double confidence;
  final int sourceRank;

  const _PoseDetection({
    required this.skeleton,
    required this.bbox,
    required this.confidence,
    required this.sourceRank,
  });

  double get area => bbox.area;
}

class _BoundingBox {
  final double x1;
  final double y1;
  final double x2;
  final double y2;

  const _BoundingBox({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
  });

  factory _BoundingBox.fromPoints(Iterable<Offset> points) {
    final xs = points.map((p) => p.dx).toList();
    final ys = points.map((p) => p.dy).toList();
    return _BoundingBox(
      x1: xs.reduce(math.min),
      y1: ys.reduce(math.min),
      x2: xs.reduce(math.max),
      y2: ys.reduce(math.max),
    );
  }

  static _BoundingBox? fromSkeleton(FencingSkeleton skeleton) {
    if (skeleton.joints.isEmpty) return null;
    return _BoundingBox.fromPoints(skeleton.joints.values);
  }

  double get width => (x2 - x1).clamp(0.0, double.infinity).toDouble();
  double get height => (y2 - y1).clamp(0.0, double.infinity).toDouble();
  double get area => width * height;
  double get centerX => (x1 + x2) / 2.0;
  double get centerY => (y1 + y2) / 2.0;
  double get diagonal => math.sqrt(width * width + height * height);
}

class _Keypoint {
  final double x;
  final double y;
  final double confidence;

  const _Keypoint({
    required this.x,
    required this.y,
    required this.confidence,
  });
}
