/// YOLOv8-Pose processing service — converts MethodChannel parsed keypoints to fencing skeleton maps.
///
/// COCO Keypoints index mapping (17 keypoints):
///   0: nose
///   5: left shoulder,   6: right shoulder
///   7: left elbow,      8: right elbow
///   9: left wrist,     10: right wrist
///  11: left hip,       12: right hip
///  13: left knee,      14: right knee
///  15: left ankle,     16: right ankle

library;

import 'dart:ui' show Offset;
import 'package:flutter/services.dart';
import 'pose_service.dart';

class YoloPoseService {
  static const MethodChannel _channel = MethodChannel('fencing_coach/yolo_pose');
  
  bool _isLoaded = false;
  bool get isLoaded => _isLoaded;

  Future<bool> load() async {
    try {
      final result = await _channel.invokeMethod<bool>('load');
      _isLoaded = result == true;
      return _isLoaded;
    } catch (e) {
      _isLoaded = false;
      return false;
    }
  }

  /// Process raw camera image bytes and return the resolved FencingSkeleton.
  Future<FencingSkeleton?> processImageBytes({
    required Uint8List bytes,
    required int width,
    required int height,
    required int bytesPerRow,
    required String targetSide,
    required bool isFrontCamera,
  }) async {
    if (!_isLoaded) return null;
    
    try {
      final List<dynamic>? rawKeypoints = await _channel.invokeMethod<List<dynamic>>(
        'detectPose',
        {
          'bytes': bytes,
          'width': width,
          'height': height,
          'bytesPerRow': bytesPerRow,
        },
      );

      if (rawKeypoints == null || rawKeypoints.isEmpty) return null;

      // Map dynamic list to Keypoint map
      final Map<int, _Keypoint> keypoints = {};
      for (final item in rawKeypoints) {
        final Map map = item as Map;
        final idx = map['index'] as int;
        keypoints[idx] = _Keypoint(
          x: (map['x'] as num).toDouble(),
          y: (map['y'] as num).toDouble(),
          confidence: (map['confidence'] as num).toDouble(),
        );
      }

      return _buildFencingSkeleton(keypoints, targetSide, isFrontCamera);
    } catch (e) {
      return null;
    }
  }

  FencingSkeleton? _buildFencingSkeleton(
    Map<int, _Keypoint> keypoints,
    String targetSide,
    bool isFrontCamera,
  ) {
    // Width and height of normalized space is 1.0 (since Swift returns coordinates relative to [0, 1])
    const double imgW = 1.0;
    const double imgH = 1.0;

    Offset? _lm(int idx) {
      final kp = keypoints[idx];
      if (kp == null) return null;
      if (kp.confidence < 0.3) return null;
      
      // Perform rotation mapping from 1.0x1.0 landscape (from sensor bytes) to portrait
      final rawX = kp.x;
      final rawY = kp.y;

      if (isFrontCamera) {
        // Front camera (270 degrees rotation + mirrored)
        final x = imgH - rawY; // 1.0 - rawY
        final y = imgW - rawX; // 1.0 - rawX
        return Offset(x, y);
      } else {
        // Back camera (90 degrees rotation)
        final x = imgH - rawY; // 1.0 - rawY
        final y = rawX;
        return Offset(x, y);
      }
    }

    // Extract all joints
    final nose = _lm(0);
    final leftShoulder = _lm(5);
    final rightShoulder = _lm(6);
    final leftElbow = _lm(7);
    final rightElbow = _lm(8);
    final leftWrist = _lm(9);
    final rightWrist = _lm(10);
    final leftHip = _lm(11);
    final rightHip = _lm(12);
    final leftKnee = _lm(13);
    final rightKnee = _lm(14);
    final leftAnkle = _lm(15);
    final rightAnkle = _lm(16);

    // Map to fencing-relative skeleton
    // For a 'left' fencer (facing right on screen):
    //   front_wrist = right wrist (sword arm)
    //   front_elbow = right elbow
    //   front_shoulder = right shoulder
    //   front hip/knee/ankle = LEFT (leading) leg
    Offset? frontWrist, frontElbow, frontShoulder;
    if (targetSide == 'left') {
      frontWrist = rightWrist;
      frontElbow = rightElbow;
      frontShoulder = rightShoulder;
    } else {
      frontWrist = leftWrist;
      frontElbow = leftElbow;
      frontShoulder = leftShoulder;
    }

    // Build the skeleton map
    final skelMap = <String, Offset>{};
    void add(String key, Offset? v) {
      if (v != null) skelMap[key] = v;
    }

    add('nose', nose);
    add('front_wrist', frontWrist);
    add('front_elbow', frontElbow);
    add('front_shoulder', frontShoulder);
    add('left_hip', leftHip);
    add('right_hip', rightHip);
    add('left_knee', leftKnee);
    add('right_knee', rightKnee);
    add('left_ankle', leftAnkle);
    add('right_ankle', rightAnkle);
    // Extra joints for drawing
    add('left_shoulder', leftShoulder);
    add('right_shoulder', rightShoulder);
    add('left_elbow', leftElbow);
    add('right_elbow', rightElbow);
    add('left_wrist', leftWrist);
    add('right_wrist', rightWrist);

    // Need at least hips to be useful
    if (leftHip == null && rightHip == null) return null;

    // Compute nose→front_ankle scale for normalization
    Offset? frontAnkle =
        targetSide == 'left' ? leftAnkle : rightAnkle;
    double? scale;
    if (nose != null && frontAnkle != null) {
      scale = (frontAnkle.dy - nose.dy).abs();
      if (scale < 1e-6) scale = null;
    }

    return FencingSkeleton(
      joints: skelMap,
      nose: nose,
      scale: scale,
      imageWidth: imgH,  // Rotated width (1.0)
      imageHeight: imgW, // Rotated height (1.0)
    );
  }
}

class _Keypoint {
  final double x;
  final double y;
  final double confidence;

  _Keypoint({
    required this.x,
    required this.y,
    required this.confidence,
  });
}
