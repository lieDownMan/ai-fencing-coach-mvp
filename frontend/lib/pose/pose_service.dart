/// Pose processing service — converts ML Kit landmarks to fencing skeleton maps.
///
/// ML Kit BlazePose landmark indices (33 landmarks):
/// https://developers.google.com/ml-kit/vision/pose-detection
///   0  = nose
///  11  = left shoulder,  12 = right shoulder
///  13  = left elbow,     14 = right elbow
///  15  = left wrist,     16 = right wrist
///  23  = left hip,       24 = right hip
///  25  = left knee,      26 = right knee
///  27  = left ankle,     28 = right ankle

library;

import 'dart:ui' show Offset;
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum in-frame likelihood to accept a landmark.
const double kMinLandmarkConfidence = 0.3;

// ---------------------------------------------------------------------------
// PoseService — detects poses and maps landmarks to fencing skeleton
// ---------------------------------------------------------------------------

class PoseService {
  late final PoseDetector _detector;
  bool _isInitialized = false;

  PoseService() {
    _detector = PoseDetector(
      options: PoseDetectorOptions(
        mode: PoseDetectionMode.stream,
        model: PoseDetectionModel.accurate,
      ),
    );
    _isInitialized = true;
  }

  bool get isInitialized => _isInitialized;

  /// Process an [InputImage] and return the resolved fencing skeleton.
  ///
  /// Returns `null` if no pose found or insufficient landmarks.
  Future<FencingSkeleton?> processImage(
    InputImage image, {
    required String targetSide,
    required double imageWidth,
    required double imageHeight,
  }) async {
    if (!_isInitialized) return null;

    List<Pose> poses;
    try {
      poses = await _detector.processImage(image);
    } catch (e) {
      return null;
    }

    if (poses.isEmpty) return null;

    // Pick the pose with the highest average landmark confidence
    final pose = poses.reduce((a, b) {
      final aConf = _avgConf(a);
      final bConf = _avgConf(b);
      return aConf >= bConf ? a : b;
    });

    return _buildFencingSkeleton(pose, targetSide, imageWidth, imageHeight);
  }

  double _avgConf(Pose pose) {
    if (pose.landmarks.isEmpty) return 0;
    final values = pose.landmarks.values
        .map((l) => (l.likelihood ?? 0).toDouble())
        .toList();
    return values.reduce((a, b) => a + b) / values.length;
  }

  FencingSkeleton? _buildFencingSkeleton(
    Pose pose,
    String targetSide,
    double imgW,
    double imgH,
  ) {
    Offset? _lm(PoseLandmarkType type) {
      final lm = pose.landmarks[type];
      if (lm == null) return null;
      if ((lm.likelihood ?? 0) < kMinLandmarkConfidence) return null;
      // Landmarks are in image pixel coords
      return Offset(lm.x.toDouble(), lm.y.toDouble());
    }

    // Extract all joints
    final nose = _lm(PoseLandmarkType.nose);
    final leftShoulder = _lm(PoseLandmarkType.leftShoulder);
    final rightShoulder = _lm(PoseLandmarkType.rightShoulder);
    final leftElbow = _lm(PoseLandmarkType.leftElbow);
    final rightElbow = _lm(PoseLandmarkType.rightElbow);
    final leftWrist = _lm(PoseLandmarkType.leftWrist);
    final rightWrist = _lm(PoseLandmarkType.rightWrist);
    final leftHip = _lm(PoseLandmarkType.leftHip);
    final rightHip = _lm(PoseLandmarkType.rightHip);
    final leftKnee = _lm(PoseLandmarkType.leftKnee);
    final rightKnee = _lm(PoseLandmarkType.rightKnee);
    final leftAnkle = _lm(PoseLandmarkType.leftAnkle);
    final rightAnkle = _lm(PoseLandmarkType.rightAnkle);

    // Map to fencing-relative skeleton
    // For a 'left' fencer (facing right on screen):
    //   front_wrist = right wrist (sword arm)
    //   front_elbow = right elbow
    //   front_shoulder = right shoulder
    //   front hip/knee/ankle = LEFT (leading) leg
    //
    // For a 'right' fencer (facing left on screen):
    //   front_wrist = left wrist
    //   front_elbow = left elbow
    //   front_shoulder = left shoulder
    //   front hip/knee/ankle = RIGHT (leading) leg

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
      imageWidth: imgW,
      imageHeight: imgH,
    );
  }

  void dispose() {
    _detector.close();
    _isInitialized = false;
  }
}

// ---------------------------------------------------------------------------
// FencingSkeleton — immutable snapshot of one frame's pose
// ---------------------------------------------------------------------------

class FencingSkeleton {
  final Map<String, Offset> joints;
  final Offset? nose;
  final double? scale; // nose-to-front-ankle pixel distance
  final double imageWidth;
  final double imageHeight;

  const FencingSkeleton({
    required this.joints,
    required this.nose,
    required this.scale,
    required this.imageWidth,
    required this.imageHeight,
  });

  Offset? operator [](String key) => joints[key];
}
