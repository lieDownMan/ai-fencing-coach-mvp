/// FenceNet action classifier using TFLite.
/// Input:  [1, 18, 28]  float32  (9 joints × 2 coords × 28 frames)
/// Output: [1,  6]      float32  logits for ["R","IS","WW","JS","SF","SB"]

library;

import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Offset;
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

const List<String> kFenceNetClassNames = ['R', 'IS', 'WW', 'JS', 'SF', 'SB'];

// Model input shape
const int kWindowSize = 28;
const int kNumJoints = 9;
const int kNumChannels = 18; // 9 joints × 2 coords

// Joint order expected by FenceNetV2 (matches training data)
const List<String> kModelJoints = [
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

const double kConfidenceThreshold = 0.6;

class FenceNetClassifier {
  Interpreter? _interpreter;
  bool _isLoaded = false;

  bool get isLoaded => _isLoaded;

  Future<void> load() async {
    try {
      // Check if asset exists before loading
      try {
        await rootBundle.load('assets/models/fencenet_v2.tflite');
      } catch (e) {
        // TFLite model not available (not yet converted), use rule-based fallback
        _isLoaded = false;
        return;
      }
      _interpreter = await Interpreter.fromAsset(
        'assets/models/fencenet_v2.tflite',
      );
      _isLoaded = true;
    } catch (e) {
      _isLoaded = false;
    }
  }

  /// Classify a 28-frame window of skeletons.
  ///
  /// [window] should contain exactly 28 skeleton maps.
  /// Returns null if model not loaded or window is invalid.
  ({String action, double confidence})? classify(
    List<Map<String, Offset>> window, {
    required String targetSide,
    required Offset? referenceNose,
    required double? referenceScale,
  }) {
    if (!_isLoaded || _interpreter == null) return null;
    if (window.length < kWindowSize) return null;

    // Build (18, 28) float32 tensor
    // Order: channel-first — matches training code (.T after reshape)
    final input = Float32List(kNumChannels * kWindowSize);

    for (int t = 0; t < kWindowSize; t++) {
      final skel = window[t];
      for (int j = 0; j < kNumJoints; j++) {
        final jointName = kModelJoints[j];
        final pt = skel[jointName];
        double x = 0, y = 0;
        if (pt != null && referenceNose != null && referenceScale != null && referenceScale > 1e-6) {
          x = (pt.dx - referenceNose.dx) / referenceScale;
          y = (pt.dy - referenceNose.dy) / referenceScale;
        }
        // channels-first layout: input[channel * kWindowSize + t]
        input[(j * 2) * kWindowSize + t] = x;
        input[(j * 2 + 1) * kWindowSize + t] = y;
      }
    }

    // Run inference
    final inputTensor = [input.reshape([1, kNumChannels, kWindowSize])];
    final outputTensor = [List.filled(kFenceNetClassNames.length, 0.0)];

    try {
      _interpreter!.runForMultipleInputs(inputTensor, {0: outputTensor});
    } catch (e) {
      return null;
    }

    final logits = outputTensor[0] as List<double>;

    // Softmax
    final maxLogit = logits.reduce(math.max);
    final exps = logits.map((l) => math.exp(l - maxLogit)).toList();
    final sumExps = exps.reduce((a, b) => a + b);
    final probs = exps.map((e) => e / sumExps).toList();

    final maxProb = probs.reduce(math.max);
    final maxIdx = probs.indexOf(maxProb);

    if (maxProb < kConfidenceThreshold) {
      return (action: 'Idle', confidence: maxProb);
    }
    return (action: kFenceNetClassNames[maxIdx], confidence: maxProb);
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _isLoaded = false;
  }
}

// ---------------------------------------------------------------------------
// Extension for reshape (since tflite_flutter doesn't expose this cleanly)
// ---------------------------------------------------------------------------

extension ListReshape on Float32List {
  List<dynamic> reshape(List<int> shape) {
    return _reshapeHelper(this, shape) as List<dynamic>;
  }

  static dynamic _reshapeHelper(dynamic data, List<int> shape) {
    if (shape.length == 1) return data;
    final int count = (data as dynamic).length as int;
    final int size = count ~/ shape[0];
    return List<dynamic>.generate(
      shape[0],
      (i) => _reshapeHelper(
        data is Float32List
            ? data.sublist(i * size, (i + 1) * size)
            : (data as List<dynamic>).sublist(i * size, (i + 1) * size),
        shape.sublist(1),
      ),
    );
  }
}


