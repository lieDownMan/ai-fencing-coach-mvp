/// Dart client for the FenceNet CoreML model via MethodChannel.
/// Communicates with FenceNetBridge.swift on iOS.

library;

import 'package:flutter/services.dart';

const List<String> kFenceNetClasses = ['R', 'IS', 'WW', 'JS', 'SF', 'SB'];

class FenceNetChannel {
  static const MethodChannel _channel =
      MethodChannel('fencing_coach/fencenet');

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

  /// Classify a 28-frame skeleton window.
  ///
  /// [flatInput] must be Float32List of length 504 (18 channels × 28 frames),
  /// laid out in channel-first order: [ch0_t0, ch0_t1, ..., ch17_t27].
  Future<({String action, double confidence})> classify(
    List<double> flatInput,
  ) async {
    assert(flatInput.length == 18 * 28,
        'Expected 504 values, got ${flatInput.length}');
    try {
      final result = await _channel.invokeMethod<Map>('classify', {
        'input': flatInput,
      });
      if (result == null) return (action: 'Idle', confidence: 0.0);
      return (
        action: result['action'] as String? ?? 'Idle',
        confidence: (result['confidence'] as num?)?.toDouble() ?? 0.0,
      );
    } catch (e) {
      return (action: 'Idle', confidence: 0.0);
    }
  }
}
