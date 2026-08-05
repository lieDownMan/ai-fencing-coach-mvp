/// Replays a video's raw pose detections through the EXACT live pipeline:
/// YoloPoseService parsing + target tracking, then the HeuristicsEngine over
/// the same rolling window the app uses (60-frame buffer of person-frames,
/// evaluated every 10 pose frames after a 28-frame warmup, fps measured from
/// the last ≤30 pose timestamps — mirrors main.dart's _onCameraFrame /
/// _effectivePoseFps / _classifyAndEvaluate).
///
/// Threshold changes only need [evaluate] re-run — pure Dart math, instant.

library;

import '../heuristics/heuristics_engine.dart';
import '../pose/yolo_pose_service.dart';
import 'video_pose_extractor.dart';

/// A frame where the tracker produced a skeleton (the only frames the live
/// pipeline buffers).
class PoseFrame {
  final int tMs;
  final Skeleton joints;

  const PoseFrame({required this.tMs, required this.joints});
}

/// One evaluation point (window ending at [endMs]).
class WindowEval {
  final int startMs;
  final int endMs;
  final double fps;
  final Map<String, double> metrics;
  final List<String> triggered; // error keys from evaluateWindow

  const WindowEval({
    required this.startMs,
    required this.endMs,
    required this.fps,
    required this.metrics,
    required this.triggered,
  });
}

class ReplayResult {
  final List<PoseFrame> poseFrames;
  final List<WindowEval> windows;
  final int totalVideoFrames;

  const ReplayResult({
    required this.poseFrames,
    required this.windows,
    required this.totalVideoFrames,
  });
}

/// Mirrors the live pipeline's cadence (main.dart).
const int kWindowLen = 60;
const int kEvalStride = 10;
const int kWarmupFrames = 28; // classifier-buffer gate before first eval
const int kFpsWindow = 30; // timestamps used for effective-fps estimate

/// Run target tracking over the raw detections. Sequential + stateful, same
/// as live: frames with no tracked person are dropped from the buffer.
List<PoseFrame> trackSkeletons(VideoPoseData data, String targetSide) {
  final service = YoloPoseService();
  final out = <PoseFrame>[];
  for (final frame in data.frames) {
    final skel = service.processDetectionList(
      frame.detections,
      targetSide: targetSide,
    );
    if (skel != null) {
      out.add(PoseFrame(tMs: frame.tMs, joints: skel.joints));
    }
  }
  return out;
}

/// Evaluate every window position over the tracked pose frames.
/// [action] is the FenceNet class assumed for the cue being tuned (the tuner
/// has no classifier; each cue maps to the action context that would run its
/// check live — see kCueAction in tuner_cues.dart).
ReplayResult evaluateReplay({
  required VideoPoseData data,
  required List<PoseFrame> poseFrames,
  required HeuristicsConfig config,
  required String targetSide,
  required String action,
}) {
  final engine = HeuristicsEngine(
    targetSide: targetSide,
    trainingMode: 'Target Practice',
    config: config,
  );

  final windows = <WindowEval>[];
  for (int f = 1; f <= poseFrames.length; f++) {
    if (f % kEvalStride != 0 || f < kWarmupFrames) continue;

    final start = f - kWindowLen < 0 ? 0 : f - kWindowLen;
    final window = poseFrames.sublist(start, f);
    final skeletons = [for (final p in window) p.joints];

    // Effective fps from the last ≤30 pose timestamps (main._effectivePoseFps).
    final fpsStart = f - kFpsWindow < 0 ? 0 : f - kFpsWindow;
    final stamps = poseFrames.sublist(fpsStart, f);
    double fps = 30.0;
    if (stamps.length >= 5) {
      final spanMs = stamps.last.tMs - stamps.first.tMs;
      if (spanMs > 0) fps = (stamps.length - 1) * 1000.0 / spanMs;
    }

    windows.add(WindowEval(
      startMs: window.first.tMs,
      endMs: window.last.tMs,
      fps: fps,
      metrics: engine.computeWindowMetrics(skeletons, fps: fps),
      triggered: engine.evaluateWindow(
        action: action,
        skeletons: skeletons,
        fps: fps,
      ),
    ));
  }

  return ReplayResult(
    poseFrames: poseFrames,
    windows: windows,
    totalVideoFrames: data.frames.length,
  );
}
