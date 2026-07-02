/// Custom painter that draws the fencing pose skeleton on top of camera preview.

library;

import 'package:flutter/material.dart';
import 'pose_service.dart';

// ---------------------------------------------------------------------------
// Skeleton connection groups (for color coding)
// ---------------------------------------------------------------------------

// Upper body: green
const List<(String, String)> kUpperConnections = [
  ('nose', 'left_shoulder'),
  ('nose', 'right_shoulder'),
  ('left_shoulder', 'right_shoulder'),
  ('left_shoulder', 'left_elbow'),
  ('left_elbow', 'left_wrist'),
  ('right_shoulder', 'right_elbow'),
  ('right_elbow', 'right_wrist'),
];

// Torso: green (links upper to lower)
const List<(String, String)> kTorsoConnections = [
  ('left_shoulder', 'left_hip'),
  ('right_shoulder', 'right_hip'),
  ('left_hip', 'right_hip'),
];

// Lower body: blue
const List<(String, String)> kLowerConnections = [
  ('left_hip', 'left_knee'),
  ('left_knee', 'left_ankle'),
  ('right_hip', 'right_knee'),
  ('right_knee', 'right_ankle'),
];

// Upper body joint names
const Set<String> kUpperJoints = {
  'nose',
  'left_shoulder', 'right_shoulder',
  'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist',
  'front_shoulder', 'front_elbow', 'front_wrist',
};

// Lower body joint names
const Set<String> kLowerJoints = {
  'left_hip', 'right_hip',
  'left_knee', 'right_knee',
  'left_ankle', 'right_ankle',
};

// ---------------------------------------------------------------------------
// PosePainter
// ---------------------------------------------------------------------------

class PosePainter extends CustomPainter {
  final FencingSkeleton skeleton;
  final Size imageSize;
  final String? triggeredError;
  final String currentAction;
  final bool isFrontCamera;

  PosePainter({
    required this.skeleton,
    required this.imageSize,
    this.triggeredError,
    required this.currentAction,
    this.isFrontCamera = false,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final scaleX = size.width / skeleton.imageWidth;
    final scaleY = size.height / skeleton.imageHeight;

    Offset mapPoint(Offset pt) => Offset(pt.dx * scaleX, pt.dy * scaleY);

    // Error flash overrides all colors with red
    final bool hasError = triggeredError != null;

    // ── Paint factories ────────────────────────────────────────────────────

    Paint bonePaint(Color color) => Paint()
      ..color = hasError ? Colors.redAccent.withAlpha(220) : color
      ..strokeWidth = 3.0
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    Paint jointPaint(Color color) => Paint()
      ..color = hasError ? Colors.red.withAlpha(220) : color
      ..style = PaintingStyle.fill;

    // Upper body: vivid green
    final upperPaint = bonePaint(const Color(0xFF00E676).withAlpha(220)); // green accent
    // Torso: green (same as upper)
    final torsoPaint = bonePaint(const Color(0xFF00E676).withAlpha(180));
    // Lower body: vivid blue
    final lowerPaint = bonePaint(const Color(0xFF2979FF).withAlpha(220)); // blue accent

    // Joint dot paints
    final upperJointPaint  = jointPaint(const Color(0xFF69FF47).withAlpha(240)); // bright green
    final lowerJointPaint  = jointPaint(const Color(0xFF448AFF).withAlpha(240)); // bright blue
    final frontJointPaint  = Paint()
      ..color = hasError ? Colors.red : const Color(0xFFFF6600).withAlpha(240)
      ..style = PaintingStyle.fill;

    final joints = skeleton.joints;

    // ── Draw connections ───────────────────────────────────────────────────

    void drawConnections(List<(String, String)> pairs, Paint paint) {
      for (final (a, b) in pairs) {
        final ptA = joints[a];
        final ptB = joints[b];
        if (ptA != null && ptB != null) {
          canvas.drawLine(mapPoint(ptA), mapPoint(ptB), paint);
        }
      }
    }

    drawConnections(kUpperConnections, upperPaint);
    drawConnections(kTorsoConnections, torsoPaint);
    drawConnections(kLowerConnections, lowerPaint);

    // ── Draw joints ────────────────────────────────────────────────────────

    for (final entry in joints.entries) {
      final name = entry.key;
      final pt = mapPoint(entry.value);
      final isFront = name.startsWith('front_');

      if (isFront) {
        // Sword-arm joints: orange highlight with glow ring
        canvas.drawCircle(pt, 8.0, Paint()
          ..color = const Color(0xFFFF6600).withAlpha(60)
          ..style = PaintingStyle.fill);
        canvas.drawCircle(pt, 5.5, frontJointPaint);
      } else if (kLowerJoints.contains(name)) {
        canvas.drawCircle(pt, 4.0, lowerJointPaint);
      } else {
        canvas.drawCircle(pt, 4.0, upperJointPaint);
      }
    }
  }

  @override
  bool shouldRepaint(PosePainter oldDelegate) =>
      oldDelegate.skeleton != skeleton ||
      oldDelegate.triggeredError != triggeredError ||
      oldDelegate.currentAction != currentAction ||
      oldDelegate.isFrontCamera != isFrontCamera;
}

// ---------------------------------------------------------------------------
// Warning overlay painter (flash animation for triggered errors)
// ---------------------------------------------------------------------------

class WarningFlashPainter extends CustomPainter {
  final double opacity; // 0.0–1.0

  const WarningFlashPainter({required this.opacity});

  @override
  void paint(Canvas canvas, Size size) {
    if (opacity <= 0) return;
    final paint = Paint()
      ..color = Colors.red.withAlpha((opacity * 60).toInt())
      ..style = PaintingStyle.fill;
    canvas.drawRect(Offset.zero & size, paint);

    // Draw red border
    final borderPaint = Paint()
      ..color = Colors.red.withAlpha((opacity * 200).toInt())
      ..strokeWidth = 4
      ..style = PaintingStyle.stroke;
    canvas.drawRect(Offset.zero & size, borderPaint);
  }

  @override
  bool shouldRepaint(WarningFlashPainter old) => old.opacity != opacity;
}
