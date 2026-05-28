/// Custom painter that draws the fencing pose skeleton on top of camera preview.

library;

import 'package:flutter/material.dart';
import 'pose_service.dart';

// ---------------------------------------------------------------------------
// Connection pairs for skeleton lines
// ---------------------------------------------------------------------------

const List<(String, String)> kSkeletonConnections = [
  // Torso
  ('left_shoulder', 'right_shoulder'),
  ('left_shoulder', 'left_hip'),
  ('right_shoulder', 'right_hip'),
  ('left_hip', 'right_hip'),
  // Left arm
  ('left_shoulder', 'left_elbow'),
  ('left_elbow', 'left_wrist'),
  // Right arm
  ('right_shoulder', 'right_elbow'),
  ('right_elbow', 'right_wrist'),
  // Left leg
  ('left_hip', 'left_knee'),
  ('left_knee', 'left_ankle'),
  // Right leg
  ('right_hip', 'right_knee'),
  ('right_knee', 'right_ankle'),
  // Nose to shoulders
  ('nose', 'left_shoulder'),
  ('nose', 'right_shoulder'),
];

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

    // Choose skeleton color based on action
    final Color boneColor = triggeredError != null
        ? Colors.redAccent.withAlpha(230)
        : const Color(0xFF00E5FF).withAlpha(200);

    final Paint bonePaint = Paint()
      ..color = boneColor
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final Paint jointPaint = Paint()
      ..color = Colors.white.withAlpha(220)
      ..style = PaintingStyle.fill;

    final Paint frontJointPaint = Paint()
      ..color = const Color(0xFFFF6600).withAlpha(240)
      ..style = PaintingStyle.fill;

    final joints = skeleton.joints;

    // Draw connections
    for (final (a, b) in kSkeletonConnections) {
      final ptA = joints[a];
      final ptB = joints[b];
      if (ptA != null && ptB != null) {
        canvas.drawLine(mapPoint(ptA), mapPoint(ptB), bonePaint);
      }
    }

    // Draw joints
    for (final entry in joints.entries) {
      final name = entry.key;
      final pt = mapPoint(entry.value);
      final isFront = name.startsWith('front_');
      final radius = isFront ? 5.0 : 3.5;
      canvas.drawCircle(pt, radius, isFront ? frontJointPaint : jointPaint);
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
