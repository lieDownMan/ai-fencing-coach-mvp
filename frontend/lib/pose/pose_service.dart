import 'dart:ui' show Offset;

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
