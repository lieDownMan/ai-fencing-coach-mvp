/// Geometric posture heuristics for fencing — Dart port of heuristics_engine.py
///
/// All computations are frame-by-frame or over sliding windows of raw skeletons.
/// Skeletons are represented as `Map<String, Offset>` where keys match the
/// ML-Kit landmark names we resolve into logical fencing joints.

library;

import 'dart:math' as math;
import 'dart:ui' show Offset;
import 'package:flutter/foundation.dart';

// ---------------------------------------------------------------------------
// Constants — mirror of heuristics_engine.py
// ---------------------------------------------------------------------------

const int kBouncMinPelvisSamples = 5;
const double kBounceRatioThreshold = 0.33;
const double kLungeKneeMinAngleDeg = 90.0;
const int kGuardDroppedThresholdFrames = 10;
const int kGuardDroppedFreeBoutingThresholdFrames = 20;
const double kFootBeforeHandMinDisplacementPx = 0.01; // normalized [0,1] space (≈ 6px in 640px)
const double kStanceTooHighAngleDeg = 170.0;
const double kIncompleteArmExtensionAngleDeg = 155.0;
const int kOverParryMinWristSamples = 5;
const double kOverParryShoulderMultiplier = 2.0;
const double kOverParryRatioThreshold = 2.0;
const double kStepShoulderProxyMultiplier = 2.5;
const double kStepMinShoulderWidthPx = 0.01;  // normalized [0,1] space (≈ 6px in 640px)
const double kWideStepRatioThreshold = 3.0;
const double kNarrowStepRatioThreshold = 1.0;
const double kComMinBaseWidthPx = 0.01;       // normalized [0,1] space (≈ 6px in 640px)
const double kComInFrontRatioThreshold = 0.65;
const double kComLeaningBackRatioThreshold = 0.35;

// ---------------------------------------------------------------------------
// Detected action classes (from FenceNetV2 class names)
// ---------------------------------------------------------------------------

const List<String> kClassNames = ['R', 'IS', 'WW', 'JS', 'SF', 'SB'];
const Set<String> kOffensiveActions = {'R', 'IS', 'WW', 'JS'};
const Set<String> kFootworkActions = {'SF', 'SB'};

// ---------------------------------------------------------------------------
// Skeleton type alias
// ---------------------------------------------------------------------------

typedef Skeleton = Map<String, Offset>;

// ---------------------------------------------------------------------------
// Math utilities
// ---------------------------------------------------------------------------

/// Angle ABC (vertex at B) in degrees.
double calcAngle(Offset a, Offset b, Offset c) {
  final ba = Offset(a.dx - b.dx, a.dy - b.dy);
  final bc = Offset(c.dx - b.dx, c.dy - b.dy);
  final normBa = math.sqrt(ba.dx * ba.dx + ba.dy * ba.dy);
  final normBc = math.sqrt(bc.dx * bc.dx + bc.dy * bc.dy);
  if (normBa < 1e-8 || normBc < 1e-8) return 180.0;
  final dot = ba.dx * bc.dx + ba.dy * bc.dy;
  final cosAngle = (dot / (normBa * normBc)).clamp(-1.0, 1.0);
  return math.acos(cosAngle) * 180.0 / math.pi;
}

Offset? _pelvisCenter(Skeleton skel) {
  final lh = skel['left_hip'];
  final rh = skel['right_hip'];
  if (lh == null || rh == null) return null;
  return Offset((lh.dx + rh.dx) / 2, (lh.dy + rh.dy) / 2);
}

// ---------------------------------------------------------------------------
// Joint mapping helper
// ---------------------------------------------------------------------------

/// Given a targetSide ('left'|'right'), returns the logical joint names.
/// 'front' = the leading side (sword arm side).
///   left fencer  → front hip/knee/ankle = right hip/knee/ankle
///   right fencer → front hip/knee/ankle = left  hip/knee/ankle
Map<String, String> frontLimbs(String targetSide) {
  if (targetSide == 'left') {
    return {
      'hip': 'right_hip',
      'knee': 'right_knee',
      'ankle': 'right_ankle',
      'wrist': 'front_wrist',
      'elbow': 'front_elbow',
      'shoulder': 'front_shoulder',
    };
  } else {
    return {
      'hip': 'left_hip',
      'knee': 'left_knee',
      'ankle': 'left_ankle',
      'wrist': 'front_wrist',
      'elbow': 'front_elbow',
      'shoulder': 'front_shoulder',
    };
  }
}

String backAnkleName(String targetSide) =>
    targetSide == 'left' ? 'left_ankle' : 'right_ankle';

// ---------------------------------------------------------------------------
// Heuristics Engine
// ---------------------------------------------------------------------------

class HeuristicsEngine {
  final String targetSide;
  final String trainingMode;
  double? lastStepRatio;
  double? lastStepWidth;

  HeuristicsEngine({
    required this.targetSide,
    required this.trainingMode,
  });

  /// Evaluate a buffer of skeletons against the given action label.
  /// Returns a list of triggered error keys.
  List<String> evaluateWindow({
    required String action,
    required List<Skeleton> skeletons,
  }) {
    if (skeletons.isEmpty) return [];

    // Unconditionally compute live step debug metrics
    final latestSkel = skeletons.last;
    final liveLimbs = frontLimbs(targetSide);
    final liveBackAnkle = backAnkleName(targetSide);
    final fAnkle = latestSkel[liveLimbs['ankle']!];
    final bAnkle = latestSkel[liveBackAnkle];
    if (fAnkle != null && bAnkle != null) {
      lastStepWidth = (fAnkle - bAnkle).distance;
      final fShoulder = latestSkel[liveLimbs['shoulder']!];
      final pCenter = _pelvisCenter(latestSkel);
      if (fShoulder != null && pCenter != null) {
        final sw = (fShoulder.dx - pCenter.dx).abs() * kStepShoulderProxyMultiplier;
        if (sw > 1e-6) lastStepRatio = lastStepWidth! / sw;
      }
    }

    final errors = <String>[];
    final isOffensive = kOffensiveActions.contains(action);

    // Footwork-specific checks
    if (trainingMode == 'Footwork' && kFootworkActions.contains(action)) {
      _tryAdd(errors, _checkBounce(skeletons));
      _tryAdd(errors, _checkStanceTooHigh(skeletons));
      _tryAdd(errors, _checkCenterOfMass(skeletons));
    }

    // Target Practice offensive checks
    if (trainingMode == 'Target Practice' && isOffensive) {
      _tryAdd(errors, _checkLunge(skeletons));
      _tryAdd(errors, _checkFootBeforeHand(skeletons));
      _tryAdd(errors, _checkIncompleteArmExtension(skeletons));
    }

    // Guard dropped — all modes
    _tryAdd(errors, _checkGuard(skeletons));

    // Step width — all modes, unconditionally
    _tryAdd(errors, _checkStepWidth(skeletons));

    // Footwork checks in non-Footwork modes
    if (trainingMode != 'Footwork' && kFootworkActions.contains(action)) {
      _tryAdd(errors, _checkStanceTooHigh(skeletons));
      _tryAdd(errors, _checkBounce(skeletons));
      _tryAdd(errors, _checkCenterOfMass(skeletons));
    }

    // Over-parrying
    if (action == 'SB' ||
        (trainingMode == 'Free Bouting' && kFootworkActions.contains(action))) {
      _tryAdd(errors, _checkOverParrying(skeletons));
    }

    return errors;
  }

  void _tryAdd(List<String> list, String? key) {
    if (key != null) list.add(key);
  }

  // ── Rule 1: bounce_excessive ──────────────────────────────────────────────

  String? _checkBounce(List<Skeleton> skeletons) {
    final pelvisYs = <double>[];
    final allYs = <double>[];
    for (final skel in skeletons) {
      final pc = _pelvisCenter(skel);
      if (pc != null) pelvisYs.add(pc.dy);
      for (final v in skel.values) {
        allYs.add(v.dy);
      }
    }
    if (pelvisYs.length < kBouncMinPelvisSamples || allYs.length < 2) {
      return null;
    }
    final bboxHeight = allYs.reduce(math.max) - allYs.reduce(math.min);
    if (bboxHeight < 1e-4) return null;
    final deltaY = pelvisYs.reduce(math.max) - pelvisYs.reduce(math.min);
    if (deltaY > kBounceRatioThreshold * bboxHeight) {
      return 'bounce_excessive';
    }
    return null;
  }

  // ── Rule 2: lunge_overextension ───────────────────────────────────────────

  String? _checkLunge(List<Skeleton> skeletons) {
    final limbs = frontLimbs(targetSide);
    final refAnkle = skeletons[0][limbs['ankle']!];
    if (refAnkle == null) return null;

    double maxDisp = 0;
    Skeleton peakSkel = skeletons[0];
    for (final skel in skeletons) {
      final ankle = skel[limbs['ankle']!];
      if (ankle != null) {
        final d = (ankle - refAnkle).distance;
        if (d > maxDisp) {
          maxDisp = d;
          peakSkel = skel;
        }
      }
    }

    final hip = peakSkel[limbs['hip']!];
    final knee = peakSkel[limbs['knee']!];
    final ankle = peakSkel[limbs['ankle']!];
    if (hip == null || knee == null || ankle == null) return null;

    final angle = calcAngle(hip, knee, ankle);
    if (angle < kLungeKneeMinAngleDeg) return 'lunge_overextension';
    return null;
  }

  // ── Rule 3: guard_dropped ─────────────────────────────────────────────────

  String? _checkGuard(List<Skeleton> skeletons) {
    final limbs = frontLimbs(targetSide);
    int consecutive = 0;
    final threshold = trainingMode == 'Free Bouting'
        ? kGuardDroppedFreeBoutingThresholdFrames
        : kGuardDroppedThresholdFrames;

    for (final skel in skeletons) {
      final wrist = skel[limbs['wrist']!];
      final pelvis = _pelvisCenter(skel);
      if (wrist != null && pelvis != null && wrist.dy > pelvis.dy) {
        consecutive++;
        if (consecutive > threshold) return 'guard_dropped';
      } else {
        consecutive = 0;
      }
    }
    return null;
  }

  // ── Rule 4: foot_before_hand ──────────────────────────────────────────────

  String? _checkFootBeforeHand(List<Skeleton> skeletons) {
    final limbs = frontLimbs(targetSide);
    final refWrist = skeletons[0][limbs['wrist']!];
    final refAnkle = skeletons[0][limbs['ankle']!];
    if (refWrist == null || refAnkle == null) return null;

    double maxWristDisp = 0, maxAnkleDisp = 0;
    int wristPeakFrame = 0, anklePeakFrame = 0;

    for (int i = 0; i < skeletons.length; i++) {
      final skel = skeletons[i];
      final wrist = skel[limbs['wrist']!];
      final ankle = skel[limbs['ankle']!];
      if (wrist != null) {
        final d = (wrist.dx - refWrist.dx).abs();
        if (d > maxWristDisp) {
          maxWristDisp = d;
          wristPeakFrame = i;
        }
      }
      if (ankle != null) {
        final d = (ankle.dx - refAnkle.dx).abs();
        if (d > maxAnkleDisp) {
          maxAnkleDisp = d;
          anklePeakFrame = i;
        }
      }
    }

    if (maxAnkleDisp > kFootBeforeHandMinDisplacementPx &&
        maxWristDisp > kFootBeforeHandMinDisplacementPx &&
        anklePeakFrame < wristPeakFrame) {
      return 'foot_before_hand';
    }
    return null;
  }

  // ── Rule 5: stance_too_high ───────────────────────────────────────────────

  String? _checkStanceTooHigh(List<Skeleton> skeletons) {
    final limbs = frontLimbs(targetSide);
    final angles = <double>[];
    for (final skel in skeletons) {
      final hip = skel[limbs['hip']!];
      final knee = skel[limbs['knee']!];
      final ankle = skel[limbs['ankle']!];
      if (hip != null && knee != null && ankle != null) {
        angles.add(calcAngle(hip, knee, ankle));
      }
    }
    if (angles.length < 3) return null;
    final avg = angles.reduce((a, b) => a + b) / angles.length;
    if (avg > kStanceTooHighAngleDeg) return 'stance_too_high';
    return null;
  }

  // ── Rule 6: incomplete_arm_extension ──────────────────────────────────────

  String? _checkIncompleteArmExtension(List<Skeleton> skeletons) {
    final limbs = frontLimbs(targetSide);
    final refWrist = skeletons[0][limbs['wrist']!];
    if (refWrist == null) return null;

    double maxDisp = 0;
    Skeleton peakSkel = skeletons[0];
    for (final skel in skeletons) {
      final wrist = skel[limbs['wrist']!];
      if (wrist != null) {
        final d = (wrist.dx - refWrist.dx).abs();
        if (d > maxDisp) {
          maxDisp = d;
          peakSkel = skel;
        }
      }
    }

    final shoulder = peakSkel[limbs['shoulder']!];
    final elbow = peakSkel[limbs['elbow']!];
    final wrist = peakSkel[limbs['wrist']!];
    if (shoulder == null || elbow == null || wrist == null) return null;

    final angle = calcAngle(shoulder, elbow, wrist);
    if (angle < kIncompleteArmExtensionAngleDeg) return 'incomplete_arm_extension';
    return null;
  }

  // ── Rule 9: over_parrying ─────────────────────────────────────────────────

  String? _checkOverParrying(List<Skeleton> skeletons) {
    final limbs = frontLimbs(targetSide);
    double? shoulderWidth;

    for (final skel in skeletons) {
      final shoulder = skel[limbs['shoulder']!];
      final otherShoulderName =
          targetSide == 'right' ? 'left_shoulder' : 'right_shoulder';
      final otherShoulder = skel[otherShoulderName];
      if (otherShoulder == null) {
        final pelvis = _pelvisCenter(skel);
        if (shoulder != null && pelvis != null) {
          shoulderWidth =
              (shoulder.dx - pelvis.dx).abs() * kOverParryShoulderMultiplier;
          break;
        }
      } else if (shoulder != null) {
        shoulderWidth = (shoulder.dx - otherShoulder.dx).abs();
        break;
      }
    }

    if (shoulderWidth == null || shoulderWidth < 1e-6) return null;

    final wristXs = <double>[];
    for (final skel in skeletons) {
      final wrist = skel[limbs['wrist']!];
      if (wrist != null) wristXs.add(wrist.dx);
    }

    if (wristXs.length < kOverParryMinWristSamples) return null;

    final sweepRange = wristXs.reduce(math.max) - wristXs.reduce(math.min);
    if (sweepRange > kOverParryRatioThreshold * shoulderWidth) {
      return 'over_parrying';
    }
    return null;
  }

  // ── Rule 10: wide_step / narrow_step ──────────────────────────────────────

  String? _checkStepWidth(List<Skeleton> skeletons) {
    final limbs = frontLimbs(targetSide);
    final backAnkleKey = backAnkleName(targetSide);

    for (final skel in skeletons) {
      final frontAnkle = skel[limbs['ankle']!];
      final back = skel[backAnkleKey];
      final frontShoulder = skel[limbs['shoulder']!];
      final pelvis = _pelvisCenter(skel);
      if (frontAnkle == null || back == null || frontShoulder == null ||
          pelvis == null) continue;

      final sw =
          (frontShoulder.dx - pelvis.dx).abs() * kStepShoulderProxyMultiplier;
      if (sw < kStepMinShoulderWidthPx) continue;

      final stepWidth = (frontAnkle - back).distance;
      final ratio = stepWidth / sw;

      if (ratio < kNarrowStepRatioThreshold) return 'narrow_step';
    }
    return null;
  }

  // ── Rule 11: center_of_mass_in_front / leaning_backward ──────────────────

  String? _checkCenterOfMass(List<Skeleton> skeletons) {
    final limbs = frontLimbs(targetSide);
    final backAnkleKey = backAnkleName(targetSide);

    for (final skel in skeletons) {
      final frontAnkle = skel[limbs['ankle']!];
      final back = skel[backAnkleKey];
      final pelvis = _pelvisCenter(skel);
      if (frontAnkle == null || back == null || pelvis == null) continue;

      final frontX = frontAnkle.dx;
      final backX = back.dx;
      final pelvisX = pelvis.dx;
      final baseWidth = (frontX - backX).abs();
      if (baseWidth < kComMinBaseWidthPx) continue;

      double ratio;
      if (frontX > backX) {
        ratio = (pelvisX - backX) / baseWidth;
      } else {
        ratio = (backX - pelvisX) / baseWidth;
      }

      if (ratio > kComInFrontRatioThreshold) return 'center_of_mass_in_front';
      if (ratio < kComLeaningBackRatioThreshold) {
        return 'center_of_mass_leaning_backward';
      }
    }
    return null;
  }
}

// ---------------------------------------------------------------------------
// Heuristic debug metric (for debug tab)
// ---------------------------------------------------------------------------

@immutable
class HeuristicMetric {
  final String key;
  final bool triggered;
  final String primaryValue;
  final String details;

  const HeuristicMetric({
    required this.key,
    required this.triggered,
    required this.primaryValue,
    required this.details,
  });
}
