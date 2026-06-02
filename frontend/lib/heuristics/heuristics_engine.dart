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
const double kLungeKneeMinAngleDeg = 120.0; // 🎯 已改為 120 度
const int kGuardDroppedThresholdFrames = 5; // 🎯 觸發幀數調小 (原10)
const int kGuardDroppedFreeBoutingThresholdFrames = 10; // 🎯 實戰觸發幀數調小 (原20)
const double kFootBeforeHandMinDisplacementPx = 0.01; // 🎯 歸一化座標下合理值 (原5.0)
const double kStanceTooHighAngleDeg = 160.0; // 🎯 改為 160 度，更容易觸發
const double kIncompleteArmExtensionAngleDeg = 155.0;
const int kOverParryMinWristSamples = 5;
const double kOverParryShoulderMultiplier = 2.0;
const double kOverParryRatioThreshold = 3.0; // 🎯 調大門檻 (原2.0)，讓防守動作過大更難觸發
const double kStepShoulderProxyMultiplier = 2.5;
const double kStepMinShoulderWidthPx = 0.02; // 🎯 歸一化座標下合理值 (原3.0)
const double kWideStepRatioThreshold = 3.0;
const double kNarrowStepRatioThreshold = 1.2; // 🎯 歸一化座標下合理值 (原2.5)
const double kComMinBaseWidthPx = 0.03; // 🎯 歸一化座標下合理值 (原10.0)
const double kSpineForwardTiltThresholdDeg = 15.0; // 重心前傾判定角度
const double kSpineBackwardTiltThresholdDeg = 10.0; // 重心後仰判定角度
const double kShoulderForwardTiltThresholdDeg = 15.0; // 肩膀連線前傾角度門檻
const double kShoulderBackwardTiltThresholdDeg = 15.0; // 肩膀連線後仰角度門檻
const double kElbowTooAcuteMinAngleDeg = 100.0; // 手肘角度小於此值 → 手抬太高（預設 100°）

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
    final l = skel['left_hip'];
    final r = skel['right_hip'];
    if (l == null || r == null) return null;
    return Offset((l.dx + r.dx) / 2, (l.dy + r.dy) / 2);
  }

  Offset? _neckCenter(Skeleton skel) {
    final l = skel['left_shoulder'];
    final r = skel['right_shoulder'];
    if (l == null || r == null) return null;
    return Offset((l.dx + r.dx) / 2, (l.dy + r.dy) / 2);
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
  double? lastSpineTiltDeg;
  double? lastShoulderTiltDeg;
  double? lastArmAngleDeg;

  // 🎯 新增：用來計算經過的總幀數，控制前五秒的冷卻時間
  int _frameCount = 0;

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

    _frameCount++; // 每進來一個畫面，計數器 +1

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

    // 🎯 1. 身體起伏過大 (bounce_excessive)
    // 假設 30 FPS，大約 150 幀是 5 秒。前 5 秒不檢查。
    if (_frameCount > 150) {
      _tryAdd(errors, _checkBounce(skeletons));
    }

    // 🎯 2. 隨時監測：膝蓋超伸 (大於 100 度)
    _tryAdd(errors, _checkLunge(skeletons));

    // 🎯 3. 隨時監測：前傾後仰
    _tryAdd(errors, _checkCenterOfMass(skeletons));

    // 🎯 4. 隨時監測：站太高 (門檻已調為 160 度)
    _tryAdd(errors, _checkStanceTooHigh(skeletons));

    // 🎯 5. 隨時監測：持劍手掉落
    _tryAdd(errors, _checkGuard(skeletons));

    // 🎯 6. 隨時監測：步伐寬度
    _tryAdd(errors, _checkStepWidth(skeletons));

    // 🎯 7. 隨時監測：防守動作過大 (改用歐幾里得距離，偵測上下+左右)
    _tryAdd(errors, _checkOverParrying(skeletons));

    // 🎯 8. 隨時監測：手抬太高
    _tryAdd(errors, _checkHandTooHigh(skeletons));

    // ── 特定動作才觸發的檢查 ──

    // Target Practice 專屬的攻擊檢查 (腳先走、手未伸直)
    if (trainingMode == 'Target Practice' && isOffensive) {
      _tryAdd(errors, _checkFootBeforeHand(skeletons));
      _tryAdd(errors, _checkIncompleteArmExtension(skeletons));
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
  // 🎯 隨時監測：掃描 window 中膝角最小的那幀（worst-case），
  //    不再依賴腳踝位移，所以站立時也能偵測到膝蓋超伸。

  String? _checkLunge(List<Skeleton> skeletons) {
    final limbs = frontLimbs(targetSide);

    double minAngle = double.infinity;
    for (final skel in skeletons) {
      final hip   = skel[limbs['hip']!];
      final knee  = skel[limbs['knee']!];
      final ankle = skel[limbs['ankle']!];
      if (hip == null || knee == null || ankle == null) continue;
      final angle = calcAngle(hip, knee, ankle);
      if (angle < minAngle) minAngle = angle;
    }

    if (minAngle == double.infinity) return null;
    if (minAngle < kLungeKneeMinAngleDeg) return 'lunge_overextension';
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
  //
  // 使用歐幾里得距離（2D）量測手腕的移動範圍。
  // 傳統只看 x 軸的話，上下揮動的大幅防守動作會漏掉；
  // 改成計算手腕移動的 「最大 2D 位移」 來同時捕捉水平和垂直方向的過大擺動。

  String? _checkOverParrying(List<Skeleton> skeletons) {
    final limbs = frontLimbs(targetSide);
    double? shoulderWidth;

    // ── Step 1: 估算肩膀寬度作為比例基準 ────────────────────────────────────
    for (final skel in skeletons) {
      final shoulder = skel[limbs['shoulder']!];
      final otherShoulderName =
          targetSide == 'right' ? 'left_shoulder' : 'right_shoulder';
      final otherShoulder = skel[otherShoulderName];
      if (otherShoulder == null) {
        final pelvis = _pelvisCenter(skel);
        if (shoulder != null && pelvis != null) {
          // 只有單肩時，用肩到骨盆中心的距離近似
          shoulderWidth =
              (shoulder - pelvis).distance * kOverParryShoulderMultiplier;
          break;
        }
      } else if (shoulder != null) {
        // 兩肩都有時，直接用 2D 歐幾里得肩膀距離
        shoulderWidth = (shoulder - otherShoulder).distance;
        break;
      }
    }

    if (shoulderWidth == null || shoulderWidth < 1e-6) return null;

    // ── Step 2: 收集所有手腕位置 ────────────────────────────────────────────
    final wristPositions = <Offset>[];
    for (final skel in skeletons) {
      final wrist = skel[limbs['wrist']!];
      if (wrist != null) wristPositions.add(wrist);
    }

    if (wristPositions.length < kOverParryMinWristSamples) return null;

    // ── Step 3: 計算手腕的最大 2D 歐幾里得移動距離 ──────────────────────────
    // 取所有手腕位置兩兩之間的最大距離（最大 sweep 範圍）
    double maxSweep = 0.0;
    for (int i = 0; i < wristPositions.length; i++) {
      for (int j = i + 1; j < wristPositions.length; j++) {
        final d = (wristPositions[i] - wristPositions[j]).distance;
        if (d > maxSweep) maxSweep = d;
      }
    }

    if (maxSweep > kOverParryRatioThreshold * shoulderWidth) {
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
      if (ratio > kWideStepRatioThreshold) return 'wide_step'; // 🎯 修復 Bug，補上過寬檢查
    }
    return null;
  }

  // ── Rule 11: center_of_mass_in_front / leaning_backward ──────────────────

  String? _checkCenterOfMass(List<Skeleton> skeletons) {
    final limbs = frontLimbs(targetSide);
    final backAnkleKey = backAnkleName(targetSide);
    final frontShoulderKey = targetSide == 'Right' ? 'right_shoulder' : 'left_shoulder';
    final backShoulderKey = targetSide == 'Right' ? 'left_shoulder' : 'right_shoulder';

    for (final skel in skeletons) {
      // 1. Spine Tilt Check
      final frontAnkle = skel[limbs['ankle']!];
      final back = skel[backAnkleKey];
      final pelvis = _pelvisCenter(skel);
      final neck = _neckCenter(skel);
      
      if (frontAnkle != null && back != null && pelvis != null && neck != null) {
        final frontX = frontAnkle.dx;
        final backX = back.dx;
        
        // dx is horizontal difference (neck relative to pelvis)
        // dy is vertical difference (positive if neck is above pelvis, which is expected)
        final dx = neck.dx - pelvis.dx;
        final dy = pelvis.dy - neck.dy; 
        
        if (dy > 0) {
          // Angle from vertical (0 is perfectly upright, positive is neck to the right of pelvis)
          final thetaDeg = math.atan2(dx, dy) * 180 / math.pi;

          // Determine forward tilt depending on facing direction
          // If facing right (frontX > backX), positive theta means leaning forward
          // If facing left (frontX < backX), negative theta means leaning forward
          final forwardTiltDeg = (frontX > backX) ? thetaDeg : -thetaDeg;
          
          lastSpineTiltDeg = forwardTiltDeg;

          if (forwardTiltDeg > kSpineForwardTiltThresholdDeg) return 'center_of_mass_in_front';
          if (forwardTiltDeg < -kSpineBackwardTiltThresholdDeg) {
            return 'center_of_mass_leaning_backward';
          }
        }
      }

      // 2. Shoulder Line Tilt Check
      final frontShoulder = skel[frontShoulderKey];
      final backShoulder = skel[backShoulderKey];
      
      if (frontShoulder != null && backShoulder != null) {
        // dy: positive if front shoulder is lower (screen y grows downwards)
        final dy = frontShoulder.dy - backShoulder.dy;
        // dx: horizontal distance between shoulders
        final dx = (frontShoulder.dx - backShoulder.dx).abs();
        
        // Tilt angle: positive means front shoulder is lower (leaning forward)
        // negative means front shoulder is higher (leaning backward)
        if (dx > 0.001 || dy.abs() > 0.001) {
          final shoulderTiltDeg = math.atan2(dy, dx) * 180 / math.pi;
          
          lastShoulderTiltDeg = shoulderTiltDeg;
          
          if (shoulderTiltDeg > kShoulderForwardTiltThresholdDeg) return 'center_of_mass_in_front';
          if (shoulderTiltDeg < -kShoulderBackwardTiltThresholdDeg) {
            return 'center_of_mass_leaning_backward';
          }
        }
      }
    }
    return null;
  }

  // ── Rule 12: hand_too_high ───────────────────────────────────────────────

  String? _checkHandTooHigh(List<Skeleton> skeletons) {
    final limbs = frontLimbs(targetSide);

    for (final skel in skeletons) {
      final shoulder = skel[limbs['shoulder']!];
      final elbow    = skel[limbs['elbow']!];
      final wrist    = skel[limbs['wrist']!];

      if (shoulder != null && elbow != null && wrist != null) {
        // Elbow joint angle: shoulder → elbow → wrist
        // Small angle = arm is bent / hand raised
        final angleDeg = calcAngle(shoulder, elbow, wrist);
        lastArmAngleDeg = angleDeg;

        if (angleDeg < kElbowTooAcuteMinAngleDeg) {
          return 'hand_too_high';
        }
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