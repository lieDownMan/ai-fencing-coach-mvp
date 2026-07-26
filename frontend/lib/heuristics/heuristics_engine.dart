/// Geometric posture heuristics for fencing — Dart port of heuristics_engine.py
///
/// All computations are frame-by-frame or over sliding windows of raw skeletons.
/// Skeletons are represented as `Map<String, Offset>` where keys match the
/// ML-Kit landmark names we resolve into logical fencing joints.
///
/// Divergences from the Python engine (deliberate, tuned for live phone use):
///  - Frame-count thresholds are time-based (seconds × measured fps) because the
///    effective pose fps on device varies with hardware and load.
///  - Step-width / center-of-mass checks must hold for a sustained duration,
///    not a single frame, so normal mid-stride phases don't trigger them.
///  - Over-parrying measures the wrist sweep RELATIVE to the pelvis (so walking
///    doesn't count as a parry) and scales by torso length, which stays valid
///    in the side-on camera view where shoulder X-width collapses to ~0.
///  - foot_before_hand compares motion ONSET times (not peak times) with a
///    small margin, since in a correct lunge the foot can legitimately *peak*
///    (land) before the arm reaches full extension.

library;

import 'dart:math' as math;
import 'dart:ui' show Offset;
import 'package:flutter/foundation.dart';

// ---------------------------------------------------------------------------
// Fixed constants (not worth tuning)
// ---------------------------------------------------------------------------

const int kBouncMinPelvisSamples = 5;
const int kOverParryMinWristSamples = 5;
const double kStepMinShoulderWidthPx = 0.01;  // normalized [0,1] space (≈ 6px in 640px)
const double kComMinBaseWidthPx = 0.01;       // normalized [0,1] space (≈ 6px in 640px)
const double kDefaultFps = 30.0;

// ---------------------------------------------------------------------------
// Tunable thresholds
// ---------------------------------------------------------------------------

/// Every tunable threshold in one place, overridable per-instance so the
/// offline replay tool can sweep values and (later) per-user calibration can
/// scale them. The defaults are the shipped behavior — hand-tuned on-device
/// (2026-07) via the in-app Tuning tab / Mac tuning server.
class HeuristicsConfig {
  final double bounceRatioThreshold;
  final double lungeKneeMinAngleDeg;
  final double guardDroppedSeconds;
  final double guardDroppedFreeBoutingSeconds;
  final double footBeforeHandMinDisplacement; // normalized [0,1] space
  final double footBeforeHandLeadSeconds; // ankle must lead wrist by at least this
  final double stanceTooHighAngleDeg;
  final double incompleteArmExtensionAngleDeg;
  // Wrist sweep (relative to pelvis) beyond this multiple of torso length = over-parry.
  final double overParryTorsoRatioThreshold;
  final double stepShoulderProxyMultiplier;
  final double wideStepRatioThreshold;
  final double narrowStepRatioThreshold;
  final double stepSustainedSeconds; // ratio must stay out of range this long
  // Torso lean from vertical (pelvis→front-shoulder line), degrees; positive
  // = leaning toward the opponent. Beyond forward/behind backward = error.
  final double comForwardLeanDeg;
  final double comBackwardLeanDeg;
  final double comSustainedSeconds;

  const HeuristicsConfig({
    this.bounceRatioThreshold = 0.13,
    this.lungeKneeMinAngleDeg = 153.5,
    this.guardDroppedSeconds = 3.0,
    this.guardDroppedFreeBoutingSeconds = 3.0,
    // Minimum body-relative forward RISE (peak minus baseline, normalized
    // units) for a wrist-extension / attack-step movement to count at all.
    this.footBeforeHandMinDisplacement = 0.03,
    this.footBeforeHandLeadSeconds = 0.10,
    this.stanceTooHighAngleDeg = 170.0,
    this.incompleteArmExtensionAngleDeg = 101.0,
    this.overParryTorsoRatioThreshold = 0.54,
    this.stepShoulderProxyMultiplier = 2.5,
    this.wideStepRatioThreshold = 1.72,
    this.narrowStepRatioThreshold = 0.9,
    this.stepSustainedSeconds = 0.30,
    this.comForwardLeanDeg = 25.0,
    this.comBackwardLeanDeg = -10.0,
    this.comSustainedSeconds = 0.30,
  });

  /// Param-name → value map, used for persistence and the in-app tuning UI.
  Map<String, double> toMap() => {
        'bounceRatioThreshold': bounceRatioThreshold,
        'lungeKneeMinAngleDeg': lungeKneeMinAngleDeg,
        'guardDroppedSeconds': guardDroppedSeconds,
        'guardDroppedFreeBoutingSeconds': guardDroppedFreeBoutingSeconds,
        'footBeforeHandMinDisplacement': footBeforeHandMinDisplacement,
        'footBeforeHandLeadSeconds': footBeforeHandLeadSeconds,
        'stanceTooHighAngleDeg': stanceTooHighAngleDeg,
        'incompleteArmExtensionAngleDeg': incompleteArmExtensionAngleDeg,
        'overParryTorsoRatioThreshold': overParryTorsoRatioThreshold,
        'stepShoulderProxyMultiplier': stepShoulderProxyMultiplier,
        'wideStepRatioThreshold': wideStepRatioThreshold,
        'narrowStepRatioThreshold': narrowStepRatioThreshold,
        'stepSustainedSeconds': stepSustainedSeconds,
        'comForwardLeanDeg': comForwardLeanDeg,
        'comBackwardLeanDeg': comBackwardLeanDeg,
        'comSustainedSeconds': comSustainedSeconds,
      };

  /// Inverse of [toMap]; missing keys fall back to the shipped defaults.
  factory HeuristicsConfig.fromMap(Map<String, double> map) {
    const d = HeuristicsConfig();
    return HeuristicsConfig(
      bounceRatioThreshold:
          map['bounceRatioThreshold'] ?? d.bounceRatioThreshold,
      lungeKneeMinAngleDeg:
          map['lungeKneeMinAngleDeg'] ?? d.lungeKneeMinAngleDeg,
      guardDroppedSeconds: map['guardDroppedSeconds'] ?? d.guardDroppedSeconds,
      guardDroppedFreeBoutingSeconds: map['guardDroppedFreeBoutingSeconds'] ??
          d.guardDroppedFreeBoutingSeconds,
      footBeforeHandMinDisplacement: map['footBeforeHandMinDisplacement'] ??
          d.footBeforeHandMinDisplacement,
      footBeforeHandLeadSeconds:
          map['footBeforeHandLeadSeconds'] ?? d.footBeforeHandLeadSeconds,
      stanceTooHighAngleDeg:
          map['stanceTooHighAngleDeg'] ?? d.stanceTooHighAngleDeg,
      incompleteArmExtensionAngleDeg: map['incompleteArmExtensionAngleDeg'] ??
          d.incompleteArmExtensionAngleDeg,
      overParryTorsoRatioThreshold: map['overParryTorsoRatioThreshold'] ??
          d.overParryTorsoRatioThreshold,
      stepShoulderProxyMultiplier: map['stepShoulderProxyMultiplier'] ??
          d.stepShoulderProxyMultiplier,
      wideStepRatioThreshold:
          map['wideStepRatioThreshold'] ?? d.wideStepRatioThreshold,
      narrowStepRatioThreshold:
          map['narrowStepRatioThreshold'] ?? d.narrowStepRatioThreshold,
      stepSustainedSeconds:
          map['stepSustainedSeconds'] ?? d.stepSustainedSeconds,
      comForwardLeanDeg:
          map['comForwardLeanDeg'] ?? d.comForwardLeanDeg,
      comBackwardLeanDeg: map['comBackwardLeanDeg'] ??
          d.comBackwardLeanDeg,
      comSustainedSeconds: map['comSustainedSeconds'] ?? d.comSustainedSeconds,
    );
  }

  HeuristicsConfig copyWith({
    double? bounceRatioThreshold,
    double? lungeKneeMinAngleDeg,
    double? guardDroppedSeconds,
    double? guardDroppedFreeBoutingSeconds,
    double? footBeforeHandMinDisplacement,
    double? footBeforeHandLeadSeconds,
    double? stanceTooHighAngleDeg,
    double? incompleteArmExtensionAngleDeg,
    double? overParryTorsoRatioThreshold,
    double? stepShoulderProxyMultiplier,
    double? wideStepRatioThreshold,
    double? narrowStepRatioThreshold,
    double? stepSustainedSeconds,
    double? comForwardLeanDeg,
    double? comBackwardLeanDeg,
    double? comSustainedSeconds,
  }) {
    return HeuristicsConfig(
      bounceRatioThreshold: bounceRatioThreshold ?? this.bounceRatioThreshold,
      lungeKneeMinAngleDeg: lungeKneeMinAngleDeg ?? this.lungeKneeMinAngleDeg,
      guardDroppedSeconds: guardDroppedSeconds ?? this.guardDroppedSeconds,
      guardDroppedFreeBoutingSeconds:
          guardDroppedFreeBoutingSeconds ?? this.guardDroppedFreeBoutingSeconds,
      footBeforeHandMinDisplacement:
          footBeforeHandMinDisplacement ?? this.footBeforeHandMinDisplacement,
      footBeforeHandLeadSeconds:
          footBeforeHandLeadSeconds ?? this.footBeforeHandLeadSeconds,
      stanceTooHighAngleDeg:
          stanceTooHighAngleDeg ?? this.stanceTooHighAngleDeg,
      incompleteArmExtensionAngleDeg: incompleteArmExtensionAngleDeg ??
          this.incompleteArmExtensionAngleDeg,
      overParryTorsoRatioThreshold:
          overParryTorsoRatioThreshold ?? this.overParryTorsoRatioThreshold,
      stepShoulderProxyMultiplier:
          stepShoulderProxyMultiplier ?? this.stepShoulderProxyMultiplier,
      wideStepRatioThreshold:
          wideStepRatioThreshold ?? this.wideStepRatioThreshold,
      narrowStepRatioThreshold:
          narrowStepRatioThreshold ?? this.narrowStepRatioThreshold,
      stepSustainedSeconds: stepSustainedSeconds ?? this.stepSustainedSeconds,
      comForwardLeanDeg:
          comForwardLeanDeg ?? this.comForwardLeanDeg,
      comBackwardLeanDeg:
          comBackwardLeanDeg ?? this.comBackwardLeanDeg,
      comSustainedSeconds: comSustainedSeconds ?? this.comSustainedSeconds,
    );
  }
}

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

double _median(List<double> values) {
  final sorted = List<double>.from(values)..sort();
  final mid = sorted.length ~/ 2;
  if (sorted.length.isOdd) return sorted[mid];
  return (sorted[mid - 1] + sorted[mid]) / 2;
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
  final HeuristicsConfig config;
  double? lastStepRatio;
  double? lastStepWidth;

  HeuristicsEngine({
    required this.targetSide,
    required this.trainingMode,
    this.config = const HeuristicsConfig(),
  });

  /// Convert a duration threshold to a frame count at the given fps.
  static int _framesFor(double seconds, double fps) =>
      math.max(3, (seconds * fps).round());

  /// Evaluate a buffer of skeletons against the given action label.
  /// [fps] is the *effective* pose frame rate of the buffer (frames actually
  /// processed per second, not the camera rate) so duration-based thresholds
  /// stay consistent across devices.
  /// Returns a list of triggered error keys.
  List<String> evaluateWindow({
    required String action,
    required List<Skeleton> skeletons,
    double fps = kDefaultFps,
  }) {
    if (skeletons.isEmpty) return [];
    if (!fps.isFinite || fps <= 1.0) {
      fps = kDefaultFps;
    }

    // Unconditionally compute live step debug metrics
    final latestSkel = skeletons.last;
    final liveLimbs = frontLimbs(targetSide);
    final liveBackAnkle = backAnkleName(targetSide);
    final fAnkle = latestSkel[liveLimbs['ankle']!];
    final bAnkle = latestSkel[liveBackAnkle];
    if (fAnkle != null && bAnkle != null) {
      lastStepWidth = (fAnkle.dx - bAnkle.dx).abs();
      final fShoulder = latestSkel[liveLimbs['shoulder']!];
      final pCenter = _pelvisCenter(latestSkel);
      if (fShoulder != null && pCenter != null) {
        final sw = (fShoulder.dx - pCenter.dx).abs() * config.stepShoulderProxyMultiplier;
        if (sw > 1e-6) lastStepRatio = lastStepWidth! / sw;
      }
    }

    final errors = <String>[];
    final isOffensive = kOffensiveActions.contains(action);
    final isFootwork = kFootworkActions.contains(action);

    // Footwork checks — footwork actions in any mode
    if (isFootwork) {
      _tryAdd(errors, _checkBounce(skeletons));
      _tryAdd(errors, _checkStanceTooHigh(skeletons));
      _tryAdd(errors, _checkStepWidth(skeletons, fps));
      _tryAdd(errors, _checkCenterOfMass(skeletons, fps));
    }

    // Target Practice offensive checks
    if (trainingMode == 'Target Practice' && isOffensive) {
      _tryAdd(errors, _checkLunge(skeletons));
      _tryAdd(errors, _checkFootBeforeHand(skeletons, fps));
      _tryAdd(errors, _checkIncompleteArmExtension(skeletons));
    }

    // Guard dropped — all modes
    _tryAdd(errors, _checkGuard(skeletons, fps));

    // Over-parrying — defensive context: SB in all modes, SF/SB in Free Bouting
    if (action == 'SB' ||
        (trainingMode == 'Free Bouting' && isFootwork)) {
      _tryAdd(errors, _checkOverParrying(skeletons));
    }

    return errors;
  }

  void _tryAdd(List<String> list, String? key) {
    if (key != null) list.add(key);
  }

  /// Raw metric values for a window, keyed by name — the offline tuning tool
  /// replays recorded sessions through this to see metric DISTRIBUTIONS, so
  /// thresholds can be chosen from data instead of guessed.
  /// Missing joints → the metric is simply absent from the map.
  Map<String, double> computeWindowMetrics(
    List<Skeleton> skeletons, {
    double fps = kDefaultFps,
  }) {
    final m = <String, double>{};
    if (skeletons.isEmpty) return m;
    if (!fps.isFinite || fps <= 1.0) fps = kDefaultFps;
    final limbs = frontLimbs(targetSide);
    final backAnkleKey = backAnkleName(targetSide);

    // bounce: pelvis Y range / bbox height
    final pelvisYs = <double>[];
    final allYs = <double>[];
    for (final skel in skeletons) {
      final pc = _pelvisCenter(skel);
      if (pc != null) pelvisYs.add(pc.dy);
      for (final v in skel.values) {
        allYs.add(v.dy);
      }
    }
    if (pelvisYs.length >= kBouncMinPelvisSamples && allYs.length >= 2) {
      final bbox = allYs.reduce(math.max) - allYs.reduce(math.min);
      if (bbox > 1e-4) {
        m['bounce_ratio'] =
            (pelvisYs.reduce(math.max) - pelvisYs.reduce(math.min)) / bbox;
      }
    }

    // knee angles: average (stance) + at peak ankle displacement (lunge)
    final kneeAngles = <double>[];
    for (final skel in skeletons) {
      final hip = skel[limbs['hip']!];
      final knee = skel[limbs['knee']!];
      final ankle = skel[limbs['ankle']!];
      if (hip != null && knee != null && ankle != null) {
        kneeAngles.add(calcAngle(hip, knee, ankle));
      }
    }
    if (kneeAngles.length >= 3) {
      m['avg_front_knee_angle_deg'] =
          kneeAngles.reduce((a, b) => a + b) / kneeAngles.length;
      m['min_front_knee_angle_deg'] = kneeAngles.reduce(math.min);
    }

    // knee angle at the peak-ankle-displacement frame (lunge_overextension)
    final refAnkle = skeletons[0][limbs['ankle']!];
    if (refAnkle != null) {
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
      if (hip != null && knee != null && ankle != null) {
        m['lunge_knee_angle_deg'] = calcAngle(hip, knee, ankle);
      }
    }

    // arm angle at peak wrist displacement (incomplete_arm_extension)
    final refWrist = skeletons[0][limbs['wrist']!];
    if (refWrist != null) {
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
      if (shoulder != null && elbow != null && wrist != null) {
        m['arm_extension_angle_deg'] = calcAngle(shoulder, elbow, wrist);
      }
    }

    // over-parry: body-relative wrist sweep / torso length
    final relXs = <double>[];
    final torsoLens = <double>[];
    for (final skel in skeletons) {
      final wrist = skel[limbs['wrist']!];
      final pelvis = _pelvisCenter(skel);
      if (wrist == null || pelvis == null) continue;
      relXs.add(wrist.dx - pelvis.dx);
      final shoulder = skel[limbs['shoulder']!];
      if (shoulder != null) torsoLens.add((shoulder - pelvis).distance);
    }
    if (relXs.length >= kOverParryMinWristSamples && torsoLens.isNotEmpty) {
      final torso = _median(torsoLens);
      if (torso > 1e-6) {
        m['parry_sweep_torso_ratio'] =
            (relXs.reduce(math.max) - relXs.reduce(math.min)) / torso;
      }
    }

    // step ratio + torso lean across the window
    final stepRatios = <double>[];
    final leans = <double>[];
    final facing = _windowFacingSign(skeletons);
    for (final skel in skeletons) {
      final frontAnkle = skel[limbs['ankle']!];
      final back = skel[backAnkleKey];
      final frontShoulder = skel[limbs['shoulder']!];
      final pelvis = _pelvisCenter(skel);

      if (frontAnkle != null && back != null && frontShoulder != null &&
          pelvis != null) {
        final sw = (frontShoulder.dx - pelvis.dx).abs() *
            config.stepShoulderProxyMultiplier;
        if (sw >= kStepMinShoulderWidthPx) {
          stepRatios.add((frontAnkle.dx - back.dx).abs() / sw);
        }
      }

      final lean = _torsoLeanDeg(skel, facing);
      if (lean != null) leans.add(lean);
    }
    if (stepRatios.isNotEmpty) {
      m['step_ratio_min'] = stepRatios.reduce(math.min);
      m['step_ratio_max'] = stepRatios.reduce(math.max);
      m['step_ratio_median'] = _median(stepRatios);
    }
    if (leans.isNotEmpty) {
      m['torso_lean_deg_min'] = leans.reduce(math.min);
      m['torso_lean_deg_max'] = leans.reduce(math.max);
      m['torso_lean_deg_median'] = _median(leans);
    }

    // guard: longest run of wrist-below-pelvis, in seconds
    int run = 0;
    int maxRun = 0;
    for (final skel in skeletons) {
      final wrist = skel[limbs['wrist']!];
      final pelvis = _pelvisCenter(skel);
      if (wrist != null && pelvis != null && wrist.dy > pelvis.dy) {
        run++;
        if (run > maxRun) maxRun = run;
      } else {
        run = 0;
      }
    }
    m['guard_below_pelvis_max_run_s'] = maxRun / fps;

    // foot-before-hand: how much earlier the ankle's attack-step rise began
    // vs the wrist's extension rise, in seconds (positive = foot first =
    // the error direction). Body-relative + rise-onset — same algorithm as
    // _checkFootBeforeHand. Absent when either movement's rise is below the
    // noise floor.
    final (wristRel, ankleRel) = _forwardSeries(skeletons);
    final wristOn = _riseOnset(wristRel, config.footBeforeHandMinDisplacement);
    final ankleOn = _riseOnset(ankleRel, config.footBeforeHandMinDisplacement);
    if (wristOn != null && ankleOn != null) {
      m['foot_hand_lead_s'] = (wristOn.$1 - ankleOn.$1) / fps;
    }

    return m;
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
    if (deltaY > config.bounceRatioThreshold * bboxHeight) {
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
    if (angle < config.lungeKneeMinAngleDeg) return 'lunge_overextension';
    return null;
  }

  // ── Rule 3: guard_dropped ─────────────────────────────────────────────────

  String? _checkGuard(List<Skeleton> skeletons, double fps) {
    final limbs = frontLimbs(targetSide);
    int consecutive = 0;
    final seconds = trainingMode == 'Free Bouting'
        ? config.guardDroppedFreeBoutingSeconds
        : config.guardDroppedSeconds;
    final threshold = _framesFor(seconds, fps);

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
  //
  // Signals are BODY-RELATIVE forward positions (wrist/ankle X minus pelvis X,
  // signed toward the opponent) so whole-body translation — ordinary footwork
  // before the attack — doesn't register as "movement". The onset of each
  // movement is found by walking BACK from its global peak (full arm
  // extension / lunge-foot landing) to where that final rise began, which
  // anchors both events to the attack itself rather than to the arbitrary
  // start of the rolling window. Foot onset clearly before hand onset =
  // wrong sequencing.

  /// Forward direction sign: left-side fencer faces +x, right-side faces −x.
  /// Fallback only — prefer [_windowFacingSign], which reads the actual
  /// stance instead of trusting the targetSide setting.
  double get _forwardSign => targetSide == 'left' ? 1.0 : -1.0;

  /// Facing inferred from foot placement: the front (sword-side) ankle is
  /// always toward the opponent, so the median X offset front−back ankle
  /// gives the attack direction regardless of which way the fencer faces
  /// on screen or whether targetSide is set correctly.
  double _windowFacingSign(List<Skeleton> skeletons) {
    final limbs = frontLimbs(targetSide);
    final backKey = backAnkleName(targetSide);
    final diffs = <double>[];
    for (final skel in skeletons) {
      final fa = skel[limbs['ankle']!];
      final ba = skel[backKey];
      if (fa != null && ba != null) diffs.add(fa.dx - ba.dx);
    }
    if (diffs.isEmpty) return _forwardSign;
    final med = _median(diffs);
    if (med.abs() < 1e-6) return _forwardSign;
    return med > 0 ? 1.0 : -1.0;
  }

  /// Onset index of the final rise toward the series' global peak, or null
  /// if the total rise (peak − baseline-before-peak) is below [minRise].
  static (int, double)? _riseOnset(List<double?> series, double minRise) {
    int? peakIdx;
    double peakVal = -double.infinity;
    for (int i = 0; i < series.length; i++) {
      final v = series[i];
      if (v != null && v > peakVal) {
        peakVal = v;
        peakIdx = i;
      }
    }
    if (peakIdx == null) return null;

    double baseline = double.infinity;
    for (int i = 0; i <= peakIdx; i++) {
      final v = series[i];
      if (v != null && v < baseline) baseline = v;
    }
    final rise = peakVal - baseline;
    if (rise < minRise) return null;

    final onsetLevel = baseline + 0.1 * rise;
    int onset = peakIdx;
    for (int i = peakIdx; i >= 0; i--) {
      final v = series[i];
      if (v == null) continue;
      if (v <= onsetLevel) break;
      onset = i;
    }
    return (onset, rise);
  }

  /// Body-relative forward series for the front wrist and front ankle.
  (List<double?>, List<double?>) _forwardSeries(List<Skeleton> skeletons) {
    final limbs = frontLimbs(targetSide);
    final sign = _windowFacingSign(skeletons);
    final wristRel = <double?>[];
    final ankleRel = <double?>[];
    for (final skel in skeletons) {
      final pelvis = _pelvisCenter(skel);
      final wrist = skel[limbs['wrist']!];
      final ankle = skel[limbs['ankle']!];
      wristRel.add(pelvis == null || wrist == null
          ? null
          : sign * (wrist.dx - pelvis.dx));
      ankleRel.add(pelvis == null || ankle == null
          ? null
          : sign * (ankle.dx - pelvis.dx));
    }
    return (wristRel, ankleRel);
  }

  String? _checkFootBeforeHand(List<Skeleton> skeletons, double fps) {
    final (wristRel, ankleRel) = _forwardSeries(skeletons);
    final wrist = _riseOnset(wristRel, config.footBeforeHandMinDisplacement);
    final ankle = _riseOnset(ankleRel, config.footBeforeHandMinDisplacement);
    if (wrist == null || ankle == null) return null;

    final margin = _framesFor(config.footBeforeHandLeadSeconds, fps);
    if (ankle.$1 + margin <= wrist.$1) return 'foot_before_hand';
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
    if (avg > config.stanceTooHighAngleDeg) return 'stance_too_high';
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
    if (angle < config.incompleteArmExtensionAngleDeg) return 'incomplete_arm_extension';
    return null;
  }

  // ── Rule 9: over_parrying ─────────────────────────────────────────────────
  //
  // Wrist X measured RELATIVE to the pelvis center each frame, so whole-body
  // translation (advancing/retreating) doesn't register as a parry sweep.
  // Scale reference is the median torso length (front shoulder → pelvis,
  // Euclidean), which stays meaningful in the side-on view where the X-width
  // between the two shoulders collapses to ~0.

  String? _checkOverParrying(List<Skeleton> skeletons) {
    final limbs = frontLimbs(targetSide);
    final relXs = <double>[];
    final torsoLens = <double>[];

    for (final skel in skeletons) {
      final wrist = skel[limbs['wrist']!];
      final pelvis = _pelvisCenter(skel);
      if (wrist == null || pelvis == null) continue;
      relXs.add(wrist.dx - pelvis.dx);
      final shoulder = skel[limbs['shoulder']!];
      if (shoulder != null) {
        torsoLens.add((shoulder - pelvis).distance);
      }
    }

    if (relXs.length < kOverParryMinWristSamples || torsoLens.isEmpty) {
      return null;
    }
    final torso = _median(torsoLens);
    if (torso < 1e-6) return null;

    final sweepRange = relXs.reduce(math.max) - relXs.reduce(math.min);
    if (sweepRange > config.overParryTorsoRatioThreshold * torso) {
      return 'over_parrying';
    }
    return null;
  }

  // ── Rule 10: wide_step / narrow_step ──────────────────────────────────────
  //
  // The ratio must stay out of the healthy band for a sustained duration —
  // feet naturally pass close together mid-stride, and a single such frame
  // is normal footwork, not an error.

  String? _checkStepWidth(List<Skeleton> skeletons, double fps) {
    final limbs = frontLimbs(targetSide);
    final backAnkleKey = backAnkleName(targetSide);
    final sustained = _framesFor(config.stepSustainedSeconds, fps);
    int wideRun = 0;
    int narrowRun = 0;

    for (final skel in skeletons) {
      final frontAnkle = skel[limbs['ankle']!];
      final back = skel[backAnkleKey];
      final frontShoulder = skel[limbs['shoulder']!];
      final pelvis = _pelvisCenter(skel);
      if (frontAnkle == null || back == null || frontShoulder == null ||
          pelvis == null) continue;

      final sw =
          (frontShoulder.dx - pelvis.dx).abs() * config.stepShoulderProxyMultiplier;
      if (sw < kStepMinShoulderWidthPx) continue;

      final stepWidth = (frontAnkle.dx - back.dx).abs();
      final ratio = stepWidth / sw;

      if (ratio > config.wideStepRatioThreshold) {
        wideRun++;
        narrowRun = 0;
        if (wideRun >= sustained) return 'wide_step';
      } else if (ratio < config.narrowStepRatioThreshold) {
        narrowRun++;
        wideRun = 0;
        if (narrowRun >= sustained) return 'narrow_step';
      } else {
        wideRun = 0;
        narrowRun = 0;
      }
    }
    return null;
  }

  // ── Rule 11: center_of_mass_in_front / leaning_backward ──────────────────
  //
  // Torso lean angle from vertical (pelvis center → front shoulder), signed
  // toward the opponent: leaning too far forward/backward = the error. Same
  // sustained-duration requirement as step width so a transient lean during
  // a step's transfer phase doesn't trigger.

  /// Torso lean from vertical in degrees; positive = toward the opponent
  /// ([facingSign] from [_windowFacingSign]). Null when joints are missing
  /// or the pose is degenerate.
  double? _torsoLeanDeg(Skeleton skel, double facingSign) {
    final limbs = frontLimbs(targetSide);
    final shoulder = skel[limbs['shoulder']!];
    final pelvis = _pelvisCenter(skel);
    if (shoulder == null || pelvis == null) return null;
    final vertical = pelvis.dy - shoulder.dy; // >0: shoulder above pelvis
    if (vertical <= 1e-6) return null;
    final forward = facingSign * (shoulder.dx - pelvis.dx);
    return math.atan2(forward, vertical) * 180.0 / math.pi;
  }

  String? _checkCenterOfMass(List<Skeleton> skeletons, double fps) {
    final sustained = _framesFor(config.comSustainedSeconds, fps);
    final facing = _windowFacingSign(skeletons);
    int frontRun = 0;
    int backRun = 0;

    for (final skel in skeletons) {
      final lean = _torsoLeanDeg(skel, facing);
      if (lean == null) continue;

      if (lean > config.comForwardLeanDeg) {
        frontRun++;
        backRun = 0;
        if (frontRun >= sustained) return 'center_of_mass_in_front';
      } else if (lean < config.comBackwardLeanDeg) {
        backRun++;
        frontRun = 0;
        if (backRun >= sustained) return 'center_of_mass_leaning_backward';
      } else {
        frontRun = 0;
        backRun = 0;
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
