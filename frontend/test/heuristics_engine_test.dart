import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/heuristics/heuristics_engine.dart';
import 'package:frontend/tuning/tuning_specs.dart';

/// Synthetic skeleton for a LEFT-side fencer (front limbs = right_*) in
/// normalized [0,1] coordinates, standing in a sound en-garde:
/// bent front knee (~122°), straight-ish arm, wrist above pelvis,
/// pelvis centered between the ankles, step ratio ≈ 2 (healthy band 1–3).
Skeleton makeStance({
  double dx = 0.0, // whole-body horizontal translation
  double frontAnkleX = 0.60,
  double backAnkleX = 0.40,
  double pelvisY = 0.50,
  double wristX = 0.74,
  double wristY = 0.45,
  double shoulderX = 0.54, // front shoulder X — controls torso lean
  bool straightLeg = false,
}) {
  final knee = straightLeg
      ? const Offset(0.56, 0.65) // collinear hip(0.52,0.50)→ankle(0.60,0.80) → 180°
      : const Offset(0.62, 0.65); // bent (~122°)
  return {
    'left_hip': Offset(0.48 + dx, pelvisY),
    'right_hip': Offset(0.52 + dx, pelvisY),
    'right_knee': Offset(knee.dx + dx, knee.dy),
    'right_ankle': Offset(frontAnkleX + dx, 0.80),
    'left_ankle': Offset(backAnkleX + dx, 0.80),
    'front_shoulder': Offset(shoulderX + dx, 0.35),
    'left_shoulder': Offset(0.50 + dx, 0.35),
    'right_shoulder': Offset(shoulderX + dx, 0.35),
    'front_elbow': Offset(0.64 + dx, 0.40),
    'front_wrist': Offset(wristX + dx, wristY),
  };
}

List<Skeleton> repeat(Skeleton s, int n) => List.generate(n, (_) => Map.of(s));

/// Stance sitting inside every tuned healthy band — step ratio ≈1.33
/// (narrow 0.9 – wide 1.72) and torso lean ≈11.3° (back 5.5° – front 14°).
/// Use for "nothing should trigger" filler frames.
Skeleton goodStance() => makeStance(
    frontAnkleX: 0.55, backAnkleX: 0.45, shoulderX: 0.53);

void main() {
  final engine = HeuristicsEngine(targetSide: 'left', trainingMode: 'Footwork');

  group('baseline', () {
    test('sound en-garde during SF triggers nothing', () {
      final errors = engine.evaluateWindow(
        action: 'SF',
        skeletons: repeat(goodStance(), 40),
        fps: 30,
      );
      expect(errors, isEmpty);
    });

    test('no person in most of the window → no warnings, no metrics', () {
      // 30 empty (no detection) frames + 10 real ones: majority has no
      // person, so the engine must stay silent even though the real frames
      // alone would trigger stance_too_high.
      final skels = [
        ...repeat(<String, Offset>{}, 30),
        ...repeat(makeStance(straightLeg: true), 10),
      ];
      final errors =
          engine.evaluateWindow(action: 'SF', skeletons: skels, fps: 30);
      expect(errors, isEmpty);
      expect(engine.computeWindowMetrics(skels, fps: 30), isEmpty);
    });
  });

  group('wide_step / narrow_step', () {
    test('sustained wide stance triggers wide_step (regression: missing in old port)', () {
      // ankles 0.30–0.70 → step 0.40, shoulder proxy 0.10 → ratio 4.0 > 3.0
      final wide = makeStance(frontAnkleX: 0.70, backAnkleX: 0.30);
      final errors = engine.evaluateWindow(
        action: 'SF',
        skeletons: repeat(wide, 40),
        fps: 30,
      );
      expect(errors, contains('wide_step'));
    });

    test('sustained narrow stance triggers narrow_step', () {
      // ankles 0.46–0.54 → step 0.08, ratio 0.8 < 1.0
      final narrow = makeStance(frontAnkleX: 0.54, backAnkleX: 0.46);
      final errors = engine.evaluateWindow(
        action: 'SF',
        skeletons: repeat(narrow, 40),
        fps: 30,
      );
      expect(errors, contains('narrow_step'));
    });

    test('transient feet-together mid-stride does NOT trigger narrow_step', () {
      // 0.3s sustained @30fps = 9 frames; only 4 narrow frames here.
      final skels = [
        ...repeat(goodStance(), 20),
        ...repeat(makeStance(frontAnkleX: 0.54, backAnkleX: 0.46), 4),
        ...repeat(goodStance(), 20),
      ];
      final errors =
          engine.evaluateWindow(action: 'SF', skeletons: skels, fps: 30);
      expect(errors, isNot(contains('narrow_step')));
    });
  });

  group('center_of_mass (torso lean)', () {
    test('sustained forward lean triggers center_of_mass_in_front', () {
      // shoulder x 0.64, pelvis (0.50, 0.50) → lean atan2(0.14, 0.15) ≈ 43° > 25°
      final leaning = makeStance(shoulderX: 0.64);
      final errors = engine.evaluateWindow(
        action: 'SF',
        skeletons: repeat(leaning, 40),
        fps: 30,
      );
      expect(errors, contains('center_of_mass_in_front'));
    });

    test('sustained backward lean triggers center_of_mass_leaning_backward', () {
      // shoulder x 0.46 → lean atan2(-0.04, 0.15) ≈ -14.9° < -10°
      final leaning = makeStance(shoulderX: 0.46);
      final errors = engine.evaluateWindow(
        action: 'SF',
        skeletons: repeat(leaning, 40),
        fps: 30,
      );
      expect(errors, contains('center_of_mass_leaning_backward'));
    });

    test('transient lean during a step does NOT trigger', () {
      final skels = [
        ...repeat(goodStance(), 20),
        ...repeat(makeStance(shoulderX: 0.64), 4),
        ...repeat(goodStance(), 20),
      ];
      final errors =
          engine.evaluateWindow(action: 'SF', skeletons: skels, fps: 30);
      expect(errors, isNot(contains('center_of_mass_in_front')));
    });

    test('baseline en-garde lean (~15°) stays inside the healthy band', () {
      final m = engine.computeWindowMetrics(repeat(makeStance(), 10), fps: 30);
      expect(m['torso_lean_deg_median'], closeTo(14.93, 0.05));
    });

    test('facing the other way: forward lean is still positive and triggers '
        '(regression: sign was hardcoded from targetSide)', () {
      // Mirror the whole skeleton in X — fencer now faces −x.
      Skeleton mirror(Skeleton s) =>
          s.map((k, v) => MapEntry(k, Offset(1.0 - v.dx, v.dy)));
      final leaningMirrored = mirror(makeStance(shoulderX: 0.64));
      final m = engine.computeWindowMetrics(
          repeat(leaningMirrored, 40), fps: 30);
      expect(m['torso_lean_deg_median'], closeTo(43.0, 0.5));
      final errors = engine.evaluateWindow(
        action: 'SF',
        skeletons: repeat(leaningMirrored, 40),
        fps: 30,
      );
      expect(errors, contains('center_of_mass_in_front'));
    });
  });

  group('over_parrying', () {
    test('whole-body retreat does NOT trigger (regression: absolute-X sweep)', () {
      // Body translates 0.4 in X across the window — wrist moves with it.
      final skels = List.generate(
          40, (i) => makeStance(dx: -0.01 * i)); // retreat over the window
      final errors =
          engine.evaluateWindow(action: 'SB', skeletons: skels, fps: 30);
      expect(errors, isNot(contains('over_parrying')));
    });

    test('large wrist sweep relative to body triggers', () {
      // Torso length ≈ hypot(0.04, 0.15) ≈ 0.155; threshold 1.2× ≈ 0.186.
      // Sweep wrist X ±0.12 around the body → range 0.24 > 0.186.
      final skels = List.generate(40, (i) {
        final sweep = (i % 2 == 0) ? -0.12 : 0.12;
        return makeStance(wristX: 0.74 + sweep, wristY: 0.45);
      });
      final errors =
          engine.evaluateWindow(action: 'SB', skeletons: skels, fps: 30);
      expect(errors, contains('over_parrying'));
    });
  });

  group('guard_dropped (time-based)', () {
    test('wrist below pelvis for >3s triggers at 30fps', () {
      final dropped = makeStance(wristY: 0.60); // below pelvis (0.50)
      final errors = engine.evaluateWindow(
        action: 'SF',
        skeletons: repeat(dropped, 100), // 100 frames @30fps ≈ 3.3s > 3.0s
        fps: 30,
      );
      expect(errors, contains('guard_dropped'));
    });

    test('same frame count at high fps does NOT trigger (shorter wall time)', () {
      final dropped = makeStance(wristY: 0.60);
      final errors = engine.evaluateWindow(
        action: 'SF',
        skeletons: repeat(dropped, 100), // 100 frames @60fps ≈ 1.7s < 3.0s
        fps: 60,
      );
      expect(errors, isNot(contains('guard_dropped')));
    });
  });

  group('HeuristicsConfig + tuning specs', () {
    test('toMap/fromMap roundtrip preserves every value', () {
      final original = const HeuristicsConfig().copyWith(
        stanceTooHighAngleDeg: 163.5,
        overParryTorsoRatioThreshold: 0.95,
        guardDroppedSeconds: 0.5,
      );
      final restored = HeuristicsConfig.fromMap(original.toMap());
      expect(restored.toMap(), equals(original.toMap()));
    });

    test('every tuning spec maps to a real config param and metric name', () {
      final params = const HeuristicsConfig().toMap().keys.toSet();
      for (final spec in kTuningSpecs) {
        expect(params, contains(spec.paramName),
            reason: '${spec.errorKey} points at unknown param');
        // apply() must change exactly that param
        final changed =
            spec.apply(const HeuristicsConfig(), spec.min).toMap();
        expect(changed[spec.paramName], spec.min);
      }
    });
  });

  group('stance_too_high', () {
    test('straight front leg triggers', () {
      final tall = makeStance(straightLeg: true);
      final errors = engine.evaluateWindow(
        action: 'SF',
        skeletons: repeat(tall, 10),
        fps: 30,
      );
      expect(errors, contains('stance_too_high'));
    });
  });

  group('foot_before_hand (onset-based)', () {
    final tpEngine =
        HeuristicsEngine(targetSide: 'left', trainingMode: 'Target Practice');

    test('foot starts clearly before hand → triggers', () {
      final skels = List.generate(40, (i) {
        // ankle moves from frame 5, wrist from frame 25 — 20-frame lead
        // (0.67s @30fps) > tuned leadSeconds 0.37s (11-frame margin)
        final ankleShift = i >= 5 ? 0.05 : 0.0;
        final wristShift = i >= 25 ? 0.05 : 0.0;
        final s = makeStance(wristX: 0.74 + wristShift);
        s['right_ankle'] =
            Offset(s['right_ankle']!.dx + ankleShift, s['right_ankle']!.dy);
        return s;
      });
      final errors =
          tpEngine.evaluateWindow(action: 'R', skeletons: skels, fps: 30);
      expect(errors, contains('foot_before_hand'));
    });

    test('lead below the 0.37s threshold does NOT trigger', () {
      final skels = List.generate(30, (i) {
        // 10-frame lead = 0.33s < 0.37s margin
        final ankleShift = i >= 5 ? 0.05 : 0.0;
        final wristShift = i >= 15 ? 0.05 : 0.0;
        final s = makeStance(wristX: 0.74 + wristShift);
        s['right_ankle'] =
            Offset(s['right_ankle']!.dx + ankleShift, s['right_ankle']!.dy);
        return s;
      });
      final errors =
          tpEngine.evaluateWindow(action: 'R', skeletons: skels, fps: 30);
      expect(errors, isNot(contains('foot_before_hand')));
    });

    test('retreat then simultaneous lunge does NOT trigger '
        '(regression: min-baseline anchored ankle onset to the retreat)', () {
      final skels = List.generate(60, (i) {
        // guard → retreat (frames 10–19, front ankle pulled in) → guard →
        // lunge at 45 with foot and hand together; frame 30 is a one-frame
        // ankle glitch the 5-frame median filter must absorb.
        var ankleX = 0.60;
        var wristX = 0.74;
        if (i >= 10 && i < 20) ankleX = 0.53;
        if (i == 30) ankleX = 0.52;
        if (i >= 45) {
          ankleX = 0.75;
          wristX = 0.79;
        }
        final s = makeStance(wristX: wristX);
        s['right_ankle'] = Offset(ankleX, s['right_ankle']!.dy);
        return s;
      });
      final errors =
          tpEngine.evaluateWindow(action: 'R', skeletons: skels, fps: 30);
      expect(errors, isNot(contains('foot_before_hand')));
      final m = tpEngine.computeWindowMetrics(skels, fps: 30);
      expect(m['foot_hand_lead_s'], closeTo(0.0, 1e-6));
    });

    test('foot_hand_lead_s metric reports the onset gap in seconds', () {
      final skels = List.generate(30, (i) {
        final ankleShift = i >= 5 ? 0.05 : 0.0;
        final wristShift = i >= 15 ? 0.05 : 0.0;
        final s = makeStance(wristX: 0.74 + wristShift);
        s['right_ankle'] =
            Offset(s['right_ankle']!.dx + ankleShift, s['right_ankle']!.dy);
        return s;
      });
      final m = tpEngine.computeWindowMetrics(skels, fps: 30);
      // ankle onset frame 5, wrist onset frame 15 → 10 frames @30fps ≈ 0.333s
      expect(m['foot_hand_lead_s'], closeTo(10 / 30, 1e-6));
    });

    test('simultaneous onset does NOT trigger', () {
      final skels = List.generate(30, (i) {
        final shift = i >= 5 ? 0.05 : 0.0;
        final s = makeStance(wristX: 0.74 + shift);
        s['right_ankle'] =
            Offset(s['right_ankle']!.dx + shift, s['right_ankle']!.dy);
        return s;
      });
      final errors =
          tpEngine.evaluateWindow(action: 'R', skeletons: skels, fps: 30);
      expect(errors, isNot(contains('foot_before_hand')));
    });
  });
}
