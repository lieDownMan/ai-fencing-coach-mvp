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
    'front_shoulder': Offset(0.54 + dx, 0.35),
    'left_shoulder': Offset(0.50 + dx, 0.35),
    'right_shoulder': Offset(0.54 + dx, 0.35),
    'front_elbow': Offset(0.64 + dx, 0.40),
    'front_wrist': Offset(wristX + dx, wristY),
  };
}

List<Skeleton> repeat(Skeleton s, int n) => List.generate(n, (_) => Map.of(s));

void main() {
  final engine = HeuristicsEngine(targetSide: 'left', trainingMode: 'Footwork');

  group('baseline', () {
    test('sound en-garde during SF triggers nothing', () {
      final errors = engine.evaluateWindow(
        action: 'SF',
        skeletons: repeat(makeStance(), 40),
        fps: 30,
      );
      expect(errors, isEmpty);
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
        ...repeat(makeStance(), 20),
        ...repeat(makeStance(frontAnkleX: 0.54, backAnkleX: 0.46), 4),
        ...repeat(makeStance(), 20),
      ];
      final errors =
          engine.evaluateWindow(action: 'SF', skeletons: skels, fps: 30);
      expect(errors, isNot(contains('narrow_step')));
    });
  });

  group('center_of_mass', () {
    test('sustained forward pelvis triggers center_of_mass_in_front', () {
      // pelvis x=0.5 with ankles 0.30–0.52 → ratio (0.5-0.3)/0.22 ≈ 0.91 > 0.65
      final leaning = makeStance(frontAnkleX: 0.52, backAnkleX: 0.30);
      final errors = engine.evaluateWindow(
        action: 'SF',
        skeletons: repeat(leaning, 40),
        fps: 30,
      );
      expect(errors, contains('center_of_mass_in_front'));
    });

    test('transient shift during a step does NOT trigger', () {
      final skels = [
        ...repeat(makeStance(), 20),
        ...repeat(makeStance(frontAnkleX: 0.52, backAnkleX: 0.30), 4),
        ...repeat(makeStance(), 20),
      ];
      final errors =
          engine.evaluateWindow(action: 'SF', skeletons: skels, fps: 30);
      expect(errors, isNot(contains('center_of_mass_in_front')));
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
    test('wrist below pelvis for >0.35s triggers at 30fps', () {
      final dropped = makeStance(wristY: 0.60); // below pelvis (0.50)
      final errors = engine.evaluateWindow(
        action: 'SF',
        skeletons: repeat(dropped, 15), // 15 frames @30fps = 0.5s > 0.35s
        fps: 30,
      );
      expect(errors, contains('guard_dropped'));
    });

    test('same frame count at high fps does NOT trigger (shorter wall time)', () {
      final dropped = makeStance(wristY: 0.60);
      final errors = engine.evaluateWindow(
        action: 'SF',
        skeletons: repeat(dropped, 15), // 15 frames @60fps = 0.25s < 0.35s
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
      final skels = List.generate(30, (i) {
        // ankle moves from frame 5, wrist from frame 15 (10-frame lead > 3-frame margin)
        final ankleShift = i >= 5 ? 0.05 : 0.0;
        final wristShift = i >= 15 ? 0.05 : 0.0;
        final s = makeStance(wristX: 0.74 + wristShift);
        s['right_ankle'] =
            Offset(s['right_ankle']!.dx + ankleShift, s['right_ankle']!.dy);
        return s;
      });
      final errors =
          tpEngine.evaluateWindow(action: 'R', skeletons: skels, fps: 30);
      expect(errors, contains('foot_before_hand'));
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
