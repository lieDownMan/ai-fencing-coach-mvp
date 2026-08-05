/// Cue metadata for the video threshold-tuner: filename → cues, the action
/// context each cue's check runs under live, and the auxiliary (non-primary)
/// HeuristicsConfig params worth co-tuning per cue.

library;

import 'tuning_specs.dart';

/// FenceNet action assumed when evaluating a cue in the tuner (the check must
/// be reachable in evaluateWindow under trainingMode 'Target Practice'):
///  - footwork checks (bounce/stance/step/CoM) run on footwork actions → SF
///  - offensive checks (lunge/foot-before-hand/arm) run on offensive → WW
///  - over_parrying runs on SB
///  - guard_dropped runs under any action
const Map<String, String> kCueAction = {
  'bounce_excessive': 'SF',
  'stance_too_high': 'SF',
  'narrow_step': 'SF',
  'wide_step': 'SF',
  'center_of_mass_in_front': 'SF',
  'center_of_mass_leaning_backward': 'SF',
  'guard_dropped': 'SF',
  'over_parrying': 'SB',
  'lunge_overextension': 'WW',
  'foot_before_hand': 'WW',
  'incomplete_arm_extension': 'WW',
};

/// Auxiliary config params shown as extra sliders when tuning a cue. These
/// affect the metric computation or the sustained/onset logic rather than the
/// primary threshold itself.
class AuxParamSpec {
  final String paramName;
  final double min;
  final double max;
  final int decimals;
  final String hint;

  const AuxParamSpec({
    required this.paramName,
    required this.min,
    required this.max,
    this.decimals = 2,
    required this.hint,
  });
}

const Map<String, List<AuxParamSpec>> kCueAuxParams = {
  'foot_before_hand': [
    AuxParamSpec(
      paramName: 'footBeforeHandMinDisplacement',
      min: 0.005,
      max: 0.10,
      decimals: 3,
      hint: '動作幅度門檻：手/腳的前伸量至少要這麼大才算「有動作」',
    ),
  ],
  'narrow_step': [
    AuxParamSpec(
      paramName: 'stepSustainedSeconds',
      min: 0.05,
      max: 1.0,
      hint: '比例要持續超出範圍這麼多秒才觸發',
    ),
    AuxParamSpec(
      paramName: 'stepShoulderProxyMultiplier',
      min: 1.0,
      max: 4.0,
      decimals: 1,
      hint: '肩寬代理倍率（側面視角肩寬會塌縮，用肩-骨盆距×倍率代替）',
    ),
  ],
  'wide_step': [
    AuxParamSpec(
      paramName: 'stepSustainedSeconds',
      min: 0.05,
      max: 1.0,
      hint: '比例要持續超出範圍這麼多秒才觸發',
    ),
    AuxParamSpec(
      paramName: 'stepShoulderProxyMultiplier',
      min: 1.0,
      max: 4.0,
      decimals: 1,
      hint: '肩寬代理倍率（側面視角肩寬會塌縮，用肩-骨盆距×倍率代替）',
    ),
  ],
  'center_of_mass_in_front': [
    AuxParamSpec(
      paramName: 'comSustainedSeconds',
      min: 0.05,
      max: 1.0,
      hint: '傾角要持續超過閾值這麼多秒才觸發',
    ),
  ],
  'center_of_mass_leaning_backward': [
    AuxParamSpec(
      paramName: 'comSustainedSeconds',
      min: 0.05,
      max: 1.0,
      hint: '傾角要持續超過閾值這麼多秒才觸發',
    ),
  ],
};

/// Mirror of main.dart's kErrorLabels (kept local so the tuner entry point
/// doesn't have to compile the whole live app).
const Map<String, String> kCueLabels = {
  'lunge_overextension': '長刺過度前傾 (Lunge Overextension)',
  'guard_dropped': '持劍手掉落 (Guard Dropped)',
  'bounce_excessive': '步伐上下浮動 (Excessive Bounce)',
  'foot_before_hand': '手腳順序錯誤 (Foot Before Hand)',
  'over_parrying': '防守動作太大 (Over-Parrying)',
  'stance_too_high': '預備姿勢沒蹲好 (Stance Too High)',
  'incomplete_arm_extension': '手沒有伸直 (Incomplete Extension)',
  'wide_step': '步伐太大 (Wide Step)',
  'narrow_step': '步伐太小 (Narrow Step)',
  'center_of_mass_in_front': '重心向前 (CoM Forward)',
  'center_of_mass_leaning_backward': '重心向後 (CoM Backward)',
};

/// Cues matched from a video filename (e.g. `narrow_step_wide_step.MOV` →
/// [narrow_step, wide_step]; `center_of_mass.MOV` → both CoM cues).
List<String> cuesForVideoName(String fileName) {
  final base = fileName.toLowerCase();
  final out = <String>[];
  for (final spec in kTuningSpecs) {
    if (base.contains(spec.errorKey)) out.add(spec.errorKey);
  }
  if (base.contains('center_of_mass')) {
    for (final key in [
      'center_of_mass_in_front',
      'center_of_mass_leaning_backward',
    ]) {
      if (!out.contains(key)) out.add(key);
    }
  }
  return out;
}
