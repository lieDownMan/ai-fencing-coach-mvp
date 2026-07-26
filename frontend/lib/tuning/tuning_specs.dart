/// Per-error specs for the in-app Tuning tab: which live metric to watch,
/// which HeuristicsConfig parameter it compares against, the trigger
/// direction, and a sensible slider range for realtime adjustment.

library;

import '../heuristics/heuristics_engine.dart';

enum TriggerDirection { above, below }

class TuningSpec {
  final String errorKey;
  final String metricKey; // key in HeuristicsEngine.computeWindowMetrics()
  final String paramName; // key in HeuristicsConfig.toMap()
  final TriggerDirection direction;
  final double min;
  final double max;
  final String unit;
  final int decimals;
  final String hint; // one-line usage hint shown in the UI

  const TuningSpec({
    required this.errorKey,
    required this.metricKey,
    required this.paramName,
    required this.direction,
    required this.min,
    required this.max,
    required this.unit,
    this.decimals = 2,
    required this.hint,
  });

  double thresholdOf(HeuristicsConfig c) => c.toMap()[paramName]!;

  HeuristicsConfig apply(HeuristicsConfig c, double value) {
    final map = c.toMap();
    map[paramName] = value;
    return HeuristicsConfig.fromMap(map);
  }

  bool wouldTrigger(double metric, double threshold) =>
      direction == TriggerDirection.above
          ? metric > threshold
          : metric < threshold;
}

const List<TuningSpec> kTuningSpecs = [
  TuningSpec(
    errorKey: 'stance_too_high',
    metricKey: 'avg_front_knee_angle_deg',
    paramName: 'stanceTooHighAngleDeg',
    direction: TriggerDirection.above,
    min: 140,
    max: 180,
    unit: '°',
    decimals: 1,
    hint: '維持 en garde，看膝角數字。180=站直、標準蹲姿 120–140。',
  ),
  TuningSpec(
    errorKey: 'lunge_overextension',
    metricKey: 'lunge_knee_angle_deg',
    paramName: 'lungeKneeMinAngleDeg',
    direction: TriggerDirection.below,
    min: 60,
    max: 120,
    unit: '°',
    decimals: 1,
    hint: '做弓步，數字是最深那幀的前膝角。90=小腿垂直。',
  ),
  TuningSpec(
    errorKey: 'incomplete_arm_extension',
    metricKey: 'arm_extension_angle_deg',
    paramName: 'incompleteArmExtensionAngleDeg',
    direction: TriggerDirection.below,
    min: 120,
    max: 180,
    unit: '°',
    decimals: 1,
    hint: '出手刺擊，數字是最遠那幀的手臂角。伸直=170–180。',
  ),
  TuningSpec(
    errorKey: 'bounce_excessive',
    metricKey: 'bounce_ratio',
    paramName: 'bounceRatioThreshold',
    direction: TriggerDirection.above,
    min: 0.10,
    max: 0.60,
    unit: '',
    hint: '做步伐，數字是骨盆起伏佔身高比例。',
  ),
  TuningSpec(
    errorKey: 'guard_dropped',
    metricKey: 'guard_below_pelvis_max_run_s',
    paramName: 'guardDroppedSeconds',
    direction: TriggerDirection.above,
    min: 0.10,
    max: 2.00,
    unit: 's',
    hint: '手垂低於骨盆，數字是連續低垂秒數。(Free Bouting 另有放寬值)',
  ),
  TuningSpec(
    errorKey: 'over_parrying',
    metricKey: 'parry_sweep_torso_ratio',
    paramName: 'overParryTorsoRatioThreshold',
    direction: TriggerDirection.above,
    min: 0.40,
    max: 2.50,
    unit: '×軀幹',
    hint: '做防守揮劍，數字是手腕橫掃範圍 ÷ 軀幹長。',
  ),
  TuningSpec(
    errorKey: 'narrow_step',
    metricKey: 'step_ratio_median',
    paramName: 'narrowStepRatioThreshold',
    direction: TriggerDirection.below,
    min: 0.30,
    max: 2.00,
    unit: '×肩寬',
    hint: '維持窄站姿看數字。正常 en garde 約 1.5–2.5。',
  ),
  TuningSpec(
    errorKey: 'wide_step',
    metricKey: 'step_ratio_median',
    paramName: 'wideStepRatioThreshold',
    direction: TriggerDirection.above,
    min: 2.00,
    max: 4.50,
    unit: '×肩寬',
    hint: '維持寬站姿看數字。正常 en garde 約 1.5–2.5。',
  ),
  TuningSpec(
    errorKey: 'center_of_mass_in_front',
    metricKey: 'com_ratio_median',
    paramName: 'comInFrontRatioThreshold',
    direction: TriggerDirection.above,
    min: 0.50,
    max: 0.90,
    unit: '',
    hint: '重心前傾看數字。0.5=正中、1.0=完全壓在前腳。',
  ),
  TuningSpec(
    errorKey: 'center_of_mass_leaning_backward',
    metricKey: 'com_ratio_median',
    paramName: 'comLeaningBackRatioThreshold',
    direction: TriggerDirection.below,
    min: 0.10,
    max: 0.50,
    unit: '',
    hint: '重心後仰看數字。0.5=正中、0.0=完全壓在後腳。',
  ),
  TuningSpec(
    errorKey: 'foot_before_hand',
    metricKey: 'foot_hand_lead_s',
    paramName: 'footBeforeHandLeadSeconds',
    direction: TriggerDirection.above,
    min: 0.00,
    max: 0.50,
    unit: 's',
    hint: '做刺靶動作才有值：正數=腳比手先動幾秒（錯誤方向）。',
  ),
];

TuningSpec specForError(String errorKey) =>
    kTuningSpecs.firstWhere((s) => s.errorKey == errorKey);
