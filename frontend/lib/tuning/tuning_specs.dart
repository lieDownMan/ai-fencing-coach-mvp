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
    min: 130,
    max: 170,
    unit: '°',
    decimals: 1,
    hint: '做弓步，數字是最深那幀的前膝角。90=小腿垂直。',
  ),
  TuningSpec(
    errorKey: 'incomplete_arm_extension',
    metricKey: 'arm_extension_angle_deg',
    paramName: 'incompleteArmExtensionAngleDeg',
    direction: TriggerDirection.below,
    min: 70,
    max: 120,
    unit: '°',
    decimals: 1,
    hint: '出手刺擊，數字是最遠那幀的手臂角。伸直=170–180。',
  ),
  TuningSpec(
    errorKey: 'bounce_excessive',
    metricKey: 'bounce_ratio',
    paramName: 'bounceRatioThreshold',
    direction: TriggerDirection.above,
    min: 0.05,
    max: 0.60,
    unit: '',
    hint: '做步伐，數字是骨盆起伏佔身高比例。',
  ),
  TuningSpec(
    errorKey: 'guard_dropped',
    metricKey: 'guard_elbow_angle_deg_median',
    paramName: 'guardElbowAngleDeg',
    direction: TriggerDirection.above,
    min: 90,
    max: 180,
    unit: '°',
    decimals: 1,
    hint: '前手手肘角度：持劍彎曲 ~90–120°、手垂放伸直 >150°。'
        '另需手腕低於手肘（直臂前刺不算）、持續 guardDroppedSeconds 秒。',
  ),
  TuningSpec(
    errorKey: 'over_parrying',
    metricKey: 'parry_sweep_torso_ratio',
    paramName: 'overParryTorsoRatioThreshold',
    direction: TriggerDirection.above,
    min: 0.20,
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
    min: 1.00,
    max: 3.50,
    unit: '×肩寬',
    hint: '維持寬站姿看數字。正常 en garde 約 1.5–2.5。',
  ),
  TuningSpec(
    errorKey: 'center_of_mass_in_front',
    metricKey: 'torso_lean_deg_median',
    paramName: 'comForwardLeanDeg',
    direction: TriggerDirection.above,
    min: 5,
    max: 45,
    unit: '°',
    decimals: 1,
    hint: '軀幹前傾角（骨盆→肩膀 vs 鉛直線）。0=直立、正=朝對手傾。',
  ),
  TuningSpec(
    errorKey: 'center_of_mass_leaning_backward',
    metricKey: 'torso_lean_deg_median',
    paramName: 'comBackwardLeanDeg',
    direction: TriggerDirection.below,
    min: -30,
    max: 15,
    unit: '°',
    decimals: 1,
    hint: '軀幹傾角，負=向後仰。低於閾值觸發。',
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
