import 'dart:math' as math;
import 'dart:ui' show Offset;

import 'pose_service.dart';

double? calcAngle(Offset? p1, Offset? p2, Offset? p3) {
  if (p1 == null || p2 == null || p3 == null) return null;
  final v1x = p1.dx - p2.dx;
  final v1y = p1.dy - p2.dy;
  final v2x = p3.dx - p2.dx;
  final v2y = p3.dy - p2.dy;

  final dot = v1x * v2x + v1y * v2y;
  final mag1 = math.sqrt(v1x * v1x + v1y * v1y);
  final mag2 = math.sqrt(v2x * v2x + v2y * v2y);
  if (mag1 == 0 || mag2 == 0) return null;

  final cosTheta = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
  return math.acos(cosTheta) * 180.0 / math.pi;
}

Offset? pelvisCenter(FencingSkeleton skeleton) {
  final lHip = skeleton['left_hip'];
  final rHip = skeleton['right_hip'];
  if (lHip == null && rHip == null) return null;
  if (lHip == null) return rHip;
  if (rHip == null) return lHip;
  return Offset((lHip.dx + rHip.dx) / 2, (lHip.dy + rHip.dy) / 2);
}

class ActivityGatekeeper {
  static const stateIdle = 'IDLE';
  static const stateChecking = 'CHECKING';
  static const stateActive = 'ACTIVE';

  final int fps;
  final double activeKneeAngleDeg;
  final double idleKneeAngleDeg;
  final double motionThresholdNorm; 

  String state = stateIdle;
  int frameCount = 0;

  int _activeTriggerCount = 0;
  late int _activeTriggerThreshold;
  int _idleTriggerCount = 0;
  late int _idleTriggerThreshold;

  Offset? _lastPelvisCenter;
  Map<String, dynamic> lastReasons = {};

  ActivityGatekeeper({
    this.fps = 30,
    this.activeKneeAngleDeg = 176.0,
    this.idleKneeAngleDeg = 178.0,
    this.motionThresholdNorm = 0.005,
  }) {
    _activeTriggerThreshold = 5;
    _idleTriggerThreshold = 2 * fps;
  }

  void reset() {
    state = stateIdle;
    _activeTriggerCount = 0;
    _idleTriggerCount = 0;
    _lastPelvisCenter = null;
  }

  bool shouldExtractPose() {
    frameCount++;
    if (state == stateIdle) {
      final skipRate = math.max(1, fps ~/ 5);
      return (frameCount % skipRate) == 1;
    }
    return true;
  }

  double? _getKneeAngle(FencingSkeleton skeleton, String targetSide) {
    final isLeft = targetSide == 'left';
    final hip = skeleton[isLeft ? 'left_hip' : 'right_hip'];
    final knee = skeleton[isLeft ? 'left_knee' : 'right_knee'];
    final ankle = skeleton[isLeft ? 'left_ankle' : 'right_ankle'];
    return calcAngle(hip, knee, ankle);
  }

  double? _getShoulderWidth(FencingSkeleton skeleton) {
    final ls = skeleton['left_shoulder'];
    final rs = skeleton['right_shoulder'];
    if (ls == null || rs == null) return null; 
    final dx = ls.dx - rs.dx;
    final dy = ls.dy - rs.dy;
    return math.sqrt(dx * dx + dy * dy);
  }

  bool update(FencingSkeleton? targetSkeleton, String targetSide) {
    if (targetSkeleton == null) {
      if (state == stateActive) {
        _idleTriggerCount++;
        if (_idleTriggerCount >= _idleTriggerThreshold) {
          state = stateIdle;
          _idleTriggerCount = 0;
        }
      } else if (state == stateChecking) {
        state = stateIdle;
        _activeTriggerCount = 0;
      }
      
      lastReasons = {
        'has_target': false,
        'state': state,
        'reason': 'missing_target',
      };
      return state == stateActive;
    }

    final kneeAngle = _getKneeAngle(targetSkeleton, targetSide) ?? 180.0;
    
    final shoulderWidth = _getShoulderWidth(targetSkeleton);
    final isTurnedBack = (shoulderWidth ?? 1.0) < 0.05;

    final pelvis = pelvisCenter(targetSkeleton);
    double pelvisMotion = 0.0;
    if (pelvis != null && _lastPelvisCenter != null) {
      final dx = pelvis.dx - _lastPelvisCenter!.dx;
      final dy = pelvis.dy - _lastPelvisCenter!.dy;
      pelvisMotion = math.sqrt(dx * dx + dy * dy);
    }
    
    final moving = _lastPelvisCenter == null || pelvisMotion >= motionThresholdNorm;
    if (pelvis != null) {
      _lastPelvisCenter = pelvis;
    }

    final enGardePosture = kneeAngle < activeKneeAngleDeg;
    final enGarde = enGardePosture && (moving || state != stateIdle || _activeTriggerCount > 0);
    final standingUp = kneeAngle > idleKneeAngleDeg;
    final stopCondition = standingUp || isTurnedBack;

    if (state == stateIdle) {
      if (enGarde) {
        state = stateChecking;
        _activeTriggerCount = 1;
      }
    } else if (state == stateChecking) {
      if (enGarde) {
        _activeTriggerCount++;
        if (_activeTriggerCount >= _activeTriggerThreshold) {
          state = stateActive;
          _idleTriggerCount = 0;
        }
      } else {
        state = stateIdle;
        _activeTriggerCount = 0;
      }
    } else if (state == stateActive) {
      if (stopCondition) {
        _idleTriggerCount++;
        if (_idleTriggerCount >= _idleTriggerThreshold) {
          state = stateIdle;
          _idleTriggerCount = 0;
        }
      } else {
        _idleTriggerCount = 0;
      }
    }

    lastReasons = {
      'has_target': true,
      'state': state,
      'knee_angle': kneeAngle,
      'en_garde': enGarde,
      'en_garde_posture': enGardePosture,
      'standing_up': standingUp,
      'turned_back': isTurnedBack,
      'moving': moving,
    };

    return state == stateActive;
  }
}
