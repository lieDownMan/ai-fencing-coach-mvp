/// Dart port of backend/src/realtime/feedback_scheduler.py
///
/// Priority scheduler for realtime coaching feedback.
/// Handles error aging, persistence, novelty, repeat penalties, and cooldowns.
/// Mirrors the Python FeedbackScheduler exactly so behaviour stays in sync
/// with the main branch.

library;

import 'dart:math' as math;

// ---------------------------------------------------------------------------
// Error weights — mirrors feedback_config.py DEFAULT_ERROR_WEIGHTS
// ---------------------------------------------------------------------------

const Map<String, double> kDefaultErrorWeights = {
  'foot_before_hand': 5.0,
  'lunge_overextension': 9.5,
  'incomplete_arm_extension': 9.0,
  'guard_dropped': 9.7,
  'stance_too_high': 10.0,
  'bounce_excessive': 6.5,
  'center_of_mass_in_front': 6.0,
  'center_of_mass_leaning_backward': 6.0,
  'over_parrying': 5.0,
  'wide_step': 4.0,
  'narrow_step': 9.0,
  'hand_too_high': 8.0,
};

// ---------------------------------------------------------------------------
// Data classes
// ---------------------------------------------------------------------------

/// One visual feedback row, equivalent to Python FeedbackItem.
class FeedbackItem {
  final String errorKey;
  final double score;
  final bool triggered;   // still actively detected in the current window
  final bool focused;     // the user has marked this as a focus error
  final int activeCount;
  final int spokenCount;
  final double cooldownRemaining;
  final double queuedSeconds;

  const FeedbackItem({
    required this.errorKey,
    required this.score,
    required this.triggered,
    required this.focused,
    required this.activeCount,
    required this.spokenCount,
    required this.cooldownRemaining,
    required this.queuedSeconds,
  });
}

/// Scheduler output for one inference tick, equivalent to Python FeedbackDecision.
class FeedbackDecision {
  /// The error key chosen for voice output this tick (null = silence).
  final String? voiceErrorKey;

  /// Visual items sorted by dynamic priority (up to visualTopN entries).
  final List<FeedbackItem> visualItems;

  const FeedbackDecision({
    this.voiceErrorKey,
    required this.visualItems,
  });
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

class _FeedbackState {
  final String errorKey;
  final double baseWeight;

  int skippedCount = 0;
  int activeCount = 0;
  int spokenCount = 0;
  double? firstPendingAt;
  double? lastSeenAt;
  double? lastSpokenAt;

  _FeedbackState({required this.errorKey, required this.baseWeight});
}

// ---------------------------------------------------------------------------
// FeedbackScheduler
// ---------------------------------------------------------------------------

/// Dynamic-priority queue with aging for realtime fencing feedback.
///
/// Voice feedback is intentionally narrow: one cue per scheduling decision,
/// with per-error and global cooldowns.  Visual feedback is broader: callers
/// receive the top-N current/queued issues sorted by dynamic priority.
class FeedbackScheduler {
  // Tuning knobs
  final double agingFactor;
  final double persistenceFactor;
  final double noveltyBonus;
  final double repeatPenalty;
  final double voiceCooldownSeconds;
  final double globalVoiceCooldownSeconds;
  final int minActiveCount;
  final int visualTopN;
  final double pendingTtlSeconds;
  final double focusBoost;

  // Weights per error key
  final Map<String, double> weights;

  // User preferences
  final Set<String> focusErrors;
  final Set<String> muteErrors;
  final Set<String> onlyErrors;

  // Internal queue
  final Map<String, _FeedbackState> _states = {};
  double? _lastVoiceAt;

  FeedbackScheduler({
    this.agingFactor = 2.0,
    this.persistenceFactor = 0.25,
    this.noveltyBonus = 0.75,
    this.repeatPenalty = 1.0,
    this.voiceCooldownSeconds = 4.0,
    this.globalVoiceCooldownSeconds = 1.2,
    this.minActiveCount = 1,
    this.visualTopN = 3,
    this.pendingTtlSeconds = 5.0,
    this.focusBoost = 4.0,
    Map<String, double>? weights,
    Set<String>? focusErrors,
    Set<String>? muteErrors,
    Set<String>? onlyErrors,
  })  : weights = {...kDefaultErrorWeights, ...?weights},
        focusErrors = focusErrors ?? {},
        muteErrors = muteErrors ?? {},
        onlyErrors = onlyErrors ?? {};

  /// Current time in fractional seconds (monotonic-like via epoch).
  static double _now() => DateTime.now().millisecondsSinceEpoch / 1000.0;

  // ── Public API ─────────────────────────────────────────────────────────────

  /// Feed the currently detected error keys into the scheduler.
  /// Returns a [FeedbackDecision] with an optional voice cue and sorted visual items.
  FeedbackDecision update(Iterable<String> activeErrorKeys) {
    final now = _now();

    // Collect allowed active errors
    final activeSet = <String>{};
    for (final key in activeErrorKeys) {
      if (_allows(key)) activeSet.add(key);
    }

    // Increment active counts for currently-firing errors
    for (final key in activeSet) {
      final state = _stateFor(key);
      if (!_isPending(state, now)) {
        state.firstPendingAt = now;
        state.skippedCount = 0;
        state.activeCount = 0;
      }
      state.lastSeenAt = now;
      state.activeCount++;
    }

    // Reset states for errors that have left the pending window
    for (final state in _states.values) {
      if (!activeSet.contains(state.errorKey) && !_isPending(state, now)) {
        state.activeCount = 0;
        state.skippedCount = 0;
        state.firstPendingAt = null;
      }
    }

    // Rank all pending states
    final pendingStates =
        _states.values.where((s) => _isPending(s, now)).toList();
    final scores = {for (final s in pendingStates) s.errorKey: _score(s)};

    pendingStates.sort((a, b) {
      final cmp = scores[b.errorKey]!.compareTo(scores[a.errorKey]!);
      if (cmp != 0) return cmp;
      final bw = b.baseWeight.compareTo(a.baseWeight);
      if (bw != 0) return bw;
      final ac = b.activeCount.compareTo(a.activeCount);
      if (ac != 0) return ac;
      return a.errorKey.compareTo(b.errorKey);
    });

    // Pick voice cue
    final voiceState = _selectVoiceState(pendingStates, scores, now);

    // Build visual items (top N)
    final visualItems = pendingStates
        .take(visualTopN)
        .map((s) => _makeItem(s,
            score: scores[s.errorKey]!,
            triggered: activeSet.contains(s.errorKey),
            now: now))
        .toList();

    // Commit voice decision
    String? voiceErrorKey;
    if (voiceState != null) {
      voiceErrorKey = voiceState.errorKey;
      voiceState.lastSpokenAt = now;
      voiceState.spokenCount++;
      voiceState.skippedCount = 0;
      _lastVoiceAt = now;

      for (final s in pendingStates) {
        if (s.errorKey != voiceState.errorKey) s.skippedCount++;
      }
    }

    return FeedbackDecision(voiceErrorKey: voiceErrorKey, visualItems: visualItems);
  }

  /// Clear all state (call when session resets or settings change).
  void reset() {
    _states.clear();
    _lastVoiceAt = null;
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  bool _allows(String errorKey) {
    if (muteErrors.contains(errorKey)) return false;
    if (onlyErrors.isNotEmpty && !onlyErrors.contains(errorKey)) return false;
    return true;
  }

  _FeedbackState _stateFor(String errorKey) {
    return _states.putIfAbsent(
      errorKey,
      () => _FeedbackState(
        errorKey: errorKey,
        baseWeight: weights[errorKey] ?? 5.0,
      ),
    );
  }

  bool _isPending(_FeedbackState state, double now) {
    if (state.lastSeenAt == null) return false;
    return (now - state.lastSeenAt!) <= pendingTtlSeconds;
  }

  double _score(_FeedbackState state) {
    final novelty = state.spokenCount == 0 ? noveltyBonus : 0.0;
    final rPenalty = repeatPenalty * math.min(state.spokenCount, 3).toDouble();
    final persistence =
        persistenceFactor * math.min(state.activeCount, 8).toDouble();
    final fBoost = focusErrors.contains(state.errorKey) ? focusBoost : 0.0;
    return state.baseWeight +
        agingFactor * state.skippedCount +
        persistence +
        novelty +
        fBoost -
        rPenalty;
  }

  double _cooldownRemaining(_FeedbackState state, double now) {
    if (state.lastSpokenAt == null) return 0.0;
    return math.max(0.0, voiceCooldownSeconds - (now - state.lastSpokenAt!));
  }

  double _globalCooldownRemaining(double now) {
    if (_lastVoiceAt == null) return 0.0;
    return math.max(
        0.0, globalVoiceCooldownSeconds - (now - _lastVoiceAt!));
  }

  _FeedbackState? _selectVoiceState(
    List<_FeedbackState> rankedStates,
    Map<String, double> scores,
    double now,
  ) {
    if (_globalCooldownRemaining(now) > 0) return null;

    final eligible = rankedStates
        .where((s) =>
            s.activeCount >= minActiveCount &&
            _cooldownRemaining(s, now) <= 0)
        .toList();
    if (eligible.isEmpty) return null;

    // Pick highest-scored eligible state (list is already sorted, so first wins)
    return eligible.first;
  }

  FeedbackItem _makeItem(
    _FeedbackState state, {
    required double score,
    required bool triggered,
    required double now,
  }) {
    final queuedSeconds = state.firstPendingAt != null
        ? math.max(0.0, now - state.firstPendingAt!)
        : 0.0;
    return FeedbackItem(
      errorKey: state.errorKey,
      score: score,
      triggered: triggered,
      focused: focusErrors.contains(state.errorKey),
      activeCount: state.activeCount,
      spokenCount: state.spokenCount,
      cooldownRemaining: _cooldownRemaining(state, now),
      queuedSeconds: queuedSeconds,
    );
  }
}
