"""Priority scheduler for realtime coaching feedback.

The realtime pipeline can detect several posture errors in the same window.
This module keeps voice feedback focused on one cue at a time while still
letting the UI show the highest-priority current and queued issues.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

# Keep these imports available from feedback_scheduler for older callers while
# the source of truth lives in feedback_config.
from src.realtime.feedback_config import (
    DEFAULT_ERROR_WEIGHTS,
    ERROR_AVAILABILITY_BY_MODE,
    FUTURE_ERROR_KEYS,
    TRAINING_MODES,
    available_error_keys_for_mode,
    filter_error_keys_for_mode,
    is_error_available_for_mode,
    normalize_error_keys,
    supported_modes_for_error_key,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PLAYBOOK_PATH = _REPO_ROOT / "coach_playbook.json"


@dataclass(frozen=True)
class FeedbackPreferences:
    """User-selected feedback emphasis and filtering settings."""

    focus_errors: tuple[str, ...] = field(default_factory=tuple)
    mute_errors: tuple[str, ...] = field(default_factory=tuple)
    only_errors: tuple[str, ...] = field(default_factory=tuple)
    focus_boost: float = 4.0
    training_mode: Optional[str] = None

    @classmethod
    def build(
        cls,
        *,
        focus_errors: Optional[Iterable[str]] = None,
        mute_errors: Optional[Iterable[str]] = None,
        only_errors: Optional[Iterable[str]] = None,
        only_selected: bool = False,
        focus_boost: float = 4.0,
        training_mode: Optional[str] = None,
    ) -> "FeedbackPreferences":
        if training_mode is not None:
            focus = tuple(filter_error_keys_for_mode(focus_errors, training_mode)[0])
            mute = tuple(filter_error_keys_for_mode(mute_errors, training_mode)[0])
        else:
            focus = tuple(normalize_error_keys(focus_errors))
            mute = tuple(normalize_error_keys(mute_errors))

        only_source = focus if only_selected and focus else only_errors
        if training_mode is not None:
            only = tuple(filter_error_keys_for_mode(only_source, training_mode)[0])
        else:
            only = tuple(normalize_error_keys(only_source))

        focus = tuple(key for key in focus if key not in mute)
        only = tuple(key for key in only if key not in mute)
        return cls(
            focus_errors=focus,
            mute_errors=mute,
            only_errors=only,
            focus_boost=float(focus_boost),
            training_mode=training_mode,
        )

    @property
    def focus_set(self) -> set[str]:
        return set(self.focus_errors)

    @property
    def mute_set(self) -> set[str]:
        return set(self.mute_errors)

    @property
    def only_set(self) -> set[str]:
        return set(self.only_errors)

    @property
    def is_active(self) -> bool:
        return bool(self.focus_errors or self.mute_errors or self.only_errors)

    def allows(self, error_key: str) -> bool:
        if self.training_mode is not None and not is_error_available_for_mode(error_key, self.training_mode):
            return False
        if error_key in self.mute_set:
            return False
        only = self.only_set
        return not only or error_key in only

    def is_focused(self, error_key: str) -> bool:
        return error_key in self.focus_set


def build_feedback_preferences(
    *,
    focus_errors: Optional[Iterable[str]] = None,
    mute_errors: Optional[Iterable[str]] = None,
    only_errors: Optional[Iterable[str]] = None,
    only_selected: bool = False,
    focus_boost: float = 4.0,
    training_mode: Optional[str] = None,
) -> FeedbackPreferences:
    return FeedbackPreferences.build(
        focus_errors=focus_errors,
        mute_errors=mute_errors,
        only_errors=only_errors,
        only_selected=only_selected,
        focus_boost=focus_boost,
        training_mode=training_mode,
    )


def feedback_priority(
    error_key: str,
    preferences: Optional[FeedbackPreferences] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> float:
    base_weights = dict(DEFAULT_ERROR_WEIGHTS)
    if weights:
        base_weights.update({key: float(value) for key, value in weights.items()})
    score = float(base_weights.get(error_key, 5.0))
    if preferences and preferences.is_focused(error_key):
        score += preferences.focus_boost
    return score


def filter_posture_errors(
    posture_errors: Iterable[Mapping[str, Any]],
    preferences: Optional[FeedbackPreferences] = None,
) -> List[Dict[str, Any]]:
    """Filter posture errors for presentation without mutating detections."""
    if preferences is None:
        preferences = FeedbackPreferences()

    visible_errors: List[Dict[str, Any]] = []
    for err in posture_errors:
        error_key = str(err.get("error_key") or err.get("error") or "")
        if not error_key or not preferences.allows(error_key):
            continue
        item = dict(err)
        item["feedback_focused"] = preferences.is_focused(error_key)
        item["feedback_priority"] = feedback_priority(error_key, preferences)
        visible_errors.append(item)
    return visible_errors


def sort_posture_errors_for_feedback(
    posture_errors: Iterable[Mapping[str, Any]],
    preferences: Optional[FeedbackPreferences] = None,
) -> List[Dict[str, Any]]:
    """Sort errors so focused/high-priority overlapping alerts win in the UI."""
    return sorted(
        [dict(err) for err in posture_errors],
        key=lambda err: (
            -float(err.get("feedback_priority", feedback_priority(str(err.get("error_key") or err.get("error") or ""), preferences))),
            int(err.get("start_frame") or 0),
            str(err.get("error_key") or err.get("error") or ""),
        ),
    )


@dataclass(frozen=True)
class FeedbackItem:
    """One visual feedback row returned to OpenCV/web UI callers."""

    error_key: str
    message: str
    label: str
    priority: str
    score: float
    base_weight: float
    skipped_count: int
    active_count: int
    spoken_count: int
    cooldown_remaining: float
    triggered: bool
    queued_seconds: float
    focused: bool

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["score"] = round(float(self.score), 2)
        data["base_weight"] = round(float(self.base_weight), 2)
        data["cooldown_remaining"] = round(float(self.cooldown_remaining), 2)
        data["queued_seconds"] = round(float(self.queued_seconds), 2)
        return data


@dataclass(frozen=True)
class FeedbackDecision:
    """Scheduler output for one inference tick."""

    voice_error_key: Optional[str]
    voice_message: Optional[str]
    visual_items: List[FeedbackItem]


@dataclass
class _FeedbackState:
    error_key: str
    base_weight: float
    label: str
    message: str
    skipped_count: int = 0
    active_count: int = 0
    spoken_count: int = 0
    first_pending_at: Optional[float] = None
    last_seen_at: Optional[float] = None
    last_spoken_at: Optional[float] = None


class FeedbackScheduler:
    """Dynamic priority queue with aging for realtime fencing feedback.

    Voice feedback is intentionally narrow: one cue per scheduling decision,
    with per-error and global cooldowns. Visual feedback is broader: callers
    receive the top N current/queued issues sorted by dynamic priority.
    """

    def __init__(
        self,
        *,
        playbook: Optional[Mapping[str, Any]] = None,
        playbook_path: Optional[str | Path] = None,
        weights: Optional[Mapping[str, float]] = None,
        aging_factor: float = 2.0,
        persistence_factor: float = 0.25,
        novelty_bonus: float = 0.75,
        repeat_penalty: float = 1.0,
        voice_cooldown_seconds: float = 4.0,
        global_voice_cooldown_seconds: float = 1.2,
        min_active_count: int = 1,
        visual_top_n: int = 3,
        pending_ttl_seconds: float = 5.0,
        focus_errors: Optional[Iterable[str]] = None,
        mute_errors: Optional[Iterable[str]] = None,
        only_errors: Optional[Iterable[str]] = None,
        only_selected: bool = False,
        focus_boost: float = 4.0,
        training_mode: Optional[str] = None,
    ) -> None:
        self.playbook = dict(playbook) if playbook is not None else self._load_playbook(playbook_path)
        self.weights = dict(DEFAULT_ERROR_WEIGHTS)
        if weights:
            self.weights.update({key: float(value) for key, value in weights.items()})

        self.aging_factor = float(aging_factor)
        self.persistence_factor = float(persistence_factor)
        self.novelty_bonus = float(novelty_bonus)
        self.repeat_penalty = float(repeat_penalty)
        self.voice_cooldown_seconds = float(voice_cooldown_seconds)
        self.global_voice_cooldown_seconds = float(global_voice_cooldown_seconds)
        self.min_active_count = max(1, int(min_active_count))
        self.visual_top_n = max(1, int(visual_top_n))
        self.pending_ttl_seconds = max(0.0, float(pending_ttl_seconds))
        self.preferences = build_feedback_preferences(
            focus_errors=focus_errors,
            mute_errors=mute_errors,
            only_errors=only_errors,
            only_selected=only_selected,
            focus_boost=focus_boost,
            training_mode=training_mode,
        )

        self._states: Dict[str, _FeedbackState] = {}
        self._last_voice_at: Optional[float] = None

    def update(
        self,
        active_error_keys: Iterable[str],
        *,
        now: Optional[float] = None,
    ) -> FeedbackDecision:
        """Update the queue and return the visual list plus optional voice cue."""
        now = time.monotonic() if now is None else float(now)
        active_keys = [
            key for key in normalize_error_keys(active_error_keys)
            if self.preferences.allows(key)
        ]
        active_set = set(active_keys)

        for key in active_keys:
            state = self._state_for(key)
            was_pending = self._is_pending(state, now)
            if not was_pending:
                state.first_pending_at = now
                state.skipped_count = 0
                state.active_count = 0
            state.last_seen_at = now
            state.active_count += 1

        for key, state in self._states.items():
            if key in active_set:
                continue
            if not self._is_pending(state, now):
                state.active_count = 0
                state.skipped_count = 0
                state.first_pending_at = None

        pending_states = [
            state for state in self._states.values()
            if self._is_pending(state, now)
        ]
        scores = {state.error_key: self._score(state) for state in pending_states}
        ranked_states = sorted(
            pending_states,
            key=lambda state: (
                scores[state.error_key],
                state.base_weight,
                state.active_count,
                state.error_key,
            ),
            reverse=True,
        )

        voice_state = self._select_voice_state(ranked_states, scores, now)
        visual_items = [
            self._make_item(
                state,
                score=scores[state.error_key],
                priority="primary" if index == 0 else "secondary",
                triggered=state.error_key in active_set,
                now=now,
            )
            for index, state in enumerate(ranked_states[: self.visual_top_n])
        ]

        voice_error_key = None
        voice_message = None
        if voice_state is not None:
            voice_error_key = voice_state.error_key
            voice_message = voice_state.message
            voice_state.last_spoken_at = now
            voice_state.spoken_count += 1
            voice_state.skipped_count = 0
            self._last_voice_at = now

            for state in pending_states:
                if state.error_key != voice_state.error_key:
                    state.skipped_count += 1

        return FeedbackDecision(
            voice_error_key=voice_error_key,
            voice_message=voice_message,
            visual_items=visual_items,
        )

    def reset(self) -> None:
        self._states.clear()
        self._last_voice_at = None

    def set_preferences(
        self,
        *,
        focus_errors: Optional[Iterable[str]] = None,
        mute_errors: Optional[Iterable[str]] = None,
        only_errors: Optional[Iterable[str]] = None,
        only_selected: bool = False,
        focus_boost: Optional[float] = None,
        training_mode: Optional[str] = None,
        reset_queue: bool = True,
    ) -> None:
        next_training_mode = self.preferences.training_mode if training_mode is None else training_mode
        self.preferences = build_feedback_preferences(
            focus_errors=focus_errors,
            mute_errors=mute_errors,
            only_errors=only_errors,
            only_selected=only_selected,
            focus_boost=self.preferences.focus_boost if focus_boost is None else focus_boost,
            training_mode=next_training_mode,
        )
        self._purge_disabled_states()
        if reset_queue:
            self._last_voice_at = None

    @staticmethod
    def _load_playbook(path: Optional[str | Path]) -> Dict[str, Any]:
        target = Path(path) if path is not None else _DEFAULT_PLAYBOOK_PATH
        if not target.exists():
            return {}
        with open(target, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _state_for(self, error_key: str) -> _FeedbackState:
        state = self._states.get(error_key)
        if state is not None:
            return state

        entry = self.playbook.get(error_key, {})
        label = str(entry.get("error_name") or error_key)
        message = str(entry.get("short_cue") or label or error_key)
        state = _FeedbackState(
            error_key=error_key,
            base_weight=self._base_weight(error_key, entry),
            label=label,
            message=message,
        )
        self._states[error_key] = state
        return state

    def _base_weight(self, error_key: str, entry: Mapping[str, Any]) -> float:
        for field in ("weight", "importance", "severity"):
            value = entry.get(field)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return float(self.weights.get(error_key, 5.0))

    def _is_pending(self, state: _FeedbackState, now: float) -> bool:
        if state.last_seen_at is None:
            return False
        return (now - state.last_seen_at) <= self.pending_ttl_seconds

    def _score(self, state: _FeedbackState) -> float:
        novelty = self.novelty_bonus if state.spoken_count == 0 else 0.0
        repeat_penalty = self.repeat_penalty * min(state.spoken_count, 3)
        persistence = self.persistence_factor * min(state.active_count, 8)
        focus_boost = self.preferences.focus_boost if self.preferences.is_focused(state.error_key) else 0.0
        return (
            state.base_weight
            + self.aging_factor * state.skipped_count
            + persistence
            + novelty
            + focus_boost
            - repeat_penalty
        )

    def _purge_disabled_states(self) -> None:
        self._states = {
            key: state for key, state in self._states.items()
            if self.preferences.allows(key)
        }

    def _cooldown_remaining(self, state: _FeedbackState, now: float) -> float:
        if state.last_spoken_at is None:
            return 0.0
        elapsed = now - state.last_spoken_at
        return max(0.0, self.voice_cooldown_seconds - elapsed)

    def _global_cooldown_remaining(self, now: float) -> float:
        if self._last_voice_at is None:
            return 0.0
        elapsed = now - self._last_voice_at
        return max(0.0, self.global_voice_cooldown_seconds - elapsed)

    def _select_voice_state(
        self,
        ranked_states: List[_FeedbackState],
        scores: Mapping[str, float],
        now: float,
    ) -> Optional[_FeedbackState]:
        if self._global_cooldown_remaining(now) > 0:
            return None

        eligible = [
            state for state in ranked_states
            if state.active_count >= self.min_active_count
            and self._cooldown_remaining(state, now) <= 0
        ]
        if not eligible:
            return None

        return max(
            eligible,
            key=lambda state: (
                scores[state.error_key],
                state.base_weight,
                state.active_count,
                state.error_key,
            ),
        )

    def _make_item(
        self,
        state: _FeedbackState,
        *,
        score: float,
        priority: str,
        triggered: bool,
        now: float,
    ) -> FeedbackItem:
        queued_seconds = 0.0
        if state.first_pending_at is not None:
            queued_seconds = max(0.0, now - state.first_pending_at)
        return FeedbackItem(
            error_key=state.error_key,
            message=state.message,
            label=state.label,
            priority=priority,
            score=score,
            base_weight=state.base_weight,
            skipped_count=state.skipped_count,
            active_count=state.active_count,
            spoken_count=state.spoken_count,
            cooldown_remaining=self._cooldown_remaining(state, now),
            triggered=triggered,
            queued_seconds=queued_seconds,
            focused=self.preferences.is_focused(state.error_key),
        )
