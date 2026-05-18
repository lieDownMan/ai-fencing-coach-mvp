"""Shared feedback configuration for realtime and post-session coaching.

The scheduler, CLI, Gradio app, and browser realtime UI all use this module so
error weights and mode availability do not drift apart.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional


DEFAULT_ERROR_WEIGHTS: Dict[str, float] = {
    "foot_before_hand": 5.0,
    "lunge_overextension": 9.5,
    "incomplete_arm_extension": 9.0,
    "guard_dropped": 9.7,
    "stance_too_high": 10.0,
    "bounce_excessive": 6.5,
    "center_of_mass_in_front": 6.0,
    "center_of_mass_leaning_backward": 6.0,
    "over_parrying": 5.0,
    "wide_step": 4.0,
    "narrow_step": 9.0,
    "wide_disengage": 4.0,
}

TRAINING_MODES: tuple[str, ...] = ("Footwork", "Target Practice", "Free Bouting")

# Present in the playbook/weights but not emitted by the current heuristic engine.
FUTURE_ERROR_KEYS: set[str] = {"wide_disengage"}

# Mode-level availability mirrors HeuristicsEngine._check_rules. This is less
# strict than action-level gating because the UI cannot know the next action.
ERROR_AVAILABILITY_BY_MODE: Dict[str, set[str]] = {
    "Footwork": {
        "guard_dropped",
        "bounce_excessive",
        "stance_too_high",
        "wide_step",
        "narrow_step",
        "center_of_mass_in_front",
        "center_of_mass_leaning_backward",
        "over_parrying",
    },
    "Target Practice": {
        "guard_dropped",
        "lunge_overextension",
        "foot_before_hand",
        "incomplete_arm_extension",
        "bounce_excessive",
        "stance_too_high",
        "wide_step",
        "narrow_step",
        "center_of_mass_in_front",
        "center_of_mass_leaning_backward",
        "over_parrying",
    },
    "Free Bouting": {
        "guard_dropped",
        "bounce_excessive",
        "stance_too_high",
        "wide_step",
        "narrow_step",
        "center_of_mass_in_front",
        "center_of_mass_leaning_backward",
        "over_parrying",
    },
}


def normalize_error_keys(keys: Optional[Iterable[str]]) -> List[str]:
    """Return unique, non-empty error keys while preserving input order."""
    unique: List[str] = []
    seen = set()
    for key in keys or []:
        if not key:
            continue
        clean_key = str(key).strip()
        if not clean_key or clean_key in seen:
            continue
        seen.add(clean_key)
        unique.append(clean_key)
    return unique


def available_error_keys_for_mode(
    training_mode: Optional[str] = None,
    *,
    include_future: bool = False,
) -> List[str]:
    """Return error keys that can be emitted in the selected training mode."""
    if training_mode in ERROR_AVAILABILITY_BY_MODE:
        keys = set(ERROR_AVAILABILITY_BY_MODE[str(training_mode)])
    else:
        keys = set(DEFAULT_ERROR_WEIGHTS.keys())

    if include_future:
        keys.update(FUTURE_ERROR_KEYS)
    else:
        keys.difference_update(FUTURE_ERROR_KEYS)

    return sorted(key for key in keys if key in DEFAULT_ERROR_WEIGHTS)


def supported_modes_for_error_key(
    error_key: str,
    *,
    include_future: bool = False,
) -> List[str]:
    """Return training modes where an error can be configured."""
    modes = [
        mode for mode in TRAINING_MODES
        if error_key in available_error_keys_for_mode(mode, include_future=include_future)
    ]
    if include_future and error_key in FUTURE_ERROR_KEYS:
        return list(TRAINING_MODES)
    return modes


def is_error_available_for_mode(
    error_key: str,
    training_mode: Optional[str] = None,
    *,
    include_future: bool = False,
) -> bool:
    return error_key in set(
        available_error_keys_for_mode(training_mode, include_future=include_future)
    )


def filter_error_keys_for_mode(
    keys: Optional[Iterable[str]],
    training_mode: Optional[str] = None,
    *,
    include_future: bool = False,
) -> tuple[List[str], List[str]]:
    """Split requested error keys into allowed and rejected lists."""
    normalized = normalize_error_keys(keys)
    allowed_set = set(
        available_error_keys_for_mode(training_mode, include_future=include_future)
    )
    allowed = [key for key in normalized if key in allowed_set]
    rejected = [key for key in normalized if key not in allowed_set]
    return allowed, rejected
