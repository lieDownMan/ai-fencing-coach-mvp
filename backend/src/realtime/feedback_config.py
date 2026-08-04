"""Shared feedback configuration for realtime and post-session coaching.

The scheduler, CLI, Gradio app, and browser realtime UI all use this module so
error weights and mode availability do not drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


TRAINING_MODES: tuple[str, ...] = ("Footwork", "Target Practice", "Free Bouting")


@dataclass(frozen=True)
class FeedbackErrorConfig:
    """UI/scheduler metadata for one coaching error key."""

    weight: float
    modes: tuple[str, ...] = TRAINING_MODES



# Add or tune feedback errors here. The scheduler, CLI, Gradio app, and web UI
# derive their weights and supported-mode lists from this single registry.
FEEDBACK_ERROR_CONFIG: Dict[str, FeedbackErrorConfig] = {
    "foot_before_hand": FeedbackErrorConfig(5.0, ("Target Practice",)),
    "lunge_overextension": FeedbackErrorConfig(9.5, ("Target Practice",)),
    "incomplete_arm_extension": FeedbackErrorConfig(9.0, ("Target Practice",)),
    "guard_dropped": FeedbackErrorConfig(9.7),
    "stance_too_high": FeedbackErrorConfig(10.0),
    "bounce_excessive": FeedbackErrorConfig(6.5),
    "center_of_mass_in_front": FeedbackErrorConfig(6.0),
    "center_of_mass_leaning_backward": FeedbackErrorConfig(6.0),
    "over_parrying": FeedbackErrorConfig(5.0),
    "wide_step": FeedbackErrorConfig(4.0),
    "narrow_step": FeedbackErrorConfig(9.0),
}

DEFAULT_ERROR_WEIGHTS: Dict[str, float] = {
    key: config.weight
    for key, config in FEEDBACK_ERROR_CONFIG.items()
}

# Mode-level availability mirrors HeuristicsEngine._check_rules. This is less
# strict than action-level gating because the UI cannot know the next action.
ERROR_AVAILABILITY_BY_MODE: Dict[str, set[str]] = {
    mode: {
        key for key, config in FEEDBACK_ERROR_CONFIG.items()
        if mode in config.modes and not config.future
    }
    for mode in TRAINING_MODES
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
) -> List[str]:
    """Return error keys that can be emitted in the selected training mode."""
    if training_mode in TRAINING_MODES:
        keys = {
            key for key, config in FEEDBACK_ERROR_CONFIG.items()
            if str(training_mode) in config.modes
        }
    else:
        keys = set(FEEDBACK_ERROR_CONFIG.keys())

    return sorted(keys)


def supported_modes_for_error_key(
    error_key: str,
) -> List[str]:
    """Return training modes where an error can be configured."""
    config = FEEDBACK_ERROR_CONFIG.get(error_key)
    if config is None:
        return []
    return list(config.modes)


def is_error_available_for_mode(
    error_key: str,
    training_mode: Optional[str] = None,
) -> bool:
    return error_key in set(
        available_error_keys_for_mode(training_mode)
    )


def filter_error_keys_for_mode(
    keys: Optional[Iterable[str]],
    training_mode: Optional[str] = None,
) -> tuple[List[str], List[str]]:
    """Split requested error keys into allowed and rejected lists."""
    normalized = normalize_error_keys(keys)
    allowed_set = set(
        available_error_keys_for_mode(training_mode)
    )
    allowed = [key for key in normalized if key in allowed_set]
    rejected = [key for key in normalized if key not in allowed_set]
    return allowed, rejected
