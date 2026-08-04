import json
from pathlib import Path

from inference.heuristics_engine import (
    FUTURE_HEURISTIC_ERROR_KEYS,
    HEURISTIC_ERROR_KEYS,
)
from src.realtime.feedback_config import (
    DEFAULT_ERROR_WEIGHTS,
    ERROR_AVAILABILITY_BY_MODE,
    FEEDBACK_ERROR_CONFIG,
    FUTURE_ERROR_KEYS,
    TRAINING_MODES,
    available_error_keys_for_mode,
    supported_modes_for_error_key,
)


def test_feedback_registry_derives_weights_and_future_keys():
    assert DEFAULT_ERROR_WEIGHTS == {
        key: config.weight
        for key, config in FEEDBACK_ERROR_CONFIG.items()
    }
    assert FUTURE_ERROR_KEYS == {
        key for key, config in FEEDBACK_ERROR_CONFIG.items()
        if config.future
    }


def test_feedback_registry_matches_playbook_keys():
    playbook_path = Path(__file__).resolve().parents[1] / "backend" / "coach_playbook.json"
    if not playbook_path.exists():
        playbook_path = Path(__file__).resolve().parents[1] / "coach_playbook.json"
    playbook_keys = set(json.loads(playbook_path.read_text(encoding="utf-8")))

    assert set(FEEDBACK_ERROR_CONFIG) == playbook_keys


def test_feedback_registry_covers_implemented_and_future_heuristics():
    assert set(HEURISTIC_ERROR_KEYS).issubset(FEEDBACK_ERROR_CONFIG)
    assert set(FUTURE_HEURISTIC_ERROR_KEYS) == FUTURE_ERROR_KEYS


def test_mode_availability_is_derived_from_registry():
    for mode in TRAINING_MODES:
        assert ERROR_AVAILABILITY_BY_MODE[mode] == {
            key for key, config in FEEDBACK_ERROR_CONFIG.items()
            if mode in config.modes and not config.future
        }

    assert supported_modes_for_error_key("lunge_overextension") == ["Target Practice"]
    assert supported_modes_for_error_key("stance_too_high") == list(TRAINING_MODES)
