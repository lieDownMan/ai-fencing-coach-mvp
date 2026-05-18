from src.realtime.feedback_scheduler import (
    FeedbackScheduler,
    available_error_keys_for_mode,
    build_feedback_preferences,
)


PLAYBOOK = {
    "major": {"short_cue": "fix major", "error_name": "Major"},
    "minor": {"short_cue": "fix minor", "error_name": "Minor"},
    "guard_dropped": {"short_cue": "guard up", "error_name": "Guard"},
}


def _scheduler(**kwargs):
    defaults = {
        "playbook": PLAYBOOK,
        "weights": {"major": 10.0, "minor": 5.0, "guard_dropped": 7.0},
        "voice_cooldown_seconds": 0.0,
        "global_voice_cooldown_seconds": 0.0,
        "persistence_factor": 0.0,
        "novelty_bonus": 0.0,
        "repeat_penalty": 0.0,
    }
    defaults.update(kwargs)
    return FeedbackScheduler(**defaults)


def test_highest_weight_is_spoken_and_primary_visual():
    scheduler = _scheduler()

    decision = scheduler.update(["minor", "major"], now=0.0)

    assert decision.voice_error_key == "major"
    assert decision.voice_message == "fix major"
    assert decision.visual_items[0].error_key == "major"
    assert [item.error_key for item in decision.visual_items] == ["major", "minor"]


def test_aging_eventually_prevents_starvation():
    scheduler = _scheduler(aging_factor=3.0)

    assert scheduler.update(["major", "minor"], now=0.0).voice_error_key == "major"
    assert scheduler.update(["major", "minor"], now=1.0).voice_error_key == "major"

    decision = scheduler.update(["major", "minor"], now=2.0)

    assert decision.voice_error_key == "minor"


def test_same_error_cooldown_blocks_repeat_voice_but_keeps_visual():
    scheduler = _scheduler(voice_cooldown_seconds=10.0)

    assert scheduler.update(["guard_dropped"], now=0.0).voice_error_key == "guard_dropped"
    decision = scheduler.update(["guard_dropped"], now=1.0)

    assert decision.voice_error_key is None
    assert decision.visual_items[0].error_key == "guard_dropped"
    assert decision.visual_items[0].cooldown_remaining == 9.0


def test_visual_feedback_can_show_multiple_ranked_items():
    scheduler = _scheduler(visual_top_n=2)

    decision = scheduler.update(["minor", "major", "guard_dropped"], now=0.0)

    assert len(decision.visual_items) == 2
    assert decision.visual_items[0].priority == "primary"
    assert decision.visual_items[1].priority == "secondary"


def test_focus_errors_get_priority_boost():
    scheduler = _scheduler(focus_errors=["minor"], focus_boost=6.0)

    decision = scheduler.update(["major", "minor"], now=0.0)

    assert decision.voice_error_key == "minor"
    assert decision.visual_items[0].focused is True


def test_muted_errors_are_hidden_and_silent():
    scheduler = _scheduler(mute_errors=["major"])

    decision = scheduler.update(["major", "minor"], now=0.0)

    assert decision.voice_error_key == "minor"
    assert [item.error_key for item in decision.visual_items] == ["minor"]


def test_only_selected_limits_feedback_to_focus_list():
    scheduler = _scheduler(focus_errors=["minor"], only_selected=True)

    decision = scheduler.update(["major", "minor"], now=0.0)

    assert decision.voice_error_key == "minor"
    assert [item.error_key for item in decision.visual_items] == ["minor"]


def test_mode_availability_excludes_unimplemented_future_errors():
    footwork_errors = available_error_keys_for_mode("Footwork")

    assert "stance_too_high" in footwork_errors
    assert "wide_disengage" not in footwork_errors


def test_mode_preferences_reject_unavailable_errors():
    preferences = build_feedback_preferences(
        focus_errors=["lunge_overextension", "stance_too_high"],
        mute_errors=["foot_before_hand"],
        training_mode="Footwork",
    )

    assert preferences.focus_errors == ("stance_too_high",)
    assert preferences.mute_errors == ()
    assert preferences.allows("stance_too_high") is True
    assert preferences.allows("lunge_overextension") is False


def test_scheduler_filters_active_errors_by_training_mode():
    scheduler = _scheduler(training_mode="Footwork")

    decision = scheduler.update(["lunge_overextension", "stance_too_high"], now=0.0)

    assert decision.voice_error_key == "stance_too_high"
    assert [item.error_key for item in decision.visual_items] == ["stance_too_high"]
