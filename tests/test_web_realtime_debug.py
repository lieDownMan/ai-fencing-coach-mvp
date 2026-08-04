import json

import web_realtime


def test_status_includes_debug_fields_before_pipeline_initializes():
    response = web_realtime.status()
    payload = json.loads(response.body)

    assert payload["debug_enabled"] is True
    assert payload["debug_window_size"] == web_realtime.DEBUG_WINDOW_SIZE
    assert payload["heuristic_metrics"] == []
    assert payload["feedback_items"] == []
    assert payload["feedback_preferences"]["focus_errors"] == []
    assert payload["target_lock"]["mode"] == "none"


def test_heuristic_options_include_all_and_specific_heuristics():
    html = web_realtime._heuristic_options_html()

    assert 'value="all"' in html
    assert 'value="stance_too_high"' in html


def test_feedback_error_options_include_known_errors():
    html = web_realtime._feedback_error_checkboxes_html("focus_errors", ["stance_too_high"])

    assert 'name="focus_errors"' in html
    assert 'value="stance_too_high" checked' in html
    assert 'value="guard_dropped"' in html


def test_feedback_error_options_include_mode_metadata():
    html = web_realtime._feedback_error_checkboxes_html("focus_errors", [])

    assert 'data-modes="Footwork|Target Practice|Free Bouting"' in html
