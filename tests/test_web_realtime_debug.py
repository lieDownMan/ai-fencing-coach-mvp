import json

import web_realtime


def test_status_includes_debug_fields_before_pipeline_initializes():
    response = web_realtime.status()
    payload = json.loads(response.body)

    assert payload["debug_enabled"] is True
    assert payload["debug_window_size"] == web_realtime.DEBUG_WINDOW_SIZE
    assert payload["heuristic_metrics"] == []
    assert payload["target_lock"]["mode"] == "none"


def test_heuristic_options_include_all_and_specific_heuristics():
    html = web_realtime._heuristic_options_html()

    assert 'value="all"' in html
    assert 'value="stance_too_high"' in html
