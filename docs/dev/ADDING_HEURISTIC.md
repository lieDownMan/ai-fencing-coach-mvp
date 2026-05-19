# Adding or Tuning Heuristics

Use this checklist when changing coaching errors.

## Tune Feedback Weight

Edit one file:

```python
src/realtime/feedback_config.py
```

Change the `weight` in `FEEDBACK_ERROR_CONFIG`. Higher weight means the error
starts higher in the voice/text feedback queue. The realtime scheduler still
adds aging, persistence, focus boost, and cooldown effects on top.

## Change a Heuristic Threshold

Edit the named constants near the top of:

```python
inference/heuristics_engine.py
```

The heuristic debug tools import those same constants, so the visualizer
threshold labels and metric-trigger checks stay aligned with runtime behavior.

## Add a New Heuristic

1. Add the error key and default feedback metadata in `src/realtime/feedback_config.py`.
2. Add the coaching copy in `coach_playbook.json`.
3. Add the error key to `HEURISTIC_ERROR_KEYS` in `inference/heuristics_engine.py`.
4. Implement a `_check_*` method in `HeuristicsEngine` and call it from `_check_rules`.
5. Add a matching metric function in `inference/heuristic_debug.py` so clip and realtime visualizers can show raw values.
6. Add or update `HEURISTIC_NOTES` in `heuristic_visualizer.py`.
7. Add tests in `tests/test_feedback_config.py` and a focused heuristic test.

If the playbook entry exists but detection is not implemented yet, mark it as
`future=True` in `FEEDBACK_ERROR_CONFIG` and add it to
`FUTURE_HEURISTIC_ERROR_KEYS`. Future errors stay hidden from mode-aware UI
choices unless code explicitly asks for future entries.
