from types import SimpleNamespace

from llm_agent import LLMAgent


def _sample_actions():
    return [
        {"action": "R"},
        {"action": "SF"},
        {"action": "SB"},
    ]


def _sample_errors():
    return [
        {"error_key": "guard_dropped"},
        {"error_key": "guard_dropped"},
        {"error_key": "foot_before_hand"},
    ]


def test_summary_without_llm_uses_playbook_counts(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    agent = LLMAgent()

    summary = agent.generate_summary(
        {"handedness": "right", "height_cm": 180},
        "Free Bouting",
        _sample_actions(),
        _sample_errors(),
    )
    aggregated = agent._aggregate_playbook_errors(_sample_errors())

    assert "LLM Agent disabled" not in summary
    assert aggregated[0]["key"] == "guard_dropped"
    assert aggregated[0]["count"] == 2
    assert "practice" in aggregated[0]
    assert aggregated[1]["key"] == "foot_before_hand"
    assert aggregated[1]["count"] == 1
    assert "Free Bouting" in summary
    assert "2" in summary
    assert "1" in summary


def test_summary_prompt_with_llm_includes_playbook_context(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    captured = {}

    class FakeModel:
        def generate_content(self, prompt):
            captured["prompt"] = prompt
            return SimpleNamespace(text="LLM summary")

    agent = LLMAgent()
    agent.enabled = True
    agent.model = FakeModel()

    summary = agent.generate_summary(
        {"handedness": "left", "height_cm": 170},
        "Target Practice",
        _sample_actions(),
        _sample_errors(),
    )

    assert summary == "LLM summary"
    assert "[COACH PLAYBOOK CONTEXT]" in captured["prompt"]
    assert "error_key: guard_dropped" in captured["prompt"]
    assert "frequency: 2" in captured["prompt"]
    assert "short_cue:" in captured["prompt"]
    assert "practice:" in captured["prompt"]
    assert "practice recommendation" in captured["prompt"]
    assert "List every detected problem" in captured["prompt"]


def test_summary_can_force_playbook_when_llm_is_available(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    class FakeModel:
        def generate_content(self, prompt):
            raise AssertionError("Gemini should not be called")

    agent = LLMAgent()
    agent.enabled = True
    agent.model = FakeModel()

    summary = agent.generate_summary(
        {"handedness": "left", "height_cm": 170},
        "Target Practice",
        _sample_actions(),
        _sample_errors(),
        use_llm=False,
    )

    assert "Gemini summary failed" not in summary
    assert "Target Practice" in summary
    assert "2" in summary


def test_summary_aggregation_prioritizes_focus_errors(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    agent = LLMAgent()

    aggregated = agent._aggregate_playbook_errors(
        _sample_errors(),
        focus_errors=["foot_before_hand"],
    )

    assert aggregated[0]["key"] == "foot_before_hand"
    assert aggregated[0]["focused"] is True
