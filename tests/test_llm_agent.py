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

    assert "LLM Agent disabled" not in summary
    assert "持劍手掉落：2 次" in summary
    assert "手腳順序錯誤：1 次" in summary
    assert "手抬起來，劍尖指著對手" in summary
    assert "手要先伸" in summary


def test_summary_prompt_with_llm_includes_playbook_context(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    captured = {}

    class FakeModels:
        def generate_content(self, model, contents):
            captured["model"] = model
            captured["contents"] = contents
            return SimpleNamespace(text="LLM summary")

    agent = LLMAgent()
    agent.enabled = True
    agent.model_name = "test-model"
    agent.client = SimpleNamespace(models=FakeModels())

    summary = agent.generate_summary(
        {"handedness": "left", "height_cm": 170},
        "Target Practice",
        _sample_actions(),
        _sample_errors(),
    )

    assert summary == "LLM summary"
    assert captured["model"] == "test-model"
    assert "[COACH PLAYBOOK CONTEXT]" in captured["contents"]
    assert "frequency: 2" in captured["contents"]
    assert "problem: 持劍手掉落" in captured["contents"]
    assert "short_cue: 手抬起來，劍尖指著對手" in captured["contents"]
    assert "List every detected problem" in captured["contents"]


def test_summary_can_force_playbook_when_llm_is_available(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    class FakeModels:
        def generate_content(self, model, contents):
            raise AssertionError("Gemini should not be called")

    agent = LLMAgent()
    agent.enabled = True
    agent.client = SimpleNamespace(models=FakeModels())

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
