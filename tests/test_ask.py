import sys
from types import SimpleNamespace

import app.ask as ask_module


def test_ask_reports_response_time_and_token_counts(monkeypatch, capsys):
    class FakeModels:
        def generate_content(self, *, model, contents):
            assert model == "test-model"
            assert contents == "test question"
            return SimpleNamespace(
                text="test answer",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=12,
                    candidates_token_count=34,
                ),
            )

    class FakeClient:
        models = FakeModels()

    timestamps = iter([10.0, 11.25])
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: FakeClient())
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.time, "perf_counter", lambda: next(timestamps))

    assert ask_module.ask("test question", model="test-model") == "test answer"
    output = capsys.readouterr().out
    assert "response_seconds=1.25" in output
    assert "input_tokens=12 output_tokens=34" in output


def test_main_runs_each_question_with_each_model(monkeypatch, capsys):
    calls = []

    def fake_ask(question, model):
        calls.append((question, model))
        return "test answer"

    monkeypatch.setattr(ask_module, "ask", fake_ask)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "app.ask",
            "Question one",
            "Question two",
            "--model",
            "model-a",
            "--model",
            "model-b",
        ],
    )

    ask_module.main()

    assert calls == [
        ("Question one", "model-a"),
        ("Question one", "model-b"),
        ("Question two", "model-a"),
        ("Question two", "model-b"),
    ]
    assert capsys.readouterr().out.count("test answer") == 4