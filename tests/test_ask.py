import sys
from types import SimpleNamespace

import app.ask as ask_module


def test_ask_reports_response_time_and_token_counts(monkeypatch, capsys):
    class FakeModels:
        def generate_content(self, *, model, contents, config):
            assert model == "test-model"
            assert contents == "test question"
            assert config is None
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

    def fake_ask(question, model, temperature=None):
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


def test_main_repeats_questions_with_temperature(monkeypatch, capsys):
    calls = []

    def fake_ask(question, model, temperature):
        calls.append((question, model, temperature))
        return "test answer"

    monkeypatch.setattr(ask_module, "ask", fake_ask)
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.ask", "Same question", "--model", "model-a", "--repeat", "3", "--temperature", "0.8"],
    )

    ask_module.main()

    assert calls == [("Same question", "model-a", 0.8)] * 3
    assert capsys.readouterr().out.count("test answer") == 3


def test_ask_passes_temperature_to_generation_config(monkeypatch):
    captured = {}

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            captured["temperature"] = config.temperature
            return SimpleNamespace(text="test answer", usage_metadata=None)

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: FakeClient())
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")

    assert ask_module.ask("test question", temperature=0.8) == "test answer"
    assert captured["temperature"] == 0.8


def test_interactive_chat_reuses_chat_for_follow_up_questions(monkeypatch, capsys):
    sent_questions = []
    created_chats = []

    class FakeChat:
        def send_message(self, question):
            sent_questions.append(question)
            return SimpleNamespace(
                text=f"answer to {question}",
                usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=7),
            )

    class FakeChats:
        def create(self, *, model, config):
            assert model == "test-model"
            assert config is None
            chat = FakeChat()
            created_chats.append(chat)
            return chat

    class FakeClient:
        chats = FakeChats()

    questions = iter(["First question", "Follow-up", "exit"])
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: FakeClient())
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr("builtins.input", lambda prompt: next(questions))

    ask_module.interactive_chat(model="test-model")

    assert sent_questions == ["First question", "Follow-up"]
    assert len(created_chats) == 1
    output = capsys.readouterr().out
    assert "answer to Follow-up" in output
    assert output.count("[metrics]") == 2


def test_main_routes_interactive_mode(monkeypatch, capsys):
    calls = []

    def fake_interactive_chat(model, temperature):
        calls.append((model, temperature))

    monkeypatch.setattr(ask_module, "interactive_chat", fake_interactive_chat)
    monkeypatch.setattr(sys, "argv", ["app.ask", "--interactive", "--model", "model-a"])

    ask_module.main()

    assert calls == [("model-a", None)]