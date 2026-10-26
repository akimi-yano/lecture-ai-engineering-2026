"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --model gemini-3.5-flash --model gemini-3.1-flash-lite "LoRAとは？" "RAGとは？"

既定ではツールも会話履歴も持たない「素の」呼び出し。
--interactive を指定すると、会話履歴を保って続けて質問できる。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai import errors
from google.genai import types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, model: str = MODEL, temperature: float | None = None) -> str:
    client = genai.Client(api_key=get_api_key())
    started_at = time.perf_counter()
    config = types.GenerateContentConfig(temperature=temperature) if temperature is not None else None
    response = client.models.generate_content(model=model, contents=question, config=config)
    elapsed_seconds = time.perf_counter() - started_at

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage is not None else "unknown"
    output_tokens = usage.candidates_token_count if usage is not None else "unknown"
    print(
        f"[metrics] model={model} response_seconds={elapsed_seconds:.2f} "
        f"input_tokens={input_tokens} output_tokens={output_tokens}"
    )

    return response.text


def interactive_chat(model: str = MODEL, temperature: float | None = None) -> None:
    client = genai.Client(api_key=get_api_key())
    config = types.GenerateContentConfig(temperature=temperature) if temperature is not None else None
    chat = client.chats.create(model=model, config=config)
    print("会話を開始しました。終了するには exit と入力してください。")

    while True:
        try:
            question = input("質問> ").strip()
        except EOFError:
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        started_at = time.perf_counter()
        response = chat.send_message(question)
        elapsed_seconds = time.perf_counter() - started_at
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage is not None else "unknown"
        output_tokens = usage.candidates_token_count if usage is not None else "unknown"
        print(
            f"[metrics] model={model} response_seconds={elapsed_seconds:.2f} "
            f"input_tokens={input_tokens} output_tokens={output_tokens}"
        )
        print(response.text)


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する（応答時間とトークン数を計測）")
    parser.add_argument("questions", nargs="*", help="質問。複数指定すると順に実行する")
    parser.add_argument("--interactive", action="store_true", help="会話履歴を保って対話する")
    parser.add_argument("--repeat", type=int, default=1, help="各質問・各モデルでの実行回数")
    parser.add_argument("--temperature", type=float, help="生成時の temperature（0〜2）")
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="使用モデル。複数回指定すると同じ質問を各モデルで実行する",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat は1以上を指定してください")
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        parser.error("--temperature は0〜2を指定してください")

    questions = args.questions or [DEFAULT_QUESTION]
    models = args.models or [MODEL]
    try:
        if args.interactive:
            if args.questions:
                parser.error("--interactive では質問を引数に指定せず、対話中に入力してください")
            for model in models:
                print(f"\n[model] {model}")
                interactive_chat(model=model, temperature=args.temperature)
            return

        for question in questions:
            for model in models:
                for run in range(1, args.repeat + 1):
                    print(f"\n[質問] {question} (model={model}, run={run}/{args.repeat})")
                    print(ask(question, model=model, temperature=args.temperature))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
