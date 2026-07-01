from __future__ import annotations

import argparse
from pathlib import Path

from agent.qwen_client import DEFAULT_QWEN_MODEL, QwenPlusClient
from reading_router.final_runner import (
    format_final_summary,
    run_final_router,
)


def main() -> None:
    args = _parse_args()
    qwen_extra_body = {} if args.enable_thinking else {"enable_thinking": False}

    def client_factory() -> QwenPlusClient:
        return QwenPlusClient(
            model=args.model,
            api_key_env=args.api_key_env,
            api_key_env_fallbacks=args.api_key_env_fallbacks,
            base_url_env=args.base_url_env,
            default_base_url=args.default_base_url,
            temperature=args.temperature,
            extra_body=qwen_extra_body,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
        )

    summaries = run_final_router(
        input_path=args.input,
        out_dir=args.out_dir,
        client_factory=client_factory,
        model=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
        qwen_extra_body=qwen_extra_body,
    )
    print(format_final_summary(summaries))
    print(f"outputs: {args.out_dir / 'reading_routes.jsonl'}")
    print(f"tokens:  {args.out_dir / 'route_summary.csv'}")
    print(f"meta:    {args.out_dir / 'run_meta.json'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the retained final reading-router prompts.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/questions.jsonl"),
        help="Question JSON/JSONL with qid, question, answer_format, and options where needed.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/reading_router_final"),
        help="Directory for reading_routes.jsonl, route_summary.csv, and run_meta.json.",
    )
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Do not pass enable_thinking=false. Default keeps thinking disabled for router runs.",
    )
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-key-env-fallbacks", nargs="*", default=["QWEN_API_KEY"])
    parser.add_argument("--base-url-env", default="QWEN_BASE_URL")
    parser.add_argument(
        "--default-base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
