from __future__ import annotations

import argparse
from pathlib import Path

from agent.qwen_client import QwenPlusClient
from reading_router.experiments.experiment import format_token_summary, run_comparison
from reading_router.experiments.routes import READING_MODEL_PRESETS, ModelRoute, resolve_routes


def main() -> None:
    args = _parse_args()
    routes = resolve_routes(preset=args.preset, models=args.models)

    def client_factory(route: ModelRoute) -> QwenPlusClient:
        return QwenPlusClient(
            model=route.model,
            api_key_env=args.api_key_env,
            api_key_env_fallbacks=args.api_key_env_fallbacks,
            base_url_env=args.base_url_env,
            default_base_url=args.default_base_url,
            temperature=args.temperature,
            extra_body=route.extra_body,
            max_tokens=route.max_tokens if route.max_tokens is not None else args.max_tokens,
            timeout_seconds=args.timeout_seconds,
        )

    summaries = run_comparison(
        input_path=args.input,
        out_dir=args.out_dir,
        routes=routes,
        client_factory=client_factory,
        batch_size=args.batch_size,
        limit=args.limit,
        prompt_style=args.prompt_style,
        output_style=args.output_style,
    )
    print(format_token_summary(summaries))
    print(f"outputs: {args.out_dir / 'reading_chains.jsonl'}")
    print(f"tokens:  {args.out_dir / 'token_summary.csv'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run question-reading chain model comparisons.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("eval/fake_gold/gptpro_reading_trace_request.json"),
        help="Question JSON/JSONL. Supports a top-level questions/items array or JSONL rows.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/reading_router"),
        help="Directory for reading_chains.jsonl and token_summary.csv.",
    )
    parser.add_argument(
        "--preset",
        default="reading_core",
        choices=sorted(READING_MODEL_PRESETS),
        help="Named model comparison preset. Ignored when --models is provided.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated manual model specs. Use model or route_name=model. Overrides --preset.",
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt-style", choices=["full", "compact", "compact_rel", "reasoning_md"], default="full")
    parser.add_argument("--output-style", choices=["full", "compact"], default="full")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
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
