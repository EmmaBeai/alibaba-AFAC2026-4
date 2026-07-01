from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agent.qwen_client import DEFAULT_QWEN_MODEL, QwenPlusClient
from reading_router.experiments.prompts import COMPACT_READING_SYSTEM_PROMPT, build_reasoning_md_reading_prompt


NON_YEAR_NUMBER_RE = re.compile(r"(?<!\d)(?!(?:19|20)\d{2}(?!\d))\d+(?:\.\d+)?%?")


def is_mcq_question(question: Mapping[str, Any]) -> bool:
    return str(question.get("answer_format", "")).lower() == "mcq"


def has_non_year_number_in_q_options(question: Mapping[str, Any]) -> bool:
    return bool(NON_YEAR_NUMBER_RE.search(_question_and_options_text(question)))


def select_mcq_questions(
    questions: Sequence[Mapping[str, Any]],
    *,
    exclude_numeric_q_options: bool = False,
) -> list[Mapping[str, Any]]:
    selected = [question for question in questions if is_mcq_question(question)]
    if exclude_numeric_q_options:
        return [question for question in selected if not has_non_year_number_in_q_options(question)]
    return selected


def build_mcq_routes_prompt(questions: Sequence[Mapping[str, Any]]) -> str:
    payload = [_mcq_input(question) for question in questions]
    return (
        "你是single-choice evidence router。输入qid、q、options。\n"
        "只做证据路由，不答题，不判断A/B/C/D。目标是为“唯一正确项”找到最少裁决证据。\n\n"
        "输出JSON数组:\n"
        '[{"qid":"...","decision":"...","routes":["A ...","B ...","C ...","D ..."],"risk":"..."}]\n\n'
        "字段:\n"
        "decision: 本题唯一正确项的裁决轴；如果四个选项是不同事实，写“分别核对四项事实，唯一完全准确项为准”并点出口径。\n"
        "routes: 每个选项一个最少证据定位短语，含对象锚点+待核对事实+关键限定词。\n"
        "risk: 最容易误选的一点，如部分正确当全对、跨文档套用、期限/单位/金额/比例/例外条件混淆。\n\n"
        "要求:\n"
        "route写短语，不写完整解释。\n"
        "必须原样保留选项里的数字、比例、期限、比较方向、否定和全称限定词，如高于/低于/超过/不足/未/不/均/所有/至少。\n"
        "不要把具体裁决点改写成泛泛的“情况/对比/规定”。\n"
        "禁补题外事实。\n"
        f"Q={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def build_general_reasoning_placeholder(questions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "prompts": {"reasoning_md": build_reasoning_md_reading_prompt(questions)},
        "usage": {},
        "raw_content": {},
        "parsed": {},
    }


def run_mcq_routes(questions: Sequence[Mapping[str, Any]], *, client: Any) -> dict[str, Any]:
    prompt = build_mcq_routes_prompt(questions)
    completion = client.raw_completion(system="单选题证据路由助手。只输出JSON。", prompt=prompt)
    return {
        "prompts": {"mcq_routes": prompt},
        "usage": {"mcq_routes": _completion_usage(completion)},
        "raw_content": {"mcq_routes": completion.raw_content},
        "parsed": {"mcq_routes": _completion_items_or_empty(completion)},
    }


def run_general_reasoning(questions: Sequence[Mapping[str, Any]], *, client: Any) -> dict[str, Any]:
    prompt = build_reasoning_md_reading_prompt(questions)
    completion = client.raw_completion(system=COMPACT_READING_SYSTEM_PROMPT, prompt=prompt)
    return {
        "prompts": {"reasoning_md": prompt},
        "usage": {"reasoning_md": _completion_usage(completion)},
        "raw_content": {"reasoning_md": completion.raw_content},
        "parsed": {"reasoning_md": _completion_items_or_empty(completion)},
    }


def build_mcq_artifact(
    questions: Sequence[Mapping[str, Any]],
    *,
    model: str | None = None,
    qwen_extra_body: Mapping[str, Any] | None = None,
    numeric_q_options_qids: Sequence[str] = (),
    mcq_after_numeric_qids: Sequence[str] = (),
    mcq_routes: Mapping[str, Any] | None = None,
    reasoning_baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "mcq_question_count": len(questions),
        "mcq_qids": [str(question.get("qid", "")) for question in questions],
        "numeric_q_options_qids": list(numeric_q_options_qids),
        "mcq_after_numeric_qids": list(mcq_after_numeric_qids),
    }
    if model:
        artifact["model"] = model
    if qwen_extra_body:
        artifact["qwen_extra_body"] = dict(qwen_extra_body)
    if mcq_routes:
        artifact["mcq_routes"] = _stage_artifact(mcq_routes)
    if reasoning_baseline:
        artifact["reasoning_baseline"] = _stage_artifact(reasoning_baseline)
    return artifact


def load_questions(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("questions") or payload.get("items") or []
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError(f"Question payload must be a list in {path}")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def write_mcq_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_qwen_extra_body(*, disable_thinking: bool) -> dict[str, Any]:
    if disable_thinking:
        return {"enable_thinking": False}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare single-choice evidence routing against fixed reasoning.")
    parser.add_argument("--input", type=Path, default=Path("data/questions.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/reading_router_mcq_prompt/mcq_compare.json"))
    parser.add_argument("--run-max", action="store_true", help="Call qwen max and print returned token usage.")
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--disable-thinking", action="store_true", help="Pass enable_thinking=false to Qwen.")
    parser.add_argument(
        "--exclude-numeric-q-options",
        action="store_true",
        help="Simulate numeric-first routing by excluding mcq questions with non-year digits in q+options.",
    )
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-key-env-fallbacks", nargs="*", default=["QWEN_API_KEY"])
    parser.add_argument("--base-url-env", default="QWEN_BASE_URL")
    parser.add_argument(
        "--default-base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    args = parser.parse_args()

    all_questions = load_questions(args.input)
    all_mcq = select_mcq_questions(all_questions)
    numeric_q_options_qids = [
        str(question.get("qid", ""))
        for question in all_mcq
        if has_non_year_number_in_q_options(question)
    ]
    mcq_after_numeric_qids = [
        str(question.get("qid", ""))
        for question in all_mcq
        if not has_non_year_number_in_q_options(question)
    ]
    selected = select_mcq_questions(
        all_questions,
        exclude_numeric_q_options=args.exclude_numeric_q_options,
    )
    qwen_extra_body = build_qwen_extra_body(disable_thinking=args.disable_thinking)
    if args.run_max:
        client = QwenPlusClient(
            model=args.model,
            api_key_env=args.api_key_env,
            api_key_env_fallbacks=args.api_key_env_fallbacks,
            base_url_env=args.base_url_env,
            default_base_url=args.default_base_url,
            temperature=args.temperature,
            extra_body=qwen_extra_body,
            timeout_seconds=args.timeout_seconds,
        )
        mcq_routes = run_mcq_routes(selected, client=client)
        reasoning_baseline = run_general_reasoning(selected, client=client)
    else:
        mcq_routes = {
            "prompts": {"mcq_routes": build_mcq_routes_prompt(selected)},
            "usage": {},
            "raw_content": {},
            "parsed": {},
        }
        reasoning_baseline = build_general_reasoning_placeholder(selected)

    artifact = build_mcq_artifact(
        selected,
        model=args.model if args.run_max else None,
        qwen_extra_body=qwen_extra_body,
        numeric_q_options_qids=numeric_q_options_qids,
        mcq_after_numeric_qids=mcq_after_numeric_qids,
        mcq_routes=mcq_routes,
        reasoning_baseline=reasoning_baseline,
    )
    write_mcq_artifact(args.out, artifact)
    summary: dict[str, Any] = {
        "out": str(args.out),
        "mcq_question_count": len(selected),
        "mcq_qids": artifact["mcq_qids"],
        "numeric_q_options_qids": artifact["numeric_q_options_qids"],
        "mcq_after_numeric_qids": artifact["mcq_after_numeric_qids"],
    }
    if args.run_max:
        summary.update(
            {
                "model": args.model,
                "qwen_extra_body": qwen_extra_body,
                "mcq_routes_usage": artifact["mcq_routes"]["usage"],
                "reasoning_baseline_usage": artifact["reasoning_baseline"]["usage"],
                "usage_totals": {
                    "mcq_routes": _total_usage(artifact["mcq_routes"]["usage"]),
                    "reasoning_baseline": _total_usage(artifact["reasoning_baseline"]["usage"]),
                },
            }
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2 if args.run_max else None))


def _mcq_input(question: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "qid": str(question.get("qid", "")),
        "q": str(question.get("question", "")),
        "options": _compact_options(question.get("options")),
    }


def _compact_options(options: Any) -> list[str]:
    if isinstance(options, Mapping):
        return [f"{key} {value}" for key, value in options.items()]
    if isinstance(options, list):
        return [str(option) for option in options if str(option)]
    return []


def _question_and_options_text(question: Mapping[str, Any]) -> str:
    options = question.get("options")
    option_text: str
    if isinstance(options, Mapping):
        option_text = " ".join(str(value) for value in options.values())
    elif isinstance(options, list):
        option_text = " ".join(str(value) for value in options)
    else:
        option_text = str(options or "")
    return f"{question.get('question', '')} {option_text}"


def _stage_artifact(stage: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prompts": dict(stage.get("prompts", {})),
        "usage": dict(stage.get("usage", {})),
        "raw_content": dict(stage.get("raw_content", {})),
        "parsed": dict(stage.get("parsed", {})),
    }


def _completion_usage(completion: Any) -> dict[str, int]:
    prompt_tokens = int(completion.prompt_tokens)
    completion_tokens = int(completion.completion_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _total_usage(usage: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    return {
        "prompt_tokens": sum(int(item.get("prompt_tokens", 0)) for item in usage.values()),
        "completion_tokens": sum(int(item.get("completion_tokens", 0)) for item in usage.values()),
        "total_tokens": sum(int(item.get("total_tokens", 0)) for item in usage.values()),
    }


def _completion_items_or_empty(completion: Any) -> list[dict[str, Any]]:
    payload = getattr(completion, "payload", {}) or {}
    items = payload.get("items")
    if isinstance(items, list):
        return [dict(item) for item in items if isinstance(item, Mapping)]
    if isinstance(payload, Mapping) and "qid" in payload:
        return [dict(payload)]
    return []


if __name__ == "__main__":
    main()
