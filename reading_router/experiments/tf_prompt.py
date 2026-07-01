from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agent.qwen_client import DEFAULT_QWEN_MODEL, QwenPlusClient
from reading_router.experiments.prompts import COMPACT_READING_SYSTEM_PROMPT, build_reasoning_md_reading_prompt


def is_tf_question(question: Mapping[str, Any]) -> bool:
    return str(question.get("answer_format", "")).lower() == "tf"


def select_tf_questions(questions: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [question for question in questions if is_tf_question(question)]


def build_tf_reading_prompt(questions: Sequence[Mapping[str, Any]]) -> str:
    payload = [_tf_input(question) for question in questions]
    return (
        "你是判断题读题助手。输入为qid、q。\n"
        "只读判断题题干，不答题。每题写3句reading_chain:\n"
        "1先自然说明“这是判断题，陈述说……”，保留题干里的主体、时间、数字、术语和限定词;\n"
        "2再拆出需要分别核对的条件、文档归属、比较关系、存在/未出现、且/或关系;\n"
        "3最后写最容易误读的点，如只验一半、把一份文档套到另一份、相似术语当原文、口径/单位/时间混淆。\n"
        'JSON数组:[{"qid":"...","reading_chain":["...","...","..."]}]\n'
        "禁补题外事实，不写正确/错误。\n"
        f"Q={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def build_tf_router_prompt(questions: Sequence[Mapping[str, Any]]) -> str:
    return build_tf_reading_prompt(questions)


def build_general_reasoning_placeholder(questions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "prompts": {"reasoning_md": build_reasoning_md_reading_prompt(questions)},
        "usage": {},
        "raw_content": {},
        "parsed": {},
    }


def run_tf_reading(questions: Sequence[Mapping[str, Any]], *, client: Any) -> dict[str, Any]:
    prompt = build_tf_reading_prompt(questions)
    completion = client.raw_completion(system="判断题读题助手。只输出JSON。", prompt=prompt)
    return {
        "prompts": {"tf_reading": prompt},
        "usage": {"tf_reading": _completion_usage(completion)},
        "raw_content": {"tf_reading": completion.raw_content},
        "parsed": {"tf_reading": _completion_items_or_empty(completion)},
    }


def run_tf_router(questions: Sequence[Mapping[str, Any]], *, client: Any) -> dict[str, Any]:
    return run_tf_reading(questions, client=client)


def run_general_reasoning(questions: Sequence[Mapping[str, Any]], *, client: Any) -> dict[str, Any]:
    prompt = build_reasoning_md_reading_prompt(questions)
    completion = client.raw_completion(system=COMPACT_READING_SYSTEM_PROMPT, prompt=prompt)
    return {
        "prompts": {"reasoning_md": prompt},
        "usage": {"reasoning_md": _completion_usage(completion)},
        "raw_content": {"reasoning_md": completion.raw_content},
        "parsed": {"reasoning_md": _completion_items_or_empty(completion)},
    }


def build_tf_artifact(
    questions: Sequence[Mapping[str, Any]],
    *,
    model: str | None = None,
    qwen_extra_body: Mapping[str, Any] | None = None,
    tf_reading: Mapping[str, Any] | None = None,
    reasoning_baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "tf_question_count": len(questions),
        "tf_qids": [str(question.get("qid", "")) for question in questions],
    }
    if model:
        artifact["model"] = model
    if qwen_extra_body:
        artifact["qwen_extra_body"] = dict(qwen_extra_body)
    if tf_reading:
        artifact["tf_reading"] = _stage_artifact(tf_reading)
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


def write_tf_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_qwen_extra_body(*, disable_thinking: bool) -> dict[str, Any]:
    if disable_thinking:
        return {"enable_thinking": False}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TF evidence routing against fixed general reasoning.")
    parser.add_argument("--input", type=Path, default=Path("data/questions.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/reading_router_tf_prompt/tf_compare.json"))
    parser.add_argument("--run-max", action="store_true", help="Call qwen max and print returned token usage.")
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--disable-thinking", action="store_true", help="Pass enable_thinking=false to Qwen.")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-key-env-fallbacks", nargs="*", default=["QWEN_API_KEY"])
    parser.add_argument("--base-url-env", default="QWEN_BASE_URL")
    parser.add_argument(
        "--default-base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    args = parser.parse_args()

    selected = select_tf_questions(load_questions(args.input))
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
        tf_reading = run_tf_reading(selected, client=client)
        reasoning_baseline = run_general_reasoning(selected, client=client)
    else:
        tf_reading = {
            "prompts": {"tf_reading": build_tf_reading_prompt(selected)},
            "usage": {},
            "raw_content": {},
            "parsed": {},
        }
        reasoning_baseline = build_general_reasoning_placeholder(selected)

    artifact = build_tf_artifact(
        selected,
        model=args.model if args.run_max else None,
        qwen_extra_body=qwen_extra_body,
        tf_reading=tf_reading,
        reasoning_baseline=reasoning_baseline,
    )
    write_tf_artifact(args.out, artifact)
    summary: dict[str, Any] = {
        "out": str(args.out),
        "tf_question_count": len(selected),
        "tf_qids": artifact["tf_qids"],
    }
    if args.run_max:
        summary.update(
            {
                "model": args.model,
                "qwen_extra_body": qwen_extra_body,
                "tf_reading_usage": artifact["tf_reading"]["usage"],
                "reasoning_baseline_usage": artifact["reasoning_baseline"]["usage"],
                "usage_totals": {
                    "tf_reading": _total_usage(artifact["tf_reading"]["usage"]),
                    "reasoning_baseline": _total_usage(artifact["reasoning_baseline"]["usage"]),
                },
            }
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2 if args.run_max else None))


def _tf_input(question: Mapping[str, Any]) -> dict[str, str]:
    return {
        "qid": str(question.get("qid", "")),
        "q": str(question.get("question", "")),
    }


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
