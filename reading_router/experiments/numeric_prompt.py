from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agent.qwen_client import DEFAULT_QWEN_MODEL, QwenPlusClient
from reading_router.experiments.prompts import COMPACT_READING_SYSTEM_PROMPT, build_reasoning_md_reading_prompt


NUMERIC_QIDS = (
    "fc_a_013",
    "ins_a_001",
    "ins_a_002",
    "ins_a_003",
    "ins_a_006",
    "ins_a_011",
    "ins_a_015",
    "ins_a_020",
    "reg_a_006",
    "reg_a_010",
    "res_a_003",
    "res_a_006",
    "res_a_010",
    "res_a_013",
    "res_a_018",
)

NUMERIC_QID_SET = frozenset(NUMERIC_QIDS)
NON_YEAR_NUMBER_RE = re.compile(r"(?<!\d)(?!(?:19|20)\d{2}(?!\d))\d+(?:\.\d+)?%?")


def build_qwen_extra_body(*, disable_thinking: bool) -> dict[str, Any]:
    if disable_thinking:
        return {"enable_thinking": False}
    return {}


def build_numeric_delta_prompt(questions: Sequence[Mapping[str, Any]]) -> str:
    payload = [_numeric_delta_input(question) for question in questions]
    return (
        "你是numeric evidence router。输入为qid、q、options。\n"
        "只读题干和选项，不答题。把选项差异压成后续找证据用的三槽。\n\n"
        "输出JSON数组:\n"
        '[{"qid":"...","delta":"...","need":"...","risk":"..."}]\n\n'
        "字段:\n"
        "delta: 选项分歧或TF待验点；写出候选值/结论，如金额、排序、可赔/不赔、适用/不适用、正误条件；不要只概括题目任务。\n"
        "need: 裁决分歧所需证据槽；写具体条款、公式、字段、免赔额、比例、时间、主体、补偿/扣减/叠加规则。\n"
        "risk: 最大误判点；写清错因，如规则混用、扣减顺序、对象归属、单位/日期、只验证部分条件。\n"
        "要求: 短句;不复述题干;不逐项解释;不答题;禁补题外事实。\n"
        f"Q={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def build_three_stage_read_q_prompt(questions: Sequence[Mapping[str, Any]]) -> str:
    payload = [_read_q_input(question) for question in questions]
    return (
        "补全每项的read_q_output。输入字段是read_q_input。\n"
        '返回JSON数组:[{"qid":"...","read_q":"..."}]\n'
        "read_q只基于read_q_input.q,自然写清最终要算/比/排什么,以及题干给出的对象、数字和条件。禁补题外事实。\n"
        f"Q={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def build_three_stage_stmt_prompt(
    questions: Sequence[Mapping[str, Any]],
    read_q_by_qid: Mapping[str, str],
) -> str:
    payload = [_stmt_input(question, read_q_by_qid) for question in questions]
    return (
        "补全每项的stmt_output。输入字段是stmt_input,里面已有q,o,read_q。\n"
        '返回JSON数组:[{"qid":"...","stmt":["A...","B...","C...","D..."]}]\n'
        "stmt按选项顺序把每个选项自然改写成待核对陈述,保留选项中的关键金额、排序、合计或比较关系;不判断真假,不代入计算。禁补题外事实。\n"
        f"Q={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def build_three_stage_read_stmt_prompt(
    read_q_by_qid: Mapping[str, str],
    stmt_by_qid: Mapping[str, Sequence[str]],
) -> str:
    qids = [qid for qid in read_q_by_qid if qid in stmt_by_qid]
    payload = [
        {
            "qid": qid,
            "read_stmt_input": {
                "read_q": read_q_by_qid[qid],
                "stmt": list(stmt_by_qid[qid]),
            },
        }
        for qid in qids
    ]
    return (
        "补全每项的read_stmt_output。输入字段是read_stmt_input,里面已有read_q,stmt。\n"
        '返回JSON数组:[{"qid":"...","read_stmt":"..."}]\n'
        "read_stmt基于read_q和stmt,自然概括这些陈述共同要求核对的计算口径、变量归属、排序/合计/扣减关系和最容易算错点。禁补题外事实。\n"
        f"Q={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def run_three_stage_numeric(questions: Sequence[Mapping[str, Any]], *, client: Any) -> dict[str, Any]:
    prompts: dict[str, str] = {}
    usage: dict[str, dict[str, int]] = {}
    raw_content: dict[str, str] = {}
    parsed: dict[str, list[dict[str, Any]]] = {}

    read_q_prompt = build_three_stage_read_q_prompt(questions)
    read_q_completion = client.raw_completion(system="计算题阅读助手。只输出JSON。", prompt=read_q_prompt)
    prompts["read_q"] = read_q_prompt
    usage["read_q"] = _completion_usage(read_q_completion)
    raw_content["read_q"] = read_q_completion.raw_content
    parsed["read_q"] = _completion_items(read_q_completion, stage="read_q")
    read_q_by_qid = _field_by_qid(parsed["read_q"], "read_q")

    stmt_prompt = build_three_stage_stmt_prompt(questions, read_q_by_qid)
    stmt_completion = client.raw_completion(system="计算题阅读助手。只输出JSON。", prompt=stmt_prompt)
    prompts["stmt"] = stmt_prompt
    usage["stmt"] = _completion_usage(stmt_completion)
    raw_content["stmt"] = stmt_completion.raw_content
    parsed["stmt"] = _completion_items(stmt_completion, stage="stmt")
    stmt_by_qid = _stmt_by_qid(parsed["stmt"])

    read_stmt_prompt = build_three_stage_read_stmt_prompt(read_q_by_qid, stmt_by_qid)
    read_stmt_completion = client.raw_completion(system="计算题阅读助手。只输出JSON。", prompt=read_stmt_prompt)
    prompts["read_stmt"] = read_stmt_prompt
    usage["read_stmt"] = _completion_usage(read_stmt_completion)
    raw_content["read_stmt"] = read_stmt_completion.raw_content
    parsed["read_stmt"] = _completion_items(read_stmt_completion, stage="read_stmt")

    return {
        "prompts": prompts,
        "usage": usage,
        "raw_content": raw_content,
        "parsed": parsed,
    }


def run_numeric_delta(questions: Sequence[Mapping[str, Any]], *, client: Any) -> dict[str, Any]:
    prompt = build_numeric_delta_prompt(questions)
    completion = client.raw_completion(system="选项差异阅读助手。只输出JSON。", prompt=prompt)
    return {
        "prompts": {"numeric_delta": prompt},
        "usage": {"numeric_delta": _completion_usage(completion)},
        "raw_content": {"numeric_delta": completion.raw_content},
        "parsed": {"numeric_delta": _completion_items(completion, stage="numeric_delta")},
    }


def run_reasoning_baseline(questions: Sequence[Mapping[str, Any]], *, client: Any) -> dict[str, Any]:
    prompt = build_reasoning_md_reading_prompt(questions)
    completion = client.raw_completion(system=COMPACT_READING_SYSTEM_PROMPT, prompt=prompt)
    return {
        "prompts": {"reasoning_md": prompt},
        "usage": {"reasoning_md": _completion_usage(completion)},
        "raw_content": {"reasoning_md": completion.raw_content},
        "parsed": {"reasoning_md": _completion_items(completion, stage="reasoning_md")},
    }


def build_prompt_artifact(
    questions: Sequence[Mapping[str, Any]],
    *,
    model: str | None = None,
    qwen_extra_body: Mapping[str, Any] | None = None,
    numeric_delta: Mapping[str, Any] | None = None,
    three_stage: Mapping[str, Any] | None = None,
    reasoning_baseline: Mapping[str, Any] | None = None,
    tf_questions: Sequence[Mapping[str, Any]] = (),
    excluded_tf_questions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "question_count": len(questions),
        "qids": [str(question.get("qid", "")) for question in questions],
        "tf_qids": [str(question.get("qid", "")) for question in tf_questions],
    }
    if excluded_tf_questions:
        artifact["excluded_tf_qids"] = [str(question.get("qid", "")) for question in excluded_tf_questions]
    if model:
        artifact["model"] = model
    if qwen_extra_body:
        artifact["qwen_extra_body"] = dict(qwen_extra_body)
    if numeric_delta:
        artifact["numeric_delta"] = {
            "prompts": dict(numeric_delta.get("prompts", {})),
            "usage": dict(numeric_delta.get("usage", {})),
            "raw_content": dict(numeric_delta.get("raw_content", {})),
            "parsed": dict(numeric_delta.get("parsed", {})),
        }
    if three_stage:
        artifact["three_stage"] = {
            "prompts": dict(three_stage.get("prompts", {})),
            "usage": dict(three_stage.get("usage", {})),
            "raw_content": dict(three_stage.get("raw_content", {})),
            "parsed": dict(three_stage.get("parsed", {})),
        }
    if reasoning_baseline:
        artifact["reasoning_baseline"] = {
            "prompts": dict(reasoning_baseline.get("prompts", {})),
            "usage": dict(reasoning_baseline.get("usage", {})),
            "raw_content": dict(reasoning_baseline.get("raw_content", {})),
            "parsed": dict(reasoning_baseline.get("parsed", {})),
        }
    return artifact


def write_prompt_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def select_numeric_questions(questions: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [question for question in questions if str(question.get("qid", "")) in NUMERIC_QID_SET]


def select_mcq_numeric_q_options_questions(questions: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        question
        for question in questions
        if str(question.get("answer_format", "")).lower() == "mcq"
        and has_non_year_number_in_q_options(question)
    ]


def select_numeric_non_tf_questions(questions: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [question for question in select_numeric_questions(questions) if not is_tf_question(question)]


def select_numeric_tf_questions(questions: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [question for question in select_numeric_questions(questions) if is_tf_question(question)]


def is_tf_question(question: Mapping[str, Any]) -> bool:
    return str(question.get("answer_format", "")).lower() == "tf"


def has_non_year_number_in_q_options(question: Mapping[str, Any]) -> bool:
    return bool(NON_YEAR_NUMBER_RE.search(_question_and_options_text(question)))


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


def _read_q_input(question: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "qid": str(question.get("qid", "")),
        "read_q_input": {
            "q": str(question.get("question", "")),
        },
    }


def _stmt_input(question: Mapping[str, Any], read_q_by_qid: Mapping[str, str]) -> dict[str, Any]:
    qid = str(question.get("qid", ""))
    return {
        "qid": qid,
        "stmt_input": {
            "q": str(question.get("question", "")),
            "o": _compact_options(question.get("options")),
            "read_q": read_q_by_qid.get(qid, ""),
        },
    }


def _numeric_delta_input(question: Mapping[str, Any]) -> dict[str, Any]:
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
    if isinstance(options, Mapping):
        option_text = " ".join(str(value) for value in options.values())
    elif isinstance(options, list):
        option_text = " ".join(str(value) for value in options)
    else:
        option_text = str(options or "")
    return f"{question.get('question', '')} {option_text}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 15-question numeric delta routing against fixed reasoning.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/questions.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/reading_router_numeric_prompt/delta_compare_15.json"),
    )
    parser.add_argument("--run-max", action="store_true", help="Call qwen max and print returned token usage.")
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--disable-thinking", action="store_true", help="Pass enable_thinking=false to Qwen.")
    parser.add_argument(
        "--mcq-numeric-q-options",
        action="store_true",
        help="Select mcq questions with non-year digits in q+options instead of the hardcoded numeric qids.",
    )
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-key-env-fallbacks", nargs="*", default=["QWEN_API_KEY"])
    parser.add_argument("--base-url-env", default="QWEN_BASE_URL")
    parser.add_argument(
        "--default-base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    args = parser.parse_args()
    questions = load_questions(args.input)
    if args.mcq_numeric_q_options:
        selected = select_mcq_numeric_q_options_questions(questions)
    else:
        selected = select_numeric_questions(questions)
    tf_questions = [question for question in selected if is_tf_question(question)]
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
        numeric_delta = run_numeric_delta(selected, client=client)
        reasoning_baseline = run_reasoning_baseline(selected, client=client)
        artifact = build_prompt_artifact(
            selected,
            model=args.model,
            qwen_extra_body=qwen_extra_body,
            numeric_delta=numeric_delta,
            reasoning_baseline=reasoning_baseline,
            tf_questions=tf_questions,
        )
        write_prompt_artifact(args.out, artifact)
        summary: dict[str, Any] = {
            "out": str(args.out),
            "model": args.model,
            "qwen_extra_body": qwen_extra_body,
            "question_count": len(selected),
            "tf_qids": artifact["tf_qids"],
            "numeric_delta_usage": artifact["numeric_delta"]["usage"],
            "reasoning_baseline_usage": artifact["reasoning_baseline"]["usage"],
            "usage_totals": {
                "numeric_delta": _total_usage(artifact["numeric_delta"]["usage"]),
                "reasoning_baseline": _total_usage(artifact["reasoning_baseline"]["usage"]),
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    numeric_delta_preview = {
        "prompts": {"numeric_delta": build_numeric_delta_prompt(selected)},
        "usage": {},
        "raw_content": {},
        "parsed": {},
    }
    reasoning_preview = {
        "prompts": {"reasoning_md": build_reasoning_md_reading_prompt(selected)},
        "usage": {},
        "raw_content": {},
        "parsed": {},
    }
    artifact = build_prompt_artifact(
        selected,
        qwen_extra_body=qwen_extra_body,
        numeric_delta=numeric_delta_preview,
        reasoning_baseline=reasoning_preview,
        tf_questions=tf_questions,
    )
    write_prompt_artifact(args.out, artifact)
    print(json.dumps({
        "out": str(args.out),
        "question_count": len(selected),
        "qids": artifact["qids"],
        "tf_qids": artifact["tf_qids"],
    }, ensure_ascii=False))
    print("\n--- numeric delta prompt ---")
    print(numeric_delta_preview["prompts"]["numeric_delta"])
    print("\n--- reasoning baseline prompt ---")
    print(reasoning_preview["prompts"]["reasoning_md"])


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


def _completion_items(completion: Any, *, stage: str) -> list[dict[str, Any]]:
    payload = getattr(completion, "payload", {}) or {}
    items = payload.get("items")
    if isinstance(items, list):
        return [dict(item) for item in items if isinstance(item, Mapping)]
    if isinstance(payload, Mapping) and "qid" in payload:
        return [dict(payload)]
    raise RuntimeError(f"Qwen {stage} stage did not return a JSON array.")


def _field_by_qid(items: Sequence[Mapping[str, Any]], field: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for item in items:
        qid = str(item.get("qid", ""))
        if qid:
            output[qid] = str(item.get(field, ""))
    return output


def _stmt_by_qid(items: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for item in items:
        qid = str(item.get("qid", ""))
        stmt = item.get("stmt")
        if qid and isinstance(stmt, list):
            output[qid] = [str(value) for value in stmt]
    return output


if __name__ == "__main__":
    main()
