from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.schemas import TokenUsage
from reading_router.final_prompts import (
    GENERAL_REASONING,
    MCQ_ROUTE,
    MULTI_ROUTE_V2,
    TF_READING,
    build_prompt,
    route_question,
)


FINAL_ROUTER_SYSTEM_PROMPT = "你是读题证据路由助手。只输出JSON数组，不答题。"
PIPELINE_ROUTES = (TF_READING, MULTI_ROUTE_V2, MCQ_ROUTE, GENERAL_REASONING)


@dataclass(slots=True)
class FinalRouteSummary:
    route: str
    question_count: int = 0
    valid_items: int = 0
    invalid_items: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def avg_total_tokens(self) -> float:
        if self.question_count == 0:
            return 0.0
        return self.usage.total_tokens / self.question_count


@dataclass(frozen=True, slots=True)
class RoutedQuestion:
    index: int
    route: str
    question: dict[str, Any]


ClientFactory = Callable[[], Any]


def run_final_router(
    *,
    input_path: Path,
    out_dir: Path,
    client_factory: ClientFactory,
    model: str,
    batch_size: int = 10,
    limit: int | None = None,
    qwen_extra_body: Mapping[str, Any] | None = None,
) -> list[FinalRouteSummary]:
    questions = load_questions(input_path, limit=limit)
    routed = route_questions(questions)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summaries = {route: FinalRouteSummary(route=route) for route in PIPELINE_ROUTES}
    client = client_factory()

    for route in PIPELINE_ROUTES:
        route_items = [item for item in routed if item.route == route]
        summaries[route].question_count = len(route_items)
        for batch_index, batch in enumerate(_batched(route_items, batch_size), start=1):
            try:
                prompt = build_prompt(route, [item.question for item in batch])
                completion = client.raw_completion(system=FINAL_ROUTER_SYSTEM_PROMPT, prompt=prompt)
                summaries[route].usage.add(completion.prompt_tokens, completion.completion_tokens)
                items = _payload_items(getattr(completion, "payload", {}) or {})
                batch_rows = _rows_for_batch(
                    route=route,
                    batch=batch,
                    batch_index=batch_index,
                    items=items,
                    prompt_tokens=completion.prompt_tokens,
                    completion_tokens=completion.completion_tokens,
                    raw_content=completion.raw_content,
                    reasoning_content=getattr(completion, "reasoning_content", ""),
                    valid=True,
                )
            except Exception as exc:
                batch_rows = _error_rows_for_batch(
                    route=route,
                    batch=batch,
                    batch_index=batch_index,
                    error=str(exc),
                )
            for row in batch_rows:
                if row["valid"]:
                    summaries[route].valid_items += 1
                else:
                    summaries[route].invalid_items += 1
            rows.extend(batch_rows)

    rows.sort(key=lambda row: int(row["input_index"]))
    _write_jsonl(out_dir / "reading_routes.jsonl", rows)
    _write_jsonl(out_dir / "model_outputs.jsonl", _model_output_rows(rows))
    _write_route_summary(summaries.values(), out_dir / "route_summary.csv")
    _write_run_meta(
        path=out_dir / "run_meta.json",
        input_path=input_path,
        model=model,
        batch_size=batch_size,
        question_count=len(questions),
        qwen_extra_body=qwen_extra_body or {},
        summaries=summaries.values(),
    )
    return list(summaries.values())


def load_questions(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("questions") or payload.get("items") or []
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(f"Unsupported question payload in {path}")
    if not isinstance(rows, list):
        raise ValueError(f"Question payload must be a list in {path}")
    questions = [_normalize_question(row) for row in rows if isinstance(row, Mapping)]
    return questions[:limit] if limit is not None else questions


def route_questions(
    questions: Sequence[Mapping[str, Any]],
) -> list[RoutedQuestion]:
    routed: list[RoutedQuestion] = []
    for index, question in enumerate(questions):
        normalized = dict(question)
        route = route_question(normalized)
        routed.append(RoutedQuestion(index=index, route=route, question=normalized))
    return routed


def format_final_summary(summaries: Sequence[FinalRouteSummary]) -> str:
    lines = ["final reading_router token summary:"]
    for summary in summaries:
        lines.append(
            "  "
            f"route={summary.route} "
            f"questions={summary.question_count} valid={summary.valid_items} invalid={summary.invalid_items} "
            f"prompt={summary.usage.prompt_tokens} completion={summary.usage.completion_tokens} "
            f"total={summary.usage.total_tokens} avg_total={summary.avg_total_tokens:.2f}"
        )
    return "\n".join(lines)


def _normalize_question(row: Mapping[str, Any]) -> dict[str, Any]:
    if "qid" not in row or "question" not in row:
        raise ValueError(f"Question row must contain qid and question: {row}")
    question = dict(row)
    question["qid"] = str(question["qid"])
    question["question"] = str(question["question"])
    question["answer_format"] = str(question.get("answer_format", ""))
    if "doc_ids" in question and not isinstance(question["doc_ids"], list):
        question["doc_ids"] = [str(question["doc_ids"])]
    return question


def _payload_items(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("items", "results", "questions"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    if "qid" in payload:
        return [dict(payload)]
    return []


def _rows_for_batch(
    *,
    route: str,
    batch: Sequence[RoutedQuestion],
    batch_index: int,
    items: Sequence[Mapping[str, Any]],
    prompt_tokens: int,
    completion_tokens: int,
    raw_content: str,
    reasoning_content: str,
    valid: bool,
) -> list[dict[str, Any]]:
    by_qid = {str(item.get("qid", "")): dict(item) for item in items if item.get("qid")}
    rows: list[dict[str, Any]] = []
    for routed in batch:
        question = routed.question
        qid = str(question["qid"])
        item = by_qid.get(qid, {})
        schema_issues = _schema_issues(route=route, qid=qid, item=item)
        rows.append(
            {
                "input_index": routed.index,
                "route": route,
                "batch_index": batch_index,
                "qid": qid,
                "domain": question.get("domain", ""),
                "split": question.get("split", ""),
                "answer_format": question.get("answer_format", ""),
                "type": question.get("type", question.get("raw_type", "")),
                "raw_type": question.get("raw_type", ""),
                "doc_ids": question.get("doc_ids", []),
                "question": question.get("question", ""),
                "router_output": item,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "valid": valid,
                "issues": [],
                "schema_issues": schema_issues,
                "raw_content": raw_content,
                "reasoning_content": reasoning_content,
            }
        )
    return rows


def _error_rows_for_batch(
    *,
    route: str,
    batch: Sequence[RoutedQuestion],
    batch_index: int,
    error: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for routed in batch:
        question = routed.question
        rows.append(
            {
                "input_index": routed.index,
                "route": route,
                "batch_index": batch_index,
                "qid": str(question["qid"]),
                "domain": question.get("domain", ""),
                "split": question.get("split", ""),
                "answer_format": question.get("answer_format", ""),
                "type": question.get("type", question.get("raw_type", "")),
                "raw_type": question.get("raw_type", ""),
                "doc_ids": question.get("doc_ids", []),
                "question": question.get("question", ""),
                "router_output": {},
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "valid": False,
                "issues": ["request_failed"],
                "schema_issues": [],
                "raw_content": "",
                "reasoning_content": "",
                "error": error,
            }
        )
    return rows


def _schema_issues(*, route: str, qid: str, item: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    if not item:
        return ["missing_item"]
    if str(item.get("qid", "")) != qid:
        issues.append("qid_mismatch")
    if route == TF_READING:
        if not _has_nonempty_list_or_string(item.get("reading_chain")):
            issues.append("missing_reading_chain")
    elif route == MULTI_ROUTE_V2:
        if not _has_nonempty_list_or_string(item.get("routes")):
            issues.append("missing_routes")
        if not str(item.get("risk", "")).strip():
            issues.append("missing_risk")
    elif route == MCQ_ROUTE:
        if not str(item.get("decision", "")).strip():
            issues.append("missing_decision")
        if not _has_nonempty_list_or_string(item.get("routes")):
            issues.append("missing_routes")
        if not str(item.get("risk", "")).strip():
            issues.append("missing_risk")
    elif route == GENERAL_REASONING:
        if not _has_nonempty_list_or_string(item.get("reading_chain")):
            issues.append("missing_reading_chain")
    return issues


def _has_nonempty_list_or_string(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    return False


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _model_output_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        output = row.get("router_output") if isinstance(row.get("router_output"), Mapping) else {}
        output_rows.append({"qid": str(row.get("qid", "")), **_drop_duplicate_qid(output)})
    return output_rows


def _drop_duplicate_qid(output: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in output.items() if key != "qid"}


def _write_route_summary(summaries: Iterable[FinalRouteSummary], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "route",
                "question_count",
                "valid_items",
                "invalid_items",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "avg_total_tokens",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "route": summary.route,
                    "question_count": summary.question_count,
                    "valid_items": summary.valid_items,
                    "invalid_items": summary.invalid_items,
                    "prompt_tokens": summary.usage.prompt_tokens,
                    "completion_tokens": summary.usage.completion_tokens,
                    "total_tokens": summary.usage.total_tokens,
                    "avg_total_tokens": f"{summary.avg_total_tokens:.2f}",
                }
            )


def _write_run_meta(
    *,
    path: Path,
    input_path: Path,
    model: str,
    batch_size: int,
    question_count: int,
    qwen_extra_body: Mapping[str, Any],
    summaries: Iterable[FinalRouteSummary],
) -> None:
    route_counts = {summary.route: summary.question_count for summary in summaries}
    meta = {
        "input": str(input_path),
        "model": model,
        "batch_size": batch_size,
        "question_count": question_count,
        "qwen_extra_body": dict(qwen_extra_body),
        "route_counts": route_counts,
    }
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _batched(rows: Sequence[RoutedQuestion], batch_size: int) -> Iterable[list[RoutedQuestion]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for index in range(0, len(rows), batch_size):
        yield list(rows[index:index + batch_size])
