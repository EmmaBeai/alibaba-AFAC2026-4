from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.schemas import TokenUsage
from reading_router.experiments.prompts import (
    COMPACT_READING_SYSTEM_PROMPT,
    QUESTION_FIELDS,
    READING_SYSTEM_PROMPT,
    build_compact_rel_reading_prompt,
    build_compact_reading_prompt,
    build_reading_prompt,
    build_reasoning_md_reading_prompt,
)
from reading_router.experiments.routes import ModelRoute


@dataclass(slots=True)
class RouteSummary:
    route: str
    model: str
    role: str
    note: str
    question_count: int
    valid_items: int
    invalid_items: int
    usage: TokenUsage

    @property
    def avg_total_tokens(self) -> float:
        if self.question_count == 0:
            return 0.0
        return self.usage.total_tokens / self.question_count


ClientFactory = Callable[[ModelRoute], Any]


def run_comparison(
    *,
    input_path: Path,
    out_dir: Path,
    routes: Sequence[ModelRoute],
    client_factory: ClientFactory,
    batch_size: int = 10,
    limit: int | None = None,
    prompt_style: str = "full",
    output_style: str = "full",
) -> list[RouteSummary]:
    questions = load_questions(input_path, limit=limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "reading_chains.jsonl"
    summary_path = out_dir / "token_summary.csv"

    summaries: list[RouteSummary] = []
    with output_path.open("w", encoding="utf-8") as output_file:
        for route in routes:
            client = client_factory(route)
            usage = TokenUsage()
            valid_items = 0
            invalid_items = 0
            for batch_index, batch in enumerate(_batched(questions, batch_size), start=1):
                try:
                    prompt = _build_prompt(batch, prompt_style=prompt_style)
                    system = _system_prompt(prompt_style=prompt_style)
                    if hasattr(client, "raw_completion"):
                        completion = client.raw_completion(system=system, prompt=prompt)
                    else:
                        completion = client.json_completion(system=system, prompt=prompt)
                    usage.add(completion.prompt_tokens, completion.completion_tokens)
                    items = _payload_items(completion.payload)
                    rows = _rows_for_batch(
                        route=route,
                        batch=batch,
                        batch_index=batch_index,
                        items=items,
                        prompt_tokens=completion.prompt_tokens,
                        completion_tokens=completion.completion_tokens,
                        raw_content=completion.raw_content,
                        require_summary=prompt_style == "full",
                    )
                except Exception as exc:
                    rows = _error_rows_for_batch(
                        route=route,
                        batch=batch,
                        batch_index=batch_index,
                        error=str(exc),
                    )
                for row in rows:
                    if row["valid"]:
                        valid_items += 1
                    else:
                        invalid_items += 1
                    output_file.write(json.dumps(_output_row(row, output_style), ensure_ascii=False) + "\n")
            summaries.append(
                RouteSummary(
                    route=route.name,
                    model=route.model,
                    role=route.role,
                    note=route.note,
                    question_count=len(questions),
                    valid_items=valid_items,
                    invalid_items=invalid_items,
                    usage=usage,
                )
            )
    write_token_summary(summaries, summary_path)
    return summaries


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
    questions = [_normalize_question(row) for row in rows]
    return questions[:limit] if limit is not None else questions


def write_token_summary(summaries: Sequence[RouteSummary], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "route",
                "model",
                "role",
                "note",
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
                    "model": summary.model,
                    "role": summary.role,
                    "note": summary.note,
                    "question_count": summary.question_count,
                    "valid_items": summary.valid_items,
                    "invalid_items": summary.invalid_items,
                    "prompt_tokens": summary.usage.prompt_tokens,
                    "completion_tokens": summary.usage.completion_tokens,
                    "total_tokens": summary.usage.total_tokens,
                    "avg_total_tokens": f"{summary.avg_total_tokens:.2f}",
                }
            )


def format_token_summary(summaries: Sequence[RouteSummary]) -> str:
    lines = ["reading_router token summary:"]
    for summary in summaries:
        lines.append(
            "  "
            f"route={summary.route} model={summary.model} "
            f"role={summary.role} "
            f"questions={summary.question_count} valid={summary.valid_items} invalid={summary.invalid_items} "
            f"prompt={summary.usage.prompt_tokens} completion={summary.usage.completion_tokens} "
            f"total={summary.usage.total_tokens} avg_total={summary.avg_total_tokens:.2f}"
        )
    return "\n".join(lines)


def _build_prompt(batch: Sequence[Mapping[str, Any]], *, prompt_style: str) -> str:
    if prompt_style == "full":
        return build_reading_prompt(batch)
    if prompt_style == "compact":
        return build_compact_reading_prompt(batch)
    if prompt_style == "compact_rel":
        return build_compact_rel_reading_prompt(batch)
    if prompt_style == "reasoning_md":
        return build_reasoning_md_reading_prompt(batch)
    raise ValueError(f"Unknown prompt_style: {prompt_style}")


def _system_prompt(*, prompt_style: str) -> str:
    if prompt_style == "full":
        return READING_SYSTEM_PROMPT
    if prompt_style in {"compact", "compact_rel", "reasoning_md"}:
        return COMPACT_READING_SYSTEM_PROMPT
    raise ValueError(f"Unknown prompt_style: {prompt_style}")


def _output_row(row: Mapping[str, Any], output_style: str) -> dict[str, Any]:
    if output_style == "full":
        return dict(row)
    if output_style == "compact":
        return {
            "qid": row.get("qid", ""),
            "reading_chain": row.get("reading_chain", []),
        }
    raise ValueError(f"Unknown output_style: {output_style}")


def _normalize_question(row: Mapping[str, Any]) -> dict[str, Any]:
    question = {field: row[field] for field in QUESTION_FIELDS if field in row}
    if "qid" not in question or "question" not in question:
        raise ValueError(f"Question row must contain qid and question: {row}")
    question["qid"] = str(question["qid"])
    question["question"] = str(question["question"])
    if "doc_ids" in question and not isinstance(question["doc_ids"], list):
        question["doc_ids"] = [str(question["doc_ids"])]
    return question


def _payload_items(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("items", "results", "questions"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if "qid" in payload:
        return [dict(payload)]
    return []


def _rows_for_batch(
    *,
    route: ModelRoute,
    batch: Sequence[Mapping[str, Any]],
    batch_index: int,
    items: Sequence[Mapping[str, Any]],
    prompt_tokens: int,
    completion_tokens: int,
    raw_content: str,
    require_summary: bool,
) -> list[dict[str, Any]]:
    by_qid = {str(item.get("qid", "")): item for item in items if item.get("qid")}
    rows: list[dict[str, Any]] = []
    for question in batch:
        qid = str(question["qid"])
        item = by_qid.get(qid, {})
        reading_chain = _normalize_chain(item.get("reading_chain"))
        reading_summary = str(item.get("reading_summary") or "")
        notes = item.get("notes", "")
        schema_issues = _schema_issues(
            qid=qid,
            reading_chain=reading_chain,
            reading_summary=reading_summary,
            item=item,
            require_summary=require_summary,
        )
        row: dict[str, Any] = {
            "route": route.name,
            "model": route.model,
            "role": route.role,
            "route_note": route.note,
            "batch_index": batch_index,
            "qid": qid,
            "domain": question.get("domain", ""),
            "split": question.get("split", ""),
            "answer_format": question.get("answer_format", ""),
            "type": question.get("type", ""),
            "doc_ids": question.get("doc_ids", []),
            "question": question.get("question", ""),
            "reading_chain": reading_chain,
            "reading_summary": reading_summary,
            "notes": notes,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "valid": True,
            "issues": [],
            "schema_issues": schema_issues,
            "raw_content": raw_content,
        }
        rows.append(row)
    return rows


def _error_rows_for_batch(
    *,
    route: ModelRoute,
    batch: Sequence[Mapping[str, Any]],
    batch_index: int,
    error: str,
) -> list[dict[str, Any]]:
    return [
        {
            "route": route.name,
            "model": route.model,
            "role": route.role,
            "route_note": route.note,
            "batch_index": batch_index,
            "qid": str(question["qid"]),
            "domain": question.get("domain", ""),
            "split": question.get("split", ""),
            "answer_format": question.get("answer_format", ""),
            "type": question.get("type", ""),
            "doc_ids": question.get("doc_ids", []),
            "question": question.get("question", ""),
            "reading_chain": [],
            "reading_summary": "",
            "notes": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "valid": False,
            "issues": ["request_failed"],
            "schema_issues": [],
            "raw_content": "",
            "error": error,
        }
        for question in batch
    ]


def _normalize_chain(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _schema_issues(
    *,
    qid: str,
    reading_chain: Sequence[str],
    reading_summary: str,
    item: Mapping[str, Any],
    require_summary: bool = True,
) -> list[str]:
    issues: list[str] = []
    if not item:
        issues.append("missing_item")
    if str(item.get("qid", "")) != qid:
        issues.append("qid_mismatch")
    if not reading_chain:
        issues.append("missing_reading_chain")
    if require_summary and not reading_summary:
        issues.append("missing_reading_summary")
    return issues


def _batched(rows: Sequence[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for index in range(0, len(rows), batch_size):
        yield list(rows[index:index + batch_size])
