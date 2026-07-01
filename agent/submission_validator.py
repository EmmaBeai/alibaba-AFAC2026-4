from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from agent.schemas import Question

REQUIRED_COLUMNS = ["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"]


def validate_submission(
    answer_csv: Path | str,
    evidence_json: Path | str,
    questions: list[Question],
) -> dict[str, Any]:
    answer_path = Path(answer_csv)
    evidence_path = Path(evidence_json)
    rows, duplicate_qids, columns = _read_submission_rows(answer_path)
    summary = rows.pop("summary", None)
    expected = {question.qid: question for question in questions}
    missing_qids = sorted(set(expected) - set(rows))
    extra_qids = sorted(set(rows) - set(expected))
    invalid_answers = _invalid_answers(rows, expected)
    evidence_by_qid = _read_evidence(evidence_path)
    evidence_stats = _evidence_stats(evidence_by_qid, rows)
    token_stats = _token_stats(rows, summary)
    column_errors = [column for column in REQUIRED_COLUMNS if column not in columns]
    ok = not (
        column_errors
        or duplicate_qids
        or missing_qids
        or extra_qids
        or invalid_answers
    )
    return {
        "ok": ok,
        "answer": {
            "rows": len(rows),
            "expected_rows": len(expected),
            "missing_qids": missing_qids,
            "extra_qids": extra_qids,
            "duplicate_qids": duplicate_qids,
            "invalid_answers": invalid_answers,
            "missing_columns": column_errors,
        },
        "page": {"with_page": evidence_stats["with_page"]},
        "block_cell": {"with_block_or_cell": evidence_stats["with_block_or_cell"]},
        "citation": {
            "with_citation": evidence_stats["with_citation"],
            "missing_evidence_qids": evidence_stats["missing_evidence_qids"],
        },
        "token": token_stats,
    }


def _read_submission_rows(path: Path) -> tuple[dict[str, dict[str, str]], list[str], list[str]]:
    rows: dict[str, dict[str, str]] = {}
    duplicate_qids: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        for row in reader:
            qid = (row.get("qid") or "").strip()
            if not qid:
                continue
            if qid in rows:
                duplicate_qids.append(qid)
            rows[qid] = {key: (value or "").strip() for key, value in row.items()}
    return rows, sorted(set(duplicate_qids)), columns


def _invalid_answers(rows: dict[str, dict[str, str]], questions: dict[str, Question]) -> list[dict[str, str]]:
    invalid: list[dict[str, str]] = []
    for qid, row in rows.items():
        question = questions.get(qid)
        if question is None:
            continue
        answer = row.get("answer", "")
        option_keys = set(question.options)
        if not answer:
            invalid.append({"qid": qid, "answer": answer, "reason": "empty"})
            continue
        unknown = [value for value in answer if value not in option_keys]
        duplicate = len(set(answer)) != len(answer)
        if unknown:
            invalid.append({"qid": qid, "answer": answer, "reason": "unknown_option"})
        elif duplicate:
            invalid.append({"qid": qid, "answer": answer, "reason": "duplicate_option"})
    return invalid


def _read_evidence(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    evidence_by_qid: dict[str, list[dict[str, Any]]] = {}
    for item in payload:
        qid = str(item.get("qid", ""))
        evidence = item.get("evidence_retrieval") or []
        evidence_by_qid[qid] = evidence if isinstance(evidence, list) else []
    return evidence_by_qid


def _evidence_stats(
    evidence_by_qid: dict[str, list[dict[str, Any]]],
    rows: dict[str, dict[str, str]],
) -> dict[str, Any]:
    with_page = 0
    with_block_or_cell = 0
    with_citation = 0
    missing_evidence_qids: list[str] = []
    for qid in sorted(rows):
        evidence = evidence_by_qid.get(qid, [])
        if not evidence:
            missing_evidence_qids.append(qid)
            continue
        evidence_items = list(_iter_dicts(evidence))
        has_page = any(_has_any(item, ["top_page", "page", "page_number"]) for item in evidence_items)
        has_block = any(
            _has_any(item, ["top_unit_id", "unit_id", "block_id", "cell_id", "table_cell_id"])
            for item in evidence_items
        )
        has_doc = any(_has_any(item, ["top_doc_id", "doc_id", "document_id", "citation"]) for item in evidence_items)
        if has_page:
            with_page += 1
        if has_block:
            with_block_or_cell += 1
        if has_doc and (has_page or has_block):
            with_citation += 1
    return {
        "with_page": with_page,
        "with_block_or_cell": with_block_or_cell,
        "with_citation": with_citation,
        "missing_evidence_qids": missing_evidence_qids,
    }


def _has_any(item: dict[str, Any], keys: list[str]) -> bool:
    for key in keys:
        value = item.get(key)
        if value is not None and value != "" and value != []:
            return True
    return False


def _iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _iter_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_dicts(item)


def _token_stats(rows: dict[str, dict[str, str]], summary: dict[str, str] | None) -> dict[str, int]:
    row_prompt = sum(_int(row.get("prompt_tokens")) for row in rows.values())
    row_completion = sum(_int(row.get("completion_tokens")) for row in rows.values())
    row_total = sum(_int(row.get("total_tokens")) for row in rows.values())
    if summary:
        prompt = _int(summary.get("prompt_tokens"))
        completion = _int(summary.get("completion_tokens"))
        total = _int(summary.get("total_tokens"))
    else:
        prompt = row_prompt
        completion = row_completion
        total = row_total
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "row_prompt_tokens": row_prompt,
        "row_completion_tokens": row_completion,
        "row_total_tokens": row_total,
    }


def _int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0
