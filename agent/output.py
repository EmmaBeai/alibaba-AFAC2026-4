from __future__ import annotations

import csv
import json
from pathlib import Path

from agent.schemas import AnswerResult, TokenUsage


def load_completed_results(csv_path: Path, evidence_path: Path) -> list[AnswerResult]:
    if not csv_path.exists():
        return []
    evidence_by_qid = {}
    if evidence_path.exists():
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence_by_qid = {item["qid"]: item.get("evidence_retrieval", []) for item in payload}
    results: list[AnswerResult] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["qid"] == "summary" or not row.get("answer"):
                continue
            results.append(
                AnswerResult(
                    qid=row["qid"],
                    answer=row["answer"],
                    evidence_retrieval=evidence_by_qid.get(row["qid"], []),
                    usage=TokenUsage(
                        prompt_tokens=int(row.get("prompt_tokens") or 0),
                        completion_tokens=int(row.get("completion_tokens") or 0),
                    ),
                )
            )
    return results


def write_outputs(results: list[AnswerResult], csv_path: Path, evidence_path: Path) -> None:
    total = TokenUsage()
    for result in results:
        total.add(result.usage.prompt_tokens, result.usage.completion_tokens)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "qid": "summary",
                "answer": "",
                "prompt_tokens": total.prompt_tokens,
                "completion_tokens": total.completion_tokens,
                "total_tokens": total.total_tokens,
            }
        )
        for result in results:
            writer.writerow(
                {
                    "qid": result.qid,
                    "answer": result.answer,
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "total_tokens": result.usage.total_tokens,
                }
            )
    evidence_path.write_text(
        json.dumps(
            [
                {
                    "qid": result.qid,
                    "answer": result.answer,
                    "evidence_retrieval": result.evidence_retrieval,
                }
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
