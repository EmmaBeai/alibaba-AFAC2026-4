from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AnswerRow:
    qid: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(slots=True)
class QidDiff:
    qid: str
    left_answer: str
    right_answer: str
    changed: bool
    left_total_tokens: int
    right_total_tokens: int


@dataclass(slots=True)
class RunComparison:
    left_path: Path
    right_path: Path
    same_qids: list[str]
    changed_qids: list[str]
    missing_left: list[str]
    missing_right: list[str]
    rows: list[QidDiff]


def compare_answer_files(left_path: Path | str, right_path: Path | str) -> RunComparison:
    left_file = Path(left_path)
    right_file = Path(right_path)
    left = read_answer_rows(left_file)
    right = read_answer_rows(right_file)
    qids = sorted((set(left) | set(right)) - {"summary"})
    same_qids: list[str] = []
    changed_qids: list[str] = []
    missing_left: list[str] = []
    missing_right: list[str] = []
    rows: list[QidDiff] = []
    for qid in qids:
        left_row = left.get(qid)
        right_row = right.get(qid)
        if left_row is None:
            missing_left.append(qid)
        if right_row is None:
            missing_right.append(qid)
        changed = (
            left_row is None
            or right_row is None
            or left_row.answer != right_row.answer
        )
        if changed:
            changed_qids.append(qid)
        else:
            same_qids.append(qid)
        rows.append(
            QidDiff(
                qid=qid,
                left_answer=left_row.answer if left_row else "<missing>",
                right_answer=right_row.answer if right_row else "<missing>",
                changed=changed,
                left_total_tokens=left_row.total_tokens if left_row else 0,
                right_total_tokens=right_row.total_tokens if right_row else 0,
            )
        )
    return RunComparison(
        left_path=left_file,
        right_path=right_file,
        same_qids=same_qids,
        changed_qids=changed_qids,
        missing_left=missing_left,
        missing_right=missing_right,
        rows=rows,
    )


def compare_run_dirs(left_dir: Path | str, right_dir: Path | str) -> RunComparison:
    return compare_answer_files(Path(left_dir) / "answer.csv", Path(right_dir) / "answer.csv")


def read_answer_rows(path: Path | str) -> dict[str, AnswerRow]:
    rows: dict[str, AnswerRow] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            qid = (row.get("qid") or "").strip()
            if not qid:
                continue
            rows[qid] = AnswerRow(
                qid=qid,
                answer=(row.get("answer") or "").strip(),
                prompt_tokens=_int(row.get("prompt_tokens")),
                completion_tokens=_int(row.get("completion_tokens")),
                total_tokens=_int(row.get("total_tokens")),
            )
    return rows


def write_comparison_csv(path: Path | str, comparison: RunComparison, *, include_all: bool = False) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "qid",
                "left_answer",
                "right_answer",
                "changed",
                "left_total_tokens",
                "right_total_tokens",
                "total_delta",
            ],
        )
        writer.writeheader()
        for row in comparison.rows:
            if not include_all and not row.changed:
                continue
            writer.writerow(
                {
                    "qid": row.qid,
                    "left_answer": row.left_answer,
                    "right_answer": row.right_answer,
                    "changed": row.changed,
                    "left_total_tokens": row.left_total_tokens,
                    "right_total_tokens": row.right_total_tokens,
                    "total_delta": row.right_total_tokens - row.left_total_tokens,
                }
            )


def _int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0
