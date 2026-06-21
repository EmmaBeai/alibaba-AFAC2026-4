from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AnswerRow:
    qid: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Compare two answer.csv files.")
    parser.add_argument("left", nargs="?", default="answer_pageindex_v0.csv")
    parser.add_argument("right", nargs="?", default="answer_pageindex_bm25_v0.csv")
    parser.add_argument("--out", help="Optional CSV path for changed rows.")
    parser.add_argument("--all", action="store_true", help="Include unchanged rows in --out.")
    args = parser.parse_args()

    left_path = Path(args.left)
    right_path = Path(args.right)
    left = _read_answers(left_path)
    right = _read_answers(right_path)

    qids = sorted((set(left) | set(right)) - {"summary"})
    changed = []
    missing_left = []
    missing_right = []
    same = 0
    for qid in qids:
        left_row = left.get(qid)
        right_row = right.get(qid)
        if left_row is None:
            missing_left.append(qid)
            changed.append((qid, None, right_row))
            continue
        if right_row is None:
            missing_right.append(qid)
            changed.append((qid, left_row, None))
            continue
        if left_row.answer == right_row.answer:
            same += 1
        else:
            changed.append((qid, left_row, right_row))

    print(f"left={left_path} rows={len(left) - ('summary' in left)}")
    print(f"right={right_path} rows={len(right) - ('summary' in right)}")
    print(f"same_answers={same}")
    print(f"different_or_missing={len(changed)}")
    if missing_left:
        print(f"missing_left={len(missing_left)} preview={missing_left[:10]}")
    if missing_right:
        print(f"missing_right={len(missing_right)} preview={missing_right[:10]}")
    _print_summary_delta(left.get("summary"), right.get("summary"))
    print()
    print("qid,left_answer,right_answer,left_total,right_total,total_delta")
    for qid, left_row, right_row in changed[:200]:
        print(_format_diff_line(qid, left_row, right_row))
    if len(changed) > 200:
        print(f"... truncated {len(changed) - 200} more changed rows")

    if args.out:
        _write_diff(Path(args.out), qids, left, right, include_all=args.all)
        print(f"\nwrote {args.out}")


def _read_answers(path: Path) -> dict[str, AnswerRow]:
    rows: dict[str, AnswerRow] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            qid = row["qid"].strip()
            rows[qid] = AnswerRow(
                qid=qid,
                answer=(row.get("answer") or "").strip(),
                prompt_tokens=_int(row.get("prompt_tokens")),
                completion_tokens=_int(row.get("completion_tokens")),
                total_tokens=_int(row.get("total_tokens")),
            )
    return rows


def _int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def _print_summary_delta(left: AnswerRow | None, right: AnswerRow | None) -> None:
    if left is None or right is None:
        return
    print(
        "summary_delta="
        f"prompt {right.prompt_tokens - left.prompt_tokens:+d}, "
        f"completion {right.completion_tokens - left.completion_tokens:+d}, "
        f"total {right.total_tokens - left.total_tokens:+d}"
    )


def _format_diff_line(qid: str, left: AnswerRow | None, right: AnswerRow | None) -> str:
    left_answer = left.answer if left else "<missing>"
    right_answer = right.answer if right else "<missing>"
    left_total = left.total_tokens if left else 0
    right_total = right.total_tokens if right else 0
    return f"{qid},{left_answer},{right_answer},{left_total},{right_total},{right_total - left_total:+d}"


def _write_diff(
    path: Path,
    qids: list[str],
    left: dict[str, AnswerRow],
    right: dict[str, AnswerRow],
    *,
    include_all: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "qid",
                "left_answer",
                "right_answer",
                "changed",
                "left_prompt_tokens",
                "right_prompt_tokens",
                "prompt_delta",
                "left_completion_tokens",
                "right_completion_tokens",
                "completion_delta",
                "left_total_tokens",
                "right_total_tokens",
                "total_delta",
            ],
        )
        writer.writeheader()
        for qid in qids:
            left_row = left.get(qid)
            right_row = right.get(qid)
            changed = left_row is None or right_row is None or left_row.answer != right_row.answer
            if not include_all and not changed:
                continue
            writer.writerow(
                {
                    "qid": qid,
                    "left_answer": left_row.answer if left_row else "<missing>",
                    "right_answer": right_row.answer if right_row else "<missing>",
                    "changed": changed,
                    "left_prompt_tokens": left_row.prompt_tokens if left_row else 0,
                    "right_prompt_tokens": right_row.prompt_tokens if right_row else 0,
                    "prompt_delta": (right_row.prompt_tokens if right_row else 0)
                    - (left_row.prompt_tokens if left_row else 0),
                    "left_completion_tokens": left_row.completion_tokens if left_row else 0,
                    "right_completion_tokens": right_row.completion_tokens if right_row else 0,
                    "completion_delta": (right_row.completion_tokens if right_row else 0)
                    - (left_row.completion_tokens if left_row else 0),
                    "left_total_tokens": left_row.total_tokens if left_row else 0,
                    "right_total_tokens": right_row.total_tokens if right_row else 0,
                    "total_delta": (right_row.total_tokens if right_row else 0)
                    - (left_row.total_tokens if left_row else 0),
                }
            )


if __name__ == "__main__":
    main()
